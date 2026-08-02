
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

# ─── Fashion keyword catalogue ────────────────────────────────────────────────
# These are the search terms sent to Google Trends.
# Keep them as natural search queries — Google understands "cobalt blue fashion"
# better than just "cobalt_blue".

KEYWORD_MAP = {
    "color": {
        "cobalt_blue":     "cobalt blue fashion",
        "dusty_rose":      "dusty rose clothing",
        "olive_green":     "olive green outfit",
        "terracotta":      "terracotta color outfit",
        "butter_yellow":   "yellow clothing trend",
        "lavender":        "lavender outfit fashion",
        "off_white":       "off white clothing",
        "chocolate_brown": "chocolate brown fashion",
    },
    "silhouette": {
        "oversized":    "oversized clothing fashion",
        "slim_fit":     "slim fit outfit",
        "relaxed":      "relaxed fit clothing",
        "cropped":      "cropped top trend",
        "boxy":            "boxy shirt fashion",
        "flared":       "flared jeans fashion",
        "straight_leg": "straight leg pants fashion",
        "wide_leg":     "wide leg pants trend",
    },
    "garment_type": {
        "midi_dress":    "midi dress fashion",
        "cargo_pants":   "cargo pants trend",
        "linen_shirt":   "linen shirt outfit",
        "denim_jacket":  "denim jacket fashion",
        "knit_sweater":    "sweater fashion trend",
        "wrap_top":      "wrap top fashion",
        "trench_coat":   "trench coat fashion",
        "jogger":        "jogger pants fashion",
    },
}


# ─── Collector ────────────────────────────────────────────────────────────────

class GoogleTrendsCollector:
    """
    Fetches weekly Google Trends data for fashion keywords.

    Google Trends only allows 5 keywords per request. We fetch each
    keyword individually (with an anchor keyword to normalise scale)
    to avoid quota issues and to get consistent 0–100 scaling.
    """

    ANCHOR = "fashion"   # anchor keyword included in every batch for normalisation

    def __init__(
        self,
        geo: str = "IN",
        timeframe: str = "today 5-y",
        delay_seconds: int = 60,
    ):
        try:
            from pytrends.request import TrendReq
        except ImportError:
            logger.error("pytrends not installed. Run: pip install pytrends")
            sys.exit(1)

        self.geo       = geo
        self.timeframe = timeframe
        self.delay     = delay_seconds

        logger.info(f"[Trends] Initialising pytrends (geo={geo}, timeframe={timeframe})")
        self.pt = TrendReq(
            hl="en-US",
            tz=330,           # IST offset — change to 0 for UTC, -300 for EST
            timeout=(10, 30),
            retries=3,
            backoff_factor=0.5,
        )

    def fetch_keyword(self, keyword: str, value_name: str) -> pd.DataFrame | None:
        """
        Fetch weekly interest for a single keyword.

        Returns a DataFrame with columns: ds, y
        Returns None if the keyword has no data.
        """
        try:
            self.pt.build_payload(
                kw_list=[keyword],
                timeframe=self.timeframe,
                geo=self.geo,
                gprop="",         # web search (use 'froogle' for Shopping)
            )
            df = self.pt.interest_over_time()

            if df.empty or keyword not in df.columns:
                logger.warning(f"[Trends] No data returned for '{keyword}'")
                return None

            # interest_over_time returns weekly rows with isPartial column
            df = df[~df["isPartial"]].copy()           # drop incomplete weeks
            df = df[[keyword]].reset_index()            # keep date + keyword column
            df.columns = ["ds", "y"]
            df["ds"] = pd.to_datetime(df["ds"])
            df["y"]  = df["y"].astype(float)

            logger.success(f"[Trends] '{value_name}' → {len(df)} weeks fetched")
            return df

        except Exception as e:
            logger.error(f"[Trends] Failed to fetch '{keyword}': {e}")
            return None

    def collect_all(
        self,
        categories: list[str] | None = None,
        raw_dir: str = "data/trend_data_raw",
    ) -> pd.DataFrame:
        """
        Collect data for all keywords across all (or selected) categories.

        Returns the combined DataFrame ready to save as trend_data.csv.
        """
        Path(raw_dir).mkdir(parents=True, exist_ok=True)

        categories = categories or list(KEYWORD_MAP.keys())
        all_rows: list[pd.DataFrame] = []
        total    = sum(len(KEYWORD_MAP[c]) for c in categories if c in KEYWORD_MAP)
        fetched  = 0

        for category in categories:
            if category not in KEYWORD_MAP:
                logger.warning(f"[Trends] Unknown category '{category}' — skipping")
                continue

            logger.info(f"[Trends] ── Category: {category} ──")

            for value_name, search_term in KEYWORD_MAP[category].items():
                fetched += 1
                logger.info(
                    f"[Trends] Fetching {fetched}/{total}: "
                    f"{category}/{value_name} → '{search_term}'"
                )

                df = self.fetch_keyword(search_term, value_name)

                if df is not None and not df.empty:
                    df["category"] = category
                    df["value"]    = value_name

                    # Save raw backup
                    raw_path = Path(raw_dir) / f"{category}_{value_name}.csv"
                    df.to_csv(raw_path, index=False)

                    all_rows.append(df[["ds", "category", "value", "y"]])

                # Rate limit — Google Trends blocks if you request too fast
                if fetched < total:
                    logger.info(f"[Trends] Waiting {self.delay}s before next request…")
                    time.sleep(self.delay)

        if not all_rows:
            logger.error("[Trends] No data collected. Check your internet connection.")
            sys.exit(1)

        combined = pd.concat(all_rows, ignore_index=True)
        combined["ds"] = combined["ds"].dt.strftime("%Y-%m-%d")
        combined["y"]  = combined["y"].round(2)

        logger.success(
            f"[Trends] Collection complete — "
            f"{len(combined)} rows, "
            f"{combined['value'].nunique()} keywords, "
            f"{combined['ds'].nunique()} unique weeks"
        )
        return combined


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch real Google Trends data for FashionAI trend forecasting"
    )
    parser.add_argument(
        "--geo",
        default="IN",
        help="Country code: IN=India, US=United States, GB=UK, '\"\"'=worldwide (default: IN)",
    )
    parser.add_argument(
        "--timeframe",
        default="today 5-y",
        help="Trends timeframe (default: 'today 5-y'). Options: 'today 3-y', 'today 1-y', '2020-01-01 2024-12-31'",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["color", "silhouette", "garment_type"],
        default=None,
        help="Which categories to fetch (default: all three)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=60,
        help="Seconds to wait between API requests (default: 60). Increase if you get 429 errors.",
    )
    parser.add_argument(
        "--output",
        default="data/trend_data.csv",
        help="Output CSV path (default: data/trend_data.csv)",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/trend_data_raw",
        help="Directory for raw per-keyword CSVs (default: data/trend_data_raw)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without making any API calls",
    )
    args = parser.parse_args()

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        categories = args.categories or list(KEYWORD_MAP.keys())
        total = 0
        print("\nDRY RUN — keywords that would be fetched:\n")
        for cat in categories:
            if cat not in KEYWORD_MAP:
                continue
            print(f"  [{cat}]")
            for val, term in KEYWORD_MAP[cat].items():
                print(f"    {val:20s} → '{term}'")
                total += 1
        est_minutes = (total * args.delay) / 60
        print(f"\n  Total: {total} keywords")
        print(f"  Estimated time: ~{est_minutes:.0f} minutes "
              f"({args.delay}s delay × {total} requests)\n")
        return

    # ── Real run ──────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Backup existing synthetic data
    out_path = Path(args.output)
    if out_path.exists():
        backup = out_path.with_suffix(f".synthetic_backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
        out_path.rename(backup)
        logger.info(f"[Trends] Backed up existing CSV → {backup}")

    collector = GoogleTrendsCollector(
        geo=args.geo,
        timeframe=args.timeframe,
        delay_seconds=args.delay,
    )

    df = collector.collect_all(
        categories=args.categories,
        raw_dir=args.raw_dir,
    )

    df.to_csv(args.output, index=False)
    logger.success(f"[Trends] Saved → {args.output}  ({len(df):,} rows)")
    logger.info(
        "[Trends] Next step: retrain the Trend Oracle\n"
        "    python scripts/train.py --stage trend"
    )


if __name__ == "__main__":
    main()
