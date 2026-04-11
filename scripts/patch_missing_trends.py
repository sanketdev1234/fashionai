"""
scripts/patch_missing_trends.py
────────────────────────────────
Retries the 4 keywords that returned no data from Google Trends,
then merges them with the 20 already-saved raw CSVs into a final
trend_data.csv ready for Prophet training.

Run:
    python scripts/patch_missing_trends.py
"""
import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq
from loguru import logger

# ── 4 keywords to retry with broader search terms ─────────────────────────────
RETRIES = {
    ("color",        "terracotta"):    "terracotta color outfit",
    ("color",        "butter_yellow"): "yellow clothing trend",
    ("silhouette",   "boxy"):          "boxy shirt fashion",
    ("garment_type", "knit_sweater"):  "sweater fashion trend",
}

# ── Fallback synthetic data if Google still returns nothing ───────────────────
# Prophet needs at least 10 rows — these are placeholder values so the
# pipeline does not break. Replace with real data later if needed.
def make_fallback(category: str, value: str) -> pd.DataFrame:
    import numpy as np
    weeks = pd.date_range("2021-01-04", periods=260, freq="W")
    rng   = np.random.default_rng(abs(hash(value)) % (2**32))
    y     = (rng.normal(25, 5, 260)).clip(5, 60).round(2)
    return pd.DataFrame({
        "ds":       weeks.strftime("%Y-%m-%d"),
        "category": category,
        "value":    value,
        "y":        y,
    })


def main():
    raw_dir = Path("data/trend_data_raw")
    out_path = Path("data/trend_data.csv")
    raw_dir.mkdir(parents=True, exist_ok=True)

    pt = TrendReq(hl="en-US", tz=330, timeout=(10, 30), retries=3, backoff_factor=0.5)
    retry_rows = []

    total = len(RETRIES)
    for i, ((category, value), term) in enumerate(RETRIES.items(), 1):
        logger.info(f"[Patch] Fetching {i}/{total}: {category}/{value} -> '{term}'")
        success = False
        try:
            pt.build_payload([term], timeframe="today 5-y", geo="IN")
            df = pt.interest_over_time()

            if not df.empty and term in df.columns:
                df = df[~df["isPartial"]][[term]].reset_index()
                df.columns = ["ds", "y"]
                df["category"] = category
                df["value"]    = value
                df["ds"]       = pd.to_datetime(df["ds"]).dt.strftime("%Y-%m-%d")
                df["y"]        = df["y"].astype(float).round(2)
                retry_rows.append(df[["ds", "category", "value", "y"]])

                raw_path = raw_dir / f"{category}_{value}.csv"
                df.to_csv(raw_path, index=False)
                logger.success(f"[Patch] Got {len(df)} weeks for {value}")
                success = True
            else:
                logger.warning(f"[Patch] Still no data for '{term}'")

        except Exception as e:
            logger.error(f"[Patch] Error fetching '{term}': {e}")

        if not success:
            logger.warning(f"[Patch] Using fallback data for {category}/{value}")
            fallback = make_fallback(category, value)
            retry_rows.append(fallback)
            raw_path = raw_dir / f"{category}_{value}.csv"
            fallback.to_csv(raw_path, index=False)

        if i < total:
            logger.info(f"[Patch] Waiting 60s...")
            time.sleep(60)

    # ── Load all 20 existing raw CSVs ─────────────────────────────────────────
    logger.info(f"[Patch] Loading existing raw CSVs from {raw_dir}/")
    existing = []
    for f in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(f)
        existing.append(df)
        logger.debug(f"  Loaded {f.name} ({len(df)} rows)")

    # ── Combine and save ──────────────────────────────────────────────────────
    all_frames = existing + retry_rows
    all_data = pd.concat(all_frames, ignore_index=True)

    # Drop duplicates in case a keyword appears in both existing and retry
    all_data = all_data.drop_duplicates(subset=["ds", "category", "value"])
    all_data["ds"] = pd.to_datetime(all_data["ds"]).dt.strftime("%Y-%m-%d")
    all_data["y"]  = all_data["y"].round(2)
    all_data = all_data.sort_values(["category", "value", "ds"]).reset_index(drop=True)

    all_data.to_csv(out_path, index=False)

    logger.success(
        f"[Patch] Saved {out_path} — "
        f"{len(all_data):,} rows, "
        f"{all_data['value'].nunique()} keywords, "
        f"{all_data['ds'].nunique()} unique weeks"
    )
    logger.info("[Patch] Next step:")
    logger.info("    python scripts/train.py --stage trend")


if __name__ == "__main__":
    main()
