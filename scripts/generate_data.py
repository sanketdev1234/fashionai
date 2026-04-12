"""
scripts/generate_data.py
─────────────────────────
Synthetic data generator for Size Fit Model and Trend Oracle.

NEW in v2:
  - Brand × Category × Label Size → actual cm lookup table (size_chart.json)
  - size_data.csv now includes label_size + garment_category columns
  - Users only need to know Brand + Category + Label (S/M/L/XL)
    — the system maps to real cm internally

Run:
    python scripts/generate_data.py --output data/ --n_size 100000

Outputs:
    data/size_data.csv       — training data for fit model
    data/trend_data.csv      — weekly trend signals for Prophet
    data/size_chart.json     — brand × category × label → cm lookup table
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from loguru import logger


# ─── Size chart: Brand × Category × Label → measurements in cm ───────────────
# Based on real-world brand sizing research.
# Structure: BRAND → CATEGORY → LABEL → {shoulder, chest, torso, arm}

SIZE_CHART = {
    "BrandA": {
        "tshirt": {
            "XS": {"shoulder": 38.0, "chest": 84.0,  "torso": 62.0, "arm": 56.0},
            "S":  {"shoulder": 40.0, "chest": 88.0,  "torso": 64.0, "arm": 57.5},
            "M":  {"shoulder": 42.0, "chest": 94.0,  "torso": 66.0, "arm": 59.0},
            "L":  {"shoulder": 44.5, "chest": 100.0, "torso": 68.0, "arm": 60.5},
            "XL": {"shoulder": 47.0, "chest": 107.0, "torso": 70.0, "arm": 62.0},
            "XXL":{"shoulder": 50.0, "chest": 114.0, "torso": 72.0, "arm": 63.5},
        },
        "shirt": {
            "XS": {"shoulder": 39.0, "chest": 86.0,  "torso": 70.0, "arm": 57.0},
            "S":  {"shoulder": 41.0, "chest": 90.0,  "torso": 72.0, "arm": 58.5},
            "M":  {"shoulder": 43.0, "chest": 96.0,  "torso": 74.0, "arm": 60.0},
            "L":  {"shoulder": 45.5, "chest": 102.0, "torso": 76.0, "arm": 61.5},
            "XL": {"shoulder": 48.0, "chest": 109.0, "torso": 78.0, "arm": 63.0},
            "XXL":{"shoulder": 51.0, "chest": 116.0, "torso": 80.0, "arm": 64.5},
        },
        "jacket": {
            "XS": {"shoulder": 40.0, "chest": 88.0,  "torso": 64.0, "arm": 58.0},
            "S":  {"shoulder": 42.0, "chest": 93.0,  "torso": 66.0, "arm": 59.5},
            "M":  {"shoulder": 44.0, "chest": 99.0,  "torso": 68.0, "arm": 61.0},
            "L":  {"shoulder": 46.5, "chest": 105.0, "torso": 70.0, "arm": 62.5},
            "XL": {"shoulder": 49.0, "chest": 112.0, "torso": 72.0, "arm": 64.0},
            "XXL":{"shoulder": 52.0, "chest": 119.0, "torso": 74.0, "arm": 65.5},
        },
        "dress": {
            "XS": {"shoulder": 35.0, "chest": 80.0,  "torso": 88.0, "arm": 54.0},
            "S":  {"shoulder": 37.0, "chest": 85.0,  "torso": 91.0, "arm": 55.5},
            "M":  {"shoulder": 39.0, "chest": 91.0,  "torso": 94.0, "arm": 57.0},
            "L":  {"shoulder": 41.5, "chest": 97.0,  "torso": 97.0, "arm": 58.5},
            "XL": {"shoulder": 44.0, "chest": 104.0, "torso": 100.0,"arm": 60.0},
            "XXL":{"shoulder": 47.0, "chest": 111.0, "torso": 103.0,"arm": 61.5},
        },
    },
    "BrandB": {   # runs slightly small
        "tshirt": {
            "XS": {"shoulder": 37.0, "chest": 82.0,  "torso": 61.0, "arm": 55.0},
            "S":  {"shoulder": 39.0, "chest": 86.0,  "torso": 63.0, "arm": 56.5},
            "M":  {"shoulder": 41.0, "chest": 92.0,  "torso": 65.0, "arm": 58.0},
            "L":  {"shoulder": 43.5, "chest": 98.0,  "torso": 67.0, "arm": 59.5},
            "XL": {"shoulder": 46.0, "chest": 105.0, "torso": 69.0, "arm": 61.0},
            "XXL":{"shoulder": 49.0, "chest": 112.0, "torso": 71.0, "arm": 62.5},
        },
        "shirt": {
            "XS": {"shoulder": 38.0, "chest": 84.0,  "torso": 69.0, "arm": 56.0},
            "S":  {"shoulder": 40.0, "chest": 88.0,  "torso": 71.0, "arm": 57.5},
            "M":  {"shoulder": 42.0, "chest": 94.0,  "torso": 73.0, "arm": 59.0},
            "L":  {"shoulder": 44.5, "chest": 100.0, "torso": 75.0, "arm": 60.5},
            "XL": {"shoulder": 47.0, "chest": 107.0, "torso": 77.0, "arm": 62.0},
            "XXL":{"shoulder": 50.0, "chest": 114.0, "torso": 79.0, "arm": 63.5},
        },
        "jacket": {
            "XS": {"shoulder": 39.0, "chest": 86.0,  "torso": 63.0, "arm": 57.0},
            "S":  {"shoulder": 41.0, "chest": 91.0,  "torso": 65.0, "arm": 58.5},
            "M":  {"shoulder": 43.0, "chest": 97.0,  "torso": 67.0, "arm": 60.0},
            "L":  {"shoulder": 45.5, "chest": 103.0, "torso": 69.0, "arm": 61.5},
            "XL": {"shoulder": 48.0, "chest": 110.0, "torso": 71.0, "arm": 63.0},
            "XXL":{"shoulder": 51.0, "chest": 117.0, "torso": 73.0, "arm": 64.5},
        },
        "dress": {
            "XS": {"shoulder": 34.0, "chest": 78.0,  "torso": 87.0, "arm": 53.0},
            "S":  {"shoulder": 36.0, "chest": 83.0,  "torso": 90.0, "arm": 54.5},
            "M":  {"shoulder": 38.0, "chest": 89.0,  "torso": 93.0, "arm": 56.0},
            "L":  {"shoulder": 40.5, "chest": 95.0,  "torso": 96.0, "arm": 57.5},
            "XL": {"shoulder": 43.0, "chest": 102.0, "torso": 99.0, "arm": 59.0},
            "XXL":{"shoulder": 46.0, "chest": 109.0, "torso": 102.0,"arm": 60.5},
        },
    },
    "BrandC": {   # runs large
        "tshirt": {
            "XS": {"shoulder": 40.0, "chest": 88.0,  "torso": 64.0, "arm": 57.0},
            "S":  {"shoulder": 42.0, "chest": 93.0,  "torso": 66.0, "arm": 58.5},
            "M":  {"shoulder": 44.0, "chest": 100.0, "torso": 68.0, "arm": 60.0},
            "L":  {"shoulder": 46.5, "chest": 107.0, "torso": 70.0, "arm": 61.5},
            "XL": {"shoulder": 49.0, "chest": 115.0, "torso": 72.0, "arm": 63.0},
            "XXL":{"shoulder": 52.0, "chest": 123.0, "torso": 74.0, "arm": 64.5},
        },
        "shirt": {
            "XS": {"shoulder": 41.0, "chest": 90.0,  "torso": 72.0, "arm": 58.0},
            "S":  {"shoulder": 43.0, "chest": 95.0,  "torso": 74.0, "arm": 59.5},
            "M":  {"shoulder": 45.0, "chest": 102.0, "torso": 76.0, "arm": 61.0},
            "L":  {"shoulder": 47.5, "chest": 109.0, "torso": 78.0, "arm": 62.5},
            "XL": {"shoulder": 50.0, "chest": 117.0, "torso": 80.0, "arm": 64.0},
            "XXL":{"shoulder": 53.0, "chest": 125.0, "torso": 82.0, "arm": 65.5},
        },
        "jacket": {
            "XS": {"shoulder": 42.0, "chest": 92.0,  "torso": 66.0, "arm": 59.0},
            "S":  {"shoulder": 44.0, "chest": 97.0,  "torso": 68.0, "arm": 60.5},
            "M":  {"shoulder": 46.0, "chest": 104.0, "torso": 70.0, "arm": 62.0},
            "L":  {"shoulder": 48.5, "chest": 111.0, "torso": 72.0, "arm": 63.5},
            "XL": {"shoulder": 51.0, "chest": 119.0, "torso": 74.0, "arm": 65.0},
            "XXL":{"shoulder": 54.0, "chest": 127.0, "torso": 76.0, "arm": 66.5},
        },
        "dress": {
            "XS": {"shoulder": 37.0, "chest": 84.0,  "torso": 91.0, "arm": 55.0},
            "S":  {"shoulder": 39.0, "chest": 90.0,  "torso": 94.0, "arm": 56.5},
            "M":  {"shoulder": 41.0, "chest": 97.0,  "torso": 97.0, "arm": 58.0},
            "L":  {"shoulder": 43.5, "chest": 104.0, "torso": 100.0,"arm": 59.5},
            "XL": {"shoulder": 46.0, "chest": 112.0, "torso": 103.0,"arm": 61.0},
            "XXL":{"shoulder": 49.0, "chest": 120.0, "torso": 106.0,"arm": 62.5},
        },
    },
    "BrandD": {   # runs very small / premium fit
        "tshirt": {
            "XS": {"shoulder": 36.0, "chest": 80.0,  "torso": 60.0, "arm": 54.0},
            "S":  {"shoulder": 38.0, "chest": 84.0,  "torso": 62.0, "arm": 55.5},
            "M":  {"shoulder": 40.0, "chest": 90.0,  "torso": 64.0, "arm": 57.0},
            "L":  {"shoulder": 42.5, "chest": 96.0,  "torso": 66.0, "arm": 58.5},
            "XL": {"shoulder": 45.0, "chest": 103.0, "torso": 68.0, "arm": 60.0},
            "XXL":{"shoulder": 48.0, "chest": 110.0, "torso": 70.0, "arm": 61.5},
        },
        "shirt": {
            "XS": {"shoulder": 37.0, "chest": 82.0,  "torso": 68.0, "arm": 55.0},
            "S":  {"shoulder": 39.0, "chest": 86.0,  "torso": 70.0, "arm": 56.5},
            "M":  {"shoulder": 41.0, "chest": 92.0,  "torso": 72.0, "arm": 58.0},
            "L":  {"shoulder": 43.5, "chest": 98.0,  "torso": 74.0, "arm": 59.5},
            "XL": {"shoulder": 46.0, "chest": 105.0, "torso": 76.0, "arm": 61.0},
            "XXL":{"shoulder": 49.0, "chest": 112.0, "torso": 78.0, "arm": 62.5},
        },
        "jacket": {
            "XS": {"shoulder": 38.0, "chest": 84.0,  "torso": 62.0, "arm": 56.0},
            "S":  {"shoulder": 40.0, "chest": 89.0,  "torso": 64.0, "arm": 57.5},
            "M":  {"shoulder": 42.0, "chest": 95.0,  "torso": 66.0, "arm": 59.0},
            "L":  {"shoulder": 44.5, "chest": 101.0, "torso": 68.0, "arm": 60.5},
            "XL": {"shoulder": 47.0, "chest": 108.0, "torso": 70.0, "arm": 62.0},
            "XXL":{"shoulder": 50.0, "chest": 115.0, "torso": 72.0, "arm": 63.5},
        },
        "dress": {
            "XS": {"shoulder": 33.0, "chest": 76.0,  "torso": 86.0, "arm": 52.0},
            "S":  {"shoulder": 35.0, "chest": 81.0,  "torso": 89.0, "arm": 53.5},
            "M":  {"shoulder": 37.0, "chest": 87.0,  "torso": 92.0, "arm": 55.0},
            "L":  {"shoulder": 39.5, "chest": 93.0,  "torso": 95.0, "arm": 56.5},
            "XL": {"shoulder": 42.0, "chest": 100.0, "torso": 98.0, "arm": 58.0},
            "XXL":{"shoulder": 45.0, "chest": 107.0, "torso": 101.0,"arm": 59.5},
        },
    },
    "BrandE": {   # standard international sizing
        "tshirt": {
            "XS": {"shoulder": 38.5, "chest": 85.0,  "torso": 62.0, "arm": 56.5},
            "S":  {"shoulder": 40.5, "chest": 89.0,  "torso": 64.0, "arm": 58.0},
            "M":  {"shoulder": 42.5, "chest": 95.0,  "torso": 66.0, "arm": 59.5},
            "L":  {"shoulder": 45.0, "chest": 101.0, "torso": 68.0, "arm": 61.0},
            "XL": {"shoulder": 47.5, "chest": 108.0, "torso": 70.0, "arm": 62.5},
            "XXL":{"shoulder": 50.5, "chest": 115.0, "torso": 72.0, "arm": 64.0},
        },
        "shirt": {
            "XS": {"shoulder": 39.5, "chest": 87.0,  "torso": 70.0, "arm": 57.5},
            "S":  {"shoulder": 41.5, "chest": 91.0,  "torso": 72.0, "arm": 59.0},
            "M":  {"shoulder": 43.5, "chest": 97.0,  "torso": 74.0, "arm": 60.5},
            "L":  {"shoulder": 46.0, "chest": 103.0, "torso": 76.0, "arm": 62.0},
            "XL": {"shoulder": 48.5, "chest": 110.0, "torso": 78.0, "arm": 63.5},
            "XXL":{"shoulder": 51.5, "chest": 117.0, "torso": 80.0, "arm": 65.0},
        },
        "jacket": {
            "XS": {"shoulder": 40.5, "chest": 89.0,  "torso": 64.0, "arm": 58.5},
            "S":  {"shoulder": 42.5, "chest": 94.0,  "torso": 66.0, "arm": 60.0},
            "M":  {"shoulder": 44.5, "chest": 100.0, "torso": 68.0, "arm": 61.5},
            "L":  {"shoulder": 47.0, "chest": 106.0, "torso": 70.0, "arm": 63.0},
            "XL": {"shoulder": 49.5, "chest": 113.0, "torso": 72.0, "arm": 64.5},
            "XXL":{"shoulder": 52.5, "chest": 120.0, "torso": 74.0, "arm": 66.0},
        },
        "dress": {
            "XS": {"shoulder": 35.5, "chest": 81.0,  "torso": 89.0, "arm": 54.5},
            "S":  {"shoulder": 37.5, "chest": 86.0,  "torso": 92.0, "arm": 56.0},
            "M":  {"shoulder": 39.5, "chest": 92.0,  "torso": 95.0, "arm": 57.5},
            "L":  {"shoulder": 42.0, "chest": 98.0,  "torso": 98.0, "arm": 59.0},
            "XL": {"shoulder": 44.5, "chest": 105.0, "torso": 101.0,"arm": 60.5},
            "XXL":{"shoulder": 47.5, "chest": 112.0, "torso": 104.0,"arm": 62.0},
        },
    },
}

LABELS_ORDERED = ["XS", "S", "M", "L", "XL", "XXL"]

# Body measurement population stats (cm) — ANSUR II survey
BODY_STATS = {
    "shoulder": {"mean": 43.0, "std": 3.5, "min": 32, "max": 58},
    "chest":    {"mean": 94.0, "std": 9.0, "min": 70, "max": 140},
    "torso":    {"mean": 44.0, "std": 3.0, "min": 36, "max": 56},
    "arm":      {"mean": 58.0, "std": 3.5, "min": 46, "max": 74},
}

# Trend catalogue (unchanged)
TREND_CATALOGUE = {
    "color": [
        "cobalt_blue", "dusty_rose", "olive_green", "terracotta",
        "butter_yellow", "lavender", "off_white", "chocolate_brown",
    ],
    "silhouette": [
        "oversized", "slim_fit", "relaxed", "cropped", "boxy",
        "flared", "straight_leg", "wide_leg",
    ],
    "garment_type": [
        "midi_dress", "cargo_pants", "linen_shirt", "denim_jacket",
        "knit_sweater", "wrap_top", "trench_coat", "jogger",
    ],
}


# ─── Lookup helper (used by model at inference) ───────────────────────────────

def lookup_size(brand: str, category: str, label: str) -> dict | None:
    """
    Map brand + category + label size to actual cm measurements.
    Returns dict with shoulder, chest, torso, arm keys or None if not found.
    """
    try:
        return SIZE_CHART[brand][category][label].copy()
    except KeyError:
        return None


def get_brands() -> list[str]:
    return list(SIZE_CHART.keys())


def get_categories() -> list[str]:
    categories = set()
    for brand_data in SIZE_CHART.values():
        categories.update(brand_data.keys())
    return sorted(categories)


def get_labels(brand: str, category: str) -> list[str]:
    try:
        available = list(SIZE_CHART[brand][category].keys())
        return [l for l in LABELS_ORDERED if l in available]
    except KeyError:
        return LABELS_ORDERED


# ─── Size data generator ──────────────────────────────────────────────────────

def generate_size_data(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate n training rows.
    Each row: user body measurements + brand + category + label_size
              + resolved item cm + fit label
    """
    rng = np.random.default_rng(seed)

    brands     = list(SIZE_CHART.keys())
    categories = list(list(SIZE_CHART[brands[0]].keys()))

    rows = []
    per_combo = n // (len(brands) * len(categories) * len(LABELS_ORDERED))
    per_combo = max(per_combo, 20)

    for brand in brands:
        for category in categories:
            for label in LABELS_ORDERED:
                item = SIZE_CHART[brand][category][label]

                for _ in range(per_combo):
                    # Sample user body
                    user_shoulder = float(np.clip(
                        rng.normal(BODY_STATS["shoulder"]["mean"], BODY_STATS["shoulder"]["std"]),
                        BODY_STATS["shoulder"]["min"], BODY_STATS["shoulder"]["max"]
                    ))
                    user_chest = float(np.clip(
                        rng.normal(BODY_STATS["chest"]["mean"], BODY_STATS["chest"]["std"]),
                        BODY_STATS["chest"]["min"], BODY_STATS["chest"]["max"]
                    ))
                    user_torso = float(np.clip(
                        rng.normal(BODY_STATS["torso"]["mean"], BODY_STATS["torso"]["std"]),
                        BODY_STATS["torso"]["min"], BODY_STATS["torso"]["max"]
                    ))
                    user_arm = float(np.clip(
                        rng.normal(BODY_STATS["arm"]["mean"], BODY_STATS["arm"]["std"]),
                        BODY_STATS["arm"]["min"], BODY_STATS["arm"]["max"]
                    ))

                    # Add tiny noise to item measurements (manufacturing variance)
                    item_shoulder = item["shoulder"] + rng.normal(0, 0.5)
                    item_chest    = item["chest"]    + rng.normal(0, 0.8)
                    item_torso    = item["torso"]    + rng.normal(0, 0.5)
                    item_arm      = item["arm"]      + rng.normal(0, 0.5)

                    # Clearance
                    d_shoulder = item_shoulder - user_shoulder
                    d_chest    = item_chest    - user_chest

                    # Label
                    if d_shoulder < -1.5 or d_chest < -2.0:
                        fit_label = 0  # Too Small
                    elif d_shoulder > 6.0 or d_chest > 8.0:
                        fit_label = 2  # Too Large
                    else:
                        fit_label = 1  # Perfect Fit

                    rows.append({
                        "user_shoulder":   round(user_shoulder, 1),
                        "user_chest":      round(user_chest, 1),
                        "user_torso":      round(user_torso, 1),
                        "user_arm":        round(user_arm, 1),
                        "brand":           brand,
                        "garment_category":category,
                        "label_size":      label,
                        "item_shoulder":   round(item_shoulder, 1),
                        "item_chest":      round(item_chest, 1),
                        "item_torso":      round(item_torso, 1),
                        "item_arm":        round(item_arm, 1),
                        "label":           fit_label,
                    })

    df = pd.DataFrame(rows)
    dist = df["label"].value_counts().to_dict()
    logger.info(
        f"[DataGen] Size data: {len(df)} rows | "
        f"Too Small={dist.get(0,0)}, Fit={dist.get(1,0)}, Too Large={dist.get(2,0)}"
    )
    return df


# ─── Trend data generator (unchanged) ────────────────────────────────────────

def generate_trend_data(seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    weeks = pd.date_range("2020-01-06", periods=260, freq="W")
    rows  = []

    for category, values in TREND_CATALOGUE.items():
        for val in values:
            base      = rng.uniform(10, 60)
            amplitude = rng.uniform(5, 25)
            phase     = rng.uniform(0, 2 * np.pi)
            growth    = rng.uniform(-0.02, 0.08)

            for t, ds in enumerate(weeks):
                seasonal = amplitude * np.sin(2 * np.pi * t / 52 + phase)
                trend    = base * (1 + growth) ** (t / 52)
                noise    = rng.normal(0, 3)
                y        = max(0, trend + seasonal + noise)
                rows.append({
                    "ds":       ds.strftime("%Y-%m-%d"),
                    "category": category,
                    "value":    val,
                    "y":        round(y, 2),
                })

    df = pd.DataFrame(rows)
    logger.info(
        f"[DataGen] Trend data: {len(df)} rows, "
        f"{df['value'].nunique()} trend values, {df['ds'].nunique()} weeks"
    )
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",      default="data")
    parser.add_argument("--n_size",      type=int, default=100_000)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--skip-trend",  action="store_true",
                        help="Skip trend CSV generation — use when you already have real Google Trends data")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Always save size chart JSON
    chart_path = os.path.join(args.output, "size_chart.json")
    with open(chart_path, "w") as f:
        json.dump(SIZE_CHART, f, indent=2)
    logger.success(f"✅  Saved → {chart_path}")

    # Always generate size training data
    size_df   = generate_size_data(args.n_size, seed=args.seed)
    size_path = os.path.join(args.output, "size_data.csv")
    size_df.to_csv(size_path, index=False)
    logger.success(f"✅  Saved → {size_path}  ({len(size_df):,} rows)")

    # Only generate synthetic trend data if NOT skipping
    if args.skip_trend:
        logger.info("⏭️  Skipping trend data — existing trend_data.csv kept intact")
    else:
        trend_df   = generate_trend_data(seed=args.seed)
        trend_path = os.path.join(args.output, "trend_data.csv")
        trend_df.to_csv(trend_path, index=False)
        logger.success(f"✅  Saved → {trend_path} ({len(trend_df):,} rows)")