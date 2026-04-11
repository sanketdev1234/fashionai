"""
scripts/generate_data.py
─────────────────────────
Synthetic data generator for both the Size Fit Model and Trend Oracle.

Run:
    python scripts/generate_data.py --output data/ --n_size 100000 --n_trend 5000

Outputs:
    data/size_data.csv   — 100k rows for fit model training
    data/trend_data.csv  — 5k rows of weekly trend signals for Prophet

──────────────────────────────────────────────────────────────────────
SIZE DATA format:
  user_shoulder, user_chest, user_torso, user_arm   ← user body measurements (cm)
  item_shoulder, item_chest, item_torso, item_arm   ← garment tech spec (cm)
  brand                                              ← brand identifier
  label  (0=Too Small, 1=Perfect Fit, 2=Too Large)  ← ground truth

  Label logic:
    Clearance = item - user.
    If shoulder_clearance < -1.5 or chest_clearance < -2: Too Small
    If shoulder_clearance > 6   or chest_clearance > 8:   Too Large
    Otherwise: Perfect Fit

TREND DATA format:
  ds        ← weekly date (2020-01-06 … today)
  category  ← "color" | "silhouette" | "garment_type"
  value     ← e.g. "cobalt_blue" | "oversized" | "midi_dress"
  y         ← demand index (0–100)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── Body measurement population statistics (cm) ───────────────────────────────
# Based on anthropometric surveys (ANSUR II)
BODY_STATS = {
    "shoulder": {"mean": 43.0, "std": 3.5, "min": 32, "max": 58},
    "chest":    {"mean": 94.0, "std": 9.0, "min": 70, "max": 140},
    "torso":    {"mean": 44.0, "std": 3.0, "min": 36, "max": 56},
    "arm":      {"mean": 58.0, "std": 3.5, "min": 46, "max": 74},
}

# ── Brand-specific grading offsets (cm above labelled size) ──────────────────
BRANDS = {
    "BrandA": {"shoulder": 2.0, "chest": 4.0, "torso": 2.0, "arm": 1.5},
    "BrandB": {"shoulder": 1.5, "chest": 3.0, "torso": 1.5, "arm": 1.0},
    "BrandC": {"shoulder": 3.0, "chest": 6.0, "torso": 3.0, "arm": 2.0},   # runs large
    "BrandD": {"shoulder": 0.5, "chest": 1.5, "torso": 0.5, "arm": 0.5},   # runs small
    "BrandE": {"shoulder": 2.5, "chest": 5.0, "torso": 2.5, "arm": 1.5},
}

# ── Trend catalogue ──────────────────────────────────────────────────────────
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


# ─── Size data generator ──────────────────────────────────────────────────────

def generate_size_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Sample user bodies from Gaussian
    user_shoulder = rng.normal(BODY_STATS["shoulder"]["mean"], BODY_STATS["shoulder"]["std"], n).clip(
        BODY_STATS["shoulder"]["min"], BODY_STATS["shoulder"]["max"])
    user_chest    = rng.normal(BODY_STATS["chest"]["mean"],    BODY_STATS["chest"]["std"],    n).clip(
        BODY_STATS["chest"]["min"],    BODY_STATS["chest"]["max"])
    user_torso    = rng.normal(BODY_STATS["torso"]["mean"],    BODY_STATS["torso"]["std"],    n).clip(
        BODY_STATS["torso"]["min"],    BODY_STATS["torso"]["max"])
    user_arm      = rng.normal(BODY_STATS["arm"]["mean"],      BODY_STATS["arm"]["std"],      n).clip(
        BODY_STATS["arm"]["min"],      BODY_STATS["arm"]["max"])

    # Sample brands
    brand_names  = list(BRANDS.keys())
    brands       = rng.choice(brand_names, n)

    # Item specs = user measurement + brand offset + random noise
    item_shoulder = np.array([
        user_shoulder[i] + BRANDS[b]["shoulder"] + rng.normal(0, 1.5)
        for i, b in enumerate(brands)
    ])
    item_chest = np.array([
        user_chest[i] + BRANDS[b]["chest"] + rng.normal(0, 2.5)
        for i, b in enumerate(brands)
    ])
    item_torso = np.array([
        user_torso[i] + BRANDS[b]["torso"] + rng.normal(0, 1.0)
        for i, b in enumerate(brands)
    ])
    item_arm = np.array([
        user_arm[i] + BRANDS[b]["arm"] + rng.normal(0, 1.0)
        for i, b in enumerate(brands)
    ])

    # Label via clearance rules
    delta_shoulder = item_shoulder - user_shoulder
    delta_chest    = item_chest    - user_chest

    labels = np.ones(n, dtype=int)  # default: Perfect Fit
    labels[(delta_shoulder < -1.5) | (delta_chest < -2.0)] = 0   # Too Small
    labels[(delta_shoulder > 6.0)  | (delta_chest > 8.0)]  = 2   # Too Large

    df = pd.DataFrame({
        "user_shoulder": np.round(user_shoulder, 1),
        "user_chest":    np.round(user_chest,    1),
        "user_torso":    np.round(user_torso,    1),
        "user_arm":      np.round(user_arm,      1),
        "item_shoulder": np.round(item_shoulder, 1),
        "item_chest":    np.round(item_chest,    1),
        "item_torso":    np.round(item_torso,    1),
        "item_arm":      np.round(item_arm,      1),
        "brand":         brands,
        "label":         labels,
    })

    dist = df["label"].value_counts().to_dict()
    logger.info(f"[DataGen] Size data: {n} rows | "
                f"Too Small={dist.get(0,0)}, Fit={dist.get(1,0)}, Too Large={dist.get(2,0)}")
    return df


# ─── Trend data generator ────────────────────────────────────────────────────

def generate_trend_data(seed: int = 42) -> pd.DataFrame:
    """Generate weekly demand signals with realistic seasonality."""
    rng   = np.random.default_rng(seed)
    weeks = pd.date_range("2020-01-06", periods=260, freq="W")  # 5 years
    rows  = []

    for category, values in TREND_CATALOGUE.items():
        for val in values:
            # Each trend value has a base level + seasonal sine wave + noise
            base       = rng.uniform(10, 60)
            amplitude  = rng.uniform(5, 25)
            phase      = rng.uniform(0, 2 * np.pi)
            growth     = rng.uniform(-0.02, 0.08)   # weekly growth rate

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
    logger.info(f"[DataGen] Trend data: {len(df)} rows, "
                f"{df['value'].nunique()} trend values, {df['ds'].nunique()} weeks")
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--n_size", type=int, default=100_000, help="Number of size rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    size_path  = os.path.join(args.output, "size_data.csv")
    trend_path = os.path.join(args.output, "trend_data.csv")

    size_df  = generate_size_data(args.n_size, seed=args.seed)
    trend_df = generate_trend_data(seed=args.seed)

    size_df.to_csv(size_path,  index=False)
    trend_df.to_csv(trend_path, index=False)

    logger.success(f"✅  Saved → {size_path}  ({len(size_df):,} rows)")
    logger.success(f"✅  Saved → {trend_path} ({len(trend_df):,} rows)")
