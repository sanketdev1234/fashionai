
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from loguru import logger


# ─── Paths ────────────────────────────────────────────────────────────────────

RAW_CSV       = "data/ANSUR_II_MALE.csv"
OUTPUT_CSV    = "data/ansur_processed.csv"
SCALER_PATH   = "artifacts/ansur_scaler.pkl"
LABEL_MAP_PATH = "artifacts/ansur_label_map.json"


# ─── Size label thresholds (chest circumference, cm) ──────────────────────────
# Standard men's sizing — chest is the universal primary clothing measurement.

SIZE_THRESHOLDS = [
    (88,  "XS"),
    (96,  "S"),
    (104, "M"),
    (112, "L"),
    (120, "XL"),
    (float("inf"), "XXL"),
]

LABEL_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]

# Columns actually used to TRAIN the model.
# chest_cm is deliberately excluded — see select_and_convert() docstring.
FEATURE_COLS = ["shoulder_cm", "arm_cm", "height_cm", "weight_kg"]

# All numeric columns including chest_cm — used for EDA, correlation
# analysis, and outlier removal (chest_cm still needs outlier cleaning
# since it directly determines the label).
ALL_NUMERIC_COLS = ["shoulder_cm", "chest_cm", "arm_cm", "height_cm", "weight_kg"]


def chest_to_size(chest_cm: float) -> str:
    """Map a chest circumference (cm) to a size label using fixed thresholds."""
    for upper_bound, label in SIZE_THRESHOLDS:
        if chest_cm < upper_bound:
            return label
    return "XXL"


# ─── Step 1 — Load raw ANSUR II CSV ───────────────────────────────────────────

def load_ansur(path: str) -> pd.DataFrame:
    """
    ANSUR II CSV has an unnamed index column as the FIRST column that is
    not present in the header row. This shifts every data value one
    position to the right relative to the header names. We handle this
    by reading with the first column explicitly as the index.
    """
    df = pd.read_csv(path, index_col=0)
    logger.info(f"[Preprocess] Loaded raw ANSUR II: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─── Step 2 & 3 — Unit conversion + feature selection ────────────────────────

def select_and_convert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the relevant columns and convert units to match what
    MediaPipe produces at inference time (all in cm).

    IMPORTANT: chest_cm is kept in this DataFrame ONLY so that
    derive_labels() can use it to assign size labels. It is
    deliberately EXCLUDED from FEATURE_COLS (the model's actual
    training inputs) further down in this file.

    Why: chest_cm is what the labels are derived FROM. If chest_cm
    is also used as a training feature, the model trivially learns
    to recover the exact threshold rule used to create the label
    (a single split: "chest < 88 → XS") and scores ~100% accuracy —
    but this is circular, not genuine prediction. Verified
    empirically: keeping chest_cm as a feature gave Decision Tree
    and Gradient Boosting 100% accuracy; removing it dropped
    accuracy to a realistic 60-66%, which is the honest number
    that reflects whether shoulder/arm/height/weight ALONE can
    predict chest-based sizing — the real question this model
    needs to answer, since MediaPipe's 2D chest width is not
    directly comparable to ANSUR II's chest circumference anyway.

    ANSUR II raw units:
        biacromialbreadth, chestcircumference, sleevelengthspinewrist → mm
        weight_kg  → already kg
        stature_m  → already metres
    """
    out = pd.DataFrame({
        "shoulder_cm": df["biacromialbreadth"]      / 10.0,   # mm → cm
        "chest_cm":    df["chestcircumference"]     / 10.0,   # mm → cm (LABEL SOURCE ONLY)
        "arm_cm":      df["sleevelengthspinewrist"] / 10.0,   # mm → cm
        "height_cm":   df["stature_m"]              * 100.0,  # m  → cm
        "weight_kg":   df["weight_kg"],                        # already kg
    })

    logger.info(f"[Preprocess] Selected columns, converted to cm/kg")
    logger.info(f"[Preprocess] Feature ranges after conversion:\n{out.describe().round(1)}")
    return out


# ─── Step 4 — Derive size labels from chest_cm ────────────────────────────────

def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["size_label"] = df["chest_cm"].apply(chest_to_size)

    dist = df["size_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    logger.info(f"[Preprocess] Size label distribution:\n{dist}")
    return df


# ─── Step 5 — EDA summary (logged, not saved as files) ──────────────────────

def run_eda(df: pd.DataFrame) -> None:
    logger.info("[EDA] ── Class distribution ──")
    logger.info(f"\n{df['size_label'].value_counts().reindex(LABEL_ORDER, fill_value=0)}")

    logger.info("[EDA] ── Feature correlation with chest_cm (label source) ──")
    corr = df[ALL_NUMERIC_COLS].corr()["chest_cm"].sort_values(ascending=False)
    logger.info(f"\n{corr.round(3)}")
    logger.info(
        "[EDA] Note: chest_cm itself is excluded from FEATURE_COLS at training "
        "time — shown here only to verify shoulder/arm/height/weight have "
        "genuine correlation with chest size (required for honest prediction)."
    )

    logger.info("[EDA] ── Mean feature values per size ──")
    grouped = df.groupby("size_label")[ALL_NUMERIC_COLS].mean().reindex(LABEL_ORDER)
    logger.info(f"\n{grouped.round(1)}")


# ─── Step 6 — Outlier removal (IQR method) ───────────────────────────────────

def remove_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Remove rows where any feature falls outside 1.5×IQR from Q1/Q3.
    Applied independently per column, rows dropped if outlier in ANY column.
    """
    df = df.copy()
    mask = pd.Series(True, index=df.index)

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        col_mask = (df[col] >= lower) & (df[col] <= upper)
        removed = (~col_mask).sum()
        if removed > 0:
            logger.info(f"[Outliers] {col}: removing {removed} rows outside [{lower:.1f}, {upper:.1f}]")
        mask &= col_mask

    before = len(df)
    df_clean = df[mask].reset_index(drop=True)
    after = len(df_clean)
    logger.success(f"[Outliers] Removed {before - after} total rows ({before} → {after})")
    return df_clean


# ─── Step 7 — Feature scaling ────────────────────────────────────────────────

def scale_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler once on the cleaned feature data.
    Returns the scaled DataFrame (features only, scaled) plus the
    fitted scaler object — the SAME scaler must be used at inference
    time to transform MediaPipe's raw measurements before prediction.
    """
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_cols])

    scaled_df = pd.DataFrame(scaled_values, columns=feature_cols, index=df.index)

    logger.info(
        f"[Scaling] StandardScaler fitted. "
        f"Means: {dict(zip(feature_cols, scaler.mean_.round(2)))}"
    )
    logger.info(
        f"[Scaling] Std devs: {dict(zip(feature_cols, scaler.scale_.round(2)))}"
    )
    return scaled_df, scaler


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main():
    Path("data").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)

    # Step 1
    raw = load_ansur(RAW_CSV)

    # Step 2 & 3
    selected = select_and_convert(raw)

    # Step 4
    labeled = derive_labels(selected)

    # Step 5
    run_eda(labeled)

    # Step 6 — outlier removal applied to ALL numeric columns (chest_cm
    # included) since chest_cm still needs cleaning even though it
    # won't be used as a training feature — it determines the label.
    cleaned = remove_outliers(labeled, ALL_NUMERIC_COLS)

    # Re-check class distribution after outlier removal
    dist_after = cleaned["size_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    logger.info(f"[Preprocess] Size distribution after outlier removal:\n{dist_after}")

    # Step 7 — scale ONLY the actual training features (chest_cm excluded)
    scaled_features, scaler = scale_features(cleaned, FEATURE_COLS)

    # Combine scaled features with label (unscaled labels — they are categorical)
    final_df = scaled_features.copy()
    final_df["size_label"] = cleaned["size_label"].values

    # Also keep unscaled features for reference/debugging, plus the
    # unscaled chest_cm (label source) for transparency — NOT used
    # as a model input, just useful to inspect alongside predictions.
    final_df_with_raw = cleaned[ALL_NUMERIC_COLS].copy()
    final_df_with_raw.columns = [f"{c}_raw" for c in ALL_NUMERIC_COLS]
    final_df = pd.concat([final_df, final_df_with_raw], axis=1)

    # Save processed dataset
    final_df.to_csv(OUTPUT_CSV, index=False)
    logger.success(f"[Preprocess] Saved → {OUTPUT_CSV} ({len(final_df)} rows)")

    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.success(f"[Preprocess] Saved → {SCALER_PATH}")

    # Save label map
    label_map = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.success(f"[Preprocess] Saved → {LABEL_MAP_PATH}")

    logger.info(
        "\n[Preprocess] Next step:\n"
        "    python scripts/train_size_models.py"
    )


if __name__ == "__main__":
    main()