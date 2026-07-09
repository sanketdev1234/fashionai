"""
src/models/size_fit_model.py
─────────────────────────────
Size Predictor — ANSUR II trained classifier (v3).

Replaces the brand/category/label-size lookup approach entirely.
This version predicts a clothing size label (XS/S/M/L/XL/XXL)
directly from body measurements — no brand, no garment category,
no synthetic size chart.

Pipeline:
    MediaPipe body scan → shoulder_width_cm, arm_length_cm
    User input           → height_cm, weight_kg
            ↓
    StandardScaler.transform()  (fitted on ANSUR II, loaded from disk)
            ↓
    Best model (Logistic Regression, selected by train_size_models.py)
            ↓
    Predicted size label + per-class probability (confidence)

Why chest_width_cm from MediaPipe is NOT used as a model input:
    The model was trained on ANSUR II's shoulder/arm/height/weight
    ONLY — chest circumference was deliberately excluded from
    training features (see scripts/preprocess_ansur.py) because
    it is what the labels were derived from, and including it
    caused models to trivially recover the threshold rule rather
    than learn genuine body-proportion patterns. MediaPipe's
    chest_width_cm (2D frontal width) is also not directly
    comparable to ANSUR II's chest circumference (full wrap
    measurement), so it would need a separate conversion model
    to be usable here — out of scope for this version.

Artifacts loaded (produced by scripts/preprocess_ansur.py and
scripts/train_size_models.py):
    artifacts/best_size_model.pkl    — trained classifier
    artifacts/ansur_scaler.pkl       — fitted StandardScaler
    artifacts/ansur_label_map.json   — {label: index} mapping
    artifacts/size_model_name.txt    — human-readable model name
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ─── Constants ────────────────────────────────────────────────────────────────

FEATURE_ORDER = ["shoulder_cm", "arm_cm", "height_cm", "weight_kg"]
LABEL_ORDER   = ["XS", "S", "M", "L", "XL", "XXL"]

# Human-readable guidance shown alongside the predicted label —
# helps users unfamiliar with abstract size codes.
SIZE_DESCRIPTIONS = {
    "XS":  "Extra Small — chest typically under 88 cm",
    "S":   "Small — chest typically 88-96 cm",
    "M":   "Medium — chest typically 96-104 cm",
    "L":   "Large — chest typically 104-112 cm",
    "XL":  "Extra Large — chest typically 112-120 cm",
    "XXL": "Double Extra Large — chest typically over 120 cm",
}


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SizeRequest:
    """
    What the caller provides.

    shoulder_cm and arm_cm come from the MediaPipe body scan
    (/fit/scan endpoint). height_cm and weight_kg are entered
    manually by the user — MediaPipe cannot measure these from
    a single photo without a calibrated reference object.
    """
    shoulder_cm: float
    arm_cm:      float
    height_cm:   float
    weight_kg:   float


@dataclass
class SizeVerdict:
    predicted_size:    str                  # e.g. "L"
    confidence:        float                # probability of predicted_size, 0-1
    description:       str                  # human-readable size description
    all_probabilities: dict[str, float]      # full probability distribution
    message:           str                  # full sentence for display


# ─── Main model ───────────────────────────────────────────────────────────────

class SizeFitModel:
    """
    Loads the pre-trained ANSUR II classifier + scaler and serves
    predictions. Training happens entirely offline via
    scripts/preprocess_ansur.py and scripts/train_size_models.py —
    this class only does inference.
    """

    MODEL_PATH      = "artifacts/best_size_model.pkl"
    SCALER_PATH     = "artifacts/ansur_scaler.pkl"
    LABEL_MAP_PATH  = "artifacts/ansur_label_map.json"
    MODEL_NAME_PATH = "artifacts/size_model_name.txt"

    def __init__(self, cfg: dict | None = None):
        self.cfg            = cfg or {}
        self.model           = None
        self.scaler          = None
        self.label_map:     dict[str, int] = {}
        self.inv_label_map: dict[int, str] = {}
        self.model_name      = "unknown"
        self._is_loaded      = False

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load all artifacts produced by the offline training pipeline.
        Raises FileNotFoundError with a clear message if the
        pipeline has not been run yet.
        """
        required = [
            (self.MODEL_PATH, "trained model"),
            (self.SCALER_PATH, "scaler"),
            (self.LABEL_MAP_PATH, "label map"),
        ]
        for path_str, label in required:
            if not Path(path_str).exists():
                raise FileNotFoundError(
                    f"[SizeFit] {label} not found at {path_str}. "
                    f"Run: python scripts/preprocess_ansur.py && "
                    f"python scripts/train_size_models.py"
                )

        with open(self.MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        with open(self.SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        with open(self.LABEL_MAP_PATH) as f:
            self.label_map = json.load(f)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        if Path(self.MODEL_NAME_PATH).exists():
            self.model_name = Path(self.MODEL_NAME_PATH).read_text().strip()

        self._is_loaded = True
        logger.info(
            f"[SizeFit] Loaded '{self.model_name}' model + scaler + label map "
            f"({len(self.label_map)} classes)"
        )

    # ── Feature engineering ───────────────────────────────────────────────────

    @staticmethod
    def _build_features(req: SizeRequest) -> np.ndarray:
        """
        Build the 4-feature vector in the EXACT order the scaler
        and model were fitted on. Order mismatches silently produce
        wrong predictions — FEATURE_ORDER is the single source of
        truth shared with preprocess_ansur.py and train_size_models.py.
        """
        row = {
            "shoulder_cm": req.shoulder_cm,
            "arm_cm":      req.arm_cm,
            "height_cm":   req.height_cm,
            "weight_kg":   req.weight_kg,
        }
        ordered = [row[col] for col in FEATURE_ORDER]
        return np.array([ordered], dtype="float64")  # shape (1, 4)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, req: SizeRequest) -> SizeVerdict:
        if not self._is_loaded:
            raise RuntimeError("Call load() before predict().")

        X_raw    = self._build_features(req)
        X_scaled = self.scaler.transform(X_raw)

        raw_pred = self.model.predict(X_scaled)[0]
        if isinstance(raw_pred, (np.integer, int)):
            predicted_label = self.inv_label_map.get(int(raw_pred), str(raw_pred))
        else:
            predicted_label = str(raw_pred)

        # Per-class probabilities, if the model supports it
        all_probs: dict[str, float] = {}
        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            proba   = self.model.predict_proba(X_scaled)[0]
            classes = list(self.model.classes_)
            for cls, p in zip(classes, proba):
                cls_label = (
                    self.inv_label_map.get(int(cls), str(cls))
                    if isinstance(cls, (np.integer, int))
                    else str(cls)
                )
                all_probs[cls_label] = round(float(p), 4)
            confidence = all_probs.get(predicted_label, 1.0)

        description = SIZE_DESCRIPTIONS.get(predicted_label, "")
        message = (
            f"Based on your body scan and measurements, your recommended "
            f"size is {predicted_label} ({confidence:.0%} confidence). "
            f"{description}"
        )

        return SizeVerdict(
            predicted_size=predicted_label,
            confidence=round(confidence, 4),
            description=description,
            all_probabilities=all_probs,
            message=message,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_label_order(self) -> list[str]:
        return LABEL_ORDER
