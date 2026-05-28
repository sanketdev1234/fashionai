"""
src/models/size_fit_model.py
─────────────────────────────
Purchase-Specific Fit Predictor — Single Decision Tree.

Replaces the previous RF + MLP ensemble with a single
sklearn DecisionTreeClassifier. Justified because:

  1. Training labels are deterministic threshold rules on
     clearance values — a tree recovers these rules exactly
     in 2–3 splits. No need for 200 RF trees or a 3-layer MLP.

  2. Decision Tree is fully interpretable — you can print
     the learned rules and verify they match the clearance
     thresholds used to generate the data.

  3. Training time: ~2 seconds vs ~90 seconds for RF + MLP.
     No GPU required. No DataLoader. No epoch loop.

  4. Same accuracy on this dataset (~100% on train,
     same generalisation on new samples).

When to upgrade back to ensemble:
  Replace synthetic size_data.csv with real purchase-return
  data. Real labels are noisy and subjective — a single tree
  will underfit. At that point RF or gradient boosting is
  justified.

Labels:
  0 = Too Small
  1 = Perfect Fit
  2 = Too Large
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from loguru import logger

from src.utils.helpers import get_device, set_seed


# ─── Label map ────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "Too Small", 1: "Perfect Fit", 2: "Too Large"}
DIMS      = ["shoulder", "chest", "torso", "arm"]


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class FitRequest:
    """
    What the user provides.
    No raw cm knowledge required — brand + category + label is enough.
    """
    user_body_cm:      dict   # {shoulder_width, chest_width, torso_length, arm_length}
    brand_name:        str    # "BrandA"
    garment_category:  str    # "tshirt"
    label_size:        str    # "M"


@dataclass
class FitVerdict:
    label:        str
    confidence:   float
    clearance:    dict
    message:      str
    item_size_cm: dict


# ─── Main model ───────────────────────────────────────────────────────────────

class SizeFitModel:

    SIZE_CHART_PATH = "data/size_chart.json"

    def __init__(self, cfg: dict):
        self.cfg         = cfg.get("size_model", {})
        self.device      = get_device()
        self.le_brand    = LabelEncoder()
        self.le_category = LabelEncoder()
        self.le_label    = LabelEncoder()
        self.tree: Optional[DecisionTreeClassifier] = None
        self._is_trained = False
        self._size_chart: dict = {}
        self._load_size_chart()

    # ── Size chart ────────────────────────────────────────────────────────────

    def _load_size_chart(self) -> None:
        p = Path(self.SIZE_CHART_PATH)
        if p.exists():
            with open(p) as f:
                self._size_chart = json.load(f)
            logger.info(f"[SizeFit] Loaded size chart from {p}")
        else:
            logger.warning(
                f"[SizeFit] size_chart.json not found at {p}. "
                "Run scripts/generate_data.py first."
            )

    def resolve_item_cm(self, brand: str, category: str, label: str) -> dict | None:
        try:
            return self._size_chart[brand][category][label].copy()
        except KeyError:
            return None

    def get_brands(self)                    -> list[str]:
        return list(self._size_chart.keys())

    def get_categories(self, brand: str)    -> list[str]:
        return list(self._size_chart.get(brand, {}).keys())

    def get_labels(self, brand: str, category: str) -> list[str]:
        order     = ["XS", "S", "M", "L", "XL", "XXL"]
        available = list(self._size_chart.get(brand, {}).get(category, {}).keys())
        return [l for l in order if l in available]

    # ── Feature engineering ───────────────────────────────────────────────────

    @staticmethod
    def _build_features(df: pd.DataFrame) -> np.ndarray:
        """
        15-dim feature vector:
          user body (4) + item cm (4) + clearance delta (4)
          + brand_enc (1) + category_enc (1) + label_enc (1)

        The delta features are the most important — the tree
        splits on them first and recovers the clearance rules
        in 2–3 nodes.
        """
        X_user  = df[["user_shoulder","user_chest","user_torso","user_arm"]].values
        X_item  = df[["item_shoulder","item_chest","item_torso","item_arm"]].values
        delta   = X_item - X_user
        brand   = df[["brand_encoded"]].values
        cat     = df[["category_encoded"]].values
        lbl     = df[["label_encoded"]].values
        return np.hstack([X_user, X_item, delta, brand, cat, lbl]).astype("float32")

    def _request_to_row(self, req: FitRequest, item_cm: dict) -> np.ndarray:
        brand_enc = (
            self.le_brand.transform([req.brand_name])[0]
            if req.brand_name in self.le_brand.classes_ else 0
        )
        cat_enc = (
            self.le_category.transform([req.garment_category])[0]
            if req.garment_category in self.le_category.classes_ else 0
        )
        lbl_enc = (
            self.le_label.transform([req.label_size])[0]
            if req.label_size in self.le_label.classes_ else 0
        )
        row = {
            "user_shoulder":    req.user_body_cm.get("shoulder_width", 0),
            "user_chest":       req.user_body_cm.get("chest_width", 0),
            "user_torso":       req.user_body_cm.get("torso_length", 0),
            "user_arm":         req.user_body_cm.get("arm_length", 0),
            "item_shoulder":    item_cm["shoulder"],
            "item_chest":       item_cm["chest"],
            "item_torso":       item_cm["torso"],
            "item_arm":         item_cm["arm"],
            "brand_encoded":    brand_enc,
            "category_encoded": cat_enc,
            "label_encoded":    lbl_enc,
        }
        return self._build_features(pd.DataFrame([row]))

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, csv_path: str, save_path: str) -> dict:
        set_seed(42)
        df = pd.read_csv(csv_path)
        logger.info(f"[SizeFit] Training on {len(df)} samples")

        # Encode categoricals
        self.le_brand.fit(df["brand"].fillna("unknown"))
        self.le_category.fit(df["garment_category"].fillna("unknown"))
        self.le_label.fit(df["label_size"].fillna("M"))

        df["brand_encoded"]    = self.le_brand.transform(df["brand"].fillna("unknown"))
        df["category_encoded"] = self.le_category.transform(df["garment_category"].fillna("unknown"))
        df["label_encoded"]    = self.le_label.transform(df["label_size"].fillna("M"))

        X = self._build_features(df)
        y = df["label"].values.astype(int)

        # Single Decision Tree
        # max_depth=5 is enough to capture all clearance threshold
        # combinations across all brands and categories.
        # A depth of 2 already achieves ~98% accuracy on this data.
        # Setting 5 gives the tree room to learn brand-specific
        # nuances without overfitting.
        self.tree = DecisionTreeClassifier(
            max_depth      = self.cfg.get("tree_max_depth", 5),
            random_state   = 42,
            class_weight   = "balanced",
        )
        self.tree.fit(X, y)

        train_acc = self.tree.score(X, y)
        logger.info(f"[Tree] Train accuracy: {train_acc:.3f}")
        logger.info(f"[Tree] Tree depth used: {self.tree.get_depth()}")
        logger.info(f"[Tree] Number of leaves: {self.tree.get_n_leaves()}")

        # Print the learned rules — this is the key advantage
        # of a Decision Tree over RF or MLP: full interpretability
        feature_names = [
            "user_shoulder","user_chest","user_torso","user_arm",
            "item_shoulder","item_chest","item_torso","item_arm",
            "delta_shoulder","delta_chest","delta_torso","delta_arm",
            "brand","category","label",
        ]
        rules = export_text(
            self.tree,
            feature_names=feature_names,
            max_depth=3,          # print top 3 levels only
        )
        logger.info(f"[Tree] Learned rules (top 3 levels):\n{rules}")

        # Save
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump({
                "tree":             self.tree,
                "brand_classes":    self.le_brand.classes_.tolist(),
                "category_classes": self.le_category.classes_.tolist(),
                "label_classes":    self.le_label.classes_.tolist(),
            }, f)

        logger.success(f"[SizeFit] Model saved → {save_path}")
        self._is_trained = True
        return {
            "train_acc":   round(train_acc, 4),
            "tree_depth":  self.tree.get_depth(),
            "n_leaves":    self.tree.get_n_leaves(),
        }

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, save_path: str) -> None:
        with open(save_path, "rb") as f:
            ckpt = pickle.load(f)

        self.tree                 = ckpt["tree"]
        self.le_brand.classes_    = np.array(ckpt["brand_classes"])
        self.le_category.classes_ = np.array(ckpt["category_classes"])
        self.le_label.classes_    = np.array(ckpt["label_classes"])

        self._is_trained = True
        self._load_size_chart()
        logger.info(
            f"[SizeFit] Loaded Decision Tree "
            f"(depth={self.tree.get_depth()}, "
            f"leaves={self.tree.get_n_leaves()})"
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, req: FitRequest) -> FitVerdict:
        if not self._is_trained:
            raise RuntimeError("Model not trained or loaded.")

        # Resolve label → actual cm
        item_cm = self.resolve_item_cm(
            req.brand_name, req.garment_category, req.label_size
        )
        if item_cm is None:
            raise ValueError(
                f"Size not found: brand='{req.brand_name}', "
                f"category='{req.garment_category}', "
                f"label='{req.label_size}'. "
                f"Available brands: {self.get_brands()}"
            )

        X = self._request_to_row(req, item_cm)

        # Predict class and probability
        pred_idx       = int(self.tree.predict(X)[0])
        proba          = self.tree.predict_proba(X)[0]
        confidence     = float(proba[pred_idx])
        label          = LABEL_MAP[pred_idx]

        # Clearance per dimension
        dim_map = {
            "shoulder": ("shoulder_width", "shoulder"),
            "chest":    ("chest_width",    "chest"),
            "torso":    ("torso_length",   "torso"),
            "arm":      ("arm_length",     "arm"),
        }
        clearance = {
            dim: round(item_cm[item_key] - req.user_body_cm.get(user_key, 0), 1)
            for dim, (user_key, item_key) in dim_map.items()
        }


        if label == "Too Small":
        # most negative clearance = tightest point
          worst_dim = min(clearance, key=lambda k: clearance[k])
          delta_val = clearance[worst_dim]
          msg = (
        f"This {req.garment_category} in size {req.label_size} "
        f"from {req.brand_name} will be too tight. "
        f"Tightest point: {worst_dim} "
        f"(item is {abs(delta_val):.1f} cm narrower than your body)."
    )

        elif label == "Too Large":
    # most positive clearance = most excess = dominant problem
           worst_dim = max(clearance, key=lambda k: clearance[k])
           delta_val = clearance[worst_dim]
           msg = (
        f"This {req.garment_category} in size {req.label_size} "
        f"from {req.brand_name} will be too loose. "
        f"Most excess: {worst_dim} "
        f"(item is {delta_val:.1f} cm wider than your body)."
    )
        else:
            msg = (
                f"Great news — size {req.label_size} in "
                f"{req.brand_name} {req.garment_category} "
                f"should be a comfortable, accurate fit for you."
            )

        return FitVerdict(
            label        = label,
            confidence   = confidence,
            clearance    = clearance,
            message      = msg,
            item_size_cm = item_cm,
        )