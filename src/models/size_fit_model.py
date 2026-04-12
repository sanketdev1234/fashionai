"""
src/models/size_fit_model.py
─────────────────────────────
Purchase-Specific Fit Validator — RF + PyTorch MLP Ensemble.

NEW in v2:
  - Accepts Brand + Category + Label Size (S/M/L/XL) as input
  - Maps label → actual cm using size_chart.json lookup table
  - FitRequest now takes label_size instead of raw item_size_cm
  - Returns resolved item_cm alongside the verdict so UI can show it

Architecture:
  1. RandomForest (sklearn)  — non-linear size-variance interactions
  2. PyTorch MLP             — residual fine-grained patterns
  3. SoftVoting Ensemble     — averages class probabilities from both

Labels:
  0 → "Too Small"
  1 → "Perfect Fit"
  2 → "Too Large"
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from src.utils.helpers import get_device, set_seed


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class FitRequest:
    """
    What the user provides — brand, garment category, and label size.
    No cm knowledge required.
    """
    user_body_cm:   dict   # from body scan: {shoulder_width, chest_width, torso_length, arm_length}
    brand_name:     str    # e.g. "BrandA"
    garment_category: str  # e.g. "tshirt", "shirt", "jacket", "dress"
    label_size:     str    # e.g. "S", "M", "L", "XL"


@dataclass
class FitVerdict:
    label:       str
    confidence:  float
    clearance:   dict   # {dimension: delta_cm}
    message:     str
    item_size_cm: dict  # resolved actual measurements shown to user


LABEL_MAP = {0: "Too Small", 1: "Perfect Fit", 2: "Too Large"}
DIMS      = ["shoulder", "chest", "torso", "arm"]


# ─── MLP ──────────────────────────────────────────────────────────────────────

class FitMLP(nn.Module):
    def __init__(self, input_dim: int = 11, hidden: list[int] = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or [256, 128, 64]
        layers = []
        prev   = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Main Model ───────────────────────────────────────────────────────────────

class SizeFitModel:

    SIZE_CHART_PATH = "data/size_chart.json"

    def __init__(self, cfg: dict):
        self.cfg    = cfg["size_model"]
        self.device = get_device()
        self.le_brand    = LabelEncoder()
        self.le_category = LabelEncoder()
        self.le_label    = LabelEncoder()
        self.rf:  Optional[RandomForestClassifier] = None
        self.mlp: Optional[FitMLP] = None
        self._is_trained  = False
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
            logger.warning(f"[SizeFit] size_chart.json not found at {p}. "
                           "Run scripts/generate_data.py first.")

    def resolve_item_cm(self, brand: str, category: str, label: str) -> dict | None:
        """
        Look up actual garment measurements for a given brand/category/label.
        Returns {shoulder, chest, torso, arm} in cm, or None if not found.
        """
        try:
            return self._size_chart[brand][category][label].copy()
        except KeyError:
            return None

    def get_brands(self) -> list[str]:
        return list(self._size_chart.keys())

    def get_categories(self, brand: str) -> list[str]:
        return list(self._size_chart.get(brand, {}).keys())

    def get_labels(self, brand: str, category: str) -> list[str]:
        order = ["XS", "S", "M", "L", "XL", "XXL"]
        available = list(self._size_chart.get(brand, {}).get(category, {}).keys())
        return [l for l in order if l in available]

    # ── Feature engineering ───────────────────────────────────────────────────

    @staticmethod
    def _build_features(df: pd.DataFrame) -> np.ndarray:
        """
        Features:
          user body (4) + item cm (4) + delta/clearance (4)
          + brand_enc (1) + category_enc (1) + label_enc (1) = 15
        """
        X_user  = df[["user_shoulder", "user_chest", "user_torso", "user_arm"]].values
        X_item  = df[["item_shoulder", "item_chest", "item_torso", "item_arm"]].values
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

        # Random Forest
        self.rf = RandomForestClassifier(
            n_estimators=self.cfg.get("rf_n_estimators", 200),
            max_depth=self.cfg.get("rf_max_depth", 15),
            n_jobs=-1, random_state=42,
        )
        self.rf.fit(X, y)
        rf_acc = self.rf.score(X, y)
        logger.info(f"[RF] Train accuracy: {rf_acc:.3f}")

        # MLP
        self.mlp = FitMLP(
            input_dim=X.shape[1],
            hidden=self.cfg.get("mlp_hidden", [256, 128, 64]),
            dropout=self.cfg.get("mlp_dropout", 0.3),
        ).to(self.device)

        Xt = torch.from_numpy(X).to(self.device)
        yt = torch.from_numpy(y).long().to(self.device)
        loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.cfg.get("batch_size", 256),
            shuffle=True,
        )
        opt = torch.optim.Adam(self.mlp.parameters(), lr=self.cfg.get("lr", 1e-3))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.get("epochs", 50)
        )

        self.mlp.train()
        for epoch in range(self.cfg.get("epochs", 50)):
            total_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = F.cross_entropy(self.mlp(xb), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                logger.info(f"[MLP] Epoch {epoch+1} | loss={total_loss/len(loader):.4f}")

        # Save
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "mlp_state":               self.mlp.state_dict(),
            "brand_classes":           self.le_brand.classes_.tolist(),
            "category_classes":        self.le_category.classes_.tolist(),
            "label_classes":           self.le_label.classes_.tolist(),
            "mlp_config": {
                "input_dim": X.shape[1],
                "hidden":    self.cfg.get("mlp_hidden", [256, 128, 64]),
                "dropout":   self.cfg.get("mlp_dropout", 0.3),
            },
        }, save_path)

        rf_path = str(save_path).replace(".pt", "_rf.pkl")
        with open(rf_path, "wb") as f:
            pickle.dump(self.rf, f)

        self._is_trained = True
        logger.success(f"[SizeFit] Model saved → {save_path}")
        return {"rf_train_acc": rf_acc}

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, save_path: str) -> None:
        ckpt = torch.load(save_path, map_location=self.device, weights_only=False)

        self.le_brand.classes_    = np.array(ckpt["brand_classes"])
        self.le_category.classes_ = np.array(ckpt["category_classes"])
        self.le_label.classes_    = np.array(ckpt["label_classes"])

        mcfg     = ckpt["mlp_config"]
        self.mlp = FitMLP(**mcfg).to(self.device)
        self.mlp.load_state_dict(ckpt["mlp_state"])
        self.mlp.eval()

        rf_path = str(save_path).replace(".pt", "_rf.pkl")
        with open(rf_path, "rb") as f:
            self.rf = pickle.load(f)

        self._is_trained = True
        self._load_size_chart()
        logger.info("[SizeFit] Loaded model from disk.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, req: FitRequest) -> FitVerdict:
        if not self._is_trained:
            raise RuntimeError("Model not trained/loaded.")

        # Resolve label → actual cm
        item_cm = self.resolve_item_cm(req.brand_name, req.garment_category, req.label_size)
        if item_cm is None:
            raise ValueError(
                f"Size not found: brand='{req.brand_name}', "
                f"category='{req.garment_category}', label='{req.label_size}'. "
                f"Available brands: {self.get_brands()}"
            )

        X = self._request_to_row(req, item_cm)

        # RF probabilities
        rf_proba = self.rf.predict_proba(X)[0]

        # MLP probabilities
        with torch.no_grad():
            xt      = torch.from_numpy(X).to(self.device)
            logits  = self.mlp(xt)
            mlp_proba = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Soft vote
        ensemble_proba = (rf_proba + mlp_proba) / 2.0
        pred_idx   = int(np.argmax(ensemble_proba))
        confidence = float(ensemble_proba[pred_idx])
        label      = LABEL_MAP[pred_idx]

        # Clearance per dimension
        dim_map = {
            "shoulder": ("shoulder_width",  "shoulder"),
            "chest":    ("chest_width",     "chest"),
            "torso":    ("torso_length",    "torso"),
            "arm":      ("arm_length",      "arm"),
        }
        clearance = {}
        for dim, (user_key, item_key) in dim_map.items():
            user_val = req.user_body_cm.get(user_key, 0)
            item_val = item_cm[item_key]
            clearance[dim] = round(item_val - user_val, 1)

        # Human message
        worst_dim = min(clearance, key=lambda k: clearance[k])
        delta_val = clearance[worst_dim]
        if label == "Too Small":
            msg = (
                f"This {req.garment_category} in size {req.label_size} from {req.brand_name} "
                f"will be too tight. Tightest point: {worst_dim} "
                f"(item is {abs(delta_val):.1f} cm narrower than your body)."
            )
        elif label == "Too Large":
            msg = (
                f"This {req.garment_category} in size {req.label_size} from {req.brand_name} "
                f"will be too loose. Most excess: {worst_dim} "
                f"(item is {delta_val:.1f} cm wider than your body)."
            )
        else:
            msg = (
                f"Great news — size {req.label_size} in {req.brand_name} {req.garment_category} "
                f"should be a comfortable, accurate fit for you."
            )

        return FitVerdict(
            label=label,
            confidence=confidence,
            clearance=clearance,
            message=msg,
            item_size_cm=item_cm,
        )