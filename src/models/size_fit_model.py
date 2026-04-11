"""
src/models/size_fit_model.py
─────────────────────────────
Purchase-Specific Fit Validator — RF + PyTorch MLP Ensemble.

Architecture
────────────
1. RandomForest (sklearn)   — captures non-linear size-variance interactions
2. PyTorch MLP              — learns residuals / fine-grained patterns
3. SoftVoting Ensemble      — averages class probabilities from both models

Labels
──────
  0 → "Too Small"
  1 → "Perfect Fit"
  2 → "Too Large"

Expected Input  (FitRequest):
  user_body_cm dict  : { shoulder_width, chest_width, torso_length, arm_length }
  item_size_cm dict  : { shoulder_width, chest_width, torso_length, arm_length }
  brand_name  str    : brand identifier (encoded internally)

The model computes Physical Clearance = item_cm - user_cm for each dimension
and feeds the delta vector into the ensemble.

Expected Output (FitVerdict):
  label      : str  ("Too Small" | "Perfect Fit" | "Too Large")
  confidence : float  (0–1)
  clearance  : dict  {dimension: delta_cm}
  message    : str  — human-readable verdict

Training Data Format (size_data.csv):
  Columns:
    user_shoulder, user_chest, user_torso, user_arm,
    item_shoulder, item_chest, item_torso, item_arm,
    brand_encoded,
    label  (0 / 1 / 2)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from src.utils.helpers import get_device, set_seed


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class FitRequest:
    user_body_cm: dict   # {"shoulder_width": 44.0, "chest_width": 96.0, ...}
    item_size_cm: dict   # {"shoulder_width": 46.0, "chest_width": 100.0, ...}
    brand_name:   str = "unknown"

@dataclass
class FitVerdict:
    label:      str
    confidence: float
    clearance:  dict   # {dim: delta_cm}
    message:    str


LABEL_MAP = {0: "Too Small", 1: "Perfect Fit", 2: "Too Large"}
DIMS = ["shoulder_width", "chest_width", "torso_length", "arm_length"]


# ─── MLP ──────────────────────────────────────────────────────────────────────

class FitMLP(nn.Module):
    """
    Lightweight MLP for fit classification.
    Input dim  : len(DIMS) * 2 + 1 (user, item, brand_encoded) = 9
    Output dim : 3 (Too Small / Fit / Too Large)
    """
    def __init__(self, input_dim: int = 9, hidden: list[int] = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or [256, 128, 64]
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)          # raw logits


# ─── Main Model ───────────────────────────────────────────────────────────────

class SizeFitModel:
    """
    RF + MLP soft-voting ensemble for garment fit prediction.
    """

    def __init__(self, cfg: dict):
        self.cfg    = cfg["size_model"]
        self.device = get_device()
        self.le     = LabelEncoder()                   # brand encoder
        self.rf: Optional[RandomForestClassifier] = None
        self.mlp: Optional[FitMLP] = None
        self._is_trained = False

    # ── Feature engineering ───────────────────────────────────────────────────

    @staticmethod
    def _build_features(df: pd.DataFrame) -> np.ndarray:
        """
        Compute Physical Clearance + raw measurements.

        Columns expected: user_shoulder, user_chest, user_torso, user_arm,
                          item_shoulder, item_chest, item_torso, item_arm,
                          brand_encoded
        """
        X_user = df[["user_shoulder", "user_chest", "user_torso", "user_arm"]].values
        X_item = df[["item_shoulder", "item_chest", "item_torso", "item_arm"]].values
        delta  = X_item - X_user                    # Physical Clearance
        brand  = df[["brand_encoded"]].values
        return np.hstack([X_user, X_item, delta, brand]).astype("float32")

    def _request_to_row(self, req: FitRequest) -> np.ndarray:
        brand_enc = self.le.transform([req.brand_name])[0] if req.brand_name in self.le.classes_ else 0
        row = {
            "user_shoulder": req.user_body_cm.get("shoulder_width", 0),
            "user_chest":    req.user_body_cm.get("chest_width", 0),
            "user_torso":    req.user_body_cm.get("torso_length", 0),
            "user_arm":      req.user_body_cm.get("arm_length", 0),
            "item_shoulder": req.item_size_cm.get("shoulder_width", 0),
            "item_chest":    req.item_size_cm.get("chest_width", 0),
            "item_torso":    req.item_size_cm.get("torso_length", 0),
            "item_arm":      req.item_size_cm.get("arm_length", 0),
            "brand_encoded": brand_enc,
        }
        return self._build_features(pd.DataFrame([row]))

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, csv_path: str, save_path: str) -> dict:
        """
        Train the RF + MLP ensemble on size_data.csv.

        Input  : CSV with columns described in module docstring
        Output : saves artifacts/size_model.pt + artifacts/size_rf.pkl
                 returns dict with train metrics
        """
        set_seed(42)
        df = pd.read_csv(csv_path)
        logger.info(f"[SizeFit] Training on {len(df)} samples")

        # Encode brands
        self.le.fit(df["brand"].fillna("unknown"))
        df["brand_encoded"] = self.le.transform(df["brand"].fillna("unknown"))

        X = self._build_features(df)
        y = df["label"].values.astype(int)

        # ── Random Forest ─────────────────────────────────────────────────────
        self.rf = RandomForestClassifier(
            n_estimators=self.cfg.get("rf_n_estimators", 200),
            max_depth=self.cfg.get("rf_max_depth", 15),
            n_jobs=-1, random_state=42,
        )
        self.rf.fit(X, y)
        rf_acc = self.rf.score(X, y)
        logger.info(f"[RF] Train accuracy: {rf_acc:.3f}")

        # ── MLP ───────────────────────────────────────────────────────────────
        self.mlp = FitMLP(
            input_dim=X.shape[1],
            hidden=self.cfg.get("mlp_hidden", [256, 128, 64]),
            dropout=self.cfg.get("mlp_dropout", 0.3),
        ).to(self.device)

        Xt = torch.from_numpy(X).to(self.device)
        yt = torch.from_numpy(y).long().to(self.device)
        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=self.cfg.get("batch_size", 256),
                            shuffle=True)

        opt = torch.optim.Adam(self.mlp.parameters(), lr=self.cfg.get("lr", 1e-3))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.get("epochs", 50))

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

        # ── Save ──────────────────────────────────────────────────────────────
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "mlp_state": self.mlp.state_dict(),
            "label_encoder_classes": self.le.classes_.tolist(),
            "mlp_config": {"input_dim": X.shape[1],
                           "hidden": self.cfg.get("mlp_hidden", [256, 128, 64]),
                           "dropout": self.cfg.get("mlp_dropout", 0.3)},
        }, save_path)

        rf_path = str(save_path).replace(".pt", "_rf.pkl")
        with open(rf_path, "wb") as f:
            pickle.dump(self.rf, f)

        self._is_trained = True
        logger.success(f"[SizeFit] Model saved → {save_path}")
        return {"rf_train_acc": rf_acc}

    def load(self, save_path: str) -> None:
        """Load trained artifacts."""
        ckpt = torch.load(save_path, map_location=self.device, weights_only=True)
        self.le.classes_ = np.array(ckpt["label_encoder_classes"])

        mcfg = ckpt["mlp_config"]
        self.mlp = FitMLP(**mcfg).to(self.device)
        self.mlp.load_state_dict(ckpt["mlp_state"])
        self.mlp.eval()

        rf_path = str(save_path).replace(".pt", "_rf.pkl")
        with open(rf_path, "rb") as f:
            self.rf = pickle.load(f)

        self._is_trained = True
        logger.info("[SizeFit] Loaded model from disk.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, req: FitRequest) -> FitVerdict:
        """
        Predict fit for a single (user, item) pair.

        Input  : FitRequest
        Output : FitVerdict with label, confidence, clearance deltas, message
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained/loaded. Call train() or load() first.")

        X = self._request_to_row(req)

        # RF probabilities
        rf_proba = self.rf.predict_proba(X)[0]   # shape (3,)

        # MLP probabilities
        with torch.no_grad():
            xt = torch.from_numpy(X).to(self.device)
            logits = self.mlp(xt)
            mlp_proba = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Soft vote (equal weight)
        ensemble_proba = (rf_proba + mlp_proba) / 2.0
        pred_idx  = int(np.argmax(ensemble_proba))
        confidence = float(ensemble_proba[pred_idx])
        label      = LABEL_MAP[pred_idx]

        # Clearance per dimension
        clearance = {}
        for dim in DIMS:
            key = dim.replace("_width", "").replace("_length", "")  # shorter key
            delta = req.item_size_cm.get(dim, 0) - req.user_body_cm.get(dim, 0)
            clearance[dim] = round(delta, 1)

        # Human message
        worst_dim = min(clearance, key=lambda k: clearance[k])
        delta_val  = clearance[worst_dim]
        if label == "Too Small":
            msg = (f"This item will be too tight. "
                   f"Tightest point: {worst_dim.replace('_', ' ')} "
                   f"(item is {abs(delta_val):.1f} cm narrower than your body).")
        elif label == "Too Large":
            msg = (f"This item will be too loose. "
                   f"Most excess: {worst_dim.replace('_', ' ')} "
                   f"(item is {delta_val:.1f} cm wider than your body).")
        else:
            msg = "Great news — this item should be a comfortable, accurate fit for you."

        return FitVerdict(label=label, confidence=confidence,
                          clearance=clearance, message=msg)
