"""
tests/test_all.py
──────────────────
Unit tests for FashionAI modules.

Run:
    pytest tests/test_all.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.utils.helpers import load_config, get_device, set_seed
from src.models.size_fit_model import SizeFitModel, FitRequest, FitVerdict, LABEL_MAP


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/config.yaml")


@pytest.fixture(scope="module")
def dummy_image():
    """224×224 random RGB PIL image."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ─── Utils ────────────────────────────────────────────────────────────────────

def test_config_loads(cfg):
    assert "feature_extractor" in cfg
    assert "size_model" in cfg
    assert "paths" in cfg


def test_device_resolution():
    dev = get_device("auto")
    assert isinstance(dev, torch.device)


def test_set_seed():
    set_seed(42)
    a = torch.rand(3)
    set_seed(42)
    b = torch.rand(3)
    assert torch.allclose(a, b)


# ─── Feature Extractor ────────────────────────────────────────────────────────

def test_feature_extractor_single(cfg, dummy_image):
    from src.models.feature_extractor import FashionFeatureExtractor
    ext = FashionFeatureExtractor(cfg)
    feat = ext.extract_single(dummy_image)
    assert feat.shape[1] == 2048, "Expected 2048-dim embedding for ResNet50"
    norm = np.linalg.norm(feat, axis=1)
    assert abs(norm[0] - 1.0) < 1e-5, "Embedding should be L2-normalised"


# ─── Data Generator ───────────────────────────────────────────────────────────

def test_generate_size_data():
    from scripts.generate_data import generate_size_data
    df = generate_size_data(n=500, seed=0)
    assert len(df) == 500
    assert set(df["label"].unique()).issubset({0, 1, 2})
    assert df["user_shoulder"].between(32, 58).all()


def test_generate_trend_data():
    from scripts.generate_data import generate_trend_data
    df = generate_trend_data(seed=0)
    assert "ds" in df.columns
    assert "y" in df.columns
    assert df["y"].ge(0).all()


# ─── Size Fit Model ───────────────────────────────────────────────────────────

def test_size_fit_train_predict(cfg, tmp_path):
    from scripts.generate_data import generate_size_data
    from src.models.size_fit_model import SizeFitModel, FitRequest

    # Generate small dataset
    df = generate_size_data(n=500, seed=1)
    csv_path = str(tmp_path / "size_data.csv")
    df.to_csv(csv_path, index=False)

    model = SizeFitModel(cfg)
    save_path = str(tmp_path / "size_model.pt")

    # Override config paths to avoid writing to real artifacts
    cfg_copy = dict(cfg)
    cfg_copy["paths"] = dict(cfg["paths"])

    metrics = model.train(csv_path=csv_path, save_path=save_path)
    assert "rf_train_acc" in metrics
    assert 0.5 < metrics["rf_train_acc"] <= 1.0

    # Predict
    req = FitRequest(
        user_body_cm={"shoulder_width": 44.0, "chest_width": 94.0,
                      "torso_length": 44.0, "arm_length": 58.0},
        item_size_cm={"shoulder_width": 46.0, "chest_width": 100.0,
                      "torso_length": 46.0, "arm_length": 60.0},
        brand_name="BrandA",
    )
    verdict = model.predict(req)
    assert isinstance(verdict, FitVerdict)
    assert verdict.label in LABEL_MAP.values()
    assert 0.0 <= verdict.confidence <= 1.0
    assert isinstance(verdict.message, str)


def test_fit_model_load(cfg, tmp_path):
    """Test save → load → predict cycle."""
    from scripts.generate_data import generate_size_data
    from src.models.size_fit_model import SizeFitModel, FitRequest

    df = generate_size_data(n=300, seed=2)
    csv_path = str(tmp_path / "s.csv")
    df.to_csv(csv_path, index=False)

    model = SizeFitModel(cfg)
    save_path = str(tmp_path / "size_model.pt")
    model.train(csv_path=csv_path, save_path=save_path)

    model2 = SizeFitModel(cfg)
    model2.load(save_path)
    assert model2._is_trained

    verdict = model2.predict(FitRequest(
        user_body_cm={"shoulder_width": 50.0, "chest_width": 110.0,
                      "torso_length": 48.0, "arm_length": 62.0},
        item_size_cm={"shoulder_width": 44.0, "chest_width": 95.0,
                      "torso_length": 44.0, "arm_length": 58.0},
        brand_name="BrandD",
    ))
    # When item is smaller than user, should be "Too Small"
    assert verdict.label == "Too Small"


# ─── CV Anthropometry ─────────────────────────────────────────────────────────

def test_cv_anthropometry_imports():
    """At least confirm the module imports without error."""
    from src.models.cv_anthropometry import CVAnthropometry, AnthropometryResult
    scanner = CVAnthropometry()
    assert scanner is not None
    scanner.close()


# ─── Trend Oracle ─────────────────────────────────────────────────────────────

def test_trend_oracle_fit_forecast(cfg, tmp_path):
    from scripts.generate_data import generate_trend_data
    from src.models.trend_oracle import TrendOracle

    df = generate_trend_data(seed=3)
    csv_path = str(tmp_path / "trend.csv")
    df.to_csv(csv_path, index=False)

    cfg_copy = dict(cfg)
    cfg_copy["paths"] = dict(cfg["paths"])
    cfg_copy["paths"]["trend_model"] = str(tmp_path / "trend_model.pkl")

    oracle = TrendOracle(cfg_copy)
    oracle.fit(csv_path, save=True)
    assert oracle._is_fitted

    forecasts = oracle.forecast(horizon_days=14)
    assert len(forecasts) > 0
    for fc_df in forecasts.values():
        assert "yhat" in fc_df.columns
        assert len(fc_df) == 14

    tops = oracle.top_trends(n=5)
    assert len(tops) <= 5
    assert all("growth_pct" in t for t in tops)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
