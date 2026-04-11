"""
scripts/train.py
─────────────────
Master training script. Run each stage independently or together.

Usage:
    # Step 1 – build FAISS visual index (needs images in data/images/)
    python scripts/train.py --stage index

    # Step 2 – train size fit model (needs data/size_data.csv)
    python scripts/train.py --stage size

    # Step 3 – fit trend Oracle (needs data/trend_data.csv)
    python scripts/train.py --stage trend

    # All at once
    python scripts/train.py --stage all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from src.utils.helpers import load_config, set_seed, ensure_dirs


def train_index(cfg: dict) -> None:
    from src.models.feature_extractor import FashionFeatureExtractor
    logger.info("═══ Stage: FAISS Visual Index ═══")
    extractor = FashionFeatureExtractor(cfg)
    extractor.build_index(force_rebuild=True)
    logger.success("✅  Visual index built and saved.")


def train_size(cfg: dict) -> None:
    from src.models.size_fit_model import SizeFitModel
    logger.info("═══ Stage: Size Fit Model ═══")
    model = SizeFitModel(cfg)
    metrics = model.train(
        csv_path=cfg["paths"]["size_data"],
        save_path=cfg["paths"]["size_model"],
    )
    logger.success(f"✅  Size model trained. Metrics: {metrics}")


def train_trend(cfg: dict) -> None:
    from src.models.trend_oracle import TrendOracle
    logger.info("═══ Stage: Trend Oracle ═══")
    oracle = TrendOracle(cfg)
    oracle.fit(cfg["paths"]["trend_data"])
    logger.success("✅  Trend Oracle fitted and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["index", "size", "trend", "all"], default="all")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    ensure_dirs("artifacts", "data", "uploads")

    if args.stage in ("index", "all"):
        train_index(cfg)
    if args.stage in ("size", "all"):
        train_size(cfg)
    if args.stage in ("trend", "all"):
        train_trend(cfg)

    logger.success("🎉  All requested training stages complete.")
