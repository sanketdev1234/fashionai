"""
src/utils/helpers.py
────────────────────
Shared utilities: config loader, logger setup, device resolver.
"""
from __future__ import annotations

import random
import yaml
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Any


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML config and return as nested dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device(preference: str = "auto") -> torch.device:
    """
    Resolve compute device.

    Args:
        preference: "auto" | "cuda" | "cpu" | "mps"

    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(preference)

    logger.info(f"[Device] Using: {dev}")
    return dev


# ─── Path helpers ─────────────────────────────────────────────────────────────

def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
