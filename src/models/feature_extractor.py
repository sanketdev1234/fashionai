"""
src/models/feature_extractor.py
────────────────────────────────
PyTorch ResNet-50 visual feature extractor with FAISS vector index.

Replaces the original TensorFlow/Keras ResNet50 + sklearn NearestNeighbors.

Key improvements:
  • Pure PyTorch backbone (configurable: resnet50, efficientnet_b3, vit_b_16)
  • Batched GPU inference via DataLoader
  • FAISS index for sub-100ms cosine search at 100k+ scale
  • Persistent artifacts (embeddings.pt + faiss.index)

Expected Input  : A directory of fashion product images (JPEG/PNG).
Expected Output : embeddings.pt  → (N, 2048) float32 tensor
                  faiss.index    → binary FAISS index
                  filenames.pkl  → list of image paths (length N)

Query Input     : Single PIL Image or image file path
Query Output    : List[dict] with keys: path, score (cosine similarity)
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Union

import faiss
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from loguru import logger

from src.utils.helpers import get_device, ensure_dirs


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    """Flat image folder dataset — no subdirectory structure required."""

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, image_dir: str | Path, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.paths = [
            p for p in sorted(self.image_dir.iterdir())
            if p.suffix.lower() in self.VALID_EXTS
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        logger.info(f"[Dataset] Found {len(self.paths)} images in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot open {path}: {e}. Returning blank image.")
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, str(path)


# ─── Feature Extractor ────────────────────────────────────────────────────────

class FashionFeatureExtractor:
    """
    Extracts L2-normalised embeddings from fashion images using a
    frozen PyTorch backbone, then builds / queries a FAISS index.
    """

    BACKBONE_REGISTRY = {
        "resnet50":       lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
        "efficientnet_b3": lambda: models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT),
        "vit_b_16":       lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT),
    }

    def __init__(self, cfg: dict):
        self.cfg = cfg["feature_extractor"]
        self.paths = cfg["paths"]
        self.device = get_device(self.cfg.get("device", "auto"))

        self.model = self._build_model()
        self.transform = self._build_transform()

        self.embeddings: np.ndarray | None = None
        self.filenames: list[str] = []
        self.faiss_index: faiss.IndexFlatIP | None = None

    # ── Model ─────────────────────────────────────────────────────────────────

    def _build_model(self) -> nn.Module:
        backbone_name = self.cfg.get("backbone", "resnet50")
        factory = self.BACKBONE_REGISTRY.get(backbone_name)
        if factory is None:
            raise ValueError(f"Unknown backbone: {backbone_name}. "
                             f"Choose from {list(self.BACKBONE_REGISTRY)}")

        backbone = factory()

        # Strip the classification head → output is the feature vector
        if backbone_name == "resnet50":
            backbone.fc = nn.Identity()
        elif backbone_name == "efficientnet_b3":
            backbone.classifier = nn.Identity()
        elif backbone_name == "vit_b_16":
            backbone.heads = nn.Identity()

        # Freeze all weights (inference only)
        for p in backbone.parameters():
            p.requires_grad = False

        backbone.eval().to(self.device)
        logger.info(f"[Model] Loaded backbone: {backbone_name}")
        return backbone

    def _build_transform(self) -> transforms.Compose:
        img_size = self.cfg.get("image_size", 224)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # ── Indexing ──────────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Extract embeddings for all images in cfg.paths.images_dir and
        persist to disk. Skip if artifacts already exist unless force_rebuild=True.

        Input : images in data/images/
        Output: artifacts/embeddings.pt, artifacts/faiss.index, artifacts/filenames.pkl
        """
        emb_path   = Path(self.paths["embeddings"])
        idx_path   = Path(self.paths["faiss_index"])
        fname_path = Path(self.paths["filenames"])

        ensure_dirs(emb_path.parent)

        if not force_rebuild and emb_path.exists() and idx_path.exists():
            logger.info("[Index] Artifacts found. Loading from disk…")
            self._load_artifacts()
            return

        dataset = ImageFolderDataset(self.paths["images_dir"], self.transform)
        loader  = DataLoader(
            dataset,
            batch_size=self.cfg.get("batch_size", 64),
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=(self.device.type == "cuda"),
        )

        all_embeddings: list[np.ndarray] = []
        all_filenames:  list[str]        = []
        t0 = time.time()

        with torch.no_grad():
            for imgs, paths in tqdm(loader, desc="Extracting embeddings"):
                imgs = imgs.to(self.device)
                feats = self.model(imgs)                        # (B, D)
                feats = feats / feats.norm(dim=1, keepdim=True)  # L2 normalise
                all_embeddings.append(feats.cpu().numpy())
                all_filenames.extend(paths)

        logger.info(f"[Index] Extraction done in {time.time() - t0:.1f}s")

        embeddings_np = np.vstack(all_embeddings).astype("float32")
        self.embeddings = embeddings_np
        self.filenames  = all_filenames

        # Build FAISS inner-product index (cosine sim after L2 norm)
        dim = embeddings_np.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings_np)
        logger.info(f"[FAISS] Index built. Vectors: {self.faiss_index.ntotal}, Dim: {dim}")

        # Persist
        torch.save(torch.from_numpy(embeddings_np), emb_path)
        faiss.write_index(self.faiss_index, str(idx_path))
        with open(fname_path, "wb") as f:
            pickle.dump(self.filenames, f)
        logger.success(f"[Index] Artifacts saved to {emb_path.parent}/")

    def _load_artifacts(self) -> None:
        self.embeddings = torch.load(self.paths["embeddings"], weights_only=True).numpy()
        self.faiss_index = faiss.read_index(self.paths["faiss_index"])
        with open(self.paths["filenames"], "rb") as f:
            self.filenames = pickle.load(f)
        logger.info(f"[Index] Loaded {len(self.filenames)} embeddings from disk.")

    # ── Query ─────────────────────────────────────────────────────────────────

    def extract_single(self, img: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Extract embedding for a single image.

        Input : file path (str/Path) or PIL.Image
        Output: (1, D) float32 numpy array, L2-normalised
        """
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = feat / feat.norm(dim=1, keepdim=True)
        return feat.cpu().numpy().astype("float32")

    def recommend(
        self,
        img: Union[str, Path, Image.Image],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Visual similarity search.

        Input : single image (path or PIL Image)
        Output: list of dicts → [{"path": str, "score": float}, ...]
                score is cosine similarity in [0, 1], higher = more similar
        """
        if self.faiss_index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query = self.extract_single(img)
        scores, indices = self.faiss_index.search(query, top_k + 1)  # +1 to skip self

        results = []
        for score, idx in zip(scores[0], indices[0]):
            path = self.filenames[idx]
            # Skip exact match (score ≈ 1.0 and same path)
            if score > 0.9999:
                continue
            results.append({"path": path, "score": float(score)})
            if len(results) == top_k:
                break

        return results
