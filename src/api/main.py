"""
src/api/main.py
────────────────
FastAPI application — single modular monolith.

  POST /recommend          → Visual similarity search (image upload)
  POST /fit/scan           → CV body scan from a user photo
  POST /fit/predict        → Fit verdict: brand + category + label size (NEW)
  GET  /fit/sizes          → Available brands, categories, labels (NEW)
  GET  /trends/top         → Top-N trending fashion items
  GET  /trends/heatmap     → Demand heatmap data for a category
  GET  /health             → Health check

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.helpers import load_config, set_seed
from src.models.feature_extractor import FashionFeatureExtractor
from src.models.size_fit_model import SizeFitModel, FitRequest
from src.models.cv_anthropometry import CVAnthropometry
from src.models.trend_oracle import TrendOracle


# ─── Globals ──────────────────────────────────────────────────────────────────

extractor:    FashionFeatureExtractor | None = None
size_model:   SizeFitModel            | None = None
cv_scanner:   CVAnthropometry         | None = None
trend_oracle: TrendOracle             | None = None
CFG: dict = {}


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractor, size_model, cv_scanner, trend_oracle, CFG

    CFG = load_config("configs/config.yaml")
    set_seed(CFG["project"]["seed"])

    logger.info("Loading models…")

    extractor = FashionFeatureExtractor(CFG)
    extractor.build_index()

    size_model = SizeFitModel(CFG)
    size_path  = CFG["paths"]["size_model"]
    if Path(size_path).exists():
        size_model.load(size_path)
    else:
        logger.warning("[API] size_model.pt not found — run scripts/train.py --stage size")

    cv_scanner = CVAnthropometry(
        reference_width_cm=CFG["cv_anthropometry"]["reference_object_cm"]
    )

    trend_oracle = TrendOracle(CFG)
    trend_path   = CFG["paths"]["trend_model"]
    if Path(trend_path).exists():
        trend_oracle.load()
        trend_oracle.forecast()
    else:
        logger.warning("[API] trend_model.pkl not found — run scripts/train.py --stage trend")

    logger.success("All models loaded. API ready.")
    yield

    cv_scanner.close()
    logger.info("Shutdown complete.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FashionAI API",
    description="Visual Recommendation + Fit Prediction + Trend Forecasting",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "models": {
            "visual_index": extractor is not None and extractor.faiss_index is not None,
            "size_model":   size_model is not None and size_model._is_trained,
            "trend_oracle": trend_oracle is not None and trend_oracle._is_fitted,
        },
    }


# ─── Visual Recommendation ────────────────────────────────────────────────────

@app.post("/recommend", tags=["Recommendation"])
async def recommend(
    file: Annotated[UploadFile, File(description="Fashion product image (JPEG/PNG)")],
    top_k: int = 5,
):
    if extractor is None or extractor.faiss_index is None:
        raise HTTPException(503, "Visual index not ready.")
    try:
        contents = await file.read()
        pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Cannot parse image: {e}")
    try:
        results = extractor.recommend(pil_img, top_k=top_k)
    except Exception as e:
        raise HTTPException(500, f"Recommendation failed: {e}")
    return {"recommendations": results}


# ─── CV Body Scan ─────────────────────────────────────────────────────────────

@app.post("/fit/scan", tags=["Fit"])
async def scan_body(
    file: Annotated[UploadFile, File(description="User upper-body photo")],
    reference_px: Optional[float] = Form(default=None),
):
    if cv_scanner is None:
        raise HTTPException(503, "CV scanner not initialised.")
    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        import cv2
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(400, f"Cannot decode image: {e}")
    try:
        result = cv_scanner.measure(bgr, reference_px=reference_px, annotate=False)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return result.to_dict()


# ─── Size Chart Lookup ────────────────────────────────────────────────────────

@app.get("/fit/sizes", tags=["Fit"])
def get_sizes(brand: Optional[str] = None, category: Optional[str] = None):
    """
    Return available brands, categories, and label sizes from the size chart.

    Query params:
      brand    (optional) → filter categories for this brand
      category (optional) → filter labels for this brand+category

    Output:
      {brands, categories, labels, item_cm (if brand+category+label given)}
    """
    if size_model is None:
        raise HTTPException(503, "Size model not loaded.")

    brands = size_model.get_brands()

    if brand and category:
        labels  = size_model.get_labels(brand, category)
        cats    = size_model.get_categories(brand)
        return {"brands": brands, "categories": cats, "labels": labels}
    elif brand:
        cats = size_model.get_categories(brand)
        return {"brands": brands, "categories": cats, "labels": []}
    else:
        return {"brands": brands, "categories": [], "labels": []}


@app.get("/fit/resolve", tags=["Fit"])
def resolve_size(brand: str, category: str, label: str):
    """
    Resolve a brand + category + label to actual measurements in cm.

    Output: {shoulder, chest, torso, arm} in cm
    """
    if size_model is None:
        raise HTTPException(503, "Size model not loaded.")
    item_cm = size_model.resolve_item_cm(brand, category, label)
    if item_cm is None:
        raise HTTPException(404, f"Size not found: {brand}/{category}/{label}")
    return item_cm


# ─── Fit Prediction ───────────────────────────────────────────────────────────

class FitPredictRequest(BaseModel):
    user_body_cm:      dict   # from body scan
    brand_name:        str    # e.g. "BrandA"
    garment_category:  str    # e.g. "tshirt", "shirt", "jacket", "dress"
    label_size:        str    # e.g. "S", "M", "L", "XL"

    model_config = {"json_schema_extra": {"example": {
        "user_body_cm": {
            "shoulder_width": 44.0, "chest_width": 94.0,
            "torso_length": 44.0,   "arm_length": 58.0,
        },
        "brand_name":       "BrandA",
        "garment_category": "tshirt",
        "label_size":       "M",
    }}}


@app.post("/fit/predict", tags=["Fit"])
def predict_fit(req: FitPredictRequest):
    """
    Predict fit from brand + category + label size.

    The system maps label → actual cm using the size chart internally.
    User never needs to know the garment measurements.

    Output: {label, confidence, clearance, message, item_size_cm}
    """
    if size_model is None or not size_model._is_trained:
        raise HTTPException(503, "Size model not trained.")

    try:
        verdict = size_model.predict(FitRequest(
            user_body_cm=req.user_body_cm,
            brand_name=req.brand_name,
            garment_category=req.garment_category,
            label_size=req.label_size,
        ))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

    return {
        "label":        verdict.label,
        "confidence":   round(verdict.confidence, 3),
        "clearance":    verdict.clearance,
        "message":      verdict.message,
        "item_size_cm": verdict.item_size_cm,   # resolved cm shown to user
    }


# ─── Trend Endpoints ──────────────────────────────────────────────────────────

@app.get("/trends/top", tags=["Trends"])
def top_trends(n: int = 10):
    if trend_oracle is None or not trend_oracle._is_fitted:
        raise HTTPException(503, "Trend Oracle not fitted.")
    return {"trends": trend_oracle.top_trends(n=n)}


@app.get("/trends/heatmap", tags=["Trends"])
def trend_heatmap(category: str = "color"):
    if trend_oracle is None or not trend_oracle._is_fitted:
        raise HTTPException(503, "Trend Oracle not ready.")
    df = trend_oracle.heatmap_data(category=category)
    return {"data": df.to_dict(orient="records")}