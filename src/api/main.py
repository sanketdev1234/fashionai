"""
src/api/main.py
────────────────
FastAPI application — single modular monolith exposing three modules:

  POST /recommend          → Visual similarity search (image upload)
  POST /fit/scan           → CV body scan from a user photo
  POST /fit/predict        → Size fit verdict given measurements
  GET  /trends/top         → Top-N trending fashion items
  GET  /trends/heatmap     → Demand heatmap data for a category
  GET  /health             → Health check

All heavy models are loaded once at startup via lifespan context.

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


# ─── Globals (populated in lifespan) ─────────────────────────────────────────

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

    # Visual index
    extractor = FashionFeatureExtractor(CFG)
    extractor.build_index()   # loads from disk if artifacts exist

    # Size model
    size_model = SizeFitModel(CFG)
    size_path  = CFG["paths"]["size_model"]
    if Path(size_path).exists():
        size_model.load(size_path)
    else:
        logger.warning("[API] size_model.pt not found — run scripts/train.py --stage size")

    # CV scanner
    cv_scanner = CVAnthropometry(
        reference_width_cm=CFG["cv_anthropometry"]["reference_object_cm"]
    )

    # Trend Oracle
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
    version="1.0.0",
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
    """
    Visual similarity search.

    Input  : image file (multipart/form-data), top_k (int, default 5)
    Output : {"recommendations": [{"path": str, "score": float}, ...]}
    """
    if extractor is None or extractor.faiss_index is None:
        raise HTTPException(503, "Visual index not ready. Build it first.")

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
    """
    Extract body measurements from a user photo via MediaPipe.

    Input  : user photo (front-facing, upper body visible)
             reference_px: pixel width of A4 paper in the frame (optional but recommended)
    Output : {shoulder_width_cm, chest_width_cm, torso_length_cm, arm_length_cm, confidence}
    """
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


# ─── Fit Prediction ───────────────────────────────────────────────────────────

class FitPredictRequest(BaseModel):
    user_body_cm: dict   # {"shoulder_width": 44.0, "chest_width": 96.0, ...}
    item_size_cm: dict   # {"shoulder_width": 46.0, "chest_width": 100.0, ...}
    brand_name:   str = "unknown"

    model_config = {"json_schema_extra": {"example": {
        "user_body_cm": {"shoulder_width": 44.0, "chest_width": 94.0,
                         "torso_length": 44.0, "arm_length": 58.0},
        "item_size_cm": {"shoulder_width": 47.0, "chest_width": 102.0,
                         "torso_length": 46.0, "arm_length": 60.0},
        "brand_name": "BrandA",
    }}}


@app.post("/fit/predict", tags=["Fit"])
def predict_fit(req: FitPredictRequest):
    """
    Predict how an item will fit given user measurements and item specs.

    Input  : FitPredictRequest JSON body
    Output : {label, confidence, clearance (per dimension), message}
    """
    if size_model is None or not size_model._is_trained:
        raise HTTPException(503, "Size model not trained. Run scripts/train.py --stage size")

    try:
        verdict = size_model.predict(FitRequest(
            user_body_cm=req.user_body_cm,
            item_size_cm=req.item_size_cm,
            brand_name=req.brand_name,
        ))
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

    return {
        "label":      verdict.label,
        "confidence": round(verdict.confidence, 3),
        "clearance":  verdict.clearance,
        "message":    verdict.message,
    }


# ─── Trend Endpoints ──────────────────────────────────────────────────────────

@app.get("/trends/top", tags=["Trends"])
def top_trends(n: int = 10):
    """
    Return top-N trending fashion items ranked by forecast growth.

    Output: [{"category", "value", "growth_pct", "peak_date"}, ...]
    """
    if trend_oracle is None or not trend_oracle._is_fitted:
        raise HTTPException(503, "Trend Oracle not fitted. Run scripts/train.py --stage trend")
    return {"trends": trend_oracle.top_trends(n=n)}


@app.get("/trends/heatmap", tags=["Trends"])
def trend_heatmap(category: str = "color"):
    """
    Return demand heatmap data for a category (color | silhouette | garment_type).

    Output: [{"value", "week", "avg_demand"}, ...]
    """
    if trend_oracle is None or not trend_oracle._is_fitted:
        raise HTTPException(503, "Trend Oracle not ready.")
    df = trend_oracle.heatmap_data(category=category)
    return {"data": df.to_dict(orient="records")}
