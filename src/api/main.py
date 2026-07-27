"""
src/api/main.py
────────────────
FashionAI API v3.0.0

  POST /recommend       → Visual similarity search (image upload)
  POST /fit/scan        → CV body scan — extracts body measurements from photo
  POST /fit/predict     → Size prediction: scan results + height + weight
                          → XS / S / M / L / XL / XXL
  GET  /trends/top      → Top-N trending fashion items
  GET  /trends/heatmap  → Demand heatmap data for a category
  GET  /health          → Health check

What changed from v2:
  REMOVED: /fit/sizes   — brand/category/label dropdowns no longer needed
  REMOVED: /fit/resolve — brand size chart lookup no longer needed
  UPDATED: /fit/predict — now takes shoulder_cm + arm_cm (from scan)
                          + height_cm + weight_kg (user input)
                          returns XS/S/M/L/XL/XXL from ANSUR II model
  UPDATED: /health      — checks _is_loaded instead of _is_trained

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
from PIL import Image
from pydantic import BaseModel
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.helpers import load_config, set_seed
from src.models.feature_extractor import FashionFeatureExtractor
from src.models.size_fit_model import SizeFitModel, SizeRequest
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

    logger.info("Loading models...")

    # 1 — Visual index (unchanged)
    extractor = FashionFeatureExtractor(CFG)
    extractor.build_index()

    # 2 — ANSUR II size model
    #     Loads: best_size_model.pkl + ansur_scaler.pkl + ansur_label_map.json
    #     Trained offline by: preprocess_ansur.py + train_size_models.py
    size_model = SizeFitModel(CFG)
    try:
        size_model.load()
    except FileNotFoundError as e:
        logger.warning(str(e))

    # 3 — CV body scanner (unchanged)
    cv_scanner = CVAnthropometry(
        reference_width_cm=CFG["cv_anthropometry"]["reference_object_cm"]
    )

    # 4 — Trend oracle (unchanged)
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
    description="Visual Recommendation + ANSUR II Size Prediction + Trend Forecasting",
    version="3.0.0",
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
            "size_model":   size_model is not None and size_model._is_loaded,
            "trend_oracle": trend_oracle is not None and trend_oracle._is_fitted,
        },
    }


# ─── Visual Recommendation (unchanged) ────────────────────────────────────────

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


# ─── CV Body Scan (unchanged) ─────────────────────────────────────────────────

@app.post("/fit/scan", tags=["Fit"])
async def scan_body(
    file: Annotated[UploadFile, File(description="User upper-body photo")],
    reference_px: Optional[float] = Form(default=None),
):
    """
    Run MediaPipe pose detection on the uploaded photo.
    Returns 4 body measurements used as partial input to /fit/predict.

    Output:
        shoulder_width_cm, chest_width_cm, torso_length_cm, arm_length_cm,
        confidence (0-1)
    """
    if cv_scanner is None:
        raise HTTPException(503, "CV scanner not initialised.")
    try:
        contents = await file.read()
        pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
        rgb = np.array(pil_img)
    except Exception as e:
        raise HTTPException(400, f"Cannot decode image: {e}")
    try:
        result = cv_scanner.measure(rgb, reference_px=reference_px, annotate=False)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return result.to_dict()


# ─── Size Prediction (new: ANSUR II model) ────────────────────────────────────

class SizePredictRequest(BaseModel):
    """
    Inputs for size prediction.

    shoulder_cm and arm_cm  → from /fit/scan response
    height_cm and weight_kg → entered manually by the user in the dashboard
    """
    shoulder_cm: float
    arm_cm:      float
    height_cm:   float
    weight_kg:   float

    model_config = {"json_schema_extra": {"example": {
        "shoulder_cm": 41.5,
        "arm_cm":      89.0,
        "height_cm":   175.0,
        "weight_kg":   80.0,
    }}}


@app.post("/fit/predict", tags=["Fit"])
def predict_size(req: SizePredictRequest):
    """
    Predict clothing size label (XS/S/M/L/XL/XXL).

    Model: Logistic Regression trained on ANSUR II (4,082 real soldiers).
    Features: shoulder_cm, arm_cm, height_cm, weight_kg.

    Output:
        predicted_size    — e.g. "L"
        confidence        — probability of predicted class (0-1)
        description       — human readable size description
        all_probabilities — full probability per class
        message           — complete user-facing sentence
    """
    if size_model is None or not size_model._is_loaded:
        raise HTTPException(
            503,
            "Size model not loaded. "
            "Run: python scripts/preprocess_ansur.py && "
            "python scripts/train_size_models.py"
        )
    try:
        verdict = size_model.predict(SizeRequest(
            shoulder_cm=req.shoulder_cm,
            arm_cm=req.arm_cm,
            height_cm=req.height_cm,
            weight_kg=req.weight_kg,
        ))
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

    return {
        "predicted_size":    verdict.predicted_size,
        "confidence":        round(verdict.confidence, 3),
        "description":       verdict.description,
        "all_probabilities": verdict.all_probabilities,
        "message":           verdict.message,
    }


# ─── Trend Endpoints (unchanged) ─────────────────────────────────────────────

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