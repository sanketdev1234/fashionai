# FashionAI

A full-stack AI fashion platform with three production-ready modules powered by real Google Trends data, PyTorch, MediaPipe, and Facebook Prophet.

![Python](https://img.shields.io/badge/Python-3.10--3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

---

## Modules

### Visual Search
Upload a fashion product image → get the top-K visually similar items from a 44k-image catalogue.
Powered by **PyTorch ResNet-50** embeddings + **FAISS** cosine similarity index.

### Body Scan & Personalised Fit
Upload a front-facing photo → extract body measurements via **MediaPipe Pose Landmarker** → get a personalised fit verdict (Too Small / Perfect Fit / Too Large) from an **RF + MLP ensemble**.

### Trend Oracle
**90-day seasonal demand forecasts** for fashion colors, silhouettes, and garment types.
Trained on real **Google Trends** weekly search data (India, 5 years) via **Facebook Prophet**.

---

## Stack

| Layer | Technology |
|---|---|
| Feature extraction | PyTorch ResNet-50 |
| Vector search | FAISS (cosine similarity, 44k images) |
| Pose estimation | MediaPipe Pose Landmarker (Tasks API, 0.10+) |
| Fit prediction | Scikit-learn Random Forest + PyTorch MLP ensemble |
| Trend data | Google Trends via pytrends |
| Trend forecasting | Facebook Prophet |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |

---

## Requirements

- Python 3.10 – 3.13
- CUDA-capable GPU recommended (CPU works, slower)
- ~4 GB disk space for artifacts + dataset

---

## Setup

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/FashionAI.git
cd FashionAI
```

### 2. Virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install pytrends==4.9.2 "urllib3>=1.26.0,<2.0"
```

### 4. Add your image dataset
Place fashion product images (JPEG/PNG) in `data/images/`.

### 5. Generate synthetic training data (size + trend baseline)
```bash
python scripts/generate_data.py
```

### 6. Fetch real Google Trends data (recommended)
```bash
# Dry run first — see what will be fetched
python scripts/collect_trends.py --dry-run

# Fetch real data (~24 min, 60s delay per keyword)
python scripts/collect_trends.py --geo IN --timeframe "today 5-y"

# If any keywords return no data, patch them
python scripts/patch_missing_trends.py
```

### 7. Train all models
```bash
# Train everything
python scripts/train.py --stage all

# Or individually
python scripts/train.py --stage index    # FAISS visual index
python scripts/train.py --stage size     # RF + MLP fit model
python scripts/train.py --stage trend    # Prophet trend oracle
```

> The MediaPipe pose landmarker model (~30 MB) downloads automatically to `artifacts/` on first API startup.

---

## Running

### API server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Docs: http://127.0.0.1:8000/docs

### Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard: http://localhost:8501

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/recommend` | Visual similarity search (image upload) |
| POST | `/fit/scan` | CV body scan from user photo |
| POST | `/fit/predict` | Fit verdict from measurements |
| GET | `/trends/top` | Top-N trending fashion items |
| GET | `/trends/heatmap` | Seasonal demand heatmap |
| GET | `/health` | Health check |

---

## Project Structure

```
FashionAI/
├── configs/
│   └── config.yaml                  # Master configuration
├── dashboard/
│   └── app.py                       # Streamlit dashboard
├── docs/
│   └── GUIDE.md                     # Detailed usage guide
├── scripts/
│   ├── collect_trends.py            # Google Trends real data collector
│   ├── patch_missing_trends.py      # Retry failed trend keywords
│   ├── generate_data.py             # Synthetic data generation
│   └── train.py                     # Master training script
├── src/
│   ├── api/
│   │   └── main.py                  # FastAPI application
│   ├── models/
│   │   ├── cv_anthropometry.py      # MediaPipe body measurement (Tasks API)
│   │   ├── feature_extractor.py     # ResNet-50 + FAISS
│   │   ├── size_fit_model.py        # RF + MLP fit ensemble
│   │   └── trend_oracle.py          # Prophet forecasting
│   └── utils/
│       └── helpers.py
├── tests/
│   └── test_all.py
├── requirements.txt
└── .gitignore
```

---

## Key fixes applied (PyTorch 2.6 + MediaPipe 0.10+)

- `torch.load()` updated to `weights_only=False` for existing checkpoints (PyTorch 2.6 breaking change)
- `label_encoder_classes` saved as plain list (no numpy in checkpoint)
- `cv_anthropometry.py` fully rewritten for MediaPipe Tasks API (`mp.tasks.vision.PoseLandmarker`) — the old `mp.solutions.pose` API was removed in mediapipe 0.10
- Trend Oracle forecast values clipped to ≥ 0 (Prophet can extrapolate negative demand)
- Heatmap data normalised 0–100 per category for consistent color scale

---

## Notes

- `artifacts/` and `data/` are excluded from Git — regenerate locally using the setup steps above
- Google Trends data is relative search interest (0–100), not absolute counts — Prophet models the shape, not the magnitude
- The `uploads/` directory is a runtime temp folder for user photo uploads and is never committed