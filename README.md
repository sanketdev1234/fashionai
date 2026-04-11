# FashionAI

A full-stack AI fashion platform with three modules:

- **Visual Search** — Upload a fashion product image and get the top-K visually similar items from a 44k-image catalogue. Powered by PyTorch ResNet-50 + FAISS.
- **Body Scan & Fit** — Upload a front-facing photo to extract body measurements via MediaPipe, then get a personalised fit verdict (Too Small / Perfect Fit / Too Large) from an RF + MLP ensemble.
- **Trend Oracle** — 90-day seasonal demand forecasts for fashion colors, silhouettes, and garment types powered by Facebook Prophet.

---

## Stack

| Layer | Technology |
|---|---|
| Feature extraction | PyTorch ResNet-50 |
| Vector search | FAISS (cosine similarity) |
| Pose estimation | MediaPipe Pose Landmarker (Tasks API) |
| Fit prediction | Scikit-learn Random Forest + PyTorch MLP ensemble |
| Trend forecasting | Facebook Prophet |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |

---

## Requirements

- Python 3.10 – 3.13
- CUDA-capable GPU recommended (CPU works but is slower)
- ~4 GB disk space for model artifacts + dataset

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/FashionAI.git
cd FashionAI
```

### 2. Create a virtual environment
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
```

### 4. Add your dataset
Place your fashion product images in `data/images/`. The FAISS index expects JPEG/PNG files.

### 5. Generate training data
```bash
python scripts/generate_data.py
```
This creates `data/size_data.csv` and `data/trend_data.csv`.

### 6. Train all models
```bash
# Train everything at once
python scripts/train.py --stage all

# Or train individually
python scripts/train.py --stage index   # FAISS visual index (~44k images)
python scripts/train.py --stage size    # RF + MLP fit model
python scripts/train.py --stage trend   # Prophet trend oracle
```

> The MediaPipe pose landmarker model (~30 MB) downloads automatically to `artifacts/` on first API startup.

---

## Running

### API server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive docs: http://127.0.0.1:8000/docs

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
| POST | `/fit/scan` | CV body scan from a user photo |
| POST | `/fit/predict` | Size fit verdict given measurements |
| GET | `/trends/top` | Top-N trending fashion items |
| GET | `/trends/heatmap` | Demand heatmap for a category |
| GET | `/health` | Health check |

---

## Project Structure

```
FashionAI/
├── configs/
│   └── config.yaml          # Master configuration
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── docs/
│   └── GUIDE.md             # Detailed usage guide
├── scripts/
│   ├── generate_data.py     # Synthetic data generation
│   └── train.py             # Master training script
├── src/
│   ├── api/
│   │   └── main.py          # FastAPI application
│   ├── models/
│   │   ├── cv_anthropometry.py    # MediaPipe body measurement
│   │   ├── feature_extractor.py   # ResNet-50 + FAISS
│   │   ├── size_fit_model.py      # RF + MLP fit ensemble
│   │   └── trend_oracle.py        # Prophet forecasting
│   └── utils/
│       └── helpers.py
├── tests/
│   └── test_all.py
├── requirements.txt
└── .gitignore
```

> `artifacts/` and `data/` are excluded from Git (see `.gitignore`).
> Run the setup steps above to regenerate them locally.

---

## Notes

- Model artifacts (`artifacts/`) are excluded from Git because they are large binary files. Regenerate them by running the training script.
- The `data/images/` dataset is excluded for the same reason. Host it on Google Drive, HuggingFace Datasets, or S3 and link it here.
- `uploads/` is a runtime directory for temporary user photo uploads and is never committed.
