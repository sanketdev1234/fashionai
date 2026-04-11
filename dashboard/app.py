"""
dashboard/app.py
─────────────────
Streamlit dashboard for FashionAI.

Tabs:
  1. 👗 Visual Search    — upload a product image → get 5 visual matches
  2. 📐 Body Scan & Fit  — upload user photo → get measurements → check item fit
  3. 📈 Trend Heatmap    — forecast demand heatmaps per category

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import io
import requests

import numpy as np
import streamlit as st
from PIL import Image
import plotly.express as px
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="FashionAI",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x60?text=FashionAI", use_container_width=True)
st.sidebar.markdown("### Navigation")
tab_selection = st.sidebar.radio(
    "Choose a module",
    ["👗 Visual Search", "📐 Body Scan & Fit", "📈 Trend Heatmap"],
)

# ─── API helpers ──────────────────────────────────────────────────────────────

def api_recommend(img_bytes: bytes, top_k: int = 5) -> list[dict] | None:
    try:
        r = requests.post(f"{API_BASE}/recommend?top_k={top_k}",
                          files={"file": ("upload.jpg", img_bytes, "image/jpeg")},
                          timeout=30)
        r.raise_for_status()
        return r.json()["recommendations"]
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def api_scan(img_bytes: bytes, ref_px: float | None) -> dict | None:
    data = {}
    if ref_px:
        data["reference_px"] = ref_px
    try:
        r = requests.post(f"{API_BASE}/fit/scan",
                          files={"file": ("scan.jpg", img_bytes, "image/jpeg")},
                          data=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def api_fit_predict(user_cm: dict, item_cm: dict, brand: str) -> dict | None:
    try:
        payload = {"user_body_cm": user_cm, "item_size_cm": item_cm, "brand_name": brand}
        r = requests.post(f"{API_BASE}/fit/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def api_trends(category: str) -> list[dict] | None:
    try:
        r = requests.get(f"{API_BASE}/trends/heatmap?category={category}", timeout=15)
        r.raise_for_status()
        return r.json()["data"]
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Visual Search
# ══════════════════════════════════════════════════════════════════════════════

if tab_selection == "👗 Visual Search":
    st.title("👗 Visual Similarity Search")
    st.markdown(
        "Upload a fashion product image and get the **top-5 visually similar** "
        "items from the catalogue. Powered by a PyTorch ResNet-50 + FAISS index."
    )

    uploaded = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])
    top_k    = st.slider("Number of results", 1, 10, 5)

    if uploaded:
        col_input, col_sep, col_results = st.columns([1, 0.05, 2])
        with col_input:
            st.subheader("Query Image")
            st.image(uploaded, use_container_width=True)

        with col_results:
            st.subheader("Recommendations")
            with st.spinner("Searching catalogue…"):
                img_bytes = uploaded.read()
                results   = api_recommend(img_bytes, top_k)

            if results:
                cols = st.columns(min(top_k, 5))
                for i, rec in enumerate(results[:5]):
                    with cols[i % 5]:
                        try:
                            pil = Image.open(rec["path"])
                            st.image(pil, caption=f"Score: {rec['score']:.3f}",
                                     use_container_width=True)
                        except Exception:
                            st.warning(f"Image not found:\n{rec['path']}")
            else:
                st.info("No results returned from API. Is the server running?")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Body Scan & Fit
# ══════════════════════════════════════════════════════════════════════════════

elif tab_selection == "📐 Body Scan & Fit":
    st.title("📐 Body Scan & Personalised Fit")
    st.markdown(
        "Upload a **front-facing photo** to extract your body measurements, "
        "then enter an item's technical spec to get a real-time **Fit Verdict**."
    )

    # ── Step 1: Scan ──────────────────────────────────────────────────────────
    st.subheader("Step 1 — Scan Your Body")
    col_photo, col_ref = st.columns([2, 1])
    with col_photo:
        user_photo = st.file_uploader("Upload upper-body photo (front-facing)", type=["jpg","jpeg","png"])
    with col_ref:
        ref_px = st.number_input(
            "Width of reference object in pixels (optional)",
            min_value=0.0, value=0.0, step=1.0,
            help="Hold an A4 paper (21 cm) in front of you and enter its pixel width here for accurate scale.",
        )
        ref_px = ref_px if ref_px > 0 else None

    measurements: dict | None = None
    if user_photo:
        if st.button("🔍 Scan Body", type="primary"):
            with st.spinner("Analysing pose…"):
                measurements = api_scan(user_photo.read(), ref_px)

            if measurements:
                st.success(f"✅ Confidence: {measurements['confidence']:.0%}")
                mc = st.columns(4)
                icons = ["🦴 Shoulder", "💪 Chest", "📏 Torso", "💪 Arm"]
                keys  = ["shoulder_width_cm", "chest_width_cm", "torso_length_cm", "arm_length_cm"]
                for i, (icon, key) in enumerate(zip(icons, keys)):
                    mc[i].metric(icon, f"{measurements[key]:.1f} cm")
                st.session_state["user_cm"] = measurements

    # ── Step 2: Item spec ──────────────────────────────────────────────────────
    st.subheader("Step 2 — Enter Item Technical Spec")
    st.caption("These are the actual garment measurements from the brand's tech pack, not the labelled size.")
    ic = st.columns(4)
    item_shoulder = ic[0].number_input("Shoulder (cm)", 30.0, 70.0, 46.0, 0.5)
    item_chest    = ic[1].number_input("Chest (cm)",    60.0, 150.0, 100.0, 0.5)
    item_torso    = ic[2].number_input("Torso (cm)",    30.0, 70.0, 46.0, 0.5)
    item_arm      = ic[3].number_input("Arm (cm)",      40.0, 85.0, 60.0, 0.5)
    brand         = st.selectbox("Brand", ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE", "unknown"])

    # ── Step 3: Predict ────────────────────────────────────────────────────────
    st.subheader("Step 3 — Get Fit Verdict")
    if st.button("🎯 Predict Fit", type="primary"):
        user_cm = st.session_state.get("user_cm", {
            "shoulder_width_cm": 44.0, "chest_width_cm": 94.0,
            "torso_length_cm": 44.0, "arm_length_cm": 58.0,
        })
        body = {
            "shoulder_width": user_cm["shoulder_width_cm"],
            "chest_width":    user_cm["chest_width_cm"],
            "torso_length":   user_cm["torso_length_cm"],
            "arm_length":     user_cm["arm_length_cm"],
        }
        item = {
            "shoulder_width": item_shoulder,
            "chest_width":    item_chest,
            "torso_length":   item_torso,
            "arm_length":     item_arm,
        }

        with st.spinner("Running ensemble prediction…"):
            verdict = api_fit_predict(body, item, brand)

        if verdict:
            label_color = {
                "Perfect Fit": "green",
                "Too Small":   "red",
                "Too Large":   "orange",
            }.get(verdict["label"], "gray")

            st.markdown(
                f"<h2 style='color:{label_color};text-align:center'>"
                f"{'✅' if verdict['label']=='Perfect Fit' else '⚠️'} {verdict['label']}"
                f"  ({verdict['confidence']:.0%} confidence)</h2>",
                unsafe_allow_html=True,
            )
            st.info(verdict["message"])

            # Clearance bar chart
            cl_df = pd.DataFrame([
                {"Dimension": k.replace("_", " ").title(), "Clearance (cm)": v}
                for k, v in verdict["clearance"].items()
            ])
            fig = px.bar(
                cl_df, x="Dimension", y="Clearance (cm)",
                color="Clearance (cm)",
                color_continuous_scale=["red", "green", "orange"],
                title="Physical Clearance (item − body) per Dimension",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Trend Heatmap
# ══════════════════════════════════════════════════════════════════════════════

elif tab_selection == "📈 Trend Heatmap":
    st.title("📈 Trend Oracle — Seasonal Demand Heatmap")
    st.markdown(
        "Visualise **90-day demand forecasts** for fashion trends (colors, "
        "silhouettes, garment types) powered by Facebook Prophet."
    )

    category = st.selectbox(
        "Select trend category",
        ["color", "silhouette", "garment_type"],
    )

    if st.button("🔮 Load Forecast", type="primary"):
        with st.spinner("Fetching forecast data…"):
            data = api_trends(category)

        if data:
            df = pd.DataFrame(data)
            pivot = df.pivot_table(
                index="value", columns="week", values="avg_demand", aggfunc="mean"
            )

            fig = px.imshow(
                pivot,
                labels=dict(x="Week of Year", y=category.replace("_", " ").title(), color="Demand"),
                aspect="auto",
                color_continuous_scale="Viridis",
                title=f"Demand Forecast Heatmap — {category.replace('_',' ').title()}",
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            # Top trends table
            top_rows = (
                df.groupby("value")["avg_demand"].mean()
                  .sort_values(ascending=False)
                  .reset_index()
                  .rename(columns={"value": "Trend", "avg_demand": "Avg Demand"})
            )
            top_rows["Avg Demand"] = top_rows["Avg Demand"].round(1)
            st.subheader("🏆 Top Trends by Average Forecasted Demand")
            st.dataframe(top_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No trend data returned. Is the server running with a trained Trend Oracle?")
