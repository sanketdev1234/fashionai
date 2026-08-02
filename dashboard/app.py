
from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="FashionAI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
# st.sidebar.image(
#     "https://via.placeholder.com/200x60?text=FashionAI",
#     use_container_width=True,
# )
st.sidebar.markdown("### Navigation")
tab_selection = st.sidebar.radio(
    "Choose a module",
    ["Visual Search", "Body Scan & Size", "Trend Heatmap"],
)


# ─── API helpers ──────────────────────────────────────────────────────────────

def api_recommend(img_bytes: bytes, top_k: int = 5):
    try:
        r = requests.post(
            f"{API_BASE}/recommend?top_k={top_k}",
            files={"file": ("upload.jpg", img_bytes, "image/jpeg")},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["recommendations"]
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_scan(img_bytes: bytes, ref_px: float | None):
    data = {}
    if ref_px:
        data["reference_px"] = ref_px
    try:
        r = requests.post(
            f"{API_BASE}/fit/scan",
            files={"file": ("scan.jpg", img_bytes, "image/jpeg")},
            data=data,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_predict_size(shoulder_cm, arm_cm, height_cm, weight_kg):
    try:
        payload = {
            "shoulder_cm": shoulder_cm,
            "arm_cm":      arm_cm,
            "height_cm":   height_cm,
            "weight_kg":   weight_kg,
        }
        r = requests.post(f"{API_BASE}/fit/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_trends(category: str):
    try:
        r = requests.get(
            f"{API_BASE}/trends/heatmap?category={category}", timeout=15
        )
        r.raise_for_status()
        return r.json()["data"]
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Visual Search (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

if tab_selection == "Visual Search":
    st.title("Visual Similarity Search")
    st.markdown(
        "Upload a fashion product image and get the **top-5 visually similar** "
        "items from the catalogue. Powered by a PyTorch ResNet-50 + FAISS index."
    )

    uploaded = st.file_uploader(
        "Upload product image", type=["jpg", "jpeg", "png"]
    )
    top_k = st.slider("Number of results", 1, 5, 5)

    if uploaded:
        col_input, col_results = st.columns([1, 2])
        with col_input:
            st.subheader("Query Image")
            st.image(uploaded, use_container_width=True)

        with col_results:
            st.subheader("Recommendations")
            with st.spinner("Searching catalogue..."):
                img_bytes = uploaded.read()
                results   = api_recommend(img_bytes, top_k)

            if results:
                cols = st.columns(min(top_k, 5))
                for i, rec in enumerate(results[:5]):
                    with cols[i % 5]:
                        try:
                            pil = Image.open(rec["path"])
                            st.image(
                                pil,
                                caption=f"Score: {rec['score']:.3f}",
                                use_container_width=True,
                            )
                        except Exception:
                            st.warning(f"Image not found:\n{rec['path']}")
            else:
                st.info("No results. Is the API running?")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Body Scan & Size Prediction (v3 — completely redesigned)
# ══════════════════════════════════════════════════════════════════════════════

elif tab_selection == "Body Scan & Size":
    st.title("Body Scan & Size Recommendation")
    st.markdown(
        "Upload a **front-facing photo** to extract your body measurements, "
        "then enter your **height and weight** to get your recommended clothing size."
    )

    # ── How it works explanation ──────────────────────────────────────────────
    with st.expander("ℹ️ How does this work?", expanded=False):
        st.markdown("""
        **Step 1 — Body Scan (Computer Vision)**
        MediaPipe detects 33 skeletal keypoints in your photo and extracts:
        - Shoulder width (landmark 11 ↔ 12)
        - Arm length (landmark chain 11 → 13 → 15)

        **Step 2 — Your Measurements (Manual Input)**
        Enter your height and weight — most people know these without
        needing a tape measure.

        **Step 3 — Size Prediction (Machine Learning)**
        A Logistic Regression model trained on **4,082 real human body
        measurements** from the ANSUR II anthropometric survey predicts
        your clothing size (XS / S / M / L / XL / XXL).

        The model learns natural body proportions — taller + heavier +
        broader shoulders = larger chest = larger size. It was trained
        on real military personnel data covering the full range of
        adult male body shapes.

        **Accuracy:** ~65% exact match (6-class problem).
        Adjacent sizes (e.g. M vs L) account for most remaining predictions.
        The probability chart below shows how confident the model is for
        each possible size.
        """)

    # ── Step 1: Body Scan ────────────────────────────────────────────────────
    st.subheader("Step 1 — Scan Your Body")

    col_photo, col_ref = st.columns([2, 1])
    with col_photo:
        user_photo = st.file_uploader(
            "Upload upper-body photo (front-facing, well-lit)",
            type=["jpg", "jpeg", "png"],
            key="body_scan_photo",
        )
    # with col_ref:
    #     ref_px = st.number_input(
    #         "Width of reference object in pixels (optional)",
    #         min_value=0.0,
    #         value=0.0,
    #         step=1.0,
    #         help=(
    #             "Hold an A4 paper (21 cm wide) in front of you. "
    #             "Enter its pixel width here for more accurate measurements. "
    #             "Leave 0 to use the automatic heuristic (±15% accuracy)."
    #         ),
    #     )
        ref_px = None

    scan_result = None

    if user_photo:
        if st.button("Scan Body", type="primary"):
            with st.spinner("Analysing pose..."):
                scan_result = api_scan(user_photo.read(), ref_px)
            if scan_result:
                conf = scan_result["confidence"]
                conf_color = "green" if conf >= 0.75 else "orange" if conf >= 0.55 else "red"
                st.markdown(
                    f"<div style='padding:8px 14px;border-radius:6px;"
                    f"background:{'#e6f4ea' if conf>=0.75 else '#fef3cd'};'>"
                    f"<strong>Confidence: {conf:.0%}</strong></div>",
                    unsafe_allow_html=True,
                ) 
                st.markdown("<br>", unsafe_allow_html=True)

                mc = st.columns(4)
                mc[0].metric(" Shoulder", f"{scan_result['shoulder_width_cm']:.1f} cm")
                mc[1].metric(" Chest",    f"{scan_result['chest_width_cm']:.1f} cm")
                mc[2].metric(" Torso",    f"{scan_result['torso_length_cm']:.1f} cm")
                mc[3].metric(" Arm",      f"{scan_result['arm_length_cm']:.1f} cm")

                st.session_state["scan_result"] = scan_result
                st.success("Body scan complete. Scroll down to Step 2.")

    # ── Step 2: Height and Weight input ──────────────────────────────────────
    st.subheader("Step 2 — Enter Your Height & Weight")
    st.caption(
        "These two measurements improve size prediction accuracy significantly. "
        "Height is in **centimetres**, weight is in **kilograms** — same units as ANSUR II."
    )

    col1, col2 = st.columns(2)
    with col1:
        height_cm = st.number_input(
            "Your Height (cm)",
            min_value=140.0,
            max_value=220.0,
            value=175.0,
            step=0.5,
            help="e.g. 175 cm = 5 ft 9 in",
        )
    with col2:
        weight_kg = st.number_input(
            "Your Weight (kg)",
            min_value=40.0,
            max_value=200.0,
            value=75.0,
            step=0.5,
            help="e.g. 75 kg = 165 lbs",
        )

    # ── Step 3: Predict size ──────────────────────────────────────────────────
    st.subheader("Step 3 — Get Size Recommendation")

    scan_data = st.session_state.get("scan_result", None)

    if scan_data is None:
        st.info("Complete Step 1 first — scan your body photo to extract measurements.")
    else:
        if st.button("Predict My Size", type="primary"):
            shoulder_cm = scan_data["shoulder_width_cm"]
            arm_cm      = scan_data["arm_length_cm"]

            with st.spinner("Running size prediction model..."):
                verdict = api_predict_size(shoulder_cm, arm_cm, height_cm, weight_kg)

            if verdict:
                size       = verdict["predicted_size"]
                confidence = verdict["confidence"]
                message    = verdict["message"]
                all_probs  = verdict["all_probabilities"]
                description = verdict["description"]

                size_color = {
                    "XS": "#6C63FF",
                    "S":  "#48A999",
                    "M":  "#2196F3",
                    "L":  "#4CAF50",
                    "XL": "#FF9800",
                    "XXL":"#F44336",
                }.get(size, "#333333")

                # ── Main verdict ──────────────────────────────────────────
                st.markdown(
                    f"<h2 style='text-align:center;color:{size_color}'>"
                    f" Your Recommended Size: <strong>{size}</strong>"
                    f" ({confidence:.0%} confidence)</h2>",
                    unsafe_allow_html=True,
                )
                st.info(message)

                # ── Measurements used ─────────────────────────────────────
                st.markdown("#### Measurements Used for Prediction")
                inp_cols = st.columns(4)
                inp_cols[0].metric("Shoulder", f"{shoulder_cm:.1f} cm", help="From body scan")
                inp_cols[1].metric("Arm Length", f"{arm_cm:.1f} cm",   help="From body scan")
                inp_cols[2].metric("Height", f"{height_cm:.1f} cm",    help="Manual input")
                inp_cols[3].metric("Weight", f"{weight_kg:.1f} kg",    help="Manual input")

                # ── Probability chart ──────────────────────────────────────
                if all_probs:
                    st.markdown("#### Confidence per Size")
                    label_order = ["XS", "S", "M", "L", "XL", "XXL"]
                    probs_ordered = {
                        k: all_probs.get(k, 0.0) for k in label_order
                    }
                    prob_df = pd.DataFrame([
                        {
                            "Size":        k,
                            "Probability": round(v * 100, 1),
                            "Selected":    "✓ Predicted" if k == size else "",
                        }
                        for k, v in probs_ordered.items()
                    ])

                    colors = [
                        size_color if row["Size"] == size else "#CCCCCC"
                        for _, row in prob_df.iterrows()
                    ]

                    fig = go.Figure(go.Bar(
                        x=prob_df["Size"],
                        y=prob_df["Probability"],
                        marker_color=colors,
                        text=[f"{v:.1f}%" for v in prob_df["Probability"]],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="Size Probability Distribution",
                        xaxis_title="Size Label",
                        yaxis_title="Probability (%)",
                        yaxis=dict(range=[0, 100]),
                        showlegend=False,
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ── Size guide ─────────────────────────────────────────────
                st.markdown("#### Standard Men's Size Guide (chest circumference)")
                size_guide = pd.DataFrame([
                    {"Size": "XS",  "Chest (cm)": "< 88",    "Chest (inches)": "< 34.5"},
                    {"Size": "S",   "Chest (cm)": "88 – 96",  "Chest (inches)": "34.5 – 37.8"},
                    {"Size": "M",   "Chest (cm)": "96 – 104", "Chest (inches)": "37.8 – 41.0"},
                    {"Size": "L",   "Chest (cm)": "104 – 112","Chest (inches)": "41.0 – 44.1"},
                    {"Size": "XL",  "Chest (cm)": "112 – 120","Chest (inches)": "44.1 – 47.2"},
                    {"Size": "XXL", "Chest (cm)": "> 120",    "Chest (inches)": "> 47.2"},
                ])
                # highlight predicted row
                def highlight_predicted(row):
                    if row["Size"] == size:
                        return [f"background-color:{size_color}22;font-weight:bold"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    size_guide.style.apply(highlight_predicted, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    " Size predictions are based on ANSUR II (US Army male anthropometric data). "
                    "Actual fit may vary by brand, garment style, and personal preference. "
                    "Model accuracy: ~65% exact match on held-out test data (6-class problem)."
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Trend Heatmap (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

elif tab_selection == "Trend Heatmap":
    st.title("Trend Oracle — Seasonal Demand Heatmap")
    st.markdown(
        "Visualise **90-day demand forecasts** for fashion trends (colors, "
        "silhouettes, garment types) powered by Facebook Prophet trained on "
        "real Google Trends data (India, 5 years)."
    )

    category = st.selectbox(
        "Select trend category",
        ["color", "silhouette", "garment_type"],
    )

    if st.button(" Load Forecast", type="primary"):
        with st.spinner("Fetching forecast data..."):
            data = api_trends(category)

        if data:
            df    = pd.DataFrame(data)
            pivot = df.pivot_table(
                index="value",
                columns="week",
                values="avg_demand",
                aggfunc="mean",
            )
            fig = px.imshow(
                pivot,
                labels=dict(
                    x="Week of Year",
                    y=category.replace("_", " ").title(),
                    color="Demand",
                ),
                aspect="auto",
                color_continuous_scale="Viridis",
                title=f"Demand Forecast Heatmap — {category.replace('_', ' ').title()}",
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            top_rows = (
                df.groupby("value")["avg_demand"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"value": "Trend", "avg_demand": "Avg Demand"})
            )
            top_rows["Avg Demand"] = top_rows["Avg Demand"].round(1)
            st.subheader(" Top Trends by Average Forecasted Demand")
            st.dataframe(top_rows, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No trend data returned. "
                "Is the API running with a trained Trend Oracle?"
            )