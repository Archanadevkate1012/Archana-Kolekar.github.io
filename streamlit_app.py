"""Interactive fraud detection dashboard for batch CSV prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "refined_model.pkl"
FEATURE_PATH = ARTIFACT_DIR / "feature_names.pkl"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Fraud Detection Dashboard")
st.caption("Phase 3: refined ensemble model, outlier-aware training, and visual analytics")

if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
    st.error("Model artifacts are missing. Run `python -m training.refine_train` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    missing_columns = [col for col in features if col not in df.columns]
    if missing_columns:
        st.error(f"Uploaded file is missing required columns: {missing_columns}")
        st.stop()

    inference_df = df[features].copy()
    preds = model.predict(inference_df)
    probs = model.predict_proba(inference_df)[:, 1]

    result_df = df.copy()
    result_df["Fraud_Prediction"] = preds
    result_df["Fraud_Probability"] = probs

    left, right = st.columns(2)
    left.metric("Total Transactions", len(result_df))
    right.metric("Fraud Detected", int(result_df["Fraud_Prediction"].sum()))

    st.subheader("Prediction Preview")
    st.dataframe(result_df.head(20), use_container_width=True)

    st.subheader("Fraud Probability Distribution")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(result_df["Fraud_Probability"], bins=30, color="#4c78a8")
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Fraud Trend Over Time (Synthetic Hourly Sequence)")
    result_df["Time"] = pd.date_range(start="2024-01-01", periods=len(result_df), freq="h")
    fraud_trend = result_df.groupby(result_df["Time"].dt.date)["Fraud_Prediction"].sum()

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    fraud_trend.plot(ax=ax2)
    ax2.set_title("Fraud Trend Over Time")
    ax2.set_ylabel("Fraud Count")
    st.pyplot(fig2)
