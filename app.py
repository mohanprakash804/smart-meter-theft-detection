"""
Streamlit Dashboard: Electricity Theft Detection
==================================================
Run with: streamlit run app.py

Lets you pick any meter from the dataset (or upload your own half-hourly
CSV) and see:
- Consumption pattern chart
- Hybrid ensemble theft probability
- SHAP-based explanation of WHY it was flagged
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Smart Meter Theft Detection", layout="wide")

st.title("⚡ Smart Meter Electricity Theft Detection")
st.caption("Hybrid ML system: Isolation Forest (anomaly detection) + XGBoost (classification) + SHAP (explainability)")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.joblib")
    iso_forest = joblib.load("iso_forest.joblib")
    xgb_model = joblib.load("xgb_model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    explainer = joblib.load("shap_explainer.joblib")
    return scaler, iso_forest, xgb_model, feature_cols, explainer

@st.cache_data
def load_features():
    return pd.read_csv("meter_features.csv").fillna(0)

scaler, iso_forest, xgb_model, FEATURE_COLS, explainer = load_artifacts()
feat_df = load_features()

st.sidebar.header("Select a Meter")
meter_ids = feat_df["meter_id"].tolist()
selected_meter = st.sidebar.selectbox("Meter ID", meter_ids)

row = feat_df[feat_df["meter_id"] == selected_meter].iloc[0]
X_row = row[FEATURE_COLS].values.reshape(1, -1)
X_scaled = scaler.transform(X_row)

xgb_prob = xgb_model.predict_proba(X_scaled)[0, 1]
iso_score = -iso_forest.score_samples(X_scaled)[0]

ALPHA = 0.7
# normalize iso score roughly using training distribution stored in feat_df
all_scaled = scaler.transform(feat_df[FEATURE_COLS].fillna(0))
all_iso_scores = -iso_forest.score_samples(all_scaled)
iso_norm = (iso_score - all_iso_scores.min()) / (all_iso_scores.max() - all_iso_scores.min() + 1e-9)

ensemble_score = ALPHA * xgb_prob + (1 - ALPHA) * iso_norm

col1, col2, col3 = st.columns(3)
col1.metric("XGBoost Theft Probability", f"{xgb_prob*100:.1f}%")
col2.metric("Anomaly Score (Isolation Forest)", f"{iso_norm*100:.1f}%")
col3.metric("Final Ensemble Risk Score", f"{ensemble_score*100:.1f}%",
            delta="⚠️ FLAGGED" if ensemble_score > 0.5 else "✅ Normal")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Meter Feature Profile")
    display_df = row[FEATURE_COLS].to_frame(name="Value")
    st.dataframe(display_df, use_container_width=True)
    if row["theft_type"] != "NONE":
        st.info(f"Ground truth (simulation only): **{row['theft_type']}**")
    else:
        st.info("Ground truth (simulation only): **Honest meter**")

with right:
    st.subheader("Why this prediction? (SHAP explanation)")
    shap_val = explainer.shap_values(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_val[0],
            base_values=explainer.expected_value,
            data=X_row[0],
            feature_names=FEATURE_COLS
        ),
        show=False
    )
    st.pyplot(fig, use_container_width=True)

st.divider()
st.subheader("Population Overview")
st.bar_chart(feat_df.groupby("theft_type")["meter_id"].count())
