"""Operator dashboard for the acoustic gearbox classifier.

Run from the project folder with: streamlit run dashboard.py
The dashboard uses the same preprocessing and feature functions as training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import acoustic_preprocessing as preprocessing
from acoustic_dashboard_backend import available_models, load_cnn_model, load_models, predict_signal, read_csv_signal


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "multiclass"

st.set_page_config(page_title="Gearbox acoustic monitor", page_icon="◉", layout="wide")


st.markdown("""<style>
  .block-container {max-width: 1380px; padding-top: 2rem;}
  [data-testid="stMetric"] {background:#fff; border:1px solid #e6e9ed; padding:16px 18px; border-radius:12px;}
  .status-dot {display:inline-block;width:9px;height:9px;border-radius:50%;background:#16a36a;margin-right:8px;}
  .muted {color:#687385;font-size:0.92rem;}
</style>""", unsafe_allow_html=True)

st.title("Gearbox acoustic monitor")
st.caption("Acoustic condition check · 16 kHz mono · 3 second analysis window")

if not MODEL_DIR.exists():
    st.error(f"Model folder is missing: {MODEL_DIR}")
    st.info("Train the model first, or update MODEL_DIR at the top of dashboard.py.")
    st.stop()

try:
    svm_model, _ = load_models(str(MODEL_DIR))
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

with st.sidebar:
    st.subheader("Analysis setup")
    cnn_enabled = False
    if (MODEL_DIR / "cnn_model.keras").exists():
        cnn_enabled = st.toggle("Use CNN model", value=False, help="Loads the CNN model when enabled.")
    try:
        cnn_model = load_cnn_model(str(MODEL_DIR)) if cnn_enabled else None
    except Exception as exc:
        cnn_model = None
        st.error(str(exc))
    model_choices = available_models(svm_model, cnn_model)
    if not model_choices:
        st.error("No usable model files were found.")
        st.stop()
    selected_model = st.selectbox("Model", model_choices, index=0)
    st.markdown("<span class='status-dot'></span>Model files loaded", unsafe_allow_html=True)
    if not cnn_enabled and (MODEL_DIR / "cnn_model.keras").exists():
        st.caption("The CNN loads when selected. SVM is ready for analysis.")
    if st.button("Reload model files", use_container_width=True):
        load_models.cache_clear()
        load_cnn_model.cache_clear()
        st.rerun()
    st.caption(f"Using `{MODEL_DIR.relative_to(ROOT)}`")
    st.divider()
    st.caption("The current training script supports Normal, Overhang and Underhang classes.")

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Sensor recording")
    st.write("Upload a numeric CSV recording from the microphone or acoustic sensor. For multi-column files, the first column is analyzed.")
    upload = st.file_uploader("Choose a sensor CSV", type=["csv"], label_visibility="collapsed")
    st.caption("The model analyzes the first 3 seconds. Shorter recordings are padded to 3 seconds.")
    st.info("Live microphone capture is not configured in this project yet. Use a recorded sensor CSV for inference.", icon="ℹ️")

with right:
    st.subheader("Latest result")
    if upload is None:
        st.markdown("### Waiting for a recording")
        st.markdown("<span class='muted'>Upload a sensor CSV to run the selected model.</span>", unsafe_allow_html=True)
    else:
        try:
            raw_signal = read_csv_signal(upload.getvalue())
            result, window = predict_signal(raw_signal, selected_model, svm_model, cnn_model)
            is_normal = result.label == "Normal"
            st.markdown(f"### {'Normal operation' if is_normal else 'Fault pattern detected'}")
            st.metric("Model result", result.label)
            st.progress(min(max(result.confidence, 0.0), 1.0), text=f"Confidence · {result.confidence:.1%}")
            st.caption(f"Analyzed {result.analyzed_samples:,} samples from **{upload.name}** with the {selected_model} model.")
            probabilities = result.class_scores
        except Exception as exc:
            st.error(f"Could not analyze this recording: {exc}")
            probabilities = {}
            raw_signal = None

if upload is not None and "raw_signal" in locals() and raw_signal is not None:
    st.divider()
    chart_col, prob_col = st.columns([1.35, 0.65], gap="large")
    with chart_col:
        st.subheader("Acoustic signal")
        stride = max(1, len(raw_signal) // 2500)
        signal_frame = pd.DataFrame({"Amplitude": raw_signal[::stride]})
        signal_frame.index = signal_frame.index * stride / preprocessing.TARGET_SR
        signal_frame.index.name = "Time (seconds)"
        st.line_chart(signal_frame, height=290, color="#267c70")
    with prob_col:
        st.subheader("Class scores")
        score_rows = [{"Condition": label, "Score": float(score)} for label, score in probabilities.items()]
        if score_rows:
            scores = pd.DataFrame(score_rows).sort_values("Score", ascending=True)
            scores = scores.set_index("Condition")[["Score"]]
            st.bar_chart(scores, height=290, horizontal=True, x_label="Model score", color="#267c70")

st.divider()
st.caption("Decision support only. Validate alerts against machine inspection, especially before using this dashboard on production equipment.")
