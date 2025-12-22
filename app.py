# ==========================================
# app.py - Intelligent Health Monitoring UI
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import streamlit as st
import joblib

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="🩺 Intelligent Health Monitoring", layout="centered")

# ------------------------
# Pastel Heart Theme CSS
# ------------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #ffb3b3;  /* dim pastel red */
        color: #0b1d51;  /* dark blue text */
        font-family: 'Helvetica', sans-serif;
    }
    h1, h2, h3 {
        color: #0b1d51;
    }
    .pred-container {
        background-color: #ffd9d9;
        padding: 20px;
        border-radius: 10px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #ff7f7f;
        color: #0b1d51;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Title
# ------------------------
st.markdown("<h1 style='text-align:center;'>🩺 Intelligent Health Monitoring System ❤️</h1>", unsafe_allow_html=True)

# ------------------------
# Feature Lists
# ------------------------
ACTIVITY_FEATURES = [
    'ax_rms', 'ax_std', 'ay_energy', 'peak_count', 'az_entropy', 'ax_energy',
    'az_rms', 'ppg_rms', 'az_std', 'sma', 'ax_entropy', 'ay_entropy',
    'az_energy', 'ay_rms', 'red_ac', 'ay_std'
]
PHYSIO_FEATURES = ['ppg_rms', 'ppg_std', 'red_ac', 'peak_count', 'ir_ac']


# ------------------------
# Load Models
# ------------------------
@st.cache_resource
def load_models():
    activity_model = joblib.load("Deep Learning Module/models/activity_classification_mlp.pkl")
    activity_scaler = joblib.load("Deep Learning Module/models/activity_scaler.pkl")
    physio_model = joblib.load("Deep Learning Module/models/physio_mlp.pkl")
    physio_scaler_X = joblib.load("Deep Learning Module/models/physio_scaler_X.pkl")
    physio_scaler_y = joblib.load("Deep Learning Module/models/physio_scaler_y.pkl")
    label_encoder = joblib.load("Deep Learning Module/models/activity_label_encoder.pkl")
    return activity_model, activity_scaler, physio_model, physio_scaler_X, physio_scaler_y, label_encoder


activity_model, activity_scaler, physio_model, physio_scaler_X, physio_scaler_y, le = load_models()


# ------------------------
# Signal Filters
# ------------------------
def bandpass_ppg(sig, fs=500, lowcut=0.5, highcut=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, sig)


def highpass_accel(acc, fs=500, cutoff=0.5):
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = butter(2, high, btype='high')
    return filtfilt(b, a, acc, axis=0)


# ------------------------
# Feature Extraction
# ------------------------
def extract_features(ppg_green, ppg_red, ppg_ir, acc_window):
    feats = {}
    feats['ppg_rms'] = np.sqrt(np.mean(ppg_green ** 2))
    feats['ppg_std'] = np.std(ppg_green)
    peaks, _ = find_peaks(ppg_green, distance=100)
    feats['peak_count'] = len(peaks)
    feats['red_ac'] = np.std(ppg_red)
    feats['ir_ac'] = np.std(ppg_ir)
    axes = ['x', 'y', 'z']
    for i, axis in enumerate(axes):
        sig = acc_window[:, i]
        feats[f'a{axis}_rms'] = np.sqrt(np.mean(sig ** 2))
        feats[f'a{axis}_std'] = np.std(sig)
        feats[f'a{axis}_energy'] = np.sum(sig ** 2)
        feats[f'a{axis}_entropy'] = -np.sum(np.histogram(sig, bins=10, density=True)[0] *
                                            np.log(np.histogram(sig, bins=10, density=True)[0] + 1e-8))
    feats['sma'] = np.mean(np.abs(acc_window))
    return feats


# ------------------------
# Prediction Function
# ------------------------
def predict_health(file):
    df = pd.read_csv(file)
    required_cols = ['pleth_1', 'pleth_2', 'pleth_3', 'a_x', 'a_y', 'a_z']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Filter signals
    ppg_green = bandpass_ppg(df['pleth_3'].values)
    ppg_red = bandpass_ppg(df['pleth_1'].values)
    ppg_ir = bandpass_ppg(df['pleth_2'].values)
    acc = highpass_accel(df[['a_x', 'a_y', 'a_z']].values)

    fs = 500
    win_samples = fs * 5
    n_windows = len(ppg_green) // win_samples

    feature_rows = []
    for i in range(n_windows):
        s, e = i * win_samples, (i + 1) * win_samples
        acc_window = acc[s:e, :]
        feature_rows.append(extract_features(ppg_green[s:e], ppg_red[s:e], ppg_ir[s:e], acc_window))

    feat_df = pd.DataFrame(feature_rows)

    # ------------------------
    # Activity Prediction per window
    # ------------------------
    X_act_scaled = activity_scaler.transform(feat_df[ACTIVITY_FEATURES])
    activity_pred_idx = activity_model.predict(X_act_scaled)
    activity_pred_labels = le.inverse_transform(activity_pred_idx)

    # ------------------------
    # Physio Prediction per window
    # ------------------------
    window_pred_hr = []
    window_pred_spo2 = []
    for i in range(len(feat_df)):
        X_phy_scaled = physio_scaler_X.transform(feat_df[PHYSIO_FEATURES].iloc[[i]])
        y_scaled = physio_model.predict(X_phy_scaled)
        y = physio_scaler_y.inverse_transform(y_scaled)
        window_pred_hr.append(y[0, 0])

        # Convert predicted R ratio to SpO₂ using calibration formula
        R_ratio = y[0, 1]
        spo2 = 110 - 25 * R_ratio  # Standard calibration formula
        spo2 = np.clip(spo2, 0, 100)  # safety clamp
        window_pred_spo2.append(spo2)

    # ------------------------
    # Aggregate final prediction
    # ------------------------
    final_pred = pd.DataFrame({
        "Activity": [pd.Series(activity_pred_labels).mode()[0]],
        "HR (bpm)": [np.mean(window_pred_hr)],
        "SpO₂ (%)": [np.mean(window_pred_spo2)]
    })

    return final_pred, feat_df, ppg_green, win_samples, activity_pred_labels, window_pred_hr, window_pred_spo2


# ------------------------
# Streamlit UI
# ------------------------
uploaded_file = st.file_uploader("Upload raw CSV with PPG and accelerometer signals", type="csv")
if uploaded_file:
    with st.spinner("Processing..."):
        final_pred, window_df, ppg_green, win_samples, activity_pred_labels, window_hr, window_spo2 = predict_health(
            uploaded_file)

        # ------------------------
        # Show waveform
        # ------------------------
        st.subheader("Filtered Green PPG Waveform (First Window)")
        fig, ax = plt.subplots(figsize=(12, 4))
        peaks, _ = find_peaks(ppg_green[:win_samples], distance=100)
        ax.plot(ppg_green[:win_samples], label="Filtered Green PPG", color='#1f77b4')
        ax.scatter(peaks, ppg_green[:win_samples][peaks], color='#ff4d4d', label="Detected Peaks")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)

        # ------------------------
        # Show final aggregated prediction
        # ------------------------
        st.subheader("Final Prediction")
        st.dataframe(final_pred, use_container_width=True)

        # ------------------------
        # Window-by-window prediction (optional)
        # ------------------------
        if st.checkbox("Show Window-by-Window Predictions"):
            window_pred_df = pd.DataFrame({
                "Window": np.arange(len(window_df)),
                "Activity": activity_pred_labels,
                "HR (bpm)": window_hr,
                "SpO₂ (%)": window_spo2
            })
            st.subheader("Window-by-Window Predictions")
            st.dataframe(window_pred_df, use_container_width=True)
