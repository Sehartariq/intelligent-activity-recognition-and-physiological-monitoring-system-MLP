# ==========================================
# project.py - Intelligent Health Monitoring UI
# (UI ONLY UPDATED — LOGIC 100% UNCHANGED + XAI)
# ==========================================

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import streamlit as st
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="🩺 Intelligent Health Monitoring", layout="wide")

# ------------------------
# ORIGINAL COLORS + SMOOTH ANIMATIONS
# ------------------------
st.markdown("""
<style>
.stApp {
    background-color: #ffb3b3;
    color: #0b1d51;
    font-family: 'Helvetica', sans-serif;
    animation: fadeIn 0.8s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(15px);}
    to {opacity: 1; transform: translateY(0);}
}
h1, h2, h3 { color: #0b1d51; text-align:center; }

.card {
    background-color: #ffd9d9;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.12);
}
.card:hover {
    transform: scale(1.03);
    transition: 0.3s ease;
}

.stButton>button {
    background-color: #ff7f7f;
    color: #0b1d51;
    font-weight: bold;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Title
# ------------------------
st.markdown("<h1>🩺 Intelligent Health Monitoring System ❤️</h1>", unsafe_allow_html=True)

# ------------------------
# LOAD MODELS (UNCHANGED)
# ------------------------
@st.cache_resource
def load_models():
    activity_model = load_model("Deep Learning Module/Deep Learning Module/models/activity_classification_mlp")
    activity_scaler = joblib.load("Deep Learning Module/Deep Learning Module/models/activity_scaler.pkl")
    label_encoder = joblib.load("Deep Learning Module/Deep Learning Module/models/activity_label_encoder.pkl")
    ACTIVITY_FEATURES = joblib.load("Deep Learning Module/Deep Learning Module/models/activity_features.pkl")

    physio_model_tf = load_model("Deep Learning Module/Deep Learning Module/models/physio_mlp_tf")
    physio_scaler_X = joblib.load("Deep Learning Module/Deep Learning Module/models/physio_scaler_X_tf.pkl")
    physio_scaler_y = joblib.load("Deep Learning Module/Deep Learning Module/models/physio_scaler_y_tf.pkl")
    PHYSIO_FEATURES = ['ppg_rms', 'ppg_std', 'red_ac', 'peak_count', 'ir_ac']

    return (activity_model, activity_scaler, label_encoder, ACTIVITY_FEATURES,
            physio_model_tf, physio_scaler_X, physio_scaler_y, PHYSIO_FEATURES)

(activity_model, activity_scaler, le, ACTIVITY_FEATURES,
 physio_model, physio_scaler_X, physio_scaler_y, PHYSIO_FEATURES) = load_models()

# ------------------------
# FILTERS (UNCHANGED)
# ------------------------
def bandpass_ppg(sig, fs=500, lowcut=0.5, highcut=5):
    nyq = 0.5 * fs
    b, a = butter(2, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, sig)

def highpass_accel(acc, fs=500, cutoff=0.5):
    nyq = 0.5 * fs
    b, a = butter(2, cutoff/nyq, btype='high')
    return filtfilt(b, a, acc, axis=0)

# ------------------------
# FEATURE EXTRACTION (UNCHANGED)
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
        feats[f'a{axis}_entropy'] = -np.sum(
            np.histogram(sig, bins=10, density=True)[0] *
            np.log(np.histogram(sig, bins=10, density=True)[0] + 1e-8)
        )
    feats['sma'] = np.mean(np.abs(acc_window))
    return feats

# ------------------------
# PREDICTION FUNCTION (UNCHANGED)
# ------------------------
def predict_health(file):
    df = pd.read_csv(file)
    required_cols = ['pleth_1', 'pleth_2', 'pleth_3', 'a_x', 'a_y', 'a_z']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

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
        feature_rows.append(
            extract_features(ppg_green[s:e], ppg_red[s:e], ppg_ir[s:e], acc[s:e])
        )

    feat_df = pd.DataFrame(feature_rows)

    X_act = activity_scaler.transform(feat_df[ACTIVITY_FEATURES])
    act_idx = np.argmax(activity_model.predict(X_act), axis=1)
    act_labels = le.inverse_transform(act_idx)

    hr, spo2 = [], []
    for i in range(len(feat_df)):
        Xp = physio_scaler_X.transform(feat_df[PHYSIO_FEATURES].iloc[[i]])
        y = physio_scaler_y.inverse_transform(physio_model.predict(Xp))
        hr.append(y[0, 0])
        spo2.append(np.clip(110 - 25 * y[0, 1], 0, 100))

    final = pd.DataFrame({
        "Activity": [pd.Series(act_labels).mode()[0]],
        "HR (bpm)": [np.mean(hr)],
        "SpO₂ (%)": [np.mean(spo2)]
    })

    # Compute SHAP values for XAI
    X_scaled = physio_scaler_X.transform(feat_df[PHYSIO_FEATURES])
    explainer = shap.Explainer(physio_model, X_scaled)
    shap_values = explainer(X_scaled)


    return final, feat_df, ppg_green, win_samples, act_labels, hr, spo2, shap_values


# ------------------------
# UI
# ------------------------
uploaded_file = st.file_uploader("Upload raw CSV with PPG and accelerometer signals", type="csv")

if uploaded_file:
    with st.spinner("Processing..."):
        final_pred, feat_df, ppg_green, win_samples, acts, hr, spo2, shap_values = predict_health(uploaded_file)

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='card'><h3>🏃 Activity</h3><h2>{final_pred.iloc[0,0]}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h3>❤️ Heart Rate</h3><h2>{final_pred.iloc[0,1]:.1f} bpm</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h3>🩸 SpO₂</h3><h2>{final_pred.iloc[0,2]:.1f} %</h2></div>", unsafe_allow_html=True)

    tabs = st.tabs(["📈 PPG Waveform", "📊 Window-by-Window Predictions", "🧠 Explainable AI (XAI)"])

    with tabs[0]:
        peaks, _ = find_peaks(ppg_green[:win_samples], distance=100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=ppg_green[:win_samples],
            mode='lines',
            name='Filtered Green PPG',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=peaks,
            y=ppg_green[:win_samples][peaks],
            mode='markers',
            name='Detected Peaks',
            marker=dict(color='red', size=8)
        ))
        fig.update_layout(
            height=450,
            plot_bgcolor="#ffd9d9",
            paper_bgcolor="#ffd9d9",
            xaxis_title="Sample",
            yaxis_title="Amplitude"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.dataframe(pd.DataFrame({
            "Window": np.arange(len(hr)),
            "Activity": acts,
            "HR (bpm)": hr,
            "SpO₂ (%)": spo2
        }), use_container_width=True)

    # ------------------------
    # ------------------------
    # ------------------------
    # XAI TAB (Activity + Physio)
    # ------------------------
    with tabs[2]:
        st.markdown("<h3 style='text-align:center;'>🧠 Explainable AI: Feature Contributions (SHAP)</h3>",
                    unsafe_allow_html=True)

        if len(feat_df) > 0:
            feat_names_act = ACTIVITY_FEATURES
            feat_names_physio = PHYSIO_FEATURES

            # --- Activity Classification ---
            st.markdown(f"### 🏃 Activity Prediction: {pd.Series(acts).mode()[0]}")
            X_act_scaled = activity_scaler.transform(feat_df[ACTIVITY_FEATURES])
            explainer_act = shap.Explainer(activity_model, X_act_scaled)
            shap_vals_act = explainer_act(X_act_scaled)

            # Collapse samples (and classes if multi-class) to get mean 1D per feature
            if shap_vals_act.values.ndim == 3:  # (samples, features, classes)
                shap_act_mean = np.mean(np.abs(shap_vals_act.values), axis=(0, 2))
            else:  # (samples, features)
                shap_act_mean = np.mean(np.abs(shap_vals_act.values), axis=0)

            fig_act = px.bar(
                x=shap_act_mean,
                y=feat_names_act,
                orientation='h',
                text=np.round(shap_act_mean, 3),
                labels={'x': 'Mean |SHAP Value|', 'y': 'Feature'},
                title="Feature Contributions to Activity Prediction"
            )
            fig_act.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=400,
                plot_bgcolor="#ffd9d9",
                paper_bgcolor="#ffd9d9"
            )
            st.plotly_chart(fig_act, use_container_width=True)

            # Human-readable explanation for top 3 features
            top_idx_act = np.argsort(shap_act_mean)[::-1][:3]
            st.markdown("**Top features influencing activity prediction:**")
            for idx in top_idx_act:
                f = feat_names_act[idx]
                val = shap_act_mean[idx]
                impact = "increased" if val > 0 else "decreased"
                st.markdown(f"- **{f}** contributed to {impact} the prediction of this activity.")

            st.markdown("---")

            # --- Physiological Predictions ---
            outputs_physio = {"HR": np.mean(hr), "SpO₂": np.mean(spo2)}
            shap_vals_physio = shap_values.values  # computed in predict_health

            feature_desc_physio = {
                'ppg_rms': 'RMS amplitude of green PPG signal',
                'ppg_std': 'Standard deviation of green PPG signal',
                'peak_count': 'Number of detected peaks in green PPG',
                'red_ac': 'Std of red PPG signal',
                'ir_ac': 'Std of IR PPG signal'
            }

            for i, output_name in enumerate(outputs_physio.keys()):
                st.markdown(f"### ❤️ {output_name} Prediction: {outputs_physio[output_name]:.2f}")

                # Collapse samples to get mean 1D SHAP per feature
                if shap_vals_physio.ndim == 2:
                    shap_to_plot = shap_vals_physio[:, i]
                else:
                    shap_to_plot = np.mean(shap_vals_physio[:, :, i], axis=0)

                fig_physio = px.bar(
                    x=shap_to_plot,
                    y=feat_names_physio,
                    orientation='h',
                    text=np.round(shap_to_plot, 3),
                    labels={'x': 'SHAP Value', 'y': 'Feature'},
                    title=f"Feature Contributions to {output_name}"
                )
                fig_physio.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=400,
                    plot_bgcolor="#ffd9d9",
                    paper_bgcolor="#ffd9d9"
                )
                st.plotly_chart(fig_physio, use_container_width=True)

                # Human-readable explanation for top 3 features
                top_idx_physio = np.argsort(np.abs(shap_to_plot))[::-1][:3]
                st.markdown(f"**Top features influencing {output_name}:**")
                for idx in top_idx_physio:
                    f = feat_names_physio[idx]
                    val = shap_to_plot[idx]
                    impact = "increased" if val > 0 else "decreased"
                    st.markdown(
                        f"- **{f}** ({feature_desc_physio.get(f, 'No description')}) {impact} the predicted {output_name}.")
        else:
            st.write("No feature data available for XAI visualization.")
