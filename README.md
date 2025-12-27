🩺 Intelligent Health Monitoring System
📋 Project Overview

The Intelligent Health Monitoring System analyzes PPG (Photoplethysmography) and accelerometer signals to provide real-time health insights. The system can:

Recognize human activities: Sit, Walk, Run 🪑🚶‍♂️🏃‍♂️

Estimate Heart Rate (HR) ❤️

Estimate Blood Oxygen Saturation (SpO₂) 🩸

Provide explainable predictions using SHAP-based XAI module 🧩

It combines signal processing, feature extraction, deep learning (MLP models), and Explainable AI with an interactive Streamlit-based user interface for easy visualization and interpretation.

✨ Key Features
📈 PPG Signal Processing

Bandpass filtering for noise removal 🔇

Peak-based heart rate estimation ❤️

🩸 SpO₂ Estimation

Physiological ratio-of-ratios (R) method using red & infrared PPG signals

🏃‍♂️ Activity Recognition

Window-based accelerometer feature extraction

Multi-Layer Perceptron (MLP) classifier

Window-wise and aggregated final predictions

🧩 Explainable AI (XAI)

SHAP-based interpretation of model predictions

Understand feature contributions for activity and physiological parameter predictions

Visualize which features influence the predictions most 🔍

🖥 Interactive UI

Upload CSV files with biomedical signals 📂

Visualize filtered signals and predictions 📊

See final and per-window results

Explore XAI visualizations for better interpretability 🧠

🛠 Technologies Used

Python 🐍

NumPy, Pandas 📊

SciPy (signal processing) ⚙️

Scikit-learn (MLP models, scaling) 🤖

Matplotlib 📈

Streamlit (interactive web interface) 🌐

Joblib (model saving/loading) 💾

SHAP (Explainable AI) 🧩

📂 Project Structure

├── project.py # Streamlit application (run this to launch the project)


├── README.md

🔄 Workflow

Upload raw biomedical CSV data 📂

Signals are filtered and segmented into windows 🔄

Features are extracted per window ⚙️

Trained MLP models predict:

Activity (classification) 🏃‍♂️

Physiological R-ratio (regression) → computes HR ❤️ and SpO₂ 🩸

XAI module interprets feature contributions for better understanding 🧩

Aggregated results and explanations are displayed visually 📊

🚀 How to Run

Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run project.py

📊 Output

Visualized filtered PPG waveforms 📈

Final predicted Activity, HR, and SpO₂ 🏃‍♂️❤️🩸

Optional window-by-window predictions 🔍

SHAP-based XAI explanations showing feature contributions 🧩

📥 Dataset

This project uses the Pulse Transit Time PPG dataset from PhysioNet:
https://physionet.org/content/pulse-transit-time-ppg/1.1.0/