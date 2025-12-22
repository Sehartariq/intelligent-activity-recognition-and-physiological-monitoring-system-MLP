Title: Intelligent Health Monitoring System

 **Project Overview**

This project presents an Intelligent Health Monitoring System that analyzes PPG (Photoplethysmography) and accelerometer signals to perform:

* Human activity recognition (sit, walk, run)

* Heart Rate (HR) estimation

* Blood Oxygen Saturation (SpO₂) estimation

The system combines signal processing, feature extraction, and Machine Learning (MLP models) with an interactive Streamlit-based user interface.

 **Key Features**

**PPG Signal Processing**

Bandpass filtering for noise removal

Peak-based heart rate estimation

**SpO₂ Estimation**

Uses physiological ratio-of-ratios (R) method from red & infrared PPG

**Activity Recognition**

* Window-based accelerometer feature extraction
* Multi-Layer Perceptron (MLP) classifier

Window-wise & Final Predictions

Per-window predictions

Aggregated final health metrics

**Interactive UI**

* CSV upload
* Signal visualization
* Final and window-by-window results

**Technologies Used**

* Python
* NumPy, Pandas
* SciPy (signal processing)
* Scikit-learn (MLP models, scaling)
* Matplotlib
* Streamlit (web interface)
* Joblib (model loading)

 Project Structure
├── app.py                     # Streamlit application
├── Deep Learning Module/
│   └── models/
│       ├── activity_classification_mlp.pkl
│       ├── activity_scaler.pkl
│       ├── physio_mlp.pkl
│       ├── physio_scaler_X.pkl
│       └── physio_scaler_y.pkl
├── Data/
│   └── processed/
│       └── biomedical_preprocessed.csv
├── README.md

 **Workflow**

* Upload raw biomedical CSV data
* Signals are filtered and segmented into windows
* Features are extracted per window
* Trained MLP models predict:
* Activity (classification)
* Physiological R-ratio (regression)
* HR and SpO₂ are computed and aggregated
* Results are displayed visually

 **How to Run**

pip install -r requirements.txt

streamlit run app.py

**Output**

Filtered PPG waveform visualization

Final predicted activity, HR, and SpO₂

Optional window-by-window predictions