# FinShield-Intelligence
Institutional-grade financial risk intelligence platform featuring NIFTY50 stock crash prediction models and mutual fund risk ratings.
# 🛡️ Financial Risk Intelligence & Market Shield Platform

An integrated academic analytics web application designed for comprehensive portfolio risk assessment, featuring real-time **NIFTY50 stock crash prediction** and **mutual fund risk ratings**.

## 🚀 Key Features
* **📉 NIFTY50 Crash-Risk Forecasting**: Powered by an optimized Machine Learning classification model (Random Forest) trained on technical, historical, and macro-enhanced indicators to predict structural market stress.
* **📊 Mutual Fund Risk Analyzer**: Evaluates fund performance metrics and volatility metrics to provide robust risk ratings for investor protection.
* **🧠 Explainable Intelligence**: Provides transparent feature breakdown and probability thresholds to explain *why* a risk status was triggered.
* **🌐 Interactive Multi-Page Dashboard**: Built with Streamlit for seamless, real-time user interaction.

## 🛠️ Tech Stack
* **Language**: Python 3.8+
* **Framework**: Streamlit
* **Machine Learning**: Scikit-Learn, Joblib
* **Data Processing**: Pandas, NumPy, OpenPyXL

## 📂 Project Architecture
```text
📦 fin-risk-intelligence/
│
├── 📁 pages/
│   ├── 1_📉_Nifty50_Crash_Risk.py
│   └── 2_📊_Mutual_Fund_Risk.py
│
├── 🏠 Home.py
├── rf_model.pkl
├── NIFTY50_Val_Macro_Enhanced.xlsx
└── requirements.txt
