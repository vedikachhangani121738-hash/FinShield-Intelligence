import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="NIFTY50 Crash Risk Intelligence Dashboard", layout="wide")
st.markdown("""
    <style>
    /* Apply Times New Roman across the entire application */
    html, body, [class*="css"] {
        font-family: 'Times New Roman', Times, serif;
    }
    
    /* Enhance headers and titles */
    h1, h2, h3 {
        font-family: 'Times New Roman', Times, serif;
        font-weight: bold;
        color: #2C3E50;
    }

    /* Style metric containers like professional cards */
    [data-testid="stMetric"] {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Style dataframes for readability */
    dataframe {
        font-family: 'Times New Roman', Times, serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📉 NIFTY50 Stock Crash-Risk Prediction & Intelligence Dashboard")
st.markdown("**Academic Project Defense Demonstration | Model: Optimized Random Forest Classifier**")

@st.cache_data
def load_data():
    try:
        return pd.read_excel("NIFTY50_Val_Macro_Enhanced (3).xlsx")
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        return joblib.load("rf_model.pkl")
    except Exception:
        return None

val_df = load_data()
model = load_model()

# Automatically fetch the exact features the model was trained on
if model is not None and hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    # Safe fallback if feature_names_in_ isn't present
    feature_cols = [
        "Volatility_30D", "Market_Volatility_Index", "RSI_14", "SMA_50", "SMA_200", 
        "VWAP_20D", "Beta_60D", "Vol_x_Beta", 
        "Lagged_Return_5D", "Lagged_Volume_5D", "Volume_Spike_Ratio", "Month"
    ]

if val_df.empty or model is None:
    st.error("⚠️ Model or validation data file missing. Please make sure Steps 1 and 2 ran successfully!")
else:
    st.sidebar.header("🔍 Stock Lookup Panel")
    
    # Check available tickers
    ticker_col = "Ticker" if "Ticker" in val_df.columns else val_df.columns[0]
    selected_ticker = st.sidebar.selectbox("Select Stock Ticker", val_df[ticker_col].unique())
    
    ticker_data = val_df[val_df[ticker_col] == selected_ticker]
    
    date_col = "Date" if "Date" in ticker_data.columns else ticker_data.columns[1]
    ticker_data[date_col] = pd.to_datetime(ticker_data[date_col])
    ticker_data = ticker_data.sort_values(date_col, ascending=False)
    
    available_dates = ticker_data[date_col].dt.strftime('%Y-%m-%d').tolist()
    selected_date = st.sidebar.selectbox("Select Trading Date", available_dates)
    
    row_data = ticker_data[ticker_data[date_col].dt.strftime('%Y-%m-%d') == selected_date]
    
    if not row_data.empty:
        # Match features dynamically based on what's available
        available_features = [col for col in feature_cols if col in row_data.columns]
        X_input = row_data[available_features].fillna(0)
        
        # Pad missing columns with 0 if any feature column was missing during inference
        for col in feature_cols:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[feature_cols]

        prob = model.predict_proba(X_input.values)[:, 1][0]
        prediction = 1 if prob >= 0.30 else 0
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("🚨 **HIGH CRASH RISK DETECTED**")
                st.metric(label="Crash Probability Score", value=f"{prob:.2%}", delta="Above 30% Threshold")
            else:
                st.success("✅ **NORMAL / STABLE MARKET**")
                st.metric(label="Crash Probability Score", value=f"{prob:.2%}", delta="Below Risk Threshold", delta_color="inverse")
                
            st.markdown("---")
            st.info("**Decision Rule**: A custom **30% risk threshold** is applied to prioritize crash recall ($77\%$), ensuring maximum capital protection.")

        with col2:
            st.subheader("📊 Key Indicator Breakdown")
            st.markdown(f"Underlying quantitative indicators for **{selected_ticker}** on **{selected_date}**:")
            
            display_cols = [c for c in ["Volatility_30D", "Market_Volatility_Index", "RSI_14", "SMA_50_200_Ratio", "Volume_Spike_Ratio"] if c in row_data.columns]
            metrics_display = row_data[display_cols].T
            metrics_display.columns = ["Value"]
            st.dataframe(metrics_display, use_container_width=True)
            
            st.markdown("### 🧠 Primary Model Decision Driver")
            importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            top_feature = importances.index[0]
            st.write(f"The **Random Forest** model heavily weights **{top_feature}** and system-wide market volatility to assess structural anomaly thresholds for this symbol.")
