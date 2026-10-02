import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="FinShield | Global Crash-Risk Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Institutional UI & Times New Roman Styling
st.markdown("""
    <style>
    /* Global Font Styling */
    html, body, [class*="css"] {
        font-family: 'Times New Roman', Times, serif;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        font-family: 'Times New Roman', Times, serif;
        font-weight: 700;
        color: #1E293B;
        letter-spacing: -0.5px;
    }

    /* Professional Terminal Metric Cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Times New Roman', Times, serif;
        font-size: 14px;
        color: #64748B;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Times New Roman', Times, serif;
        font-size: 26px;
        font-weight: bold;
        color: #0F172A;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }

    .stAlert {
        border-radius: 6px;
        font-family: 'Times New Roman', Times, serif;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title & Subtitle Header
st.title("Global Stock Market Crash-Risk Prediction & Intelligence Dashboard")
st.markdown("**Institutional Research Platform | Model: Optimized Random Forest Classifier**")
st.markdown("---")

@st.cache_resource
def load_model():
    try:
        return joblib.load("rf_model.pkl")
    except Exception as e:
        st.error(f"❌ Model Loading Error: {e}")
        return None

model = load_model()

# Automatically fetch features expected by the model
if model is not None and hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    feature_cols = [
        "Volatility_30D", "Market_Volatility_Index", "RSI_14", "SMA_50", "SMA_200", 
        "VWAP_20D", "Beta_60D", "Vol_x_Beta", 
        "Lagged_Return_5D", "Lagged_Volume_5D", "Volume_Spike_Ratio", "Month"
    ]

# Sidebar Configuration Panel
st.sidebar.header("🎛️ Terminal Controls")
data_source = st.sidebar.radio("Select Data Stream", ["Validation Benchmark File", "Live Company Search (yfinance)"])

row_data = pd.DataFrame()
ticker_data = pd.DataFrame()
selected_ticker = ""
selected_date = ""
date_col = ""

if data_source == "Validation Benchmark File":
    @st.cache_data
    def load_val_data():
        try:
            return pd.read_excel("NIFTY50_Val_Macro_Enhanced (3).xlsx")
        except Exception as e:
            st.error(f"❌ Excel Loading Error: {e}")
            return pd.DataFrame()
    
    val_df = load_val_data()
    if not val_df.empty:
        ticker_col = "Ticker" if "Ticker" in val_df.columns else val_df.columns[0]
        selected_ticker = st.sidebar.selectbox("Select Asset Symbol", val_df[ticker_col].unique())
        ticker_data = val_df[val_df[ticker_col] == selected_ticker]
        
        date_col = "Date" if "Date" in ticker_data.columns else ticker_data.columns[1]
        ticker_data[date_col] = pd.to_datetime(ticker_data[date_col])
        ticker_data = ticker_data.sort_values(date_col, ascending=False)
        
        available_dates = ticker_data[date_col].dt.strftime('%Y-%m-%d').tolist()
        selected_date = st.sidebar.selectbox("Select Valuation Date", available_dates)
        row_data = ticker_data[ticker_data[date_col].dt.strftime('%Y-%m-%d') == selected_date]

else:
    st.sidebar.markdown("### 🔍 Live Company Search")
    search_query = st.sidebar.text_input("Type Company Name or Keyword", "Reliance")
    st.sidebar.caption("Examples: `Apple`, `Reliance`, `Tata`, `Microsoft`")
    
    selected_ticker = ""
    if search_query:
        try:
            search_results = yf.Search(search_query, max_results=8).quotes
            if search_results:
                options = {}
                for item in search_results:
                    if isinstance(item, dict):
                        sym = item.get('symbol')
                        name = item.get('longname', sym)
                    else:
                        sym = getattr(item, 'symbol', None)
                        name = getattr(item, 'longname', sym)
                        
                    if sym:
                        options[f"{name} ({sym})"] = sym

                if options:
                    chosen_label = st.sidebar.selectbox("Select Matching Company", list(options.keys()))
                    selected_ticker = options[chosen_label]
            else:
                st.sidebar.warning("No matching companies found. Try a different keyword.")
        except Exception as e:
            st.sidebar.error(f"Search API Error: {e}")

    if selected_ticker:
        @st.cache_data(ttl=3600)
        def fetch_live_data(symbol):
            df = yf.download(symbol, period="1y", interval="1d", progress=False)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df['Close_pct'] = df['Close'].pct_change()
            df['Volatility_30D'] = df['Close_pct'].rolling(30).std() * np.sqrt(252)
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
            df['VWAP_20D'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).mean()
            df['Beta_60D'] = 1.0  
            df['Vol_x_Beta'] = df['Volatility_30D'] * df['Beta_60D']
            df['Lagged_Return_5D'] = df['Close'].pct_change(5)
            df['Lagged_Return_10D'] = df['Close'].pct_change(10) if 'Lagged_Return_10D' in feature_cols else 0
            df['Lagged_Return_20D'] = df['Close'].pct_change(20) if 'Lagged_Return_20D' in feature_cols else 0
            df['Lagged_Volume_5D'] = df['Volume'].shift(5)
            df['Volume_Spike_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['SMA_50_200_Ratio'] = df['SMA_50'] / df['SMA_200']
            df['Market_Volatility_Index'] = 15.0  
            
            df['Date'] = df.index
            df['Month'] = df['Date'].dt.month
            return df.dropna()

        ticker_data = fetch_live_data(selected_ticker)
        if not ticker_data.empty:
            date_col = 'Date'
            available_dates = ticker_data[date_col].dt.strftime('%Y-%m-%d').tolist()
            selected_date = st.sidebar.selectbox("Select Live Trading Date", available_dates[::-1])
            row_data = ticker_data[ticker_data[date_col].dt.strftime('%Y-%m-%d') == selected_date]

# Main Dashboard View
if model is None or row_data.empty:
    st.warning("⚠️ Please select a company from the live search dropdown or validation file in the sidebar to initialize analytics.")
else:
    X_input = row_data.reindex(columns=feature_cols).fillna(0)
    
    prob = model.predict_proba(X_input.values)[:, 1][0]
    prediction = 1 if prob >= 0.30 else 0
    
    tab1, tab2, tab3 = st.tabs(["🛡️ Risk Assessment Scorecard", "📊 Technical Telemetry", "🧠 Model Explainability"])
    
    with tab1:
        col1, col2 = st.columns([1, 1.5], gap="large")
        
        with col1:
            st.subheader("System Status & Alert")
            if prediction == 1:
                st.error("🚨 **CRASH ANOMALY DETECTED**")
                st.metric(label="Calculated Risk Probability", value=f"{prob:.2%}", delta="Exceeds 30% Threshold", delta_color="inverse")
            else:
                st.success("✅ **STABLE MARKET EQUILIBRIUM**")
                st.metric(label="Calculated Risk Probability", value=f"{prob:.2%}", delta="Within Safe Tolerance", delta_color="normal")
                
            st.markdown("---")
            st.markdown("**Decision Rule Protocol**")
            st.info("A custom **30% risk threshold** is implemented to maximize tail-risk crash recall ($\approx 77\%$), prioritizing capital preservation over false alarms.")

        with col2:
            st.subheader("Asset Telemetry Overview")
            st.markdown(f"Target Symbol: **{selected_ticker}** | Timestamp: **{selected_date}**")
            
            if 'Close' in row_data.columns:
                close_val = row_data['Close'].values[0]
                st.metric(label="Latest Settlement Price", value=f"{close_val:,.2f}" if isinstance(close_val, (int, float)) else str(close_val))
            
            display_cols = [c for c in ["Volatility_30D", "RSI_14", "SMA_50_200_Ratio", "Volume_Spike_Ratio", "Market_Volatility_Index"] if c in row_data.columns]
            if display_cols:
                metrics_table = row_data[display_cols].T.rename(columns={row_data.index[0]: "Indicator Value"})
                st.dataframe(metrics_table, use_container_width=True)

    with tab2:
        st.subheader("Historical Price Action & Volatility Profile")
        if not ticker_data.empty and 'Close' in ticker_data.columns and date_col:
            chart_df = ticker_data.set_index(date_col)[['Close']]
            st.line_chart(chart_df, use_container_width=True)
        else:
            st.info("Price chart data unavailable for current selection.")

    with tab3:
        st.subheader("Feature Importance Attribution")
        st.markdown("The chart below displays the relative weight assigned by the **Random Forest Classifier** across evaluated macroeconomic and technical dimensions:")
        
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        st.bar_chart(importances, use_container_width=True)
