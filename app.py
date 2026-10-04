import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & INSTITUTIONAL TERMINAL STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinShield | Institutional Risk Intelligence Terminal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS (Dark Terminal Theme styled after Bloomberg / TradingView)

st.markdown("""
    <style>
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .risk-badge-critical {
        animation: pulse-red 2s infinite;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .stMetric, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .terminal-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-left: 5px solid #3B82F6;
        padding: 16px 22px;
        border-radius: 8px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .terminal-title {
        font-size: 24px;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .terminal-subtitle {
        font-size: 13px;
        color: #94A3B8;
        margin-top: 4px;
    }

    [data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
        color: #F8FAFC;
    }

    .risk-badge-critical {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 12px;
    }
    .risk-badge-elevated {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 12px;
    }
    .risk-badge-stable {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22C55E;
        border: 1px solid #22C55E;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 12px;
    }

    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p {
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="terminal-header">
    <div class="terminal-title">🛡️ FinShield Risk Intelligence Terminal</div>
    <div class="terminal-subtitle">Institutional Multi-Asset Tail-Risk Analytics & ML Crash Diagnostics | Model Engine: Random Forest Classifier</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL LOADING & FEATURE INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("rf_model.pkl")
    except Exception as e:
        st.error(f"❌ Model Loading Failure: {e}")
        return None

model = load_model()

# Align feature expectations strictly with model definition (`model.feature_names_in_`)
if model is not None and hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    feature_cols = [
        "Volatility_30D", "Market_Volatility_Index", "RSI_14", "SMA_50", "SMA_200", 
        "VWAP_20D", "Beta_60D", "Vol_x_Beta", 
        "Lagged_Return_5D", "Lagged_Volume_5D", "Volume_Spike_Ratio", "Month"
    ]

# -----------------------------------------------------------------------------
# 3. SIDEBAR TERMINAL CONTROLS & DATA STREAM SELECTION
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ Data Stream Controls")
data_source = st.sidebar.radio("Select Data Engine", ["Validation Benchmark File", "Live Global Search (yfinance)"])

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
            st.error(f"❌ Benchmark Dataset Load Error: {e}")
            return pd.DataFrame()
    
    val_df = load_val_data()
    if not val_df.empty:
        ticker_col = "Ticker" if "Ticker" in val_df.columns else val_df.columns[0]
        selected_ticker = st.sidebar.selectbox("Select Benchmark Asset Symbol", val_df[ticker_col].unique())
        ticker_data = val_df[val_df[ticker_col] == selected_ticker].copy()
        
        date_col = "Date" if "Date" in ticker_data.columns else ticker_data.columns[1]
        ticker_data[date_col] = pd.to_datetime(ticker_data[date_col])
        ticker_data = ticker_data.sort_values(date_col, ascending=False)
        
        available_dates = ticker_data[date_col].dt.strftime('%Y-%m-%d').tolist()
        selected_date = st.sidebar.selectbox("Valuation Timestamp", available_dates)
        row_data = ticker_data[ticker_data[date_col].dt.strftime('%Y-%m-%d') == selected_date]

else:
    st.sidebar.markdown("### 🔍 Live Global Ticker Search")
    search_query = st.sidebar.text_input("Company Name or Ticker Keyword", "Reliance")
    st.sidebar.caption("Examples: `Apple`, `Reliance`, `Tata Motors`, `Microsoft`, `NVDA`")
    
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
                    chosen_label = st.sidebar.selectbox("Select Target Equity", list(options.keys()))
                    selected_ticker = options[chosen_label]
            else:
                st.sidebar.warning("No matching equity instruments found.")
        except Exception as e:
            st.sidebar.error(f"Search API Query Error: {e}")

    if selected_ticker:
        @st.cache_data(ttl=3600)
        def fetch_live_data(symbol):
            # Using period="max" provides sufficient historical runway for 200-day moving averages
            # without running into rolling truncation bottlenecks.
            df = yf.download(symbol, period="max", interval="1d", progress=False)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df['Close_pct'] = df['Close'].pct_change()
            df['Volatility_30D'] = df['Close_pct'].rolling(30).std() * np.sqrt(252)
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss.replace(0, 1e-6))
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
            df['VWAP_20D'] = (df['Close'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum().replace(0, 1))
            df['Beta_60D'] = 1.0  
            df['Vol_x_Beta'] = df['Volatility_30D'] * df['Beta_60D']
            df['Lagged_Return_5D'] = df['Close'].pct_change(5)
            df['Lagged_Return_10D'] = df['Close'].pct_change(10) if 'Lagged_Return_10D' in feature_cols else 0
            df['Lagged_Return_20D'] = df['Close'].pct_change(20) if 'Lagged_Return_20D' in feature_cols else 0
            df['Lagged_Volume_5D'] = df['Volume'].shift(5)
            df['Volume_Spike_Ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean().replace(0, 1))
            df['SMA_50_200_Ratio'] = df['SMA_50'] / (df['SMA_200'].replace(0, 1))
            df['Market_Volatility_Index'] = 15.0  
            
            df['Date'] = df.index
            df['Month'] = df['Date'].dt.month
            return df.dropna()

        ticker_data = fetch_live_data(selected_ticker)
        if not ticker_data.empty:
            date_col = 'Date'
            available_dates = ticker_data[date_col].dt.strftime('%Y-%m-%d').tolist()
            selected_date = st.sidebar.selectbox("Live Market Session Date", available_dates[::-1])
            row_data = ticker_data[ticker_data[date_col].dt.strftime('%Y-%m-%d') == selected_date]

# -----------------------------------------------------------------------------
# 4. MAIN TERMINAL DASHBOARD
# -----------------------------------------------------------------------------
if model is None or row_data.empty:
    st.info("💡 **Terminal Ready**: Select an asset from the sidebar or search a company keyword to initialize telemetry analytics.")
else:
    # Safely reindex feature matrix to match exact training columns and prevent shape mismatch
    X_input = row_data.reindex(columns=feature_cols).fillna(0)
    
    # Calculate Crash Risk Probability
    prob = float(model.predict_proba(X_input.values)[:, 1][0])
    is_anomaly = prob >= 0.30
    
    tab1, tab2, tab3 = st.tabs([
        "🛡️ Executive Risk Scorecard", 
        "📊 Technical Telemetry & Signals", 
        "🧠 Model Attribution & Diagnostics (XAI)"
    ])
    
    # -------------------------------------------------------------------------
    # TAB 1: EXECUTIVE RISK SCORECARD
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown(f"### Asset Overview: **{selected_ticker}** | Valuation Timestamp: **{selected_date}**")
        
        crash_prob_pct = round(prob * 100, 1)
        threshold = 30.0
        is_high_risk = crash_prob_pct >= threshold

        if is_high_risk:
            bg_gradient = "linear-gradient(145deg, #2b1d1d 0%, #4a1515 100%)"
            border_color = "#EF4444"
            shadow_color = "rgba(239, 68, 68, 0.4)"
            status_text = f"▲ High Risk (Above {threshold}% Threshold)"
            status_color = "#FCA5A5"
        else:
            bg_gradient = "linear-gradient(145deg, #1b2e1b 0%, #163820 100%)"
            border_color = "#10B981"
            shadow_color = "rgba(16, 185, 129, 0.4)"
            status_text = f"▼ Safe (Below {threshold}% Threshold)"
            status_color = "#6EE7B7"

        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(
                f"""
                <div style="
                    background: {bg_gradient};
                    border: 2px solid {border_color};
                    border-radius: 10px;
                    padding: 14px 16px;
                    box-shadow: 0 6px 20px {shadow_color};
                ">
                    <div style="color: #9CA3AF; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase;">CRASH PROBABILITY</div>
                    <div style="color: #FFFFFF; font-size: 24px; font-weight: 700; margin: 4px 0 2px 0;">{crash_prob_pct}%</div>
                    <div style="color: {status_color}; font-size: 11px; font-weight: 500;">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with c2:
            if 'Close' in row_data.columns:
                close_p = row_data['Close'].values[0]
                price_str = f"${close_p:,.2f}" if isinstance(close_p, (int, float)) else str(close_p)
                st.metric(label="SETTLEMENT PRICE", value=price_str)
        
        with c3:
            if 'Volatility_30D' in row_data.columns:
                vol_val = row_data['Volatility_30D'].values[0]
                st.metric(label="ANNUALIZED VOLATILITY (30D)", value=f"{vol_val:.1%}")
            
        with c4:
            if 'RSI_14' in row_data.columns:
                rsi_val = row_data['RSI_14'].values[0]
                st.metric(label="RSI (14-DAY)", value=f"{rsi_val:.1f}")
        
        st.markdown("---")
        
        col_left, col_right = st.columns([1.1, 1.3], gap="large")
        
        with col_left:
            st.subheader("System Anomaly Classification")
            
            if prob >= 0.30:
                st.markdown('<div class="risk-badge-critical">🚨 CRITICAL TAIL-RISK ANOMALY DETECTED</div>', unsafe_allow_html=True)
                st.progress(min(int(prob * 100), 100))
                st.error("**Directive**: Model signals heightened probability of severe downside drawdown (>10% drop within 5-10 trading sessions). Preemptive risk reduction recommended.")
            elif prob >= 0.15:
                st.markdown('<div class="risk-badge-elevated">⚠ ELEVATED WATCHLIST STATUS</div>', unsafe_allow_html=True)
                st.progress(min(int(prob * 100), 100))
                st.warning("**Directive**: Asset displays moderate volatility buildup. Monitor support levels and liquidity metrics closely.")
            else:
                st.markdown('<div class="risk-badge-stable">✅ STABLE MARKET EQUILIBRIUM</div>', unsafe_allow_html=True)
                st.progress(min(int(prob * 100), 100))
                st.success("**Directive**: Risk metrics remain within normal historical tolerance bounds. Standard position limits apply.")
                
            st.markdown("#### Decision Protocol Specification")
            st.caption("""
            * **Operational Timing**: Inference executes post-market close on Day $T$, using completed session variables to project tail-risk probability for Day $T+1$.
            * **Model Threshold**: 30% Probability.
            * **Calibration Rationale**: Optimized on historical NIFTY50 market crash cycles to capture tail-risk while mitigating false positives.
            """)

        with col_right:
            st.subheader("Key Primary Risk Drivers")
            st.markdown("Automated scan of evaluated features driving current probability classification:")
            
            drivers = []
            if 'Volatility_30D' in row_data.columns and row_data['Volatility_30D'].values[0] > 0.25:
                drivers.append(("Annualized Volatility (30D)", f"{row_data['Volatility_30D'].values[0]:.1%}", "HIGH", "High price dispersion increases crash likelihood."))
            if 'RSI_14' in row_data.columns:
                r_val = row_data['RSI_14'].values[0]
                if r_val > 70:
                    drivers.append(("RSI (14-Day)", f"{r_val:.1f}", "OVERBOUGHT", "Overextended momentum increases pullback vulnerability."))
                elif r_val < 30:
                    drivers.append(("RSI (14-Day)", f"{r_val:.1f}", "OVERSOLD", "Severe momentum breakdown detected."))
            if 'Volume_Spike_Ratio' in row_data.columns and row_data['Volume_Spike_Ratio'].values[0] > 1.8:
                drivers.append(("Volume Spike Ratio", f"{row_data['Volume_Spike_Ratio'].values[0]:.2f}x", "ELEVATED", "Abnormal institutional volume outflow detected."))
            if 'Lagged_Return_5D' in row_data.columns and row_data['Lagged_Return_5D'].values[0] < -0.04:
                drivers.append(("5-Day Trailing Return", f"{row_data['Lagged_Return_5D'].values[0]:.1%}", "NEGATIVE", "Short-term downward trend momentum."))

            if drivers:
                driver_df = pd.DataFrame(drivers, columns=["Indicator", "Observed Value", "Condition", "Risk Implication"])
                st.dataframe(driver_df, use_container_width=True, hide_index=True)
            else:
                st.info("No abnormal risk factor surges detected across evaluated features for this session.")

    # -------------------------------------------------------------------------
    # TAB 2: TECHNICAL TELEMETRY & SIGNALS
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Categorized Technical Indicator Telemetry Matrix")
        
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        
        with col_t1:
            st.markdown("#### ⚡ Volatility & Risk")
            vol = row_data['Volatility_30D'].values[0] if 'Volatility_30D' in row_data.columns else 0
            beta = row_data['Beta_60D'].values[0] if 'Beta_60D' in row_data.columns else 1.0
            st.write(f"**30D Volatility:** `{vol:.2%}`")
            st.write(f"**60D Beta:** `{beta:.2f}`")
            st.write(f"**Vol x Beta Composite:** `{vol*beta:.2%}`")

        with col_t2:
            st.markdown("#### 📈 Trend & Averages")
            sma50 = row_data['SMA_50'].values[0] if 'SMA_50' in row_data.columns else 0
            sma200 = row_data['SMA_200'].values[0] if 'SMA_200' in row_data.columns else 0
            ratio = sma50 / sma200 if sma200 > 0 else 1.0
            st.write(f"**50-Day SMA:** `{sma50:,.2f}`")
            st.write(f"**200-Day SMA:** `{sma200:,.2f}`")
            st.write(f"**SMA 50/200 Ratio:** `{ratio:.3f}` ({'Golden Alignment' if ratio >= 1.0 else 'Death Alignment'})")

        with col_t3:
            st.markdown("#### 🚀 Momentum Indicators")
            rsi = row_data['RSI_14'].values[0] if 'RSI_14' in row_data.columns else 50
            ret5 = row_data['Lagged_Return_5D'].values[0] if 'Lagged_Return_5D' in row_data.columns else 0
            st.write(f"**RSI (14D):** `{rsi:.1f}`")
            st.write(f"**5-Day Return:** `{ret5:.2%}`")
            st.write(f"**Momentum Status:** `{'Strong Bullish' if rsi>60 else 'Bearish Pressure' if rsi<40 else 'Neutral'}`")

        with col_t4:
            st.markdown("#### 📊 Liquidity & Volume")
            vwap = row_data['VWAP_20D'].values[0] if 'VWAP_20D' in row_data.columns else 0
            vol_spike = row_data['Volume_Spike_Ratio'].values[0] if 'Volume_Spike_Ratio' in row_data.columns else 1.0
            st.write(f"**20D VWAP:** `{vwap:,.2f}`")
            st.write(f"**Volume Spike Ratio:** `{vol_spike:.2f}x`")
            st.write(f"**Volume Trend:** `{'Institutional Spike' if vol_spike>1.5 else 'Normal Liquidity'}`")

        st.markdown("---")
        st.subheader("Synchronized Historical Telemetry Chart")
        
        if not ticker_data.empty and 'Close' in ticker_data.columns and date_col:
            chart_df = ticker_data.set_index(date_col)
            
            c_tab1, c_tab2 = st.tabs(["Price Action & Moving Averages", "Volume Spike & Volatility Profile"])
            
            with c_tab1:
                cols_to_plot = [c for c in ['Close', 'SMA_50', 'SMA_200'] if c in chart_df.columns]
                st.line_chart(chart_df[cols_to_plot], use_container_width=True)
                
            with c_tab2:
                cols_vol = [c for c in ['Volume_Spike_Ratio', 'Volatility_30D'] if c in chart_df.columns]
                if cols_vol:
                    st.line_chart(chart_df[cols_vol], use_container_width=True)
                else:
                    st.info("Volume/Volatility trend telemetry stream unavailable.")

    # -------------------------------------------------------------------------
    # TAB 3: MODEL ATTRIBUTION & DIAGNOSTICS (XAI)
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Explainable AI (XAI) & Model Diagnostics")
        st.markdown("Quantifying global model weightings alongside local feature value deviations to provide complete auditability.")
        
        col_x1, col_x2 = st.columns([1.2, 1], gap="large")
        
        with col_x1:
            st.markdown("#### Global Feature Importance (Random Forest Weightings)")
            importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
            st.bar_chart(importances, use_container_width=True)
            
        with col_x2:
            st.markdown("#### Local Instance Input Vector")
            st.markdown("Raw numerical vector passed to the prediction engine for current session:")
            
            x_val_df = X_input.T.reset_index()
            x_val_df.columns = ["Feature Dimension", "Session Value"]
            st.dataframe(x_val_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 🔬 Model Transparency Note")
        st.caption("""
        * **Global Weights**: Derived from Gini impurity reduction across all decision trees in the ensemble model.
        * **Auditability**: Feature shapes and names are automatically synchronized via `model.feature_names_in_` to guarantee exact inference integrity.
        """)
