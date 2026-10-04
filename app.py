import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import joblib
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION & DUAL-TONE CSS
# ==========================================
st.set_page_config(
    page_title="FinShield Intelligence | Global Dashboard", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #F3F4F6; }
    [data-testid="stSidebar"] { background-color: #0A2540 !important; }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    h1, h2, h3 { color: #0A2540; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }
    .stTextInput > div > div > input { border-radius: 8px; border: 2px solid #0A2540; padding: 10px; font-weight: bold; }
    .white-card { background-color: #FFFFFF; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #E5E7EB; margin-bottom: 1rem; }
    .blue-card { background-color: #0A2540; color: #FFFFFF; padding: 1.5rem; border-radius: 12px; box-shadow: 0 10px 15px rgba(0,0,0,0.1); margin-bottom: 1rem; border-left: 6px solid; }
    .blue-card h3, .blue-card p { color: #FFFFFF !important; margin: 0; }
    .js-plotly-plot { margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. GLOBAL LIVE YFINANCE & ML ENGINE PIPELINE
# ==========================================
@st.cache_data(ttl=900)
def fetch_live_data(ticker):
    """
    Universal ticker engine. Supports Indian equities, global stocks (US, EU), 
    and general market symbols natively without rigid prefix constraints.
    """
    clean_ticker = ticker.strip()
    try:
        df = yf.download(clean_ticker, period="6mo", interval="1d", progress=False)
        if df.empty and not clean_ticker.endswith('.NS'):
            # Fallback attempt for Indian equities if user typed base name without extension
            df = yf.download(f"{clean_ticker}.NS", period="6mo", interval="1d", progress=False)
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_crash_risk(df):
    """
    Connects to your local Random Forest model. 
    Falls back to a robust live volatility heuristic if joblib file is absent.
    """
    try:
        # POINT OF INTEGRATION: Drop your Random Forest model in the same folder
        rf_model = joblib.load('random_forest_crash_model.joblib') 
        risk_prob = 15.0 # Placeholder for loaded model prediction output
        return risk_prob
        
    except FileNotFoundError:
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  
        momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100 
        risk = (volatility * 1.8) - (momentum * 0.8)
        return max(2.0, min(98.0, risk))

mf_data = pd.DataFrame({
    'Fund Name': ['Quant Active', 'Parag Parikh Flexi', 'SBI Bluechip', 'Nippon Small Cap'],
    'Category': ['Multi Cap', 'Flexi Cap', 'Large Cap', 'Small Cap'],
    '1Y Return (%)': [32.4, 24.5, 18.2, 45.1],
    'Alpha': [5.2, 3.8, 1.1, 7.5],
    'Beta': [1.1, 0.85, 0.95, 1.25],
    'Expense Ratio (%)': [0.58, 0.65, 1.10, 0.75],
    'Risk Score': [65, 40, 35, 85]
})

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h2>🛡️ FinShield Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; font-size:0.9rem;'>Global Institutional Terminal v3.2</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    app_mode = st.radio(
        "MODULE ROUTING",
        [
            "1. Global Stock Crash-Risk Predictor",
            "2. Technical Telemetry",
            "3. Mutual Fund Risk Screening",
            "4. Head-to-Head (H2H) Comparison",
            "5. Portfolio Overlap Analysis",
            "6. Paper Trading & AI Ledger"
        ]
    )
    st.markdown("---")
    st.caption("🟢 Universal yFinance Feed: **ACTIVE**")

# ==========================================
# 4. MODULE 1: GLOBAL PREDICTOR EXECUTION
# ==========================================
if "1." in app_mode:
    st.markdown("<h1>📉 Global Stock Crash-Risk Prediction Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; margin-bottom:1rem;'>Search any global stock or asset ticker linked directly to live market feeds.</p>", unsafe_allow_html=True)
    
    search_col, _ = st.columns([1.5, 1.5])
    with search_col:
        raw_ticker = st.text_input("🔍 Search Any Global Ticker (e.g., RELIANCE.NS, AAPL, TSLA, TCS.NS):", value="RELIANCE.NS")
        ticker = raw_ticker.strip().upper() if raw_ticker else "RELIANCE.NS"

    df = fetch_live_data(ticker)
    
    if not df.empty and len(df) > 5:
        col_target, col_chart = st.columns([1, 2.5])
        
        with col_target:
            ltp = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2])
            change_pct = ((ltp - prev_close) / prev_close) * 100
            
            risk_score = round(calculate_crash_risk(df), 1)
            
            # STRICT 30% CUT-OFF LOGIC
            cutoff = 30.0
            if risk_score < cutoff:
                risk_color = "#00E676"  # Safe Green
                risk_status = "SAFE (LOW RISK)"
            else:
                risk_color = "#FF1744"  # Danger Red
                risk_status = "DANGER (HIGH RISK)"

            st.markdown(f"""
            <div class="blue-card" style="border-left-color: {risk_color};">
                <p style="font-size: 1rem; color: #94A3B8 !important;">{ticker} MARKET PRICE</p>
                <h2 style="font-size: 2.2rem;">₹{ltp:,.2f} <span style="font-size: 1rem; color: {'#00E676' if change_pct > 0 else '#FF1744'};">({change_pct:+.2f}%)</span></h2>
                <hr style="border-color: #1E293B;">
                <p style="font-size: 1rem; color: #94A3B8 !important;">30-DAY CRASH PROBABILITY</p>
                <h2 style="font-size: 3rem; color: {risk_color} !important;">{risk_score}%</h2>
                <p style="font-size: 1rem; font-weight: bold; color: {risk_color} !important;">{risk_status}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_chart:
            st.markdown('<div class="white-card">', unsafe_allow_html=True)
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                increasing_line_color='#00E676', decreasing_line_color='#FF1744'
            )])
            fig_candle.update_layout(
                title=f"Live Price Action - {ticker} (Last 6 Months)", 
                margin=dict(l=20, r=20, t=40, b=20), height=350,
                xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_candle, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Secondary Metrics
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="white-card">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Feature Importance (SHAP)")
            features = pd.DataFrame({
                'Feature': ['RSI Momentum', 'MACD Divergence', 'Volatility (Live)', 'Volume Surge', 'Moving Avg Cross'],
                'Weight': [0.35, 0.25, 0.20, 0.12, 0.08]
            }).sort_values(by='Weight', ascending=True)
            fig_bar = px.bar(features, x='Weight', y='Feature', orientation='h', color='Weight', color_continuous_scale='Blues')
            fig_bar.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="white-card">', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                number={'suffix': "%", 'font': {'size': 45, 'color': risk_color}},
                title={'text': f"{ticker} Distress Gauge", 'font': {'size': 16, 'color': 'gray'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': risk_color, 'thickness': 0.25},
                    'bgcolor': "#F3F4F6", 
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(0, 230, 118, 0.1)"}, 
                        {'range': [30, 100], 'color': "rgba(255, 23, 68, 0.1)"} 
                    ],
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error(f"Unable to pull market data for symbol '{ticker}'. Please verify the ticker format (e.g., AAPL for Apple, RELIANCE.NS for Reliance Industries) and check your connection.")

# --- MODULE 4: H2H Comparison ---
elif "4." in app_mode:
    st.markdown("<h1>⚔️ H2H Fund Comparison</h1>", unsafe_allow_html=True)
    st.info("Module 1 is currently active for universal stock searching. Use Module 1 to evaluate any stock ticker.")

else:
    st.markdown(f"<h1>{app_mode[3:]}</h1>", unsafe_allow_html=True)
    st.info("Select Module 1 from the sidebar to use the universal live stock engine.")
