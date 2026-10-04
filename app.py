import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="AlphaShield | Institutional Dashboard", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Professional Institutional Styling */
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; max-width: 95%; }
    h1, h2, h3 { color: #1E3A8A; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .metric-card { 
        background-color: #f8f9fa; border-radius: 8px; padding: 15px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 4px solid #1E3A8A;
    }
    .stSelectbox label, .stRadio label { font-weight: 600; }
    /* Fix Plotly Gauge overlap by enforcing container margins */
    .js-plotly-plot { margin: 0 auto; }
    hr { margin: 1em 0; border: 0; border-top: 1px solid #e1e4e8; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOCK DATA GENERATION (Robust & Realistic)
# ==========================================
@st.cache_data
def get_stock_data(ticker):
    np.random.seed(len(ticker))
    dates = pd.date_range(end=datetime.today(), periods=90)
    base = 2840.50 if ticker == "RELIANCE" else (4120.00 if ticker == "TCS" else 1650.00)
    prices = base * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 90)))
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0, 90)),
        'Close': prices,
        'High': prices * (1 + np.random.uniform(0, 0.02, 90)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, 90)),
        'Volume': np.random.randint(500000, 5000000, 90)
    })
    return df

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
# 3. UNIFIED SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2621/2621040.png", width=50) # Placeholder logo
    st.title("AlphaShield")
    st.markdown("### Navigation Menu")
    
    # Using a single radio menu prevents the multi-select bug in Streamlit
    app_mode = st.radio(
        "Select Analytics Module:",
        [
            "1. Stock Crash-Risk Predictor",
            "2. Technical Telemetry & Indicators",
            "3. Historical Dataset Archive",
            "4. Mutual Fund Risk Screening",
            "5. Head-to-Head (H2H) Comparison",
            "6. Portfolio Overlap Analysis",
            "7. Paper Trading & AI Ledger"
        ]
    )
    st.markdown("---")
    st.caption("Status: **Market Open** 🟢")
    st.caption("Engine: Random Forest V2.4")

# ==========================================
# 4. MODULE EXECUTIONS
# ==========================================

# --- MODULE 1: Stock Crash-Risk Predictor ---
if "1." in app_mode:
    st.header("📉 NIFTY50 Stock Crash-Risk Prediction Engine")
    st.markdown("Powered by Ensembled Decision Tree Classifiers trained on historical market stress periods.")
    st.divider()

    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col1:
        ticker = st.selectbox("Select Target Stock for ML Simulation:", ["RELIANCE", "TCS", "HDFCBANK", "INFY"])
        df = get_stock_data(ticker)
        ltp = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
        change = ((ltp - prev) / prev) * 100
        
        st.markdown(f"### Target: <span style='color:green'>{ticker}</span>", unsafe_allow_html=True)
        st.metric(label="Current Market Price", value=f"₹{ltp:,.2f}", delta=f"{change:,.2f}%")
        st.metric(label="30-Day Volatility (Ann.)", value="18.4%")
        
    with col2:
        # Professional Candlestick Chart
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            increasing_line_color='#00cc96', decreasing_line_color='#ef553b'
        )])
        fig_candle.update_layout(
            title="Recent Price Action", margin=dict(l=0, r=0, t=30, b=0), height=280,
            xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    with col3:
        # FIXED GAUGE CHART: Removed overlapping titles, tightened margins, added professional colors
        risk_score = 14.0 if ticker == "RELIANCE" else np.random.randint(10, 80)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={'suffix': "%", 'font': {'size': 40, 'color': '#1E3A8A'}},
            title={'text': "Crash Risk (30D Horizon)", 'font': {'size': 16, 'color': 'gray'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "rgba(0,0,0,0.8)", 'thickness': 0.15},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e1e4e8",
                'steps': [
                    {'range': [0, 30], 'color': "#00cc96"}, # Safe Green
                    {'range': [30, 70], 'color': "#ffa15a"}, # Warn Yellow
                    {'range': [70, 100], 'color': "#ef553b"} # Danger Red
                ],
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Added Feature: SHAP Importance Graph
    st.subheader("⚙️ ML Feature Importance (SHAP Values)")
    features = pd.DataFrame({
        'Feature': ['RSI (14)', 'MACD Divergence', 'VIX Index', 'FII Flow', 'Moving Avg Cross'],
        'Importance': [0.35, 0.25, 0.20, 0.12, 0.08]
    }).sort_values(by='Importance', ascending=True)
    
    fig_bar = px.bar(features, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
    fig_bar.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

# --- MODULE 5: Head-to-Head (H2H) Comparison ---
elif "5." in app_mode:
    st.header("⚔️ Head-to-Head (H2H) Fund Comparison")
    st.markdown("Multivariate risk-adjusted return comparison for flagship schemes.")
    
    c1, c2 = st.columns(2)
    fund1 = c1.selectbox("Select Fund A", mf_data['Fund Name'], index=0)
    fund2 = c2.selectbox("Select Fund B", mf_data['Fund Name'], index=2)
    
    d1 = mf_data[mf_data['Fund Name'] == fund1].iloc[0]
    d2 = mf_data[mf_data['Fund Name'] == fund2].iloc[0]
    
    # Institutional feature addition: Radar Chart
    fig_radar = go.Figure()
    categories = ['1Y Return (%)', 'Alpha', 'Beta (Inverse)', 'Risk Efficiency', 'Cost Efficiency']
    
    # Transform metrics to 0-100 scale for radar visualization
    val1 = [d1['1Y Return (%)']*2, d1['Alpha']*10, (2-d1['Beta'])*50, 100-d1['Risk Score'], (2-d1['Expense Ratio (%)'])*50]
    val2 = [d2['1Y Return (%)']*2, d2['Alpha']*10, (2-d2['Beta'])*50, 100-d2['Risk Score'], (2-d2['Expense Ratio (%)'])*50]
    
    fig_radar.add_trace(go.Scatterpolar(r=val1, theta=categories, fill='toself', name=fund1, line_color='#1f77b4'))
    fig_radar.add_trace(go.Scatterpolar(r=val2, theta=categories, fill='toself', name=fund2, line_color='#ff7f0e'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), height=400, margin=dict(l=40, r=40, t=40, b=40))
    
    colA, colB = st.columns([1, 2])
    with colA:
        st.dataframe(pd.DataFrame({'Metric': mf_data.columns[2:], fund1: d1[2:].values, fund2: d2[2:].values}), hide_index=True)
    with colB:
        st.plotly_chart(fig_radar, use_container_width=True)

# --- MODULE 6: Portfolio Overlap ---
elif "6." in app_mode:
    st.header("🧩 Portfolio Overlap & Hidden Concentration")
    st.markdown("Exposing underlying single-stock risks across multiple active mutual funds.")
    
    # Institutional feature addition: Treemap Hierarchy
    df_overlap = pd.DataFrame({
        'Fund': ['Quant Active', 'Quant Active', 'SBI Bluechip', 'SBI Bluechip', 'Parag Parikh Flexi', 'Parag Parikh Flexi'],
        'Holding': ['HDFC Bank', 'Reliance Ind.', 'HDFC Bank', 'ICICI Bank', 'HDFC Bank', 'ITC Ltd.'],
        'Exposure (%)': [8.5, 7.2, 9.1, 5.4, 7.8, 6.5]
    })
    
    fig_tree = px.treemap(
        df_overlap, 
        path=['Holding', 'Fund'], 
        values='Exposure (%)', 
        color='Exposure (%)',
        color_continuous_scale='Blues'
    )
    fig_tree.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_tree, use_container_width=True)
    st.info("💡 **Insight:** Notice the severe hidden concentration in **HDFC Bank** across all three major cap funds.")

# --- MODULE 7: Paper Trading Ledger ---
elif "7." in app_mode:
    st.header("💼 Unified Paper Trading & AI Ledger")
    st.markdown("Test your AI-guided strategies in a risk-free sandbox environment.")
    
    col_bal, col_exec = st.columns([1, 2])
    with col_bal:
        st.metric("Total Account Value", "₹1,245,600.00", "+₹45,600 (3.8%)")
        st.metric("Available Margin", "₹320,400.00")
        
        # Portfolio Allocation Donut
        fig_pie = px.pie(values=[40, 35, 15, 10], names=['Equities', 'Mutual Funds', 'ETFs', 'Cash'], hole=0.6)
        fig_pie.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_exec:
        st.subheader("⚡ Order Execution Form")
        with st.form("order_book"):
            c1, c2, c3 = st.columns(3)
            asset = c1.selectbox("Asset", ["RELIANCE", "TCS", "Quant Active Fund"])
            qty = c2.number_input("Quantity", min_value=1, value=10)
            order_type = c3.selectbox("Order Type", ["Market", "Limit", "GTT"])
            
            submitted = st.form_submit_button("Submit Order", use_container_width=True)
            if submitted:
                st.success(f"Successfully placed {order_type} order for {qty}x {asset}.")
        
        st.subheader("Open Positions Ledger")
        ledger = pd.DataFrame({
            'Ticker': ['RELIANCE', 'HDFCBANK', 'Quant Active'],
            'Qty': [50, 120, 450],
            'Avg Cost': [2750.0, 1580.0, 410.5],
            'LTP': [2840.5, 1650.0, 432.0],
            'P&L (₹)': ['+4,525', '+8,400', '+9,675']
        })
        # Style dataframe for trading feel
        st.dataframe(ledger.style.applymap(lambda x: "color: green; font-weight: bold" if '+' in str(x) else ""), use_container_width=True, hide_index=True)

# --- FALLBACK FOR OTHER MODULES (2, 3, 4) ---
else:
    st.info(f"The `{app_mode}` module is currently routing data. Select Modules 1, 5, 6, or 7 to see the upgraded UI components.")
