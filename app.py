import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION & DUAL-TONE CSS
# ==========================================
st.set_page_config(
    page_title="AlphaShield | Institutional Dashboard", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Background */
    [data-testid="stAppViewContainer"] {
        background-color: #F3F4F6;
    }
    
    /* Premium Dark Blue Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0A2540 !important;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    
    /* Typography */
    h1, h2, h3 { color: #0A2540; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }
    
    /* Custom Card Classes */
    .white-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .blue-card {
        background-color: #0A2540;
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .blue-card h3, .blue-card p { color: #FFFFFF !important; margin: 0; }
    
    /* Fix Plotly centering */
    .js-plotly-plot { margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOCK DATA GENERATION
# ==========================================
@st.cache_data
def get_stock_data(ticker):
    np.random.seed(sum(map(ord, ticker))) # Stable seed per ticker
    dates = pd.date_range(end=datetime.today(), periods=90)
    base = 2840.50 if ticker == "RELIANCE" else (4120.00 if ticker == "TCS" else 1650.00)
    prices = base * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 90)))
    return pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0, 90)),
        'Close': prices,
        'High': prices * (1 + np.random.uniform(0, 0.02, 90)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, 90)),
        'Volume': np.random.randint(500000, 5000000, 90)
    })

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
    st.markdown("<h2>🛡️ AlphaShield</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; font-size:0.9rem;'>Institutional Terminal v2.4</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    app_mode = st.radio(
        "MODULE ROUTING",
        [
            "1. Stock Crash-Risk Predictor",
            "2. Technical Telemetry",
            "3. Mutual Fund Risk Screening",
            "4. Head-to-Head (H2H) Comparison",
            "5. Portfolio Overlap Analysis",
            "6. Paper Trading & AI Ledger"
        ]
    )
    st.markdown("---")
    st.caption("🟢 Live Data Feed Active")

# ==========================================
# 4. MODULE EXECUTIONS
# ==========================================

# --- MODULE 1: Stock Crash-Risk Predictor ---
if "1." in app_mode:
    st.markdown("<h1>📉 Stock Crash-Risk Prediction Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; margin-bottom:2rem;'>Powered by Ensembled Decision Tree Classifiers trained on market stress periods.</p>", unsafe_allow_html=True)

    col_target, col_chart = st.columns([1, 2.5])
    
    with col_target:
        ticker = st.selectbox("Target Asset", ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ADANIENT"])
        df = get_stock_data(ticker)
        ltp = df['Close'].iloc[-1]
        change = ((ltp - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        
        # Determine Dynamic Risk Color
        risk_score = 14.0 if ticker == "RELIANCE" else (82.0 if ticker == "ADANIENT" else np.random.randint(20, 60))
        cutoff = 50.0
        
        if risk_score < cutoff:
            risk_color = "#00E676"  # Bright Neon Green
            risk_status = "SAFE (LOW RISK)"
        else:
            risk_color = "#FF1744"  # Crimson Red
            risk_status = "DANGER (HIGH RISK)"

        # Dark Blue KPI Card with Dynamic Color Injection
        st.markdown(f"""
        <div class="blue-card">
            <p style="font-size: 1rem; color: #94A3B8 !important;">{ticker} MARKET PRICE</p>
            <h2 style="font-size: 2.2rem;">₹{ltp:,.2f} <span style="font-size: 1rem; color: {'#00E676' if change > 0 else '#FF1744'};">({change:+.2f}%)</span></h2>
            <hr style="border-color: #1E293B;">
            <p style="font-size: 1rem; color: #94A3B8 !important;">30-DAY CRASH PROBABILITY</p>
            <h2 style="font-size: 3rem; color: {risk_color} !important;">{risk_score}%</h2>
            <p style="font-size: 1rem; font-weight: bold; color: {risk_color} !important;">{risk_status}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_chart:
        # White Card for Charting
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            increasing_line_color='#00E676', decreasing_line_color='#FF1744'
        )])
        fig_candle.update_layout(
            title=f"90-Day Institutional Price Action - {ticker}", 
            margin=dict(l=20, r=20, t=40, b=20), height=350,
            xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_candle, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Secondary Metrics row
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Feature Importance (SHAP)")
        features = pd.DataFrame({
            'Feature': ['RSI Momentum', 'MACD Divergence', 'VIX Index', 'Institutional Flow', 'Moving Avg Cross'],
            'Weight': [0.35, 0.25, 0.20, 0.12, 0.08]
        }).sort_values(by='Weight', ascending=True)
        fig_bar = px.bar(features, x='Weight', y='Feature', orientation='h', color='Weight', color_continuous_scale='Blues')
        fig_bar.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        # Gauge matches the dynamic HTML color scheme
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={'suffix': "%", 'font': {'size': 35, 'color': '#0A2540'}},
            title={'text': "AI Distress Gauge", 'font': {'size': 16, 'color': 'gray'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "rgba(0,0,0,0.8)", 'thickness': 0.15},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e1e4e8",
                'steps': [
                    {'range': [0, 50], 'color': "#00E676"}, # Safe Green strictly under 50
                    {'range': [50, 100], 'color': "#FF1744"} # Danger Red strictly over 50
                ],
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- MODULE 4: Head-to-Head (H2H) Comparison ---
elif "4." in app_mode:
    st.markdown("<h1>⚔️ H2H Fund Comparison</h1>", unsafe_allow_html=True)
    
    col_sel1, col_sel2 = st.columns(2)
    fund1 = col_sel1.selectbox("Select Fund A", mf_data['Fund Name'], index=0)
    fund2 = col_sel2.selectbox("Select Fund B", mf_data['Fund Name'], index=2)
    
    d1 = mf_data[mf_data['Fund Name'] == fund1].iloc[0]
    d2 = mf_data[mf_data['Fund Name'] == fund2].iloc[0]
    
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    colA, colB = st.columns([1, 1.5])
    with colA:
        st.markdown("### Metrics Comparison")
        comp_df = pd.DataFrame({'Metric': mf_data.columns[2:], fund1: d1[2:].values, fund2: d2[2:].values})
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
    with colB:
        fig_radar = go.Figure()
        categories = ['1Y Return', 'Alpha', 'Beta (Inv)', 'Risk Efficiency', 'Cost Efficiency']
        val1 = [d1['1Y Return (%)']*2, d1['Alpha']*10, (2-d1['Beta'])*50, 100-d1['Risk Score'], (2-d1['Expense Ratio (%)'])*50]
        val2 = [d2['1Y Return (%)']*2, d2['Alpha']*10, (2-d2['Beta'])*50, 100-d2['Risk Score'], (2-d2['Expense Ratio (%)'])*50]
        
        fig_radar.add_trace(go.Scatterpolar(r=val1, theta=categories, fill='toself', name=fund1, line_color='#0A2540'))
        fig_radar.add_trace(go.Scatterpolar(r=val2, theta=categories, fill='toself', name=fund2, line_color='#00E676'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), height=350, margin=dict(l=40, r=40, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- MODULE 5: Portfolio Overlap ---
elif "5." in app_mode:
    st.markdown("<h1>🧩 Portfolio Overlap & Hidden Concentration</h1>", unsafe_allow_html=True)
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    
    df_overlap = pd.DataFrame({
        'Fund': ['Quant Active', 'Quant Active', 'SBI Bluechip', 'SBI Bluechip', 'Parag Parikh Flexi', 'Parag Parikh Flexi'],
        'Holding': ['HDFC Bank', 'Reliance Ind.', 'HDFC Bank', 'ICICI Bank', 'HDFC Bank', 'ITC Ltd.'],
        'Exposure (%)': [8.5, 7.2, 9.1, 5.4, 7.8, 6.5]
    })
    
    fig_tree = px.treemap(
        df_overlap, path=['Holding', 'Fund'], values='Exposure (%)', 
        color='Exposure (%)', color_continuous_scale='Blues'
    )
    fig_tree.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_tree, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- MODULE 6: Paper Trading Ledger ---
elif "6." in app_mode:
    st.markdown("<h1>💼 Unified Sandbox & Ledger</h1>", unsafe_allow_html=True)
    
    col_bal, col_exec = st.columns([1, 2])
    with col_bal:
        st.markdown("""
        <div class="blue-card">
            <p style="color:#94A3B8 !important; font-size:1rem;">TOTAL ACCOUNT VALUE</p>
            <h2 style="font-size:2.5rem;">₹1,245,600</h2>
            <p style="color:#00E676 !important; font-weight:bold;">+₹45,600 (3.8%) All Time</p>
            <hr style="border-color:#1E293B;">
            <p style="color:#94A3B8 !important;">AVAILABLE MARGIN</p>
            <h3>₹320,400</h3>
        </div>
        """, unsafe_allow_html=True)

    with col_exec:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown("### Open Positions Ledger")
        ledger = pd.DataFrame({
            'Ticker': ['RELIANCE', 'HDFCBANK', 'Quant Active'],
            'Qty': [50, 120, 450],
            'Avg Cost (₹)': [2750.0, 1580.0, 410.5],
            'LTP (₹)': [2840.5, 1650.0, 432.0],
            'P&L (₹)': ['+4,525', '+8,400', '+9,675']
        })
        st.dataframe(ledger.style.applymap(lambda x: "color: #00E676; font-weight: bold" if '+' in str(x) else ""), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- FALLBACK FOR MINOR MODULES ---
else:
    st.markdown(f"<h1>{app_mode[3:]}</h1>", unsafe_allow_html=True)
    st.info("UI rendering standard tables for this module. Please select Modules 1, 4, 5, or 6 to view the enhanced Deep Blue/White dual-tone dashboard components.")
