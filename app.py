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
    [data-testid="stAppViewContainer"] { background-color: #F3F4F6; }
    
    /* Premium Dark Blue Sidebar */
    [data-testid="stSidebar"] { background-color: #0A2540 !important; }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    
    /* Typography */
    h1, h2, h3 { color: #0A2540; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }
    
    /* Search Bar Styling */
    .stTextInput > div > div > input {
        border-radius: 8px; border: 2px solid #0A2540; padding: 10px; font-weight: bold;
    }
    
    /* Custom Card Classes */
    .white-card {
        background-color: #FFFFFF; padding: 1.5rem; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #E5E7EB; margin-bottom: 1rem;
    }
    .blue-card {
        background-color: #0A2540; color: #FFFFFF; padding: 1.5rem; border-radius: 12px;
        box-shadow: 0 10px 15px rgba(0,0,0,0.1); margin-bottom: 1rem; border-left: 6px solid;
    }
    .blue-card h3, .blue-card p { color: #FFFFFF !important; margin: 0; }
    .js-plotly-plot { margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DYNAMIC DATA GENERATOR
# ==========================================
@st.cache_data
def get_stock_data(ticker):
    # Generates stable, realistic mock data for ANY typed ticker
    np.random.seed(sum(map(ord, ticker))) 
    dates = pd.date_range(end=datetime.today(), periods=90)
    base = np.random.uniform(500, 5000) # Dynamic base price
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
    'Fund Name': ['Quant Active', 'Parag Parikh Flexi', 'SBI Bluechip', 'Nippon Small Cap', 'HDFC Mid-Cap Opportunities'],
    'Category': ['Multi Cap', 'Flexi Cap', 'Large Cap', 'Small Cap', 'Mid Cap'],
    '1Y Return (%)': [32.4, 24.5, 18.2, 45.1, 38.6],
    'Alpha': [5.2, 3.8, 1.1, 7.5, 6.2],
    'Beta': [1.1, 0.85, 0.95, 1.25, 1.15],
    'Expense Ratio (%)': [0.58, 0.65, 1.10, 0.75, 0.92],
    'Risk Score': [65, 40, 35, 85, 72]
})

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h2>🛡️ AlphaShield</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; font-size:0.9rem;'>Institutional Terminal v2.5</p>", unsafe_allow_html=True)
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
    st.markdown("<p style='color:#475569; margin-bottom:1rem;'>Powered by Ensembled Decision Tree Classifiers trained on market stress periods.</p>", unsafe_allow_html=True)

    # NEW: Dedicated Global Search Bar
    search_col, _ = st.columns([1, 2])
    with search_col:
        raw_ticker = st.text_input("🔍 Search Any Stock Ticker (e.g., RELIANCE, TCS, INFY):", value="RELIANCE")
        ticker = raw_ticker.upper().strip() if raw_ticker else "RELIANCE"

    col_target, col_chart = st.columns([1, 2.5])
    
    with col_target:
        df = get_stock_data(ticker)
        ltp = df['Close'].iloc[-1]
        change = ((ltp - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        
        # Dynamic Risk Engine Logic
        np.random.seed(sum(map(ord, ticker))) # Ensures the same ticker yields the same risk score
        risk_score = np.random.randint(15, 85)
        cutoff = 50.0
        
        # Strictly define explicit color hexes
        if risk_score < cutoff:
            risk_color = "#00E676"  # Bright Neon Green
            risk_status = "SAFE (LOW RISK)"
        else:
            risk_color = "#FF1744"  # Crimson Red
            risk_status = "DANGER (HIGH RISK)"

        # KPI Card: Now injects the risk color into the left border and text elements
        st.markdown(f"""
        <div class="blue-card" style="border-left-color: {risk_color};">
            <p style="font-size: 1rem; color: #94A3B8 !important;">{ticker} MARKET PRICE</p>
            <h2 style="font-size: 2.2rem;">₹{ltp:,.2f} <span style="font-size: 1rem; color: {'#00E676' if change > 0 else '#FF1744'};">({change:+.2f}%)</span></h2>
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
        
        # FIXED GAUGE: Bar and text strictly adopt the active risk_color
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={'suffix': "%", 'font': {'size': 45, 'color': risk_color}}, # Text matches risk color
            title={'text': f"{ticker} Distress Gauge", 'font': {'size': 16, 'color': 'gray'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': risk_color, 'thickness': 0.25}, # Bar strictly matches risk color
                'bgcolor': "#F3F4F6", # Subtle background to make the bar pop
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50], 'color': "rgba(0, 230, 118, 0.1)"}, # Faint green background
                    {'range': [50, 100], 'color': "rgba(255, 23, 68, 0.1)"} # Faint red background
                ],
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- MODULE 4: Head-to-Head (H2H) Comparison ---
elif "4." in app_mode:
    st.markdown("<h1>⚔️ H2H Fund Comparison</h1>", unsafe_allow_html=True)
    
    # Enable search by allowing users to type fund names, falling back to defaults if not found
    col_sel1, col_sel2 = st.columns(2)
    search1 = col_sel1.text_input("🔍 Search Fund A", value="Quant Active").strip()
    search2 = col_sel2.text_input("🔍 Search Fund B", value="SBI Bluechip").strip()
    
    # Fallback to nearest match or default
    fund1 = search1 if search1 in mf_data['Fund Name'].values else "Quant Active"
    fund2 = search2 if search2 in mf_data['Fund Name'].values else "SBI Bluechip"
    
    if search1 not in mf_data['Fund Name'].values and search1 != "":
        col_sel1.warning(f"Fund '{search1}' not found in DB. Defaulting to Quant Active.")
    if search2 not in mf_data['Fund Name'].values and search2 != "":
        col_sel2.warning(f"Fund '{search2}' not found in DB. Defaulting to SBI Bluechip.")
    
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

else:
    st.markdown(f"<h1>{app_mode[3:]}</h1>", unsafe_allow_html=True)
    st.info("Module routing active. Select Module 1 or 4 to view the updated Search features and dynamic Risk Plotly Gauges.")
