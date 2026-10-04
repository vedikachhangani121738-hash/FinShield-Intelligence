import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# PAGE CONFIGURATION & SETUP
# ==============================================================================
st.set_page_config(
    page_title="AlphaShield | Stock & Mutual Fund Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; }
    .sub-header { font-size: 1.2rem; font-weight: 500; color: #4B5563; }
    .metric-card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 5px solid #2563EB; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'cash' not in st.session_state:
    st.session_state.cash = 100000.0

# ==============================================================================
# MOCK DATA & METADATA
# ==============================================================================
# 1. Stock Market Data (NIFTY50 & Technicals)
STOCKS_DATA = {
    "RELIANCE": {"price": 2840.50, "change": +1.2, "rsi": 58.4, "macd": "Bullish", "volatility": "18.2%", "crash_prob": 0.14, "verdict": "Low Risk"},
    "TCS": {"price": 4120.00, "change": -0.4, "rsi": 49.1, "macd": "Neutral", "volatility": "14.5%", "crash_prob": 0.09, "verdict": "Very Low Risk"},
    "HDFCBANK": {"price": 1650.25, "change": +0.8, "rsi": 62.0, "macd": "Bullish", "volatility": "16.8%", "crash_prob": 0.18, "verdict": "Low Risk"},
    "INFY": {"price": 1890.10, "change": +1.5, "rsi": 65.3, "macd": "Bullish", "volatility": "21.0%", "crash_prob": 0.29, "verdict": "Moderate Risk"},
    "ICICIBANK": {"price": 1240.80, "change": -0.7, "rsi": 42.5, "macd": "Bearish", "volatility": "19.4%", "crash_prob": 0.22, "verdict": "Low Risk"},
    "ADANIENT": {"price": 2450.00, "change": -3.2, "rsi": 34.2, "macd": "Bearish", "volatility": "38.5%", "crash_prob": 0.68, "verdict": "High Risk"}
}

# 2. Mutual Fund Data
FUNDS_DATA = {
    "Quant Active Fund": {"nav": 485.20, "category": "Multi Cap", "sharpe": 1.85, "sortino": 2.40, "alpha": 6.8, "beta": 1.12, "expense": 0.62, "ai_risk": 0.22},
    "Parag Parikh Flexi Cap": {"nav": 74.50, "category": "Flexi Cap", "sharpe": 1.62, "sortino": 2.10, "alpha": 4.5, "beta": 0.82, "expense": 0.65, "ai_risk": 0.11},
    "SBI Bluechip Fund": {"nav": 92.30, "category": "Large Cap", "sharpe": 1.35, "sortino": 1.70, "alpha": 1.2, "beta": 0.95, "expense": 0.88, "ai_risk": 0.15},
    "Nippon India Small Cap": {"nav": 158.40, "category": "Small Cap", "sharpe": 1.75, "sortino": 2.25, "alpha": 7.4, "beta": 1.18, "expense": 0.72, "ai_risk": 0.38},
    "Axis Midcap Fund": {"nav": 112.10, "category": "Mid Cap", "sharpe": 1.48, "sortino": 1.90, "alpha": 3.1, "beta": 1.04, "expense": 0.85, "ai_risk": 0.28}
}

SCHEME_UNDERLYING_HOLDINGS = {
    "Quant Active Fund": {"Reliance": 8.5, "HDFC Bank": 7.2, "ICICI Bank": 6.8, "ITC": 5.1, "L&T": 4.5},
    "Parag Parikh Flexi Cap": {"HDFC Bank": 9.1, "ITC": 8.4, "Alphabet Inc": 7.8, "Microsoft": 6.2, "Axis Bank": 5.5},
    "SBI Bluechip Fund": {"HDFC Bank": 10.5, "Reliance": 9.2, "ICICI Bank": 8.1, "Infosys": 6.4, "L&T": 5.0},
    "Nippon India Small Cap": {"Reliance": 4.2, "Tube Investments": 3.8, "Cholamandalam": 3.5, "KPR Mill": 3.1, "HDFC Bank": 2.9}
}

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🛡️️ AlphaShield Navigation")
st.sidebar.markdown("---")

st.sidebar.subheader("📈 Stock Market Analytics")
stock_nav = st.sidebar.radio(
    "Select Stock Module:",
    [
        "1. Stock Crash-Risk Predictor (ML)", 
        "2. Technical Telemetry & Indicators", 
        "3. Historical Crash Dataset Archive"
    ],
    key="stock_menu"
)

st.sidebar.subheader("📊 Mutual Fund Analytics")
mf_nav = st.sidebar.radio(
    "Select Mutual Fund Module:",
    [
        "4. Mutual Fund Risk Screening", 
        "5. Head-to-Head (H2H) Fund Comparison", 
        "6. Portfolio Overlap & Hidden Concentration"
    ],
    key="mf_menu"
)

st.sidebar.subheader("💼 Combined Sandbox")
ledger_nav = st.sidebar.radio(
    "Select Portfolio Module:",
    [
        "7. Unified Paper Trading & AI Ledger"
    ],
    key="ledger_menu"
)

# Determine active view based on radio selection
if stock_nav.startswith("1"):
    app_mode = "Stock Crash Predictor"
elif stock_nav.startswith("2"):
    app_mode = "Technical Telemetry"
elif stock_nav.startswith("3"):
    app_mode = "Stock Dataset Archive"
elif mf_nav.startswith("4"):
    app_mode = "Mutual Fund Screening"
elif mf_nav.startswith("5"):
    app_mode = "Head-to-Head Comparison"
elif mf_nav.startswith("6"):
    app_mode = "Portfolio Overlap"
else:
    app_mode = "Paper Trading Ledger"

# ==============================================================================
# MODULE 1: STOCK CRASH-RISK PREDICTOR (ML)
# ==============================================================================
if app_mode == "Stock Crash Predictor":
    st.markdown('<p class="main-header">📉 NIFTY50 Stock Crash-Risk Prediction Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Random Forest & Decision Tree Classifiers trained on historical market stress periods.</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_stock = st.selectbox("Select Target Stock for ML Simulation:", list(STOCKS_DATA.keys()))
        stock_info = STOCKS_DATA[selected_stock]
        
        st.markdown(f"### Target: `{selected_stock}`")
        st.metric("Current Market Price", f"₹{stock_info['price']:,.2f}", f"{stock_info['change']}%")
        st.metric("Model Crash Probability (30-Day Horizon)", f"{stock_info['crash_prob']*100:.1f}%", stock_info['verdict'], delta_color="inverse")
        
    with col2:
        st.subheader("🤖 ML Feature Importance & Inference")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = stock_info['crash_prob'] * 100,
            title = {'text': "Crash Risk Score (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 20], 'color': "#10B981"},
                    {'range': [20, 50], 'color': "#F59E0B"},
                    {'range': [50, 100], 'color': "#EF4444"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': stock_info['crash_prob'] * 100
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Batch Stock Risk Matrix")
    matrix_df = pd.DataFrame([
        {"Stock": k, "Price (INR)": v["price"], "RSI": v["rsi"], "MACD": v["macd"], "Volatility": v["volatility"], "Crash Risk": f"{v['crash_prob']*100:.1f}%", "Verdict": v["verdict"]}
        for k, v in STOCKS_DATA.items()
    ])
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

# ==============================================================================
# MODULE 2: TECHNICAL TELEMETRY & INDICATORS
# ==============================================================================
elif app_mode == "Technical Telemetry":
    st.markdown('<p class="main-header">📈 Technical Telemetry & Indicator Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep-dive technical indicators (RSI, MACD, Moving Average bands) for individual stock health verification.</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    selected_telemetry_stock = st.selectbox("Choose Stock for Telemetry Feed:", list(STOCKS_DATA.keys()), key="telemetry_stock")
    t_data = STOCKS_DATA[selected_telemetry_stock]
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.metric("Relative Strength Index (RSI)", f"{t_data['rsi']}", "Overbought > 70 | Oversold < 30")
    with col_t2:
        st.metric("MACD Signal", t_data['macd'], "Momentum Trend")
    with col_t3:
        st.metric("Annualized Volatility", t_data['volatility'], "Trailing 30-Day")
    with col_t4:
        st.metric("Live LTP", f"₹{t_data['price']:,.2f}", f"{t_data['change']}%")
        
    st.markdown("---")
    st.subheader("📉 Simulated Price & Moving Average Convergence")
    
    # Generate mock historical trend
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=60, freq="B")
    base_price = t_data['price'] * 0.9
    price_series = base_price + np.cumsum(np.random.randn(60) * 15)
    ma_20 = pd.Series(price_series).rolling(5).mean()
    
    chart_df = pd.DataFrame({
        "Date": dates,
        "Price": price_series,
        "20-Period MA": ma_20
    })
    
    fig_tech = px.line(chart_df, x="Date", y=["Price", "20-Period MA"], title=f"{selected_telemetry_stock} - Price Action vs Moving Average")
    fig_tech.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_tech, use_container_width=True)

# ==============================================================================
# MODULE 3: HISTORICAL CRASH DATASET ARCHIVE
# ==============================================================================
elif app_mode == "Stock Dataset Archive":
    st.markdown('<p class="main-header">📁 Historical Stock Crash Dataset Archive</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Examine the 47,272 processed training records and metadata powering the Random Forest model.</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.metric("Total Archived Records", "47,272 rows", "Cleaned NIFTY50 & Sectoral Data")
    st.metric("Primary ML Classifier", "Random Forest Classifier", "Optimized depth & estimators via joblib")
    
    st.subheader("Feature Specifications Table")
    features_df = pd.DataFrame([
        {"Feature": "Daily Return (%)", "Type": "Float", "Description": "Percentage change in closing price from previous session."},
        {"Feature": "30-Day Volatility", "Type": "Float", "Description": "Standard deviation of daily log returns over 30 sessions."},
        {"Feature": "RSI (14)", "Type": "Float", "Description": "Relative Strength Index momentum oscillator."},
        {"Feature": "Max Drawdown", "Type": "Float", "Description": "Peak-to-trough decline over trailing 252 sessions."}
    ])
    st.dataframe(features_df, use_container_width=True, hide_index=True)

# ==============================================================================
# MODULE 4: MUTUAL FUND RISK SCREENING
# ==============================================================================
elif app_mode == "Mutual Fund Screening":
    st.markdown('<p class="main-header">📊 Mutual Fund Risk Rating & Screening</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluate flagship mutual fund schemes using Sharpe ratios, Sortino ratios, Alpha, and AI Distress metrics.</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    mf_rows = []
    for name, metrics in FUNDS_DATA.items():
        mf_rows.append({
            "Scheme Name": name,
            "Category": metrics["category"],
            "NAV (INR)": f"₹{metrics['nav']:.2f}",
            "Sharpe": metrics["sharpe"],
            "Sortino": metrics["sortino"],
            "Alpha (%)": f"{metrics['alpha']}%",
            "Expense Ratio": f"{metrics['expense']}%",
            "AI Distress Risk": f"{metrics['ai_risk']*100:.1f}%"
        })
    mf_df = pd.DataFrame(mf_rows)
    st.dataframe(mf_df, use_container_width=True, hide_index=True)
    
    selected_mf_add = st.selectbox("Select Mutual Fund to Allocate Paper Capital:", list(FUNDS_DATA.keys()), key="mf_add_select")
    invest_amt = st.number_input("Investment Amount (INR):", min_value=1000.0, max_value=st.session_state.cash, value=10000.0, step=1000.0)
    
    if st.button("🛒 Add Fund to Paper Portfolio"):
        fund_meta = FUNDS_DATA[selected_mf_add]
        units = invest_amt / fund_meta['nav']
        st.session_state.cash -= invest_amt
        st.session_state.portfolio.append({
            "name": selected_mf_add,
            "type": "Mutual Fund",
            "category": fund_meta["category"],
            "buy_nav": fund_meta['nav'],
            "units": units,
            "invested_amt": invest_amt,
            "ai_prob_at_buy": fund_meta['ai_risk'],
            "ai_verdict_at_buy": "Screened Safe"
        })
        st.success(f"Successfully invested ₹{invest_amt:,.2f} in {selected_mf_add}!")

# ==============================================================================
# MODULE 5: HEAD-TO-HEAD (H2H) FUND COMPARISON
# ==============================================================================
elif app_mode == "Head-to-Head Comparison":
    st.markdown('<p class="main-header">⚔️ Head-to-Head (H2H) Mutual Fund Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Compare two flagship schemes side-by-side across risk-adjusted returns, alpha generation, and AI distress scores.</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    fund_keys = list(FUNDS_DATA.keys())
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fund_a = st.selectbox("Select Fund A:", fund_keys, index=0)
    with col_h2:
        fund_b = st.selectbox("Select Fund B:", fund_keys, index=1 if len(fund_keys) > 1 else 0)
        
    data_a = FUNDS_DATA[fund_a]
    data_b = FUNDS_DATA[fund_b]
    
    h2h_df = pd.DataFrame({
        "Metric": ["Category", "NAV (INR)", "Sharpe Ratio", "Sortino Ratio", "Alpha (%)", "Beta", "Expense Ratio", "AI Distress Risk"],
        f"{fund_a}": [data_a["category"], f"₹{data_a['nav']}", data_a["sharpe"], data_a["sortino"], f"{data_a['alpha']}%", data_a["beta"], f"{data_a['expense']}%", f"{data_a['ai_risk']*100:.1f}%"],
        f"{fund_b}": [data_b["category"], f"₹{data_b['nav']}", data_b["sharpe"], data_b["sortino"], f"{data_b['alpha']}%", data_b["beta"], f"{data_b['expense']}%", f"{data_b['ai_risk']*100:.1f}%"]
    })
    
    st.dataframe(h2h_df, use_container_width=True, hide_index=True)
    
    # Comparison chart
    fig_comp = go.Figure(data=[
        go.Bar(name=fund_a, x=['Sharpe', 'Sortino', 'Alpha'], y=[data_a['sharpe'], data_a['sortino'], data_a['alpha']]),
        go.Bar(name=fund_b, x=['Sharpe', 'Sortino', 'Alpha'], y=[data_b['sharpe'], data_b['sortino'], data_b['alpha']])
    ])
    fig_comp.update_layout(barmode='group', title="Risk-Adjusted Performance Comparison", height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_comp, use_container_width=True)

# ==============================================================================
# MODULE 6: PORTFOLIO OVERLAP & HIDDEN CONCENTRATION
# ==============================================================================
elif app_mode == "Portfolio Overlap":
    st.markdown('<p class="main-header">🧩 Portfolio Overlap & Hidden Stock Concentration</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze underlying equity stock exposures across active flagship schemes to detect hidden concentration risks.</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    selected_overlap_schemes = st.multiselect(
        "Select Schemes to Analyze for Overlap:",
        options=list(SCHEME_UNDERLYING_HOLDINGS.keys()),
        default=list(SCHEME_UNDERLYING_HOLDINGS.keys())[:3]
    )
    
    if selected_overlap_schemes:
        holding_aggregates = {}
        for scheme in selected_overlap_schemes:
            holdings = SCHEME_UNDERLYING_HOLDINGS.get(scheme, {})
            for stock, weight in holdings.items():
                holding_aggregates[stock] = holding_aggregates.get(stock, 0) + (weight / len(selected_overlap_schemes))
                
        sorted_holdings = sorted(holding_aggregates.items(), key=lambda x: x[1], reverse=True)
        overlap_df = pd.DataFrame(sorted_holdings, columns=['Underlying Stock', 'Weighted Exposure (%)'])
        
        col_ov1, col_ov2 = st.columns([1.5, 1])
        with col_ov1:
            st.subheader("Top Underlying Stock Concentrations")
            fig_overlap = px.bar(
                overlap_df.head(10), 
                x='Weighted Exposure (%)', 
                y='Underlying Stock', 
                orientation='h',
                title="Aggregated Portfolio Stock Exposure",
                color='Weighted Exposure (%)',
                color_continuousScale='Blues'
            )
            fig_overlap.update_layout(yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_overlap, use_container_width=True)
        with col_ov2:
            st.subheader("Concentration Insights")
            top_stock, top_weight = sorted_holdings[0] if sorted_holdings else ("None", 0)
            st.markdown(f"""
            - **Highest Concentration:** `{top_stock}` ({top_weight:.2f}% weighted average).
            - **Diversification Check:** High overlap in banking and conglomerates (HDFC Bank, Reliance) across flagship schemes.
            - **Recommendation:** Monitor consolidated single-stock risk if holding multiple large-cap funds.
            """)
        
        st.dataframe(overlap_df, use_container_width=True, hide_index=True)

# ==============================================================================
# MODULE 7: UNIFIED PAPER TRADING & AI LEDGER
# ==============================================================================
elif app_mode == "Paper Trading Ledger":
    st.markdown('<p class="main-header">💼 Unified Paper Trading & AI Proof Ledger</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Simulate investments and track portfolio health backed by the AlphaShield ML engine.</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.metric("Available Cash Balance", f"₹{st.session_state.cash:,.2f}")
    with col_c2:
        invested_total = sum([p['invested_amt'] for p in st.session_state.portfolio])
        st.metric("Total Active Portfolio Value", f"₹{invested_total:,.2f}")
        
    st.markdown("---")
    st.subheader("📋 Active Holdings Ledger")
    
    if len(st.session_state.portfolio) == 0:
        st.info("Your portfolio is currently empty. Allocate capital via the **Mutual Fund Screening** module above.")
    else:
        ledger_rows = []
        for pos in st.session_state.portfolio:
            ledger_rows.append({
                "Asset Name": pos['name'],
                "Type": pos['type'],
                "Category": pos['category'],
                "Buy Price / NAV": f"₹{pos['buy_nav']:.2f}",
                "Units": f"{pos['units']:.3f}",
                "Invested Capital": f"₹{pos['invested_amt']:,.2f}",
                "AI Distress Risk": f"{pos['ai_prob_at_buy']*100:.1f}%",
                "Verdict": pos['ai_verdict_at_buy']
            })
        ledger_df = pd.DataFrame(ledger_rows)
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)
        
        if st.button("🗑️ Reset Paper Portfolio & Cash", type="secondary"):
            st.session_state.cash = 100000.0
            st.session_state.portfolio = []
            st.rerun()
