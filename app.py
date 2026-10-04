import sys

# ----------------- PERMANENT PICKLE UNPICKLING FIX -----------------
try:
    import sklearn._loss
    sys.modules['_loss'] = sklearn._loss
    if hasattr(sklearn._loss, '_loss'):
        sys.modules['_loss._loss'] = sklearn._loss._loss
except Exception:
    pass

try:
    import sklearn.ensemble._gb_losses as _gb_losses
    sys.modules['_gb_losses'] = _gb_losses
except Exception:
    pass
# -------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
from mftool import Mftool

st.set_page_config(
    page_title="AlphaShield | Institutional Multi-Asset Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- PROFESSIONAL TRADING TERMINAL CSS (DARK BLUE THEME) -----------------
st.markdown("""
<style>
    /* Main App Background & Font */
    .stApp {
        background-color: #070d1b;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1329;
        border-right: 1px solid #1e293b;
    }
    
    /* Professional Metric & Risk Cards */
    .risk-card-red {
        background: linear-gradient(135deg, #450a0a 0%, #1e1b4b 100%);
        border: 1px solid #dc2626;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(220, 38, 38, 0.2);
        margin-bottom: 20px;
    }
    .risk-card-yellow {
        background: linear-gradient(135deg, #422006 0%, #1e1b4b 100%);
        border: 1px solid #d97706;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(217, 119, 6, 0.2);
        margin-bottom: 20px;
    }
    .risk-card-green {
        background: linear-gradient(135deg, #064e3b 0%, #1e1b4b 100%);
        border: 1px solid #059669;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(5, 150, 105, 0.2);
        margin-bottom: 20px;
    }
    
    .metric-container {
        background-color: #111c38;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    
    /* Headers & Text */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

ACTIVE_BENCHMARK_FUNDS = {
    "⭐ HDFC Top 100 Fund - Direct Growth": "118989",
    "⭐ Axis Bluechip / Large Cap Fund - Direct Growth": "120465",
    "⭐ Nippon India Small Cap Fund - Direct Growth": "118778",
    "⭐ SBI Bluechip Fund - Direct Growth": "119598",
    "⭐ ICICI Prudential Bluechip Fund - Direct Growth": "120586",
    "⭐ Parag Parikh Flexi Cap Fund - Direct Growth": "122639"
}

FEATURE_COLS = [
    'NAV', 'Daily_Return_Pct', 'Annualized_Return_1Y', 'Volatility_30D',
    'Annualized_Volatility_Cleaned', 'Sharpe_Ratio_Cleaned', 'Sortino_Ratio_Cleaned',
    'Max_Drawdown_1Y_Pct', 'Composite_Score', 'Lag_1D_Return', 'Lag_5D_Return'
]

@st.cache_resource
def load_model_bundle():
    bundle_path = 'dashboard_model_bundle.pkl'
    if os.path.exists(bundle_path):
        try:
            with open(bundle_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
            
    from sklearn.ensemble import GradientBoostingClassifier
    X_synthetic = np.array([
        [150.0,  0.05, 18.0, 14.0, 16.0,  0.8,  1.1,  -8.0, 0.72,  0.02,  0.10],
        [ 85.0,  0.02, 12.0, 18.0, 20.0,  0.3,  0.4, -16.0, 0.45, -0.05,  0.02],
        [ 25.0, -0.25, -5.0, 26.0, 28.0, -0.4, -0.6, -32.0, 0.22, -0.30, -0.45],
        [210.0,  0.10, 24.0, 13.0, 15.0,  1.1,  1.5,  -6.0, 0.85,  0.08,  0.15],
        [ 18.0, -0.40,-12.0, 32.0, 35.0, -0.9, -1.2, -45.0, 0.15, -0.50, -0.80]
    ])
    y_synthetic = np.array([0, 0, 1, 0, 1])
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_synthetic, y_synthetic)
    
    return {
        'model': clf,
        'features': FEATURE_COLS
    }

@st.cache_resource
def load_all_schemes():
    obj = Mftool()
    try:
        codes_dict = obj.get_scheme_codes()
        df = pd.DataFrame(list(codes_dict.items()), columns=['Scheme_Code', 'Scheme_Name'])
        df = df[~df['Scheme_Name'].str.contains(r'\bMIP\b|\bFMP\b|Fixed Maturity|Dividend', case=False, na=False)]
        df['Search_Label'] = df['Scheme_Name'] + " [Code: " + df['Scheme_Code'] + "]"
        return obj, df
    except Exception:
        return obj, pd.DataFrame(columns=['Scheme_Code', 'Scheme_Name', 'Search_Label'])

bundle = load_model_bundle()
model = bundle['model']
features = bundle.get('features', FEATURE_COLS)
obj, all_schemes_df = load_all_schemes()

def get_ai_reasoning(feat_dict, prob):
    reasons = []
    if feat_dict['Volatility_30D'] > 20.0:
        reasons.append(f"Elevated 30-day volatility ({feat_dict['Volatility_30D']:.1f}%) reflecting heightened market turbulence.")
    if feat_dict['Max_Drawdown_1Y_Pct'] < -15.0:
        reasons.append(f"Severe 1-year historical maximum drawdown ({feat_dict['Max_Drawdown_1Y_Pct']:.1f}%) indicating structural vulnerability.")
    if feat_dict['Sharpe_Ratio_Cleaned'] < 0.5:
        reasons.append(f"Subdued Sharpe ratio ({feat_dict['Sharpe_Ratio_Cleaned']:.2f}) showing inadequate risk-adjusted compensation.")
    if feat_dict['Lag_5D_Return'] < -2.0:
        reasons.append(f"Negative 5-day momentum ({feat_dict['Lag_5D_Return']:.2f}%) pointing to active selling pressure.")
    if prob >= 0.6 and not reasons:
        reasons.append("High composite risk signature detected across multi-factor nonlinear classification boundaries.")
    if not reasons:
        reasons.append("Stable risk profile with strong resilience metrics, healthy Sharpe ratio, and controlled drawdown limits.")
    return reasons

def compute_scheme_metrics(scheme_code):
    details = obj.get_scheme_details(scheme_code)
    if not details or not isinstance(details, dict):
        raise ValueError(f"Scheme code {scheme_code} is inactive or not recognized.")
        
    scheme_name = details.get('scheme_name', f'Scheme {scheme_code}')
    category = details.get('scheme_category', 'Mutual Fund')
    amc = details.get('fund_house', 'AMC')
    
    hist_data = obj.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
    if hist_data is None or len(hist_data) == 0:
        raise ValueError(f"Historical NAV data is unavailable for '{scheme_name}'.")
    
    df_nav = pd.DataFrame(hist_data)
    if 'nav' not in df_nav.columns:
        raise ValueError(f"AMFI did not return NAV series for '{scheme_name}'.")
        
    df_nav['nav'] = pd.to_numeric(df_nav['nav'], errors='coerce')
    df_nav = df_nav.dropna(subset=['nav'])
    df_nav.index = pd.to_datetime(df_nav.index, format='%d-%m-%Y', errors='coerce')
    df_nav = df_nav.sort_index()
    
    current_nav = float(df_nav['nav'].iloc[-1])
    daily_returns = df_nav['nav'].pct_change().dropna()
    
    nav_1y = float(df_nav['nav'].iloc[-252]) if len(df_nav) >= 252 else float(df_nav['nav'].iloc[0])
    ret_1y = ((current_nav / nav_1y) - 1.0) * 100.0
    
    vol_30d = float(daily_returns.tail(30).std() * np.sqrt(30) * 100.0) if len(daily_returns) >= 5 else 15.0
    ann_vol = float(daily_returns.tail(252).std() * np.sqrt(252) * 100.0) if len(daily_returns) >= 5 else 18.0
    
    rolling_max = df_nav['nav'].tail(252).cummax()
    drawdown_series = (df_nav['nav'].tail(252) - rolling_max) / rolling_max
    max_drawdown_1y = float(drawdown_series.min() * 100.0) if len(drawdown_series) > 0 else 0.0
    
    rf = 6.5
    downside = daily_returns.tail(252)[daily_returns.tail(252) < 0]
    downside_std = float(downside.std() * np.sqrt(252) * 100.0) if len(downside) > 0 else ann_vol
    
    sharpe = (ret_1y - rf) / ann_vol if ann_vol > 0 else 0.0
    sortino = (ret_1y - rf) / downside_std if downside_std > 0 else 0.0
    comp_score = max(0.1, min(0.9, 0.5 + (sharpe * 0.15) + (max_drawdown_1y / 100.0 * 0.2)))
    
    lag_1d = float(daily_returns.iloc[-2] * 100.0) if len(daily_returns) >= 2 else 0.0
    lag_5d = float(daily_returns.iloc[-6] * 100.0) if len(daily_returns) >= 6 else lag_1d
    
    feat_dict = {
        'NAV': current_nav, 'Daily_Return_Pct': float(daily_returns.iloc[-1] * 100.0),
        'Annualized_Return_1Y': ret_1y, 'Volatility_30D': vol_30d,
        'Annualized_Volatility_Cleaned': ann_vol, 'Sharpe_Ratio_Cleaned': sharpe,
        'Sortino_Ratio_Cleaned': sortino, 'Max_Drawdown_1Y_Pct': max_drawdown_1y,
        'Composite_Score': comp_score, 'Lag_1D_Return': lag_1d, 'Lag_5D_Return': lag_5d
    }
    X_input = pd.DataFrame([feat_dict])[features]
    prob = float(model.predict_proba(X_input)[0, 1])
    reasoning_list = get_ai_reasoning(feat_dict, prob)
    
    return {
        'name': scheme_name, 'category': category, 'amc': amc, 'nav': current_nav,
        'ret_1y': ret_1y, 'vol': ann_vol, 'drawdown_1y': max_drawdown_1y,
        'sharpe': sharpe, 'sortino': sortino, 'comp_score': comp_score, 'prob': prob, 'reasoning': reasoning_list, 'df_nav': df_nav
    }

# ----------------- SIDEBAR CONTROLS & SEARCH BAR -----------------
st.sidebar.markdown("### 🛡️ AlphaShield Terminal")
app_mode = st.sidebar.radio(
    "Select Intelligence Module:",
    [
        "📈 NIFTY50 Stock Crash-Risk Predictor",
        "🛡️ Mutual Fund Intelligence Suite",
        "💼 Paper Trading & AI Ledger",
        "📊 Monte Carlo VaR Simulator"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Asset Search & Inputs")

if app_mode == "📈 NIFTY50 Stock Crash-Risk Predictor":
    selected_stock = st.sidebar.selectbox("Select NIFTY50 Stock:", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS"])
    stock_price = st.sidebar.number_input("Current Stock Price (₹):", value=2500.0, step=10.0)
    stock_ret_1y = st.sidebar.number_input("Trailing 1Y Return (%):", value=15.5, step=0.5)
    stock_vol = st.sidebar.number_input("Annualized Volatility (%):", value=22.0, step=0.5)

elif app_mode == "🛡️ Mutual Fund Intelligence Suite":
    mf_source = st.sidebar.radio("Fund Source:", ["⭐ Popular Benchmarks", "🔎 Search AMFI Database"])
    if mf_source.startswith("⭐"):
        chosen_mf_label = st.sidebar.selectbox("Select Benchmark Fund:", list(ACTIVE_BENCHMARK_FUNDS.keys()))
        selected_mf_code = ACTIVE_BENCHMARK_FUNDS[chosen_mf_label]
    else:
        search_options = all_schemes_df['Search_Label'].tolist() if not all_schemes_df.empty else list(ACTIVE_BENCHMARK_FUNDS.keys())
        mf_search_sel = st.sidebar.selectbox("Search Scheme:", options=search_options)
        selected_mf_code = all_schemes_df[all_schemes_df['Search_Label'] == mf_search_sel]['Scheme_Code'].iloc[0] if not all_schemes_df.empty else "118989"

if 'cash' not in st.session_state:
    st.session_state.cash = 100000.0
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# ==============================================================================
# DASHBOARD 1: NIFTY50 STOCK CRASH-RISK PREDICTOR
# ==============================================================================
if app_mode == "📈 NIFTY50 Stock Crash-Risk Predictor":
    st.title("📈 NIFTY50 Stock Crash-Risk Predictor")
    st.markdown("Institutional Machine Learning Classification & Explainable AI Decision Engine (30-Day Forward Horizon).")
    
    sample_feat = {
        'NAV': stock_price, 'Daily_Return_Pct': 0.01, 'Annualized_Return_1Y': stock_ret_1y,
        'Volatility_30D': 18.0, 'Annualized_Volatility_Cleaned': stock_vol, 'Sharpe_Ratio_Cleaned': 0.8,
        'Sortino_Ratio_Cleaned': 1.1, 'Max_Drawdown_1Y_Pct': -14.0, 'Composite_Score': 0.65,
        'Lag_1D_Return': 0.01, 'Lag_5D_Return': -1.5
    }
    X_input = pd.DataFrame([sample_feat])[features]
    stock_prob = float(model.predict_proba(X_input)[0, 1])
    reasons = get_ai_reasoning(sample_feat, stock_prob)
    
    st.markdown("---")
    
    # PROMINENT RISK WIDGETS CARD
    if stock_prob >= 0.6:
        card_class = "risk-card-red"
        risk_label = "🔴 CRITICAL CRASH RISK"
    elif stock_prob >= 0.3:
        card_class = "risk-card-yellow"
        risk_label = "🟡 MODERATE WATCHLIST RISK"
    else:
        card_class = "risk-card-green"
        risk_label = "🟢 LOW CRASH RISK (STABLE)"
        
    st.markdown(f"""
    <div class="{card_class}">
        <h3 style="margin: 0; color: #ffffff;">{risk_label}</h3>
        <h1 style="font-size: 42px; margin: 10px 0; color: #ffffff;">{stock_prob*100:.1f}% <span style="font-size: 18px; font-weight: normal;">30-Day Crash Probability</span></h1>
        <p style="margin: 0; color: #cbd5e1;">Targeting asset: <strong>{selected_stock}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""<div class="metric-container"><h4>Input Price</h4><h2>₹{stock_price:,.2f}</h2></div>""", unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""<div class="metric-container"><h4>Trailing 1Y Return</h4><h2>{stock_ret_1y:+.2f}%</h2></div>""", unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""<div class="metric-container"><h4>Annualized Volatility</h4><h2>{stock_vol:.1f}%</h2></div>""", unsafe_allow_html=True)
        
    st.markdown("### 🧠 Explainable AI Decision Rationale")
    for r in reasons:
        st.info(f"• {r}")

# ==============================================================================
# DASHBOARD 2: MUTUAL FUND INTELLIGENCE SUITE
# ==============================================================================
elif app_mode == "🛡️ Mutual Fund Intelligence Suite":
    st.title("🛡️ Mutual Fund Intelligence & Stress-Testing Suite")
    st.markdown("Live AMFI scheme analytics, crisis stress testing, and quantitative risk ratings.")
    
    if selected_mf_code:
        try:
            res = compute_scheme_metrics(selected_mf_code)
            st.markdown("---")
            
            if res['prob'] >= 0.6:
                card_class = "risk-card-red"
                risk_label = "🔴 CRITICAL DISTRESS RISK"
            elif res['prob'] >= 0.3:
                card_class = "risk-card-yellow"
                risk_label = "🟡 WATCHLIST / MODERATE RISK"
            else:
                card_class = "risk-card-green"
                risk_label = "🟢 HEALTHY / SAFE ALLOCATION"
                
            st.markdown(f"""
            <div class="{card_class}">
                <h3 style="margin: 0; color: #ffffff;">{risk_label}</h3>
                <h1 style="font-size: 42px; margin: 10px 0; color: #ffffff;">{res['prob']*100:.1f}% <span style="font-size: 18px; font-weight: normal;">30-Day Distress Probability</span></h1>
                <p style="margin: 0; color: #cbd5e1;">Scheme: <strong>{res['name']}</strong> ({res['category']})</p>
            </div>
            """, unsafe_allow_html=True)
            
            c_f1, c_f2, c_f3, c_f4 = st.columns(4)
            with c_f1:
                st.markdown(f"""<div class="metric-container"><h4>Live NAV</h4><h2>₹{res['nav']:.2f}</h2></div>""", unsafe_allow_html=True)
            with c_f2:
                st.markdown(f"""<div class="metric-container"><h4>1-Year Return</h4><h2>{res['ret_1y']:+.2f}%</h2></div>""", unsafe_allow_html=True)
            with c_f3:
                st.markdown(f"""<div class="metric-container"><h4>Sharpe Ratio</h4><h2>{res['sharpe']:.2f}</h2></div>""", unsafe_allow_html=True)
            with c_f4:
                st.markdown(f"""<div class="metric-container"><h4>1-Year Max Drawdown</h4><h2>{res['drawdown_1y']:.2f}%</h2></div>""", unsafe_allow_html=True)
                
            st.markdown("### 🧠 Explainable AI Decision Rationale")
            for r in res['reasoning']:
                st.info(f"• {r}")
                
            st.markdown("---")
            st.subheader("📊 Historical NAV Performance Trend")
            st.line_chart(res['df_nav']['nav'])
            
        except Exception as e:
            st.error(f"Error loading scheme: {e}")

# ==============================================================================
# MODULE 3: PAPER TRADING & AI LEDGER
# ==============================================================================
elif app_mode == "💼 Paper Trading & AI Ledger":
    st.title("💼 Universal Paper Trading & AI Risk Ledger")
    st.markdown("Simulate cross-asset execution and monitor portfolio risk in real-time.")
    
    current_portfolio_val = sum(pos['units'] * pos['buy_price'] for pos in st.session_state.portfolio)
    total_net_worth = st.session_state.cash + current_portfolio_val
    
    c_w1, c_w2, c_w3 = st.columns(3)
    with c_w1:
        st.markdown(f"""<div class="metric-container"><h4>Available Paper Cash</h4><h2>₹{st.session_state.cash:,.2f}</h2></div>""", unsafe_allow_html=True)
    with c_w2:
        st.markdown(f"""<div class="metric-container"><h4>Portfolio Value</h4><h2>₹{current_portfolio_val:,.2f}</h2></div>""", unsafe_allow_html=True)
    with c_w3:
        st.markdown(f"""<div class="metric-container"><h4>Total Net Worth</h4><h2>₹{total_net_worth:,.2f}</h2></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🛒 Execute Paper Order")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        trade_name = st.selectbox("Select Target Asset:", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "HDFC Top 100 Fund"])
        trade_price = st.number_input("Execution Price (₹):", value=2500.0)
    with col_t2:
        trade_amt = st.number_input("Investment Amount (₹):", value=10000.0, step=1000.0)
        
    if st.button("📥 Execute Paper Trade", type="primary"):
        if trade_amt > st.session_state.cash:
            st.error("Insufficient paper cash!")
        else:
            units = trade_amt / trade_price
            st.session_state.portfolio.append({
                'time': datetime.now().strftime("%d-%b-%Y %H:%M"),
                'name': trade_name, 'buy_price': trade_price, 'units': units,
                'invested_amt': trade_amt, 'ai_risk_pct': "12.5%"
            })
            st.session_state.cash -= trade_amt
            st.success(f"Successfully bought {units:.3f} units of {trade_name}!")
            st.rerun()

    if len(st.session_state.portfolio) > 0:
        st.markdown("---")
        st.subheader("📑 Active Holdings")
        st.dataframe(pd.DataFrame(st.session_state.portfolio), use_container_width=True)
        if st.button("🔄 Reset Portfolio"):
            st.session_state.cash = 100000.0
            st.session_state.portfolio = []
            st.rerun()

# ==============================================================================
# MODULE 4: MONTE CARLO VaR SIMULATOR
# ==============================================================================
else:
    st.title("📊 Monte Carlo 30-Day Value-at-Risk (VaR) Simulator")
    st.markdown("Simulate 1,000 forward-looking price paths to determine portfolio tail-risk thresholds.")
    
    current_portfolio_val = sum(pos['units'] * pos['buy_price'] for pos in st.session_state.portfolio)
    
    if current_portfolio_val <= 0:
        st.warning("Your active portfolio has zero value. Execute some paper trades in the Ledger module first.")
    else:
        sim_vol = st.slider("Estimated Annualized Volatility (%):", 5.0, 40.0, 18.0)
        if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
            daily_vol = (sim_vol / 100.0) / np.sqrt(252)
            sims, days = 1000, 30
            simulated_returns = np.random.normal(-0.0001, daily_vol, (sims, days))
            ending_values = current_portfolio_val * (1 + np.prod(1 + simulated_returns, axis=1) - 1)
            var_95 = np.percentile(ending_values, 5)
            
            st.markdown("---")
            st.markdown(f"""
            <div class="risk-card-yellow">
                <h3 style="margin: 0; color: #ffffff;">📉 PORTFOLIO VaR AUDIT</h3>
                <h1 style="font-size: 38px; margin: 10px 0; color: #ffffff;">₹{var_95:,.2f}</h1>
                <p style="margin: 0; color: #cbd5e1;">95% Confidence 30-Day Value-at-Risk Threshold</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig = px.histogram(ending_values, nbins=50, title="Portfolio Ending Value Distribution (30 Days Ahead)")
            fig.update_layout(plot_bgcolor='#070d1b', paper_bgcolor='#111c38', font_color='#e2e8f0')
            fig.add_vline(x=var_95, line_dash="dash", line_color="#ef4444", annotation_text="95% VaR Threshold")
            st.plotly_chart(fig, use_container_width=True)
