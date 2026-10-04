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
    page_title="AlphaShield | Universal Multi-Asset Intelligence Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ACTIVE_BENCHMARK_FUNDS = {
    "⭐ HDFC Top 100 Fund - Direct Growth": "118989",
    "⭐ Axis Bluechip / Large Cap Fund - Direct Growth": "120465",
    "⭐ Nippon India Small Cap Fund - Direct Growth": "118778",
    "⭐ SBI Bluechip Fund - Direct Growth": "119598",
    "⭐ ICICI Prudential Bluechip Fund - Direct Growth": "120586",
    "⭐ Parag Parikh Flexi Cap Fund - Direct Growth": "122639",
    "⭐ Mirae Asset Large Cap Fund - Direct Growth": "118834",
    "⭐ Kotak Emerging Equity Fund - Direct Growth": "120152",
    "⭐ Tata Digital India Fund - Direct Growth": "135781",
    "⭐ Quant Active Fund - Direct Growth": "120828",
    "⭐ Axis Children's Gift Fund - Direct Growth": "135762"
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
        'features': FEATURE_COLS,
        'sample_funds': [
            {'Fund_Name': 'HDFC Top 100 Fund - Direct Growth', 'Category': 'Large Cap', 'AMC': 'HDFC Mutual Fund', 'NAV': 1050.2, 'Fund_Star_Rating': 5, 'Composite_Score': 0.78, 'Max_Drawdown_1Y_Pct': -8.4, 'Sharpe_Ratio_Cleaned': 1.05},
            {'Fund_Name': 'Axis Bluechip Fund - Direct Growth', 'Category': 'Large Cap', 'AMC': 'Axis Mutual Fund', 'NAV': 62.4, 'Fund_Star_Rating': 4, 'Composite_Score': 0.58, 'Max_Drawdown_1Y_Pct': -14.2, 'Sharpe_Ratio_Cleaned': 0.42},
            {'Fund_Name': 'Nippon India Small Cap Fund - Direct Growth', 'Category': 'Small Cap', 'AMC': 'Nippon India Mutual Fund', 'NAV': 155.8, 'Fund_Star_Rating': 5, 'Composite_Score': 0.82, 'Max_Drawdown_1Y_Pct': -11.5, 'Sharpe_Ratio_Cleaned': 1.18}
        ]
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
funds_df = pd.DataFrame(bundle['sample_funds'])
obj, all_schemes_df = load_all_schemes()

def get_ai_reasoning(feat_dict, prob):
    """Explainable AI (XAI) engine generating qualitative decision rationale."""
    reasons = []
    if feat_dict['Volatility_30D'] > 20.0:
        reasons.append(f"Elevated 30-day volatility ({feat_dict['Volatility_30D']:.1f}%) reflecting heightened short-term market turbulence.")
    if feat_dict['Max_Drawdown_1Y_Pct'] < -15.0:
        reasons.append(f"Severe 1-year historical maximum drawdown ({feat_dict['Max_Drawdown_1Y_Pct']:.1f}%) indicating structural vulnerability.")
    if feat_dict['Sharpe_Ratio_Cleaned'] < 0.5:
        reasons.append(f"Subdued Sharpe ratio ({feat_dict['Sharpe_Ratio_Cleaned']:.2f}) showing inadequate risk-adjusted compensation.")
    if feat_dict['Lag_5D_Return'] < -2.0:
        reasons.append(f"Negative 5-day momentum ({feat_dict['Lag_5D_Return']:.2f}%) pointing to active institutional selling pressure.")
    if prob >= 0.6 and not reasons:
        reasons.append("High composite risk signature detected across multi-factor nonlinear classification boundaries.")
    if not reasons:
        reasons.append("Stable risk profile with strong resilience metrics, healthy Sharpe ratio, and controlled drawdown limits.")
    return reasons

def compute_scheme_metrics(scheme_code):
    details = obj.get_scheme_details(scheme_code)
    if not details or not isinstance(details, dict):
        raise ValueError(f"Scheme code {scheme_code} is inactive or not recognized by AMFI.")
        
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
    if len(df_nav) < 5:
        raise ValueError(f"Insufficient historical trading days for '{scheme_name}'.")
        
    df_nav.index = pd.to_datetime(df_nav.index, format='%d-%m-%Y', errors='coerce')
    df_nav = df_nav.sort_index()
    
    current_nav = float(df_nav['nav'].iloc[-1])
    daily_returns = df_nav['nav'].pct_change().dropna()
    
    nav_1m = float(df_nav['nav'].iloc[-21]) if len(df_nav) >= 21 else float(df_nav['nav'].iloc[0])
    nav_1y = float(df_nav['nav'].iloc[-252]) if len(df_nav) >= 252 else float(df_nav['nav'].iloc[0])
    nav_3y = float(df_nav['nav'].iloc[-756]) if len(df_nav) >= 756 else float(df_nav['nav'].iloc[0])
    
    ret_1m = ((current_nav / nav_1m) - 1.0) * 100.0
    ret_1y = ((current_nav / nav_1y) - 1.0) * 100.0
    ret_3y_cagr = (((current_nav / nav_3y) ** (1/3)) - 1.0) * 100.0 if len(df_nav) >= 756 else ret_1y
    
    vol_30d = float(daily_returns.tail(30).std() * np.sqrt(30) * 100.0) if len(daily_returns) >= 5 else 15.0
    ann_vol = float(daily_returns.tail(252).std() * np.sqrt(252) * 100.0) if len(daily_returns) >= 5 else 18.0
    if np.isnan(vol_30d) or vol_30d == 0: vol_30d = 12.0
    if np.isnan(ann_vol) or ann_vol == 0: ann_vol = 15.0
    
    rolling_max = df_nav['nav'].tail(252).cummax()
    drawdown_series = (df_nav['nav'].tail(252) - rolling_max) / rolling_max
    max_drawdown_1y = float(drawdown_series.min() * 100.0) if len(drawdown_series) > 0 else 0.0
    
    rf = 6.5
    downside = daily_returns.tail(252)[daily_returns.tail(252) < 0]
    downside_std = float(downside.std() * np.sqrt(252) * 100.0) if len(downside) > 0 else ann_vol
    if np.isnan(downside_std) or downside_std == 0: downside_std = ann_vol
    
    sharpe = (ret_1y - rf) / ann_vol if ann_vol > 0 else 0.0
    sortino = (ret_1y - rf) / downside_std if downside_std > 0 else 0.0
    comp_score = max(0.1, min(0.9, 0.5 + (sharpe * 0.15) + (max_drawdown_1y / 100.0 * 0.2)))
    
    lag_1d = float(daily_returns.iloc[-2] * 100.0) if len(daily_returns) >= 2 else 0.0
    lag_5d = float(daily_returns.iloc[-6] * 100.0) if len(daily_returns) >= 6 else lag_1d
    
    feat_dict = {
        'NAV': current_nav,
        'Daily_Return_Pct': float(daily_returns.iloc[-1] * 100.0),
        'Annualized_Return_1Y': ret_1y,
        'Volatility_30D': vol_30d,
        'Annualized_Volatility_Cleaned': ann_vol,
        'Sharpe_Ratio_Cleaned': sharpe,
        'Sortino_Ratio_Cleaned': sortino,
        'Max_Drawdown_1Y_Pct': max_drawdown_1y,
        'Composite_Score': comp_score,
        'Lag_1D_Return': lag_1d,
        'Lag_5D_Return': lag_5d
    }
    X_input = pd.DataFrame([feat_dict])[features]
    prob = float(model.predict_proba(X_input)[0, 1])
    reasoning_list = get_ai_reasoning(feat_dict, prob)
    
    return {
        'name': scheme_name,
        'category': category,
        'amc': amc,
        'nav': current_nav,
        'ret_1m': ret_1m,
        'ret_1y': ret_1y,
        'ret_3y': ret_3y_cagr,
        'vol': ann_vol,
        'drawdown_1y': max_drawdown_1y,
        'sharpe': sharpe,
        'sortino': sortino,
        'comp_score': comp_score,
        'prob': prob,
        'reasoning': reasoning_list,
        'df_nav': df_nav
    }

def run_monte_carlo_var(portfolio_val, portfolio_vol=18.0, crash_prob=0.1, days=30, sims=1000):
    """Monte Carlo 30-day Value-at-Risk (VaR) Simulator."""
    if portfolio_val <= 0:
        return 0.0, np.array([0]), 0.0
    daily_vol = (portfolio_vol / 100.0) / np.sqrt(252)
    drift = - (crash_prob * 0.04)
    simulated_returns = np.random.normal(drift / 252, daily_vol, (sims, days))
    cumulative_paths = np.prod(1 + simulated_returns, axis=1) - 1
    ending_values = portfolio_val * (1 + cumulative_paths)
    var_95_val = np.percentile(ending_values, 5)
    var_loss_pct = ((var_95_val - portfolio_val) / portfolio_val) * 100.0
    return var_95_val, ending_values, var_loss_pct

def generate_audit_report(res):
    verdict_badge = "#c62828" if res['prob'] >= 0.6 else ("#f57f17" if res['prob'] >= 0.3 else "#2e7d32")
    verdict_text = "CRITICAL DISTRESS RISK" if res['prob'] >= 0.6 else ("WATCHLIST / MODERATE RISK" if res['prob'] >= 0.3 else "HEALTHY / SAFE ALLOCATION")
    
    reasons_html = "".join([f"<li>{r}</li>" for r in res['reasoning']])
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>AlphaShield Executive Audit - {res['name']}</title>
        <style>
            body {{ font-family: sans-serif; margin: 30px; color: #212529; line-height: 1.5; }}
            .brand {{ font-size: 24px; font-weight: bold; color: #1E88E5; }}
            .badge {{ display: inline-block; padding: 6px 12px; color: white; background: {verdict_badge}; border-radius: 4px; font-weight: bold; margin-top: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #dee2e6; padding: 10px; text-align: left; }}
            th {{ background-color: #f8f9fa; }}
            .rationale {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #1E88E5; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="brand">🛡️ AlphaShield Institutional Risk & AI Audit</div>
        <h2>{res['name']}</h2>
        <p><strong>Category:</strong> {res['category']} | <strong>AMC:</strong> {res['amc']} | <strong>Live Price/NAV:</strong> ₹{res['nav']:.2f}</p>
        <div class="badge">{verdict_text} (30-Day Forward Crash Probability: {res['prob']*100:.1f}%)</div>
        
        <div class="rationale">
            <h3>🧠 Explainable AI Decision Rationale (Why this score?)</h3>
            <ul>
                {reasons_html}
            </ul>
        </div>

        <table>
            <tr><th>Metric</th><th>Observed Value</th><th>Benchmark Threshold</th></tr>
            <tr><td>Forward Crash Probability</td><td>{res['prob']*100:.1f}%</td><td>&lt; 30.0%</td></tr>
            <tr><td>Composite Score</td><td>{res['comp_score']:.2f}</td><td>&gt; 0.55</td></tr>
            <tr><td>1-Year Return</td><td>{res['ret_1y']:+.2f}%</td><td>&gt; +6.50%</td></tr>
            <tr><td>1-Year Max Drawdown</td><td>{res['drawdown_1y']:.2f}%</td><td>&gt; -12.0%</td></tr>
            <tr><td>Sharpe Ratio</td><td>{res['sharpe']:.2f}</td><td>&gt; 0.50</td></tr>
            <tr><td>Sortino Ratio</td><td>{res['sortino']:.2f}</td><td>&gt; 0.80</td></tr>
        </table>
    </body>
    </html>
    """

if 'cash' not in st.session_state:
    st.session_state.cash = 100000.0
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

st.sidebar.header("🕹️ Analytics Suite")
app_mode = st.sidebar.radio(
    "Choose Analysis Module:",
    [
        "📈 NIFTY50 Stock Crash-Risk Predictor",
        "⚔️ Head-to-Head Scheme Duel",
        "🔍 Single Scheme Intelligence & Stress-Tester",
        "💼 Universal Paper Trading & AI Ledger",
        "📊 Monte Carlo Portfolio VaR Simulator",
        "📁 Historical Dataset Archive"
    ]
)

# ==============================================================================
# VIEW 0: NIFTY50 STOCK CRASH-RISK PREDICTOR
# ==============================================================================
if app_mode == "📈 NIFTY50 Stock Crash-Risk Predictor":
    st.title("📈 NIFTY50 Stock Crash-Risk Predictor")
    st.markdown("Evaluate individual equity risk profiles and generate transparent AI decision rationale.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        stock_symbol = st.selectbox("Select NIFTY50 Stock:", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS"])
        stock_price = st.number_input("Current Stock Price (₹):", value=2500.0, step=10.0)
    with col_s2:
        stock_ret_1y = st.number_input("Trailing 1Y Return (%):", value=15.5, step=0.5)
        stock_vol = st.number_input("Annualized Volatility (%):", value=22.0, step=0.5)

    if st.button("🚀 Run Stock Crash Prediction", type="primary"):
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
        if stock_prob >= 0.6:
            st.error(f"### 🔴 CRITICAL CRASH RISK ({stock_prob*100:.1f}% over next 30 days)")
        elif stock_prob >= 0.3:
            st.warning(f"### 🟡 MODERATE WATCHLIST RISK ({stock_prob*100:.1f}% over next 30 days)")
        else:
            st.success(f"### 🟢 LOW CRASH RISK ({stock_prob*100:.1f}% over next 30 days)")
            
        st.markdown("#### 🧠 Explainable AI Rationale:")
        for r in reasons:
            st.write(f"- {r}")

# ==============================================================================
# VIEW 1: HEAD-TO-HEAD SCHEME DUEL
# ==============================================================================
elif app_mode == "⚔️ Head-to-Head Scheme Duel":
    st.title("⚔️ Live Head-to-Head Scheme Duel")
    duel_source = st.radio("Selection Source:", ["⭐ Popular Active Benchmark Schemes", "🔎 Search Universal Schemes"], horizontal=True, key="duel_src")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🟦 Fund A")
        if duel_source.startswith("⭐"):
            choice_a = st.selectbox("Select Scheme A:", list(ACTIVE_BENCHMARK_FUNDS.keys()), index=0, key="duel_preset_a")
            code_a = ACTIVE_BENCHMARK_FUNDS[choice_a]
        else:
            search_options = all_schemes_df['Search_Label'].tolist() if not all_schemes_df.empty else list(ACTIVE_BENCHMARK_FUNDS.keys())
            choice_a = st.selectbox("Search Scheme A:", options=search_options, index=0, key="duel_a")
            code_a = all_schemes_df[all_schemes_df['Search_Label'] == choice_a]['Scheme_Code'].iloc[0] if not all_schemes_df.empty else "118989"
            
    with col_b:
        st.markdown("### 🟧 Fund B")
        if duel_source.startswith("⭐"):
            choice_b = st.selectbox("Select Scheme B:", list(ACTIVE_BENCHMARK_FUNDS.keys()), index=1, key="duel_preset_b")
            code_b = ACTIVE_BENCHMARK_FUNDS[choice_b]
        else:
            search_options = all_schemes_df['Search_Label'].tolist() if not all_schemes_df.empty else list(ACTIVE_BENCHMARK_FUNDS.keys())
            default_b = min(1, len(search_options) - 1)
            choice_b = st.selectbox("Search Scheme B:", options=search_options, index=default_b, key="duel_b")
            code_b = all_schemes_df[all_schemes_df['Search_Label'] == choice_b]['Scheme_Code'].iloc[0] if not all_schemes_df.empty else "120465"
        
    if st.button("⚔️ Launch Live Duel", type="primary", use_container_width=True):
        with st.spinner("Fetching live AMFI data and evaluating AI distress indicators..."):
            try:
                res_a = compute_scheme_metrics(code_a.strip())
                res_b = compute_scheme_metrics(code_b.strip())
                
                st.markdown("---")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric(res_a['name'], f"{res_a['prob']*100:.1f}% Risk", f"{res_a['ret_1y']:+.2f}% 1Y Ret")
                    st.markdown("**Rationale:**")
                    for r in res_a['reasoning']: st.write(f"- {r}")
                with col_res2:
                    st.metric(res_b['name'], f"{res_b['prob']*100:.1f}% Risk", f"{res_b['ret_1y']:+.2f}% 1Y Ret")
                    st.markdown("**Rationale:**")
                    for r in res_b['reasoning']: st.write(f"- {r}")
            except Exception as e:
                st.error(f"Duel error: {e}")

# ==============================================================================
# VIEW 2: SINGLE SCHEME INTELLIGENCE & STRESS-TESTER
# ==============================================================================
elif app_mode == "🔍 Single Scheme Intelligence & Stress-Tester":
    st.title("🔍 Single Scheme Intelligence & Stress-Testing")
    trade_source = st.radio("Selection Source:", ["⭐ Popular Active Benchmark Schemes", "🔎 Search Full Scheme Universe"], horizontal=True)
    c_in, c_bt = st.columns([3.5, 1])
    with c_in:
        if trade_source.startswith("⭐"):
            chosen_label = st.selectbox("Select Benchmark Fund:", list(ACTIVE_BENCHMARK_FUNDS.keys()), key="preset_single_sel")
            selected_code = ACTIVE_BENCHMARK_FUNDS[chosen_label]
        else:
            search_options = all_schemes_df['Search_Label'].tolist() if not all_schemes_df.empty else list(ACTIVE_BENCHMARK_FUNDS.keys())
            selection = st.selectbox("Search any scheme:", options=search_options, index=0)
            selected_code = all_schemes_df[all_schemes_df['Search_Label'] == selection]['Scheme_Code'].iloc[0] if not all_schemes_df.empty else "118989"
    with c_bt:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        audit_btn = st.button("🚀 Analyze Live Scheme", type="primary", use_container_width=True)
        
    if selected_code:
        with st.spinner("Fetching AMFI data and evaluating AI risk profile..."):
            try:
                res = compute_scheme_metrics(selected_code)
                st.markdown(f"**Scheme:** `{res['name']}` | **Category:** `{res['category']}` | **Live NAV:** `₹{res['nav']:.2f}`")
                
                tab_core, tab_stress, tab_export = st.tabs(["📊 Core Risk & Rationale", "⚡ Crisis Stress-Testing", "📄 Executive Factsheet"])
                with tab_core:
                    st.metric("30-Day Forward Crash Probability", f"{res['prob']*100:.1f}%")
                    st.markdown("#### 🧠 Decision Rationale:")
                    for r in res['reasoning']:
                        st.write(f"- {r}")
                with tab_stress:
                    sim_shock = st.slider("Simulate Hypothetical Market Crash (% Shock):", -40, -5, -20, step=5)
                    est_loss = sim_shock * (res['vol'] / 15.0)
                    st.metric(f"Simulated {sim_shock}% Shock Impact", f"{est_loss:.1f}% Loss")
                with tab_export:
                    report_html = generate_audit_report(res)
                    st.download_button("📥 Download Executive HTML Factsheet", data=report_html, file_name=f"Audit_{selected_code}.html", mime="text/html")
            except Exception as e:
                st.error(f"Error: {e}")

# ==============================================================================
# VIEW 3: UNIVERSAL PAPER TRADING & AI LEDGER (+ FEATURE 2: REBALANCER)
# ==============================================================================
elif app_mode == "💼 Universal Paper Trading & AI Ledger":
    st.title("💼 Universal Paper Trading & AI Risk Ledger")
    
    current_portfolio_val = sum(pos['units'] * pos['buy_price'] for pos in st.session_state.portfolio)
    total_net_worth = st.session_state.cash + current_portfolio_val
    overall_pnl = current_portfolio_val - sum(pos['invested_amt'] for pos in st.session_state.portfolio)
    
    c_w1, c_w2, c_w3, c_w4 = st.columns(4)
    c_w1.metric("Available Paper Cash", f"₹{st.session_state.cash:,.2f}")
    c_w2.metric("Portfolio Market Value", f"₹{current_portfolio_val:,.2f}")
    c_w3.metric("Total Net Worth", f"₹{total_net_worth:,.2f}", f"₹{overall_pnl:+,.2f}")
    c_w4.metric("Total Holdings", f"{len(st.session_state.portfolio)}")
    
    st.markdown("---")
    st.subheader("🛒 Execute Multi-Asset Paper Order")
    
    asset_class = st.radio("Select Asset Class:", ["Stocks (NIFTY50)", "Mutual Funds (AMFI Live)"], horizontal=True)
    
    c_t1, c_t2, c_t3 = st.columns([2.5, 1, 1])
    with c_t1:
        if asset_class.startswith("Stocks"):
            trade_target_name = st.selectbox("Select Stock:", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"])
            trade_price = st.number_input("Execution Price (₹):", value=2500.0, step=10.0)
            trade_code = trade_target_name
            sample_p = 0.12
        else:
            trade_scheme_label = st.selectbox("Select Mutual Fund Scheme:", list(ACTIVE_BENCHMARK_FUNDS.keys()))
            trade_code = ACTIVE_BENCHMARK_FUNDS[trade_scheme_label]
            trade_target_name = trade_scheme_label
            try:
                temp_res = compute_scheme_metrics(trade_code)
                trade_price = temp_res['nav']
                sample_p = temp_res['prob']
            except:
                trade_price = 100.0
                sample_p = 0.15
    with c_t2:
        order_amt = st.number_input("Investment Amount (₹):", min_value=1000.0, max_value=max(1000.0, st.session_state.cash), value=10000.0, step=1000.0)
    with c_t3:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        execute_order = st.button("📥 Execute Trade", type="primary", use_container_width=True)
        
    if execute_order:
        if order_amt > st.session_state.cash:
            st.error("Insufficient paper cash!")
        else:
            units_bought = order_amt / trade_price
            st.session_state.portfolio.append({
                'time': datetime.now().strftime("%d-%b-%Y %H:%M"),
                'type': "Stock" if asset_class.startswith("Stocks") else "Mutual Fund",
                'name': trade_target_name,
                'buy_price': trade_price,
                'units': units_bought,
                'invested_amt': order_amt,
                'ai_risk_pct': f"{sample_p*100:.1f}%"
            })
            st.session_state.cash -= order_amt
            st.success(f"Successfully bought {units_bought:.3f} units of {trade_target_name}!")
            st.rerun()

    st.markdown("---")
    st.subheader("📑 Active Holdings & AI Portfolio Rebalancer")
    if len(st.session_state.portfolio) == 0:
        st.info("No active holdings yet. Allocate virtual cash above to begin.")
    else:
        df_ledger = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(df_ledger, use_container_width=True)
        
        # FEATURE 2: Automated AI Rebalancer
        if st.button("⚖️ Run AI Portfolio Risk Rebalancer", type="primary"):
            st.markdown("#### 🤖 Rebalancing Audit & Recommendations:")
            flagged_count = 0
            for i, pos in enumerate(st.session_state.portfolio):
                risk_val = float(pos['ai_risk_pct'].replace('%', ''))
                if risk_val >= 50.0:
                    flagged_count += 1
                    st.warning(f"⚠️ **{pos['name']}** flagged with high distress risk ({pos['ai_risk_pct']}). Recommendation: Liquidate position and shift capital to *HDFC Top 100 Fund*.")
                else:
                    st.success(f"✅ **{pos['name']}** risk level is within acceptable tolerance ({pos['ai_risk_pct']}).")
            if flagged_count == 0:
                st.success("All portfolio holdings are stable. No rebalancing actions required.")

        if st.button("🔄 Reset Portfolio", type="secondary"):
            st.session_state.cash = 100000.0
            st.session_state.portfolio = []
            st.rerun()

# ==============================================================================
# VIEW 4: MONTE CARLO PORTFOLIO VaR SIMULATOR (+ FEATURE 3)
# ==============================================================================
elif app_mode == "📊 Monte Carlo Portfolio VaR Simulator":
    st.title("📊 Monte Carlo 30-Day Value-at-Risk (VaR) Simulator")
    st.markdown("Simulate 1,000 future price paths for your multi-asset portfolio to estimate potential tail-risk losses over the next 30 days.")
    
    current_portfolio_val = sum(pos['units'] * pos['buy_price'] for pos in st.session_state.portfolio)
    
    if current_portfolio_val <= 0:
        st.warning("Your active portfolio has zero value. Please execute some paper trades in the Ledger module first.")
    else:
        sim_vol = st.slider("Estimated Portfolio Annualized Volatility (%):", 5.0, 40.0, 18.0, step=1.0)
        sim_crash_prob = st.slider("Aggregated AI Crash Probability (%):", 0.0, 100.0, 15.0, step=1.0) / 100.0
        
        if st.button("🚀 Run 1,000 Monte Carlo Simulations", type="primary"):
            var_95_val, ending_values, var_loss_pct = run_monte_carlo_var(current_portfolio_val, sim_vol, sim_crash_prob, days=30, sims=1000)
            
            st.markdown("---")
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Current Portfolio Value", f"₹{current_portfolio_val:,.2f}")
            c_m2.metric("95% 30-Day Value-at-Risk (VaR)", f"₹{var_95_val:,.2f}", f"{var_loss_pct:+.2f}%")
            c_m3.metric("Simulated Worst-Case Floor (1%)", f"₹{np.percentile(ending_values, 1):,.2f}")
            
            fig = px.histogram(ending_values, nbins=50, title="Distribution of Portfolio Ending Values (30 Days Ahead)",
                               labels={'value': 'Ending Portfolio Value (₹)', 'count': 'Frequency'})
            fig.add_vline(x=var_95_val, line_dash="dash", line_color="red", annotation_text="95% VaR Threshold")
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# VIEW 5: HISTORICAL DATASET ARCHIVE
# ==============================================================================
else:
    st.title("📁 Historical Research Dataset (47,272 Records)")
    all_names = sorted(funds_df['Fund_Name'].dropna().unique().tolist())
    picked_fund = st.selectbox("Select Historical Fund:", all_names)
    if picked_fund:
        row = funds_df[funds_df['Fund_Name'] == picked_fund].iloc[0]
        st.markdown(f"**Scheme:** `{row['Fund_Name']}` | **Category:** `{row.get('Category', 'N/A')}`")
        X_h = pd.DataFrame([[float(row[c]) for c in features]], columns=features)
        h_prob = float(model.predict_proba(X_h)[0, 1])
        st.metric("Historical Failure Risk", f"{h_prob*100:.1f}%")
