import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# CONFIGURATION & PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="AlphaShield | Integrated Market & Fund Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# High-Visibility Professional Styling (Fixes contrast and text visibility)
st.markdown("""
<style>
    /* Force consistent dark theme background and text visibility */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    /* Sidebar container */
    [data-testid="stSidebar"] {
        background-color: #111827;
        color: #f8fafc;
    }
    /* Metric & Card styling */
    .stMetric, [data-testid="stMetric"], .card {
        background-color: #131b2e !important;
        color: #f8fafc !important;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }
    /* Typography contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    p, span, label, .stMarkdown, div[data-testid="stMarkdownContainer"] {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INITIALIZATION (For Paper Trading Ledger)
# ==============================================================================
if "ledger" not in st.session_state:
    st.session_state.ledger = pd.DataFrame(columns=[
        "Date", "Type", "Asset", "Ticker/Code", "Quantity", "Buy Price", "Current Price", "Invested Value", "Current Value", "P&L (%)"
    ])

# ==============================================================================
# DATA FETCHING & HELPER FUNCTIONS
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="1y"):
    """Fetch live stock data from Yahoo Finance."""
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=3600)
def fetch_mf_data(code):
    """Fetch live AMFI Mutual Fund data from public API."""
    url = f"https://api.mfapi.in/mf/{code}"
    res = requests.get(url).json()
    df = pd.DataFrame(res['data'])
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['nav'] = pd.to_numeric(df['nav'])
    df = df.sort_values('date').set_index('date')
    return res['meta'], df

def compute_stock_crash_model(df):
    """Compute features and simulate Random Forest Crash Risk Probability."""
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['Vol_21'] = df['Return'].rolling(21).std() * np.sqrt(252) * 100
    df['Drawdown'] = (df['Close'] / df['Close'].cummax() - 1) * 100
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()
    df = df.dropna()
    
    X = df[['Vol_21', 'Drawdown', 'Return']].copy()
    X['Vol_21'] = X['Vol_21'].fillna(0)
    X['Drawdown'] = X['Drawdown'].fillna(0)
    X['Return'] = X['Return'].fillna(0)
    
    y = ((X['Return'] < -0.025) | (X['Vol_21'] > 40)).astype(int)
    
    if len(y.unique()) > 1:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        crash_prob = model.predict_proba(X)[:, 1][-1]
    else:
        crash_prob = 0.15
        
    return df, float(crash_prob)

def analyze_mf(code, rf_rate=0.06):
    """Analyze mutual fund performance and risk metrics."""
    meta, df = fetch_mf_data(code)
    navs = df['nav']
    ret1m = navs.iloc[-1] / navs.iloc[-30] - 1 if len(navs) >= 30 else 0
    ret1y = navs.iloc[-1] / navs.iloc[-252] - 1 if len(navs) >= 252 else navs.iloc[-1] / navs.iloc[0] - 1
    ret3y = (navs.iloc[-1] / navs.iloc[-756]) ** (1/3) - 1 if len(navs) >= 756 else np.nan
    
    daily_ret = navs.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252) * 100
    vol30 = daily_ret.iloc[-30:].std() * np.sqrt(252) * 100 if len(daily_ret) >= 30 else vol
    
    ann_ret = daily_ret.mean() * 252
    sharpe = (ann_ret - rf_rate) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    neg_ret = daily_ret[daily_ret < 0]
    sortino = (ann_ret - rf_rate) / (neg_ret.std() * np.sqrt(252)) if len(neg_ret) > 0 and neg_ret.std() > 0 else 0
    
    cum_max = navs.cummax()
    dd = ((navs - cum_max) / cum_max).min() * 100
    
    score = max(0, min(100, (sharpe * 20) + (100 - vol)))
    prob = max(0.05, min(0.95, vol / 50.0 - sharpe * 0.1))
    
    return {
        "name": meta.get('fund_name', code),
        "category": meta.get('fund_category', 'Equity Scheme'),
        "amc": meta.get('mutual_fund_family', 'N/A'),
        "nav": navs.iloc[-1],
        "ret1m": ret1m, "ret1y": ret1y, "ret3y": ret3y,
        "vol": vol, "vol30": vol30, "sharpe": sharpe, "sortino": sortino,
        "dd": dd, "score": score, "prob": prob, "navdf": df,
        "features": {"Volatility": f"{vol:.2f}%", "Max Drawdown": f"{dd:.2f}%", "Sharpe Ratio": f"{sharpe:.2f}"}
    }

def clean_text(text):
    return str(text).replace("&", "and").replace("<", "").replace(">", "")

def money(val):
    return f"₹{val:,.2f}"

def pc(val):
    return f"{val*100:.2f}%"

def status_label(prob):
    if prob < 0.3: return "🟢 Low Risk"
    elif prob < 0.6: return "🟡 Moderate Risk"
    else: return "🔴 High Crash Risk"

def gauge_chart(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Model Crash Risk Indicator (%)", 'font': {'color': 'white'}},
        number={'font': {'color': 'white'}},
        gauge={'axis': {'range': [0, 100], 'tickfont': {'color': 'white'}},
               'bar': {'color': "#3b82f6"},
               'steps': [
                   {'range': [0, 30], 'color': "rgba(40, 167, 69, 0.3)"},
                   {'range': [30, 60], 'color': "rgba(255, 193, 7, 0.3)"},
                   {'range': [60, 100], 'color': "rgba(220, 53, 69, 0.3)"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': 'white'})
    return fig

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🛡️ AlphaShield Platform")
st.sidebar.caption("Integrated Stock & Mutual Fund Intelligence")
page = st.sidebar.radio("Navigation", [
    "🏠 Executive Overview",
    "📉 Stock Crash Predictor & Analytics",
    "🔍 Fund Intelligence (AMFI)",
    "💼 Paper Trading Ledger"
])

rf_rate = st.sidebar.slider("Benchmark Risk-Free Rate (%)", 3.0, 10.0, 6.0, 0.5) / 100.0

# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if page == "🏠 Executive Overview":
    st.title("🛡️ AlphaShield Unified Financial Intelligence")
    st.markdown("Welcome to your institutional-grade research platform. Monitor live equity crash-risk diagnostics via Random Forest models alongside AMFI-backed mutual fund intelligence.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Supported Asset Classes", "Equities & Mutual Funds", "Live Feed")
    col2.metric("ML Engine", "Random Forest Classifier", "Active")
    col3.metric("Data Sources", "Yahoo Finance & AMFI India", "Real-Time")
    
    st.markdown("---")
    st.subheader("📊 Platform Core Modules")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### 📉 Stock Crash Predictor
        * **Live Ticker Analysis**: Pulls real-time NIFTY 50 and equity quotes.
        * **Machine Learning**: Predicts market crash probabilities using rolling volatility and drawdown indicators.
        * **Interactive Charts**: Full historical price action and drawdown mapping.
        """)
    with c2:
        st.markdown("""
        ### 🔍 Fund Intelligence (AMFI)
        * **Live NAV Tracking**: Direct integration with official AMFI endpoints.
        * **Risk & Performance Metrics**: Sharpe, Sortino, CAGR, and volatility diagnostics.
        * **Downloadable Factsheets**: Generate professional HTML audit reports instantly.
        """)

# ==============================================================================
# PAGE 2: STOCK CRASH PREDICTOR & ANALYTICS
# ==============================================================================
elif page == "📉 Stock Crash Predictor & Analytics":
    st.subheader("📉 Stock Crash-Risk Intelligence Dashboard")
    st.markdown("Analyze NIFTY 50 and custom equities using machine learning crash-risk classification powered by `yfinance`.")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        stock_ticker = st.text_input("Enter Equity Ticker (Yahoo Finance format)", value="RELIANCE.NS")
    with col_b:
        stock_period = st.selectbox("Historical Horizon", ["6mo", "1y", "2y", "5y"], index=1)
        
    if st.button("🚀 Run Crash Predictor Model", type="primary", use_container_width=True):
        with st.spinner(f"Fetching live data for {stock_ticker} and evaluating ML model..."):
            try:
                raw_df = fetch_stock_data(stock_ticker, period=stock_period)
                if raw_df.empty:
                    st.error("Invalid ticker or no data retrieved.")
                else:
                    df_res, prob = compute_stock_crash_model(raw_df)
                    latest_close = df_res['Close'].iloc[-1]
                    
                    st.success(f"Analysis Complete for **{stock_ticker}** | Current Price: ₹{latest_close:,.2f}")
                    
                    if prob < 0.3:
                        st.success(f"{status_label(prob)} — Model Crash Indicator: {prob*100:.1f}%")
                    elif prob < 0.6:
                        st.warning(f"{status_label(prob)} — Model Crash Indicator: {prob*100:.1f}%")
                    else:
                        st.error(f"{status_label(prob)} — Model Crash Indicator: {prob*100:.1f}%")
                        
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Annualized Volatility", f"{df_res['Vol_21'].iloc[-1]:.2f}%")
                    m2.metric("Max Drawdown", f"{df_res['Drawdown'].min():.2f}%")
                    m3.metric("50-Day MA", f"₹{df_res['MA_50'].iloc[-1]:,.2f}")
                    m4.metric("200-Day MA", f"₹{df_res['MA_200'].iloc[-1]:,.2f}")
                    
                    t1, t2, t3 = st.tabs(["📈 Price & Moving Averages", "📉 Drawdown & Volatility", "🤖 ML Risk Gauge"])
                    with t1:
                        fig = px.line(df_res, x=df_res.index, y=['Close', 'MA_50', 'MA_200'], title=f"{stock_ticker} Price & Trend")
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': 'white'})
                        st.plotly_chart(fig, use_container_width=True)
                    with t2:
                        fig_dd = px.area(df_res, x=df_res.index, y='Drawdown', title=f"{stock_ticker} Historical Drawdown (%)")
                        fig_dd.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': 'white'})
                        st.plotly_chart(fig_dd, use_container_width=True)
                    with t3:
                        c_gauge, c_info = st.columns([1, 1.5])
                        with c_gauge:
                            st.plotly_chart(gauge_chart(prob), use_container_width=True)
                        with c_info:
                            st.markdown("### Model Diagnostics")
                            st.write("The Random Forest classifier evaluates rolling volatility spikes, return anomalies, and drawdown depth to compute short-term distress probability.")
                            st.info("Disclaimer: This model is an experimental research signal, not financial advice.")
            except Exception as e:
                st.error(f"Error processing stock data: {str(e)}")

# ==============================================================================
# PAGE 3: FUND INTELLIGENCE (AMFI)
# ==============================================================================
elif page == "🔍 Fund Intelligence (AMFI)":
    st.subheader("🔍 Mutual Fund Scheme Intelligence")
    st.markdown("Live AMFI scheme analytics with risk metrics, performance attribution, and instant factsheet generation.")
    
    mf_options = {
        "SBI Blue Fund (120503)": "120503",
        "Axis Bluechip Fund (118834)": "118834",
        "Mirae Asset Large Cap (120150)": "120150",
        "ICICI Prudential Bluechip (120586)": "120586"
    }
    selected_mf_name = st.selectbox("Select Scheme or Enter Code", list(mf_options.keys()))
    code = mf_options[selected_mf_name]
    
    if st.button("🚀 Analyze Live Mutual Fund", type="primary", use_container_width=True):
        try:
            with st.spinner("Fetching live NAV history from AMFI..."):
                r = analyze_mf(code, rf_rate)
                st.markdown(f"**{clean_text(r['name'])}** · {r['category']} · {r['amc']} · NAV {money(r['nav'])}")
                
                if r["prob"] < 0.3:
                    st.success(status_label(r["prob"]) + " — " + f"model indicator {r['prob']*100:.1f}%")
                elif r["prob"] < 0.6:
                    st.warning(status_label(r["prob"]) + " — " + f"model indicator {r['prob']*100:.1f}%")
                else:
                    st.error(status_label(r["prob"]) + " — " + f"model indicator {r['prob']*100:.1f}%")
                    
                a, b, c, d, e = st.columns(5)
                a.metric("1M Return", pc(r["ret1m"]))
                b.metric("1Y Return", pc(r["ret1y"]))
                c.metric("3Y CAGR", pc(r["ret3y"]) if np.isfinite(r["ret3y"]) else "N/A")
                d.metric("Sharpe Ratio", f"{r['sharpe']:.2f}")
                e.metric("Max Drawdown", f"{r['dd']:.1f}%")
                
                t1, t2, t3, t4 = st.tabs(["📈 Performance", "🛡️️ Risk Metrics", "🤖 AI Health Score", "📄 Factsheet"])
                with t1:
                    fig_nav = px.line(r["navdf"], x=r["navdf"].index, y="nav", title="Historical NAV")
                    fig_nav.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': 'white'})
                    st.plotly_chart(fig_nav, use_container_width=True)
                    
                    dd_series = (r["navdf"].nav / r["navdf"].nav.cummax() - 1) * 100
                    fig_mf_dd = px.area(x=dd_series.index, y=dd_series.values, title="Full-History Drawdown (%)")
                    fig_mf_dd.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': 'white'})
                    st.plotly_chart(fig_mf_dd, use_container_width=True)
                with t2:
                    x_col, y_col, z_col = st.columns(3)
                    x_col.metric("Annualized Volatility", f"{r['vol']:.2f}%")
                    y_col.metric("30D Volatility", f"{r['vol30']:.2f}%")
                    z_col.metric("Sortino Ratio", f"{r['sortino']:.2f}")
                with t3:
                    x_g, y_g = st.columns([1, 1.5])
                    with x_g:
                        st.plotly_chart(gauge_chart(r["prob"]), use_container_width=True)
                    with y_g:
                        st.metric("Composite Health Score", f"{r['score']:.2f}")
                        st.dataframe(pd.DataFrame({"Feature": list(r["features"].keys()), "Value": list(r["features"].values())}), use_container_width=True, hide_index=True)
                        st.caption("The model indicator is a research signal, not a guarantee of fund distress/failure.")
                with t4:
                    html = f"""<html><body><h1>AlphaShield Factsheet</h1><h2>{clean_text(r['name'])}</h2>
                    <p>{r['category']} | {r['amc']} | NAV {money(r['nav'])}</p>
                    <table border=1 cellpadding=8><tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>1Y Return</td><td>{pc(r['ret1y'])}</td></tr><tr><td>3Y CAGR</td><td>{pc(r['ret3y']) if np.isfinite(r['ret3y']) else 'N/A'}</td></tr>
                    <tr><td>Volatility</td><td>{r['vol']:.2f}%</td></tr><tr><td>Max Drawdown</td><td>{r['dd']:.2f}%</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{r['sharpe']:.2f}</td></tr><tr><td>Sortino Ratio</td><td>{r['sortino']:.2f}</td></tr>
                    <tr><td>Composite Score</td><td>{r['score']:.2f}</td></tr><tr><td>Model Risk</td><td>{r['prob']*100:.1f}%</td></tr></table>
                    <p>This is a research document, not investment advice.</p></body></html>"""
                    st.download_button("📥 Download HTML Factsheet", html, f"AlphaShield_{code}.html", "text/html", type="primary")
        except Exception as e:
            st.error(f"Error fetching fund data: {str(e)}")

# ==============================================================================
# PAGE 4: PAPER TRADING LEDGER
# ==============================================================================
elif page == "💼 Paper Trading Ledger":
    st.subheader("💼 Unified Paper Trading & Portfolio Ledger")
    st.markdown("Seamlessly track both **Equities** and **Mutual Funds** in a single integrated portfolio ledger.")
    
    with st.form("trade_form"):
        st.markdown("### Add New Position")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            asset_type = st.selectbox("Asset Class", ["Equity (Stock)", "Mutual Fund (AMFI)"])
        with col2:
            asset_name = st.text_input("Asset Name / Ticker", value="RELIANCE.NS" if asset_type.startswith("Equity") else "120503")
        with col3:
            qty = st.number_input("Quantity / Units", min_value=0.1, value=10.0, step=1.0)
        with col4:
            buy_price = st.text_input("Buy Price (₹)", value="2500.0")
            
        submitted = st.form_submit_button("➕ Add to Ledger", type="primary")
        if submitted:
            try:
                b_price = float(buy_price)
                current_p = b_price
                if asset_type.startswith("Equity"):
                    df_live = fetch_stock_data(asset_name, period="5d")
                    if not df_live.empty:
                        current_p = float(df_live['Close'].iloc[-1])
                else:
                    _, df_mf = fetch_mf_data(asset_name)
                    if not df_mf.empty:
                        current_p = float(df_mf['nav'].iloc[-1])
                
                inv_val = qty * b_price
                curr_val = qty * current_p
                pnl_pct = ((current_p - b_price) / b_price) * 100
                
                new_row = {
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Type": asset_type,
                    "Asset": asset_name,
                    "Ticker/Code": asset_name,
                    "Quantity": qty,
                    "Buy Price": b_price,
                    "Current Price": current_p,
                    "Invested Value": inv_val,
                    "Current Value": curr_val,
                    "P&L (%)": f"{pnl_pct:+.2f}%"
                }
                st.session_state.ledger = pd.concat([st.session_state.ledger, pd.DataFrame([new_row])], ignore_index=True)
                st.success("Position successfully added to portfolio ledger!")
            except Exception as e:
                st.error(f"Could not fetch live price for validation: {str(e)}")
                
    st.markdown("---")
    st.subheader("📋 Active Portfolio Holdings")
    if st.session_state.ledger.empty:
        st.info("No positions in ledger yet. Add your first stock or mutual fund above.")
    else:
        st.dataframe(st.session_state.ledger, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear Ledger"):
            st.session_state.ledger = pd.DataFrame(columns=st.session_state.ledger.columns)
            st.rerun()
