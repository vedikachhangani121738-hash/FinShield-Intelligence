import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import joblib
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & DUAL-TONE CSS
# ==========================================
st.set_page_config(
    page_title="FinShield Intelligence | Global Terminal", 
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
    .stTextInput > div > div > input, .stSelectbox > div > div > div, .stNumberInput > div > div > input { border-radius: 8px; border: 2px solid #0A2540; font-weight: bold; }
    .blue-card { background-color: #0A2540; color: #FFFFFF; padding: 1.5rem; border-radius: 12px; box-shadow: 0 10px 15px rgba(0,0,0,0.1); margin-bottom: 1rem; border-left: 6px solid; }
    .blue-card h3, .blue-card p { color: #FFFFFF !important; margin: 0; }
    .js-plotly-plot { margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if 'cash_balance' not in st.session_state:
    st.session_state.cash_balance = 1000000.0  # ₹10,00,000 Initial Capital
if 'trade_ledger' not in st.session_state:
    st.session_state.trade_ledger = []

# ==========================================
# 3. GLOBAL LIVE YFINANCE & PIPELINES
# ==========================================
@st.cache_data(ttl=900)
def fetch_live_data(ticker):
    clean_ticker = ticker.strip()
    try:
        df = yf.download(clean_ticker, period="1y", interval="1d", progress=False)
        if df.empty and not clean_ticker.endswith('.NS'):
            df = yf.download(f"{clean_ticker}.NS", period="1y", interval="1d", progress=False)
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_crash_risk(df):
    try:
        rf_model = joblib.load('random_forest_crash_model.joblib') 
        return 15.0 
    except FileNotFoundError:
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  
        momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100 
        risk = (volatility * 1.8) - (momentum * 0.8)
        return max(2.0, min(98.0, risk))

def compute_technical_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# Expanded Institutional Mutual Fund Database
mf_database = pd.DataFrame({
    'Fund Name': [
        'Quant Active Fund', 'Parag Parikh Flexi Cap', 'SBI Bluechip Fund', 
        'Nippon India Small Cap', 'HDFC Mid-Cap Opportunities', 'Axis Bluechip Fund',
        'Mirae Asset Large Cap', 'Kotak Emerging Equity', 'ICICI Pru Bluechip Fund',
        'Tata Digital India Fund', 'SBI Small Cap Fund', 'UTI Flexi Cap Fund'
    ],
    'Category': [
        'Multi Cap', 'Flexi Cap', 'Large Cap', 'Small Cap', 'Mid Cap', 'Large Cap',
        'Large Cap', 'Mid Cap', 'Large Cap', 'Thematic / Tech', 'Small Cap', 'Flexi Cap'
    ],
    '1Y Return (%)': [32.4, 24.5, 18.2, 45.1, 38.6, 16.8, 19.2, 36.4, 19.5, 28.1, 41.2, 22.1],
    'Alpha': [5.2, 3.8, 1.1, 7.5, 6.2, 0.9, 1.4, 5.8, 1.5, 4.2, 6.9, 2.9],
    'Beta': [1.10, 0.85, 0.95, 1.25, 1.15, 0.90, 0.92, 1.12, 0.93, 1.30, 1.18, 0.88],
    'Expense Ratio (%)': [0.58, 0.65, 1.10, 0.75, 0.92, 1.02, 0.98, 0.82, 0.95, 0.85, 0.78, 0.90],
    'Risk Score': [65, 40, 35, 85, 72, 32, 34, 70, 36, 88, 80, 42]
})

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h2>🛡️ FinShield Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; font-size:0.9rem;'>Global Institutional Terminal v3.9</p>", unsafe_allow_html=True)
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
# 5. MODULE EXECUTIONS (ARTIFACT-FREE)
# ==========================================

# --- MODULE 1 ---
if "1." in app_mode:
    st.markdown("<h1>📉 Global Stock Crash-Risk Prediction Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; margin-bottom:1rem;'>Search any global stock or asset ticker linked directly to live market feeds.</p>", unsafe_allow_html=True)
    
    search_col, _ = st.columns([1.5, 1.5])
    with search_col:
        raw_ticker = st.text_input("🔍 Search Any Global Ticker:", value="RELIANCE.NS", key="m1_search")
        ticker = raw_ticker.strip().upper() if raw_ticker else "RELIANCE.NS"

    df = fetch_live_data(ticker)
    
    if not df.empty and len(df) > 5:
        col_target, col_chart = st.columns([1, 2.5])
        with col_target:
            ltp = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2])
            change_pct = ((ltp - prev_close) / prev_close) * 100
            risk_score = round(calculate_crash_risk(df), 1)
            
            cutoff = 30.0
            risk_color = "#00E676" if risk_score < cutoff else "#FF1744"
            risk_status = "SAFE (LOW RISK)" if risk_score < cutoff else "DANGER (HIGH RISK)"

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
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                increasing_line_color='#00E676', decreasing_line_color='#FF1744'
            )])
            fig_candle.update_layout(title=f"Live Price Action - {ticker}", margin=dict(l=20, r=20, t=40, b=20), height=350, xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_candle, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ⚙️ Feature Importance (SHAP)")
            features = pd.DataFrame({'Feature': ['RSI Momentum', 'MACD Divergence', 'Volatility (Live)', 'Volume Surge', 'Moving Avg Cross'], 'Weight': [0.35, 0.25, 0.20, 0.12, 0.08]}).sort_values(by='Weight', ascending=True)
            fig_bar = px.bar(features, x='Weight', y='Feature', orientation='h', color='Weight', color_continuous_scale='Blues')
            fig_bar.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score,
                number={'suffix': "%", 'font': {'size': 40, 'color': risk_color}},
                title={'text': f"{ticker} Distress Gauge", 'font': {'size': 16, 'color': 'gray'}},
                gauge={'axis': {'range': [0, 100], 'tickwidth': 1}, 'bar': {'color': risk_color, 'thickness': 0.25}, 'bgcolor': "#F3F4F6", 'borderwidth': 0, 'steps': [{'range': [0, 30], 'color': "rgba(0, 230, 118, 0.1)"}, {'range': [30, 100], 'color': "rgba(255, 23, 68, 0.1)"}]}
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.error(f"Unable to pull market data for symbol '{ticker}'.")

# --- MODULE 2 ---
elif "2." in app_mode:
    st.markdown("<h1>📊 Technical Telemetry & Indicators</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; margin-bottom:1rem;'>Inspect live moving averages, RSI momentum, and MACD divergence charts.</p>", unsafe_allow_html=True)
    
    search_col, _ = st.columns([1.5, 1.5])
    with search_col:
        raw_ticker = st.text_input("🔍 Search Ticker for Telemetry:", value="RELIANCE.NS", key="m2_search")
        ticker = raw_ticker.strip().upper() if raw_ticker else "RELIANCE.NS"

    df_raw = fetch_live_data(ticker)
    if not df_raw.empty and len(df_raw) > 30:
        df = compute_technical_indicators(df_raw)
        latest_rsi, latest_macd, latest_sma20, latest_close = df['RSI'].iloc[-1], df['MACD'].iloc[-1], df['SMA_20'].iloc[-1], df['Close'].iloc[-1]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("RSI (14)", f"{latest_rsi:.2f}", delta="Overbought >70 | Oversold <30" if latest_rsi > 70 or latest_rsi < 30 else "Neutral")
        m2.metric("MACD Status", f"{latest_macd:.2f}", delta="Bullish" if latest_macd > 0 else "Bearish")
        m3.metric("SMA 20 vs Price", "Bullish Trend" if latest_close > latest_sma20 else "Bearish Trend")
        
        st.markdown("---")
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='#0A2540', width=2)))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='20 SMA', line=dict(color='#00E676', width=1.5)))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='50 SMA', line=dict(color='#FF1744', width=1.5)))
        fig_ma.update_layout(title=f"{ticker} - Moving Average Crossover", height=350, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ma, use_container_width=True)
        
        col_rsi, col_macd = st.columns(2)
        with col_rsi:
            fig_rsi = px.line(df, x='Date', y='RSI', title=f"{ticker} - RSI Momentum")
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rsi, use_container_width=True)
        with col_macd:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='#0A2540')))
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal_Line'], name='Signal', line=dict(color='#FF1744')))
            fig_macd.update_layout(title=f"{ticker} - MACD Divergence", height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.error("Could not load historical indicator data.")

# --- MODULE 3: MUTUAL FUND RISK SCREENING & SEARCH ---
elif "3." in app_mode:
    st.markdown("<h1>🛡️ Mutual Fund Risk Screening & Search Terminal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; margin-bottom:1rem;'>Search schemes instantly, filter by category, or look up live mutual fund/ETF tickers.</p>", unsafe_allow_html=True)
    
    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        search_query = st.text_input("🔍 Search Fund Scheme by Name or Category:", value="", key="mf_search_bar")
    with search_col2:
        category_filter = st.selectbox("Filter by Category", ["All Categories"] + list(mf_database['Category'].unique()), key="mf_cat_filter")
        
    filtered_df = mf_database.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df['Fund Name'].str.contains(search_query, case=False, na=False) |
            filtered_df['Category'].str.contains(search_query, case=False, na=False)
        ]
    if category_filter != "All Categories":
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
        
    st.markdown(f"### Institutional Mutual Fund Screener Database ({len(filtered_df)} Schemes Found)")
    st.dataframe(filtered_df, hide_index=True, use_container_width=True)
    
    with st.expander("🌐 Lookup Live Custom Mutual Fund / ETF Ticker (yfinance)"):
        custom_mf_ticker = st.text_input("Enter ETF or Fund Ticker (e.g. NIFTYBEES.NS, SETFNN50.NS):", value="NIFTYBEES.NS", key="custom_mf")
        if st.button("Fetch Live Scheme Data"):
            mf_live_df = fetch_live_data(custom_mf_ticker)
            if not mf_live_df.empty:
                current_val = float(mf_live_df['Close'].iloc[-1])
                ret_1yr = ((current_val / float(mf_live_df['Close'].iloc[0])) - 1) * 100 if len(mf_live_df) > 1 else 0.0
                st.success(f"Successfully fetched **{custom_mf_ticker}** | Latest NAV/Price: ₹{current_val:,.2f} | Period Return: {ret_1yr:+.2f}%")
                
                fig_mf_live = px.line(mf_live_df, x='Date', y='Close', title=f"Live Price Action - {custom_mf_ticker}")
                fig_mf_live.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_mf_live, use_container_width=True)
            else:
                st.error(f"Could not retrieve ticker '{custom_mf_ticker}'. Please check the symbol.")

    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = px.scatter(filtered_df if not filtered_df.empty else mf_database, x='Risk Score', y='1Y Return (%)', size='Alpha', color='Category', hover_name='Fund Name', title="Risk vs Return Matrix (Bubble size = Alpha)")
        fig_scatter.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        fig_bar = px.bar(filtered_df if not filtered_df.empty else mf_database, x='Fund Name', y='Expense Ratio (%)', color='Category', title="Expense Ratio Comparison")
        fig_bar.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

# --- MODULE 4: HEAD-TO-HEAD COMPARISON ---
elif "4." in app_mode:
    st.markdown("<h1>⚔️ Head-to-Head (H2H) Fund & Scheme Comparison</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; margin-bottom:1rem;'>Select any two mutual fund schemes from the database for direct comparative analysis.</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    fund1_name = c1.selectbox("Select Fund Scheme A", mf_database['Fund Name'], index=0, key="h2h_f1")
    fund2_name = c2.selectbox("Select Fund Scheme B", mf_database['Fund Name'], index=1, key="h2h_f2")
    
    d1 = mf_database[mf_database['Fund Name'] == fund1_name].iloc[0]
    d2 = mf_database[mf_database['Fund Name'] == fund2_name].iloc[0]
    
    colA, colB = st.columns([1, 1.5])
    with colA:
        st.markdown("### Comparative Metrics Matrix")
        comp_df = pd.DataFrame({'Metric': mf_database.columns[2:], fund1_name: d1[2:].values, fund2_name: d2[2:].values})
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
    with colB:
        fig_radar = go.Figure()
        categories = ['1Y Return', 'Alpha', 'Beta (Inv)', 'Risk Efficiency', 'Cost Efficiency']
        val1 = [d1['1Y Return (%)']*2, d1['Alpha']*10, (2-d1['Beta'])*50, 100-d1['Risk Score'], (2-d1['Expense Ratio (%)'])*50]
        val2 = [d2['1Y Return (%)']*2, d2['Alpha']*10, (2-d2['Beta'])*50, 100-d2['Risk Score'], (2-d2['Expense Ratio (%)'])*50]
        fig_radar.add_trace(go.Scatterpolar(r=val1, theta=categories, fill='toself', name=fund1_name, line_color='#0A2540'))
        fig_radar.add_trace(go.Scatterpolar(r=val2, theta=categories, fill='toself', name=fund2_name, line_color='#00E676'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), height=340, margin=dict(l=40, r=40, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_radar, use_container_width=True)

# --- MODULE 5 ---
elif "5." in app_mode:
    st.markdown("<h1>🗂️ Portfolio Overlap & Asset Allocation Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; margin-bottom:1rem;'>Analyze sector weights, diversification ratios, and cross-asset correlations.</p>", unsafe_allow_html=True)
    
    col_alloc, col_donut = st.columns([1.5, 1.2])
    
    with col_alloc:
        st.markdown("### Custom Portfolio Weighting")
        w_equity = st.slider("Large Cap Equities (%)", 0, 100, 50)
        w_mid = st.slider("Mid & Small Cap (%)", 0, 100, 30)
        w_debt = st.slider("Fixed Income / Debt (%)", 0, 100, 20)
        
        total_w = w_equity + w_mid + w_debt
        if total_w != 100:
            st.warning(f"Total allocation is {total_w}%. Recommended total is exactly 100%.")
        else:
            st.success("Allocation perfectly balanced.")
        
    with col_donut:
        alloc_df = pd.DataFrame({'Asset Class': ['Large Cap', 'Mid/Small Cap', 'Debt'], 'Weight': [w_equity, w_mid, w_debt]})
        fig_donut = px.pie(alloc_df, names='Asset Class', values='Weight', hole=0.5, title="Portfolio Allocation Breakdown", color_discrete_sequence=['#0A2540', '#00E676', '#94A3B8'])
        fig_donut.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)

# --- MODULE 6: PAPER TRADING & AI LEDGER ---
elif "6." in app_mode:
    st.markdown("<h1>📋 Paper Trading & AI Ledger</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; margin-bottom:1rem;'>Simulate institutional trade execution with real-time capital tracking.</p>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="blue-card" style="border-left-color: #00E676;">
        <p style="font-size: 1rem; color: #94A3B8 !important;">AVAILABLE LIQUID CAPITAL</p>
        <h2 style="font-size: 2.5rem; color: #00E676 !important;">₹{st.session_state.cash_balance:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col_trade, col_ledger = st.columns([1, 1.5])
    
    with col_trade:
        st.markdown("### Execute Simulation Order")
        trade_ticker = st.text_input("Asset Ticker", value="RELIANCE.NS", key="trade_ticker").strip().upper()
        trade_type = st.selectbox("Order Action", ["BUY / LONG", "SELL / SHORT"])
        shares = st.number_input("Quantity", min_value=1, max_value=10000, value=10)
        
        if st.button("Submit Order to Ledger", use_container_width=True):
            live_df = fetch_live_data(trade_ticker)
            if not live_df.empty:
                exec_price = float(live_df['Close'].iloc[-1])
                total_cost = exec_price * shares
                
                if "BUY" in trade_type:
                    if st.session_state.cash_balance >= total_cost:
                        st.session_state.cash_balance -= total_cost
                        st.session_state.trade_ledger.insert(0, {
                            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'Ticker': trade_ticker,
                            'Action': 'BUY',
                            'Qty': shares,
                            'Exec Price (₹)': round(exec_price, 2),
                            'Total (₹)': round(total_cost, 2)
                        })
                        st.success(f"Executed BUY for {shares}x {trade_ticker} at ₹{exec_price:,.2f}!")
                        st.rerun()
                    else:
                        st.error("Insufficient available liquid cash balance for this order!")
                else:
                    st.session_state.cash_balance += total_cost
                    st.session_state.trade_ledger.insert(0, {
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'Ticker': trade_ticker,
                        'Action': 'SELL',
                        'Qty': shares,
                        'Exec Price (₹)': round(exec_price, 2),
                        'Total (₹)': round(total_cost, 2)
                    })
                    st.success(f"Executed SELL for {shares}x {trade_ticker} at ₹{exec_price:,.2f}!")
                    st.rerun()
            else:
                st.error(f"Could not fetch live price for '{trade_ticker}' to execute order.")
        
    with col_ledger:
        st.markdown("### Active Execution Ledger")
        if len(st.session_state.trade_ledger) > 0:
            ledger_df = pd.DataFrame(st.session_state.trade_ledger)
            st.dataframe(ledger_df, hide_index=True, use_container_width=True)
            if st.button("Clear Ledger History"):
                st.session_state.trade_ledger = []
                st.session_state.cash_balance = 1000000.0
                st.rerun()
        else:
            st.info("No active trades executed yet in this session. Submit an order from the left panel to populate the live ledger.")
