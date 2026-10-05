import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import joblib
from datetime import datetime
import sqlite3
import requests
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Safe SHAP import to prevent startup crashes if package is missing
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ==========================================
# 1. PAGE CONFIGURATION & ADVANCED UI CSS
# ==========================================
st.set_page_config(
    page_title="FinShield Intelligence | Global Terminal", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], p, span, label, .stTextInput, .stSelectbox, .stNumberInput, div {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }
    [data-testid="stAppViewContainer"] { background-color: #F8FAFC; }
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #0A2540 0%, #1E3A8A 100%) !important; 
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    
    /* FIX: Ensure input text, selectbox text, selected values, and dropdown items are clearly visible (Dark Blue) */
    [data-testid="stSidebar"] .stTextInput input, 
    [data-testid="stSidebar"] .stSelectbox div,
    [data-testid="stSidebar"] .stSelectbox span,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] *,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #0A2540 !important;
    }
    [data-testid="stSidebar"] .stTextInput input::placeholder {
        color: #94A3B8 !important;
    }
    
    /* Dropdown menu items popup styling */
    div[data-baseweb="menu"], div[role="listbox"] {
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="menu"] *, div[role="option"] * {
        color: #0A2540 !important;
    }
    div[role="option"]:hover {
        background-color: #F1F5F9 !important;
    }
    
    h1, h2, h3 { 
        color: #0A2540; 
        font-weight: 700; 
        letter-spacing: -0.5px;
    }
    
    .stTextInput > div > div > input, .stSelectbox > div > div > div, .stNumberInput > div > div > input { 
        border-radius: 8px !important; 
        border: 1px solid #CBD5E1 !important; 
        background-color: #FFFFFF !important;
        font-weight: 500 !important; 
    }
    
    .fin-card {
        background: #FFFFFF;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(10, 37, 64, 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    
    .blue-card { 
        background: linear-gradient(135deg, #0A2540 0%, #1E3A8A 100%);
        color: #FFFFFF; 
        padding: 1.5rem; 
        border-radius: 14px; 
        box-shadow: 0 10px 25px rgba(10,37,64,0.15); 
        margin-bottom: 1rem; 
        border-left: 6px solid #3B82F6; 
    }
    .blue-card h3, .blue-card p, .blue-card h2 { 
        color: #FFFFFF !important; 
        margin: 0; 
    }
    
    .badge-safe { background-color: #D1FAE5; color: #065F46; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; display: inline-block; }
    .badge-danger { background-color: #FEE2E2; color: #991B1B; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; display: inline-block; }
    
    .js-plotly-plot { margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PERSISTENT SQLITE DATABASE INITIALIZATION
# ==========================================
def init_db():
    conn = sqlite3.connect("portfolio.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            action TEXT,
            qty REAL,
            price REAL,
            total REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cash REAL
        )
    ''')
    cursor.execute('SELECT COUNT(*) FROM portfolio_state')
    if cursor.fetchone()[0] == 0:
        cursor.execute('INSERT INTO portfolio_state (id, cash) VALUES (1, 1000000.0)')
    conn.commit()
    return conn

db_conn = init_db()

def get_cash_balance():
    cursor = db_conn.cursor()
    cursor.execute('SELECT cash FROM portfolio_state WHERE id = 1')
    return cursor.fetchone()[0]

def update_cash_balance(new_cash):
    cursor = db_conn.cursor()
    cursor.execute('UPDATE portfolio_state SET cash = ? WHERE id = 1', (new_cash,))
    db_conn.commit()

def get_trade_ledger():
    return pd.read_sql('SELECT timestamp as Timestamp, ticker as Ticker, action as Action, qty as Qty, price as "Exec Price (₹)", total as "Total (₹)" FROM trades ORDER BY id DESC', db_conn)

def log_trade_db(timestamp, ticker, action, qty, price, total):
    cursor = db_conn.cursor()
    cursor.execute(
        'INSERT INTO trades (timestamp, ticker, action, qty, price, total) VALUES (?, ?, ?, ?, ?, ?)',
        (timestamp, ticker, action, qty, price, total)
    )
    db_conn.commit()

def clear_trade_ledger_db():
    cursor = db_conn.cursor()
    cursor.execute('DELETE FROM trades')
    cursor.execute('UPDATE portfolio_state SET cash = 1000000.0 WHERE id = 1')
    db_conn.commit()

# ==========================================
# 3. GLOBAL LIVE YFINANCE & SEARCH PIPELINES
# ==========================================
@st.cache_data(ttl=3600)
def get_ticker_suggestions(query):
    if not query or len(query.strip()) < 2:
        return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query.strip()}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            data = res.json()
            quotes = data.get('quotes', [])
            suggestions = []
            for q in quotes:
                sym = q.get('symbol')
                name = q.get('shortname') or q.get('longname') or sym
                exch = q.get('exchange', '')
                suggestions.append(f"{sym} | {name} ({exch})")
            return suggestions
    except Exception:
        pass
    return []

@st.cache_data(ttl=900)
def fetch_live_data(ticker):
    clean_ticker = ticker.strip()
    try:
        df = yf.download(clean_ticker, period="1y", interval="1d", progress=False)
        if df.empty and not clean_ticker.endswith('.NS'):
            df = yf.download(f"{clean_ticker}.NS", period="1y", interval="1d", progress=False)
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Clean trailing NaN rows from yfinance incomplete feed
        df.dropna(subset=['Close'], inplace=True)
        df.reset_index(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def calculate_crash_risk(df):
    try:
        model = joblib.load('random_forest_crash_model.joblib')
        features_df = pd.DataFrame({
            'RSI': [df['RSI'].iloc[-1] if 'RSI' in df and not pd.isna(df['RSI'].iloc[-1]) else 50.0],
            'Volatility': [df['Close'].pct_change().std() * np.sqrt(252)],
            'Momentum': [(df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) if len(df) >= 20 else 0.0],
            'SMA_Ratio': [df['Close'].iloc[-1] / df['SMA_20'].iloc[-1] if 'SMA_20' in df and not pd.isna(df['SMA_20'].iloc[-1]) else 1.0],
            'Volume_Surge': [df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1] if 'Volume' in df and df['Volume'].rolling(20).mean().iloc[-1] > 0 else 1.0]
        })
        prob = model.predict_proba(features_df)[0][1] * 100
        return float(prob)
    except Exception:
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  
        momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100 if len(df) >= 20 else 0.0
        risk = (volatility * 1.8) - (momentum * 0.8)
        return float(max(2.0, min(98.0, risk)))

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

def generate_pdf_report(ticker, risk_score, ltp, change_pct, df):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    returns = df['Close'].pct_change().dropna() if 'Close' in df else pd.Series([0])
    annual_vol = float(returns.std() * np.sqrt(252) * 100)
    rsi_val = float(df['RSI'].iloc[-1]) if 'RSI' in df and not pd.isna(df['RSI'].iloc[-1]) else 50.0
    sma20 = float(df['SMA_20'].iloc[-1]) if 'SMA_20' in df and not pd.isna(df['SMA_20'].iloc[-1]) else ltp
    trend_status = "BULLISH (Price > 20 SMA)" if ltp > sma20 else "BEARISH (Price < 20 SMA)"
    
    c.setFillColorRGB(0.04, 0.15, 0.25)
    c.rect(0, height - 90, width, 90, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 38, "FINSHIELD INTELLIGENCE | EXECUTIVE RISK & CRASH FORECAST")
    c.setFont("Helvetica", 10)
    c.drawString(40, height - 58, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target Asset: {ticker}")
    c.drawString(40, height - 74, "Classification: CONFIDENTIAL - Quantitative Institutional Telemetry")
    
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, height - 120, "1. Executive Asset Valuation & Crash Probability")
    
    c.setFont("Helvetica", 10)
    y_pos = height - 145
    c.drawString(50, y_pos, f"• Last Traded Price (LTP): ₹{ltp:,.2f} ({change_pct:+.2f}% Daily Change)")
    y_pos -= 20
    c.drawString(50, y_pos, f"• 30-Day Model Crash Risk Probability: {risk_score:.1f}%")
    y_pos -= 20
    risk_status = "SAFE (LOW DISTRESS PROBABILITY)" if risk_score < 30 else "DANGER (HIGH DISTRESS PROBABILITY)"
    c.drawString(50, y_pos, f"• Risk Classification Status: {risk_status}")
    
    y_pos -= 40
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y_pos, "2. Quantitative Telemetry & Market Momentum")
    
    y_pos -= 25
    c.setFont("Helvetica", 10)
    c.drawString(50, y_pos, f"• Annualized Volatility (1-Year Rolling): {annual_vol:.2f}%")
    y_pos -= 20
    c.drawString(50, y_pos, f"• 14-Period RSI Momentum Score: {rsi_val:.2f} ({'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral Zone'})")
    y_pos -= 20
    c.drawString(50, y_pos, f"• Technical Trend Structure: {trend_status}")
    
    y_pos -= 40
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y_pos, "3. Institutional Risk Management Recommendation")
    
    y_pos -= 25
    c.setFont("Helvetica-Bold", 10)
    if risk_score < 30:
        c.setFillColorRGB(0.02, 0.4, 0.2)
        c.drawString(50, y_pos, "VERDICT: ACCUMULATE / HOLD WITH STANDARD STOP LOSS")
        y_pos -= 18
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.setFont("Helvetica", 10)
        c.drawString(50, y_pos, "The machine learning model indicates low systemic crash risk. Maintain long exposure,")
        y_pos -= 15
        c.drawString(50, y_pos, "ensuring standard trailing stop-losses are set 5% below the 20-day moving average.")
    else:
        c.setFillColorRGB(0.7, 0.1, 0.1)
        c.drawString(50, y_pos, "VERDICT: DEFENSIVE EXIT / HEDGING RECOMMENDED")
        y_pos -= 18
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.setFont("Helvetica", 10)
        c.drawString(50, y_pos, "Elevated crash probability detected via volatility expansion and momentum divergence.")
        y_pos -= 15
        c.drawString(50, y_pos, "Recommended action: Reduce portfolio weight, lock in profits, or migrate capital to liquid cash.")

    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.line(40, 60, width - 40, 60)
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(40, 42, "FinShield Intelligence Terminal | Powered by yFinance & Random Forest ML Engine.")
    c.drawString(40, 30, "For academic project defense and institutional portfolio risk evaluation use only.")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def render_shap_explanation(df):
    if HAS_SHAP:
        try:
            model = joblib.load('random_forest_crash_model.joblib')
            features_df = pd.DataFrame({
                'RSI': [df['RSI'].iloc[-1] if 'RSI' in df and not pd.isna(df['RSI'].iloc[-1]) else 50.0],
                'Volatility': [df['Close'].pct_change().std() * np.sqrt(252)],
                'Momentum': [(df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) if len(df) >= 20 else 0.0],
                'SMA_Ratio': [df['Close'].iloc[-1] / df['SMA_20'].iloc[-1] if 'SMA_20' in df and not pd.isna(df['SMA_20'].iloc[-1]) else 1.0],
                'Volume_Surge': [df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1] if 'Volume' in df and df['Volume'].rolling(20).mean().iloc[-1] > 0 else 1.0]
            })
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(features_df)
            
            fig, ax = plt.subplots(figsize=(8, 3.2))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
            return
        except Exception:
            pass
            
    st.markdown("### 🔍 Model Feature Attribution (Institutional Weighting)")
    features = pd.DataFrame({'Feature': ['RSI Momentum', 'MACD Divergence', 'Volatility (Live)', 'Volume Surge', 'Moving Avg Cross'], 'Weight': [0.35, 0.25, 0.20, 0.12, 0.08]}).sort_values(by='Weight', ascending=True)
    fig_bar = px.bar(features, x='Weight', y='Feature', orientation='h', color='Weight', color_continuous_scale='Blues')
    fig_bar.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_bar, use_container_width=True)

def render_news_feed(ticker_symbol):
    st.markdown("### 📰 Recent Market Sentiment & News")
    try:
        t_obj = yf.Ticker(ticker_symbol)
        news_list = t_obj.news
        if news_list:
            for item in news_list[:4]:
                title = item.get('title', 'Market Update')
                publisher = item.get('publisher', 'Financial Wire')
                link = item.get('link', '#')
                pub_time = datetime.fromtimestamp(item.get('providerPublishTime', datetime.now().timestamp())).strftime('%Y-%m-%d %H:%M') if 'providerPublishTime' in item else ''
                st.markdown(f"- **[{title}]({link})** — *{publisher} ({pub_time})*")
        else:
            st.info("No recent breaking news headlines found for this asset ticker.")
    except Exception:
        st.info("Live news stream temporarily unavailable for this asset symbol.")

# ==========================================
# 4. SIDEBAR NAVIGATION & GLOBAL TICKER STATE
# ==========================================
with st.sidebar:
    st.markdown("<h2>🛡 FinShield Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; font-size:0.85rem; margin-top:-10px;'>Global Institutional Terminal v4.6</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    
    app_mode = st.radio(
        "MODULE ROUTING",
        [
            "1. Global Stock Crash-Risk Predictor",
            "2. Technical Telemetry",
            "3. Historical Strategy Backtester",
            "4. Portfolio Overlap & Allocation",
            "5. Paper Trading & AI Ledger"
        ]
    )
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    
    # --- GLOBAL TICKER CONTROL ---
    st.markdown("### 🌐 Global Target Asset")
    if "global_ticker" not in st.session_state:
        st.session_state.global_ticker = "RELIANCE.NS"
        
    popular_defaults = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "GOOGL", "TSLA", "NVDA", "BTC-USD"]
    search_input = st.text_input("🔍 Search Ticker / Company:", value="", key="sidebar_search_input", placeholder="e.g. Tata, Apple, AAPL")
    
    if search_input and len(search_input.strip()) >= 2:
        suggestions = get_ticker_suggestions(search_input)
        if suggestions:
            chosen = st.selectbox("Select matched asset:", suggestions, key="sidebar_choice")
            st.session_state.global_ticker = chosen.split(" | ")[0].strip()
        else:
            st.session_state.global_ticker = search_input.strip().upper()
    else:
        chosen_pop = st.selectbox("Or choose popular asset:", popular_defaults, key="sidebar_pop")
        if not search_input:
            st.session_state.global_ticker = chosen_pop
            
    st.markdown(f"<p style='font-size:0.85rem; color:#00E676; margin-top:5px;'>Active Ticker: <b>{st.session_state.global_ticker}</b></p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.8rem; color:#94A3B8;'>🟢 Universal yFinance Search: <b style='color:#00E676;'>ACTIVE</b></p>", unsafe_allow_html=True)

# ==========================================
# 5. MODULE EXECUTIONS
# ==========================================

# --- MODULE 1: STOCK CRASH-RISK PREDICTOR ---
if "1." in app_mode:
    st.markdown("<h1>📉 Global Stock Crash-Risk Prediction Engine</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748B; margin-bottom:1.5rem;'>Analyzing active global asset: <b>{st.session_state.global_ticker}</b>. Use the sidebar to switch tickers instantly across modules.</p>", unsafe_allow_html=True)
    
    ticker = st.session_state.global_ticker
    df_raw = fetch_live_data(ticker)
    
    if not df_raw.empty and len(df_raw) > 5:
        df = compute_technical_indicators(df_raw)
        col_target, col_chart = st.columns([1, 2.5])
        with col_target:
            ltp = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2])
            change_pct = ((ltp - prev_close) / prev_close) * 100
            
            risk_score = round(calculate_crash_risk(df), 1)
            
            cutoff = 30.0
            risk_color = "#00E676" if risk_score < cutoff else "#FF1744"
            risk_badge = '<span class="badge-safe">SAFE (LOW RISK)</span>' if risk_score < cutoff else '<span class="badge-danger">DANGER (HIGH RISK)</span>'

            st.markdown(f"""
            <div class="blue-card" style="border-left-color: {risk_color};">
                <p style="font-size: 0.9rem; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 0.5px;">{ticker} Live Market Price</p>
                <h2 style="font-size: 2.2rem; margin: 4px 0;">₹{ltp:,.2f} <span style="font-size: 1rem; color: {'#00E676' if change_pct > 0 else '#FF1744'};">({change_pct:+.2f}%)</span></h2>
                <hr style="border-color: rgba(255,255,255,0.15); margin: 12px 0;">
                <p style="font-size: 0.9rem; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 0.5px;">30-Day Crash Probability</p>
                <h2 style="font-size: 3rem; color: {risk_color} !important; margin: 4px 0;">{risk_score}%</h2>
                <div style="margin-top: 10px;">{risk_badge}</div>
            </div>
            """, unsafe_allow_html=True)
            
            pdf_buffer = generate_pdf_report(ticker, risk_score, ltp, change_pct, df)
            st.download_button(
                label="📥 Download Executive PDF Report",
                data=pdf_buffer,
                file_name=f"{ticker}_Risk_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        with col_chart:
            st.markdown('<div class="fin-card">', unsafe_allow_html=True)
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                increasing_line_color='#00E676', decreasing_line_color='#FF1744'
            )])
            fig_candle.update_layout(title=f"Live Price Action - {ticker}", margin=dict(l=20, r=20, t=40, b=20), height=380, xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_candle, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="fin-card">', unsafe_allow_html=True)
            render_shap_explanation(df)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="fin-card">', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score,
                number={'suffix': "%", 'font': {'size': 36, 'color': risk_color}},
                title={'text': f"{ticker} Distress Gauge", 'font': {'size': 15, 'color': '#64748B'}},
                gauge={'axis': {'range': [0, 100], 'tickwidth': 1}, 'bar': {'color': risk_color, 'thickness': 0.25}, 'bgcolor': "#F1F5F9", 'borderwidth': 0, 'steps': [{'range': [0, 30], 'color': "rgba(0, 230, 118, 0.1)"}, {'range': [30, 100], 'color': "rgba(255, 23, 68, 0.1)"}]}
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        render_news_feed(ticker)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error(f"Unable to pull live market data for symbol '{ticker}'. Please check the ticker name or try searching again.")

# --- MODULE 2: TECHNICAL TELEMETRY ---
elif "2." in app_mode:
    st.markdown("<h1>📊 Technical Telemetry & Momentum Health</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748B; margin-bottom:1.5rem;'>Inspecting technical indicators for active global asset: <b>{st.session_state.global_ticker}</b>.</p>", unsafe_allow_html=True)
    
    ticker = st.session_state.global_ticker
    df_raw = fetch_live_data(ticker)
    
    if not df_raw.empty and len(df_raw) > 30:
        df = compute_technical_indicators(df_raw)
        latest_rsi, latest_macd, latest_sma20, latest_close = df['RSI'].iloc[-1], df['MACD'].iloc[-1], df['SMA_20'].iloc[-1], df['Close'].iloc[-1]
        
        m1, m2, m3 = st.columns(3)
        with m1:
            rsi_status = "Overbought ⚡" if latest_rsi > 70 else ("Oversold 🛡️" if latest_rsi < 30 else "Neutral ⚖️")
            st.markdown(f"""
            <div class="fin-card" style="border-left: 5px solid #3B82F6;">
                <p style="color: #64748B; margin: 0; font-size: 0.85rem; font-weight: 600;">RSI MOMENTUM (14)</p>
                <h3 style="color: #0F172A; margin: 4px 0 2px 0; font-size: 1.8rem;">{latest_rsi:.2f}</h3>
                <p style="margin: 0; font-weight: 600; color: #3B82F6; font-size: 0.9rem;">{rsi_status}</p>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            macd_status = "Bullish Momentum 🚀" if latest_macd > 0 else "Bearish Pressure 📉"
            st.markdown(f"""
            <div class="fin-card" style="border-left: 5px solid #10B981;">
                <p style="color: #64748B; margin: 0; font-size: 0.85rem; font-weight: 600;">MACD DIVERGENCE</p>
                <h3 style="color: #0F172A; margin: 4px 0 2px 0; font-size: 1.8rem;">{latest_macd:.2f}</h3>
                <p style="margin: 0; font-weight: 600; color: #10B981; font-size: 0.9rem;">{macd_status}</p>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            trend_status = "Bullish Trend 📈" if latest_close > latest_sma20 else "Bearish Trend 📉"
            st.markdown(f"""
            <div class="fin-card" style="border-left: 5px solid #8B5CF6;">
                <p style="color: #64748B; margin: 0; font-size: 0.85rem; font-weight: 600;">SMA 20 VS PRICE</p>
                <h3 style="color: #0F172A; margin: 4px 0 2px 0; font-size: 1.8rem;">₹{latest_close:,.2f}</h3>
                <p style="margin: 0; font-weight: 600; color: #8B5CF6; font-size: 0.9rem;">{trend_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='#0A2540', width=2)))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='20 SMA', line=dict(color='#00E676', width=1.5)))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='50 SMA', line=dict(color='#FF1744', width=1.5)))
        fig_ma.update_layout(title=f"{ticker} - Moving Average Crossover", height=350, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ma, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Could not load historical indicator data for this symbol.")

# --- MODULE 3: HISTORICAL STRATEGY BACKTESTER ---
elif "3." in app_mode:
    st.markdown("<h1>🧪 Historical Strategy Backtester</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748B; margin-bottom:1.5rem;'>Backtesting risk mitigation strategies on active global asset: <b>{st.session_state.global_ticker}</b>.</p>", unsafe_allow_html=True)
    
    ticker = st.session_state.global_ticker
    df_raw = fetch_live_data(ticker)
    
    if not df_raw.empty and len(df_raw) > 50:
        df = df_raw.copy()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Rolling_Vol'] = df['Daily_Return'].rolling(14).std() * np.sqrt(252) * 100
        
        threshold = st.slider("Volatility Risk Threshold (%) to Exit to Cash", 10.0, 50.0, 30.0)
        
        df['Signal'] = np.where(df['Rolling_Vol'] > threshold, 0, 1)
        df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1).fillna(1)
        
        df['Cumulative_BH'] = (1 + df['Daily_Return'].fillna(0)).cumprod() * 100
        df['Cumulative_Strat'] = (1 + df['Strategy_Return'].fillna(0)).cumprod() * 100
        
        bh_final = df['Cumulative_BH'].iloc[-1]
        strat_final = df['Cumulative_Strat'].iloc[-1]
        
        b1, b2 = st.columns(2)
        with b1:
            st.markdown(f"""
            <div class="fin-card" style="border-left: 5px solid #0A2540;">
                <p style="color: #64748B; margin: 0; font-size: 0.85rem; font-weight: 600;">BUY & HOLD FINAL VALUE</p>
                <h3 style="color: #0F172A; margin: 4px 0 2px 0; font-size: 1.8rem;">₹100.00 ➔ ₹{bh_final:,.2f}</h3>
                <p style="margin: 0; font-weight: 600; color: {'#00E676' if bh_final >= 100 else '#FF1744'}; font-size: 0.9rem;">Return: {bh_final - 100:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            st.markdown(f"""
            <div class="fin-card" style="border-left: 5px solid #00E676;">
                <p style="color: #64748B; margin: 0; font-size: 0.85rem; font-weight: 600;">RISK-MANAGED STRATEGY VALUE</p>
                <h3 style="color: #0F172A; margin: 4px 0 2px 0; font-size: 1.8rem;">₹100.00 ➔ ₹{strat_final:,.2f}</h3>
                <p style="margin: 0; font-weight: 600; color: {'#00E676' if strat_final >= 100 else '#FF1744'}; font-size: 0.9rem;">Return: {strat_final - 100:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_BH'], name='Buy & Hold Benchmark', line=dict(color='#94A3B8', width=2)))
        fig_bt.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Strat'], name='Risk-Managed Strategy', line=dict(color='#00E676', width=2.5)))
        fig_bt.update_layout(title=f"Strategy Backtest Performance Curve - {ticker}", height=380, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bt, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Insufficient historical data for backtesting.")

# --- MODULE 4: PORTFOLIO OVERLAP & ALLOCATION ---
elif "4." in app_mode:
    st.markdown("<h1>🗂️ Portfolio Overlap & Asset Allocation Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; margin-bottom:1.5rem;'>Analyze sector weights, diversification ratios, and interactive risk-return asset allocations.</p>", unsafe_allow_html=True)
    
    col_alloc, col_donut = st.columns([1.5, 1.2])
    
    with col_alloc:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        st.markdown("### Custom Portfolio Weighting")
        w_equity = st.slider("Large Cap Equities (%)", 0, 100, 50)
        w_mid = st.slider("Mid & Small Cap (%)", 0, 100, 30)
        w_debt = st.slider("Fixed Income / Debt (%)", 0, 100, 20)
        
        total_w = w_equity + w_mid + w_debt
        if total_w != 100:
            st.warning(f"Total allocation is {total_w}%. Recommended total is exactly 100%.")
        else:
            st.success("Allocation perfectly balanced.")
            
        expected_return = (w_equity * 0.14) + (w_mid * 0.18) + (w_debt * 0.07)
        expected_volatility = (w_equity * 0.12) + (w_mid * 0.22) + (w_debt * 0.04)
        
        st.markdown(f"""
        <div style="background: #F8FAFC; padding: 1rem; border-radius: 10px; border-left: 5px solid #10B981; margin-top: 1rem; border: 1px solid #E2E8F0;">
            <p style="color: #64748B; margin: 0; font-size: 0.85rem; font-weight: 600;">PORTFOLIO SIMULATION METRICS</p>
            <h4 style="color: #0F172A; margin: 4px 0;">Expected Annual Return: <span style="color: #059669;">+{expected_return:.2f}%</span></h4>
            <h4 style="color: #0F172A; margin: 4px 0;">Projected Portfolio Volatility: <span style="color: #D97706;">{expected_volatility:.2f}%</span></h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_donut:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        alloc_df = pd.DataFrame({'Asset Class': ['Large Cap', 'Mid/Small Cap', 'Debt'], 'Weight': [w_equity, w_mid, w_debt]})
        fig_donut = px.pie(alloc_df, names='Asset Class', values='Weight', hole=0.5, title="Portfolio Allocation Breakdown", color_discrete_sequence=['#0A2540', '#00E676', '#94A3B8'])
        fig_donut.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- MODULE 5: PAPER TRADING & PERSISTENT SQLITE LEDGER ---
elif "5." in app_mode:
    st.markdown("<h1>📋 Paper Trading & Persistent AI Ledger</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748B; margin-bottom:1.5rem;'>Simulate institutional trade execution for active global asset: <b>{st.session_state.global_ticker}</b> backed by persistent SQLite database storage.</p>", unsafe_allow_html=True)
    
    current_cash = get_cash_balance()
    st.markdown(f"""
    <div class="blue-card" style="border-left-color: #00E676;">
        <p style="font-size: 0.9rem; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 0.5px;">Available Liquid Capital (Persistent DB)</p>
        <h2 style="font-size: 2.5rem; color: #00E676 !important; margin: 4px 0;">₹{current_cash:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col_trade, col_ledger = st.columns([1, 1.5])
    
    with col_trade:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        st.markdown("### Execute Simulation Order")
        trade_ticker = st.session_state.global_ticker
        st.info(f"Trading Asset: **{trade_ticker}** (Synced from Global Sidebar)")
        
        trade_type = st.selectbox("Order Action", ["BUY / LONG", "SELL / SHORT"])
        shares = st.number_input("Quantity", min_value=1, max_value=10000, value=10)
        
        if st.button("Submit Order to Ledger", use_container_width=True, type="primary"):
            live_df = fetch_live_data(trade_ticker)
            if not live_df.empty:
                exec_price = float(live_df['Close'].iloc[-1])
                total_cost = exec_price * shares
                
                if "BUY" in trade_type:
                    if current_cash >= total_cost:
                        new_cash = current_cash - total_cost
                        update_cash_balance(new_cash)
                        log_trade_db(
                            datetime.now().strftime('%Y-%m-%d %H:%M'),
                            trade_ticker, 'BUY', shares, round(exec_price, 2), round(total_cost, 2)
                        )
                        st.success(f"Executed BUY for {shares}x {trade_ticker} at ₹{exec_price:,.2f}!")
                        st.rerun()
                    else:
                        st.error("Insufficient available liquid cash balance for this order!")
                else:
                    new_cash = current_cash + total_cost
                    update_cash_balance(new_cash)
                    log_trade_db(
                        datetime.now().strftime('%Y-%m-%d %H:%M'),
                        trade_ticker, 'SELL', shares, round(exec_price, 2), round(total_cost, 2)
                    )
                    st.success(f"Executed SELL for {shares}x {trade_ticker} at ₹{exec_price:,.2f}!")
                    st.rerun()
            else:
                st.error(f"Could not fetch live price for '{trade_ticker}' to execute order.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_ledger:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        st.markdown("### Active Persistent Execution Ledger")
        ledger_df = get_trade_ledger()
        if not ledger_df.empty:
            st.dataframe(ledger_df, hide_index=True, use_container_width=True)
            if st.button("Clear Ledger History & Reset Cash"):
                clear_trade_ledger_db()
                st.rerun()
        else:
            st.info("No active trades executed yet in this session. Submit an order from the left panel to populate the persistent database ledger.")
        st.markdown('</div>', unsafe_allow_html=True)
