import sys, os, pickle, math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# --- sklearn pickle compatibility ---
try:
    import sklearn._loss
    sys.modules["_loss"] = sklearn._loss
    if hasattr(sklearn._loss, "_loss"):
        sys.modules["_loss._loss"] = sklearn._loss._loss
except Exception:
    pass
try:
    import sklearn.ensemble._gb_losses as _gb_losses
    sys.modules["_gb_losses"] = _gb_losses
except Exception:
    pass

try:
    from mftool import Mftool
except Exception:
    Mftool = None

st.set_page_config(
    page_title="FinShield | Stock Market & Mutual Fund Intelligence",
    page_icon="🛡️", layout="wide", initial_sidebar_state="expanded"
)

# ============================ STYLE ============================
st.markdown("""
<style>
.stApp{background:#f5f7fb}
.block-container{max-width:1500px;padding-top:1rem}
.hero{padding:1.5rem;border:1px solid #e2e8f0;border-radius:22px;
background:linear-gradient(135deg,#fff,#eef4ff);box-shadow:0 10px 30px #0f172a0c;margin-bottom:1rem}
.hero h1{margin:0;font-size:2.25rem;letter-spacing:-.04em}
.hero p{color:#64748b;margin:.35rem 0 0}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:1rem;
box-shadow:0 5px 18px #0f172a08}
.note{color:#64748b;font-size:.78rem}
.insight{border-left:4px solid #2563eb;background:#eff6ff;padding:.8rem 1rem;border-radius:10px}
.warn{border-left:4px solid #f59e0b;background:#fffbeb;padding:.8rem 1rem;border-radius:10px}
.good{border-left:4px solid #22c55e;background:#f0fdf4;padding:.8rem 1rem;border-radius:10px}
.bad{border-left:4px solid #ef4444;background:#fef2f2;padding:.8rem 1rem;border-radius:10px}
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "NAV","Daily_Return_Pct","Annualized_Return_1Y","Volatility_30D",
    "Annualized_Volatility_Cleaned","Sharpe_Ratio_Cleaned",
    "Sortino_Ratio_Cleaned","Max_Drawdown_1Y_Pct","Composite_Score",
    "Lag_1D_Return","Lag_5D_Return"
]

ACTIVE = {
"HDFC Top 100 Fund - Direct Growth":"119018",
"HDFC Mid-Cap Opportunities Fund - Direct Growth":"118989",
"ICICI Prudential Bluechip Fund - Direct Growth":"120586",
"SBI Bluechip Fund - Direct Growth":"119598",
"Nippon India Small Cap Fund - Direct Growth":"118778",
"Parag Parikh Flexi Cap Fund - Direct Growth":"122639",
"Axis Large Cap Fund - Direct Growth":"120465",
"Mirae Asset Large Cap Fund - Direct Growth":"118834",
"Kotak Emerging Equity Fund - Direct Growth":"120152",
"Quant Active Fund - Direct Growth":"120828"
}

# Retained from original app; replace with verified latest monthly holdings in production.
HOLDINGS = {
"HDFC Top 100 Fund - Direct Growth":{"HDFC Bank":9.8,"Reliance Industries":8.2,"ICICI Bank":7.4,"Infosys":6.1,"Larsen & Toubro":4.5,"ITC Ltd":3.8,"TCS":3.5,"Bharti Airtel":3.2,"Axis Bank":2.9,"State Bank of India":2.6},
"ICICI Prudential Bluechip Fund - Direct Growth":{"ICICI Bank":8.9,"Reliance Industries":7.5,"HDFC Bank":7.1,"Larsen & Toubro":5.2,"Infosys":4.8,"Bharti Airtel":3.9,"Axis Bank":3.4,"Maruti Suzuki":2.8,"TCS":2.5,"UltraTech Cement":2.2},
"SBI Bluechip Fund - Direct Growth":{"HDFC Bank":8.6,"ICICI Bank":7.2,"Reliance Industries":6.8,"Infosys":5.4,"Larsen & Toubro":4.3,"ITC Ltd":3.9,"State Bank of India":3.5,"TCS":3.1,"Bharti Airtel":2.8,"Axis Bank":2.0},
"Parag Parikh Flexi Cap Fund - Direct Growth":{"HDFC Bank":7.8,"Bajaj Holdings":6.5,"ITC Ltd":6.2,"Power Grid Corp":5.4,"ICICI Bank":5.1,"Coal India":4.7,"Alphabet Inc (Google)":4.2,"Microsoft Corp":3.8,"Axis Bank":3.2,"Maruti Suzuki":2.9},
"Axis Large Cap Fund - Direct Growth":{"Bajaj Finance":8.5,"ICICI Bank":8.1,"HDFC Bank":7.9,"Tata Consultancy Services":6.2,"Infosys":5.8,"Avenue Supermarts (DMart)":5.1,"Reliance Industries":4.6,"Kotak Mahindra Bank":4.2,"Nestle India":3.5,"Titan Company":3.1},
"Mirae Asset Large Cap Fund - Direct Growth":{"HDFC Bank":9.2,"ICICI Bank":8.5,"Reliance Industries":7.9,"Infosys":6.4,"Larsen & Toubro":4.7,"TCS":3.9,"Axis Bank":3.5,"State Bank of India":3.1,"Bharti Airtel":2.9,"ITC Ltd":2.6},
"HDFC Mid-Cap Opportunities Fund - Direct Growth":{"Indian Hotels":4.8,"Tata Communications":4.2,"Bharat Electronics":3.9,"Federal Bank":3.7,"Max Healthcare":3.5,"Coforge Ltd":3.1,"Balkrishna Industries":2.9,"Apollo Tyres":2.6,"Sundram Fasteners":2.4,"HDFC Bank":1.8},
"Kotak Emerging Equity Fund - Direct Growth":{"Supreme Industries":4.6,"Cummins India":4.1,"Schaeffler India":3.8,"Solar Industries":3.5,"Thermax Ltd":3.2,"Persistent Systems":3.0,"Bharat Forge":2.8,"Trent Ltd":2.5,"Atul Ltd":2.3,"Federal Bank":2.1},
"Nippon India Small Cap Fund - Direct Growth":{"Tube Investments":3.2,"HDFC Bank":2.8,"Apar Industries":2.5,"KPIT Technologies":2.3,"Multi Commodity Exchange":2.1,"Carborundum Universal":1.9,"CreditAccess Grameen":1.8,"Magellanic Cloud":1.7,"Poonawalla Fincorp":1.6,"Timken India":1.5},
"Quant Active Fund - Direct Growth":{"Reliance Industries":9.4,"Jio Financial Services":6.8,"Adani Power":5.9,"HDFC Bank":5.4,"Tata Power":4.8,"Steel Authority of India":4.2,"Larsen & Toubro":3.9,"Sun Pharma":3.4,"Aurobindo Pharma":3.1,"Punjab National Bank":2.8}
}

# ============================ STATE ============================
for k,v in {"cash":100000.0,"portfolio":[],"watchlist":[]}.items():
    if k not in st.session_state: st.session_state[k]=v

# ============================ MODEL ============================
@st.cache_resource
def load_model():
    if os.path.exists("dashboard_model_bundle.pkl"):
        try:
            with open("dashboard_model_bundle.pkl","rb") as f:
                b=pickle.load(f)
            return b["model"], b.get("features",FEATURES), "dashboard_model_bundle.pkl", b
        except Exception:
            pass
    from sklearn.ensemble import GradientBoostingClassifier
    X=np.array([
    [150,.05,18,14,16,.8,1.1,-8,.72,.02,.10],
    [85,.02,12,18,20,.3,.4,-16,.45,-.05,.02],
    [25,-.25,-5,26,28,-.4,-.6,-32,.22,-.30,-.45],
    [210,.10,24,13,15,1.1,1.5,-6,.85,.08,.15],
    [18,-.40,-12,32,35,-.9,-1.2,-45,.15,-.50,-.80],
    [120,.04,16,15,17,.6,.8,-10,.65,.01,.04],
    [55,-.10,5,23,26,-.1,-.2,-24,.32,-.12,-.18],
    [300,.12,30,11,13,1.5,1.9,-5,.90,.10,.20]])
    y=np.array([0,0,1,0,1,0,1,0])
    m=GradientBoostingClassifier(n_estimators=120,learning_rate=.05,max_depth=2,random_state=42)
    m.fit(X,y)
    return m,FEATURES,"Fallback synthetic demonstration model",{}

model,features,model_source,bundle=load_model()

# ============================ AMFI ============================
@st.cache_resource
def load_amfi():
    if Mftool is None:return None,pd.DataFrame()
    o=Mftool()
    try:
        d=o.get_scheme_codes()
        df=pd.DataFrame(list(d.items()),columns=["Scheme_Code","Scheme_Name"])
        df["Scheme_Code"]=df["Scheme_Code"].astype(str)
        df["Scheme_Name"]=df["Scheme_Name"].astype(str)
        df=df[~df["Scheme_Name"].str.contains(r"\bMIP\b|\bFMP\b|Fixed Maturity|Dividend|IDCW",case=False,na=False)]
        df["Search_Label"]=df["Scheme_Name"]+" [Code: "+df["Scheme_Code"]+"]"
        return o,df
    except Exception:return o,pd.DataFrame()
amfi,all_schemes=load_amfi()

def get_nav(code):
    if amfi is None: raise RuntimeError("Install mftool: pip install mftool")
    d=amfi.get_scheme_details(str(code))
    if not d: raise ValueError(f"Scheme {code} is inactive/unrecognized by AMFI.")
    h=amfi.get_scheme_historical_nav(str(code),as_Dataframe=True)
    if h is None or len(h)<30: raise ValueError("Insufficient NAV history returned.")
    h=pd.DataFrame(h).copy()
    h["nav"]=pd.to_numeric(h["nav"],errors="coerce")
    h=h.dropna(subset=["nav"])
    h.index=pd.to_datetime(h.index,dayfirst=True,errors="coerce")
    h=h[~h.index.isna()].sort_index()
    return d,h

def analyze(code,rf=6.5):
    d,h=get_nav(code)
    r=h.nav.pct_change().dropna()
    nav=float(h.nav.iloc[-1])
    n21=float(h.nav.iloc[-21]); n252=float(h.nav.iloc[-252]) if len(h)>=252 else float(h.nav.iloc[0])
    n756=float(h.nav.iloc[-756]) if len(h)>=756 else float(h.nav.iloc[0])
    n1260=float(h.nav.iloc[-1260]) if len(h)>=1260 else float(h.nav.iloc[0])
    r1m=(nav/n21-1)*100; r1y=(nav/n252-1)*100
    r3=((nav/n756)**(1/3)-1)*100 if len(h)>=756 else np.nan
    r5=((nav/n1260)**(1/5)-1)*100 if len(h)>=1260 else np.nan
    vol=float(r.tail(252).std()*np.sqrt(252)*100)
    vol30=float(r.tail(30).std()*np.sqrt(252)*100)
    vol=15 if not np.isfinite(vol) or vol<=0 else vol
    vol30=12 if not np.isfinite(vol30) or vol30<=0 else vol30
    dd=(h.nav.tail(252)/h.nav.tail(252).cummax()-1)*100
    maxdd=float(dd.min())
    down=np.minimum(r.tail(252).values,0)
    downside=float(np.sqrt(np.mean(down**2))*np.sqrt(252)*100)
    downside=vol if not np.isfinite(downside) or downside<=0 else downside
    sharpe=(r1y-rf)/vol
    sortino=(r1y-rf)/downside
    rc=np.clip((r1y-rf)/25,-1,1); vc=np.clip((18-vol)/18,-1,1)
    dc=np.clip((15+maxdd)/15,-1,1); sc=np.clip(sharpe/2,-1,1)
    score=float(np.clip(.5+.35*(.35*rc+.25*vc+.2*dc+.2*sc),.05,.95))
    feat={"NAV":nav,"Daily_Return_Pct":r.iloc[-1]*100,"Annualized_Return_1Y":r1y,
          "Volatility_30D":vol30,"Annualized_Volatility_Cleaned":vol,
          "Sharpe_Ratio_Cleaned":sharpe,"Sortino_Ratio_Cleaned":sortino,
          "Max_Drawdown_1Y_Pct":maxdd,"Composite_Score":score,
          "Lag_1D_Return":r.iloc[-1]*100,
          "Lag_5D_Return":((h.nav.iloc[-1]/h.nav.iloc[-6])-1)*100}
    X=pd.DataFrame([feat])
    for f in features:
        if f not in X:X[f]=0
    try:p=float(model.predict_proba(X[features])[0,1])
    except Exception:p=float(np.clip(.7-score,.02,.98))
    return {"code":str(code),"name":d.get("scheme_name","Unknown Scheme"),
            "category":d.get("scheme_category","Mutual Fund"),"amc":d.get("fund_house","AMC"),
            "nav":nav,"ret1m":r1m,"ret1y":r1y,"ret3y":r3,"ret5y":r5,
            "vol":vol,"vol30":vol30,"dd":maxdd,"sharpe":sharpe,"sortino":sortino,
            "score":score,"prob":np.clip(p,0,1),"navdf":h,"features":feat}

def clean(s):return str(s).replace("⭐ ","")
def status(p):return "🔴 Critical" if p>=.6 else ("🟡 Watchlist" if p>=.3 else "🟢 Healthy")
def money(x):return f"₹{float(x):,.2f}"
def pc(x):return f"{float(x):+.2f}%"

def selector(label,key):
    source=st.radio("Source",["⭐ Popular active","🔎 Full AMFI universe"],horizontal=True,key=key+"_src")
    if source.startswith("⭐") or all_schemes.empty:
        name=st.selectbox(label,list(ACTIVE),key=key)
        return ACTIVE[name],name
    name=st.selectbox(label,all_schemes.Search_Label.tolist(),key=key)
    return str(all_schemes.loc[all_schemes.Search_Label==name,"Scheme_Code"].iloc[0]),name

def gauge(p,title="Model Risk Indicator"):
    f=go.Figure(go.Indicator(mode="gauge+number",value=p*100,number={"suffix":"%"},
      title={"text":title},gauge={"axis":{"range":[0,100]},"bar":{"color":"#b91c1c" if p>=.6 else "#b45309" if p>=.3 else "#15803d"},
      "steps":[{"range":[0,30],"color":"#dcfce7"},{"range":[30,60],"color":"#fef3c7"},{"range":[60,100],"color":"#fee2e2"}]}))
    f.update_layout(height=290,margin=dict(l=10,r=10,t=40,b=5),template="plotly_white")
    return f

def explain(r):
    good=[];bad=[]
    (good if r["ret1y"]>6.5 else bad).append("1Y return is above the 6.5% reference hurdle.")
    (good if r["sharpe"]>.5 else bad).append("Sharpe is above the 0.50 reference level.")
    (good if r["dd"]>-12 else bad).append("1Y drawdown is within the -12% reference range.")
    (good if r["vol"]<18 else bad).append("Annualized volatility is below 18%.")
    if good: st.markdown("<div class='good'><b>What looks good</b><br>• "+"<br>• ".join(good)+"</div>",unsafe_allow_html=True)
    if bad: st.markdown("<div class='warn'><b>What to watch</b><br>• "+"<br>• ".join(bad)+"</div>",unsafe_allow_html=True)

# ============================ SIDEBAR ============================
with st.sidebar:
    st.markdown("## 🛡️ FinShield")
    st.caption("Universal Mutual Fund Intelligence")
    page=st.radio("Module",[
        "🏠 Home","⚔️ Fund Duel","🔍 Fund Intelligence","📊 Advanced Analytics",
        "⚡ Stress Lab","💼 Paper Portfolio","🧩 Overlap Analyzer",
        "💰 SIP & Goal Planner","⭐ Watchlist","📁 Historical Dataset","ℹ️ Methodology"])
    rf=st.number_input("Risk-free rate (%)",0.,15.,6.5,.25)
    st.markdown("---")
    st.success("AMFI loaded") if amfi else st.error("mftool unavailable")
    st.caption(f"Model: {model_source}")

st.markdown("""<div class="hero"><h1>🛡️ FinShield</h1>
<p>Live NAV intelligence • risk analytics • fund comparison • stress testing • portfolio overlap • paper trading • SIP planning</p></div>""",unsafe_allow_html=True)

# ============================ HOME ============================
if page=="🏠 Home":
    st.subheader("Investor-first mutual fund research")
    a,b,c,d=st.columns(4)
    a.metric("AMFI schemes loaded",f"{len(all_schemes):,}")
    b.metric("Popular benchmark set",len(ACTIVE))
    c.metric("Analytics", "20+")
    d.metric("Paper cash",money(st.session_state.cash))
    st.markdown("### 🚀 Core workflow")
    st.markdown("""<div class="insight"><b>Identify → Measure → Compare → Stress-test → Check overlap → Plan → Paper-track</b><br>
    Use the dashboard to understand a fund before deciding whether it deserves further due diligence.</div>""",unsafe_allow_html=True)
    st.markdown("### ⭐ Benchmark schemes")
    st.dataframe(pd.DataFrame({"Scheme":list(ACTIVE),"AMFI Code":list(ACTIVE.values())}),use_container_width=True,hide_index=True)
    st.markdown("### ⚠️ Important")
    st.warning("FinShield is a research dashboard, not personalized investment advice. Verify the latest official factsheet, portfolio, risk-o-meter, expense ratio, exit load and scheme documents.")

# ============================ DUEL ============================
elif page=="⚔️ Fund Duel":
    st.subheader("⚔️ Head-to-Head Scheme Duel")
    c1,c2=st.columns(2)
    with c1: ca,la=selector("Fund A","duela")
    with c2: cb,lb=selector("Fund B","duelb")
    if st.button("⚔️ Launch Live Duel",type="primary",use_container_width=True):
        try:
            with st.spinner("Fetching AMFI histories..."):
                ra,rb=analyze(ca,rf),analyze(cb,rf)
            winner=ra if ra["prob"]<rb["prob"] else rb if rb["prob"]<ra["prob"] else None
            if winner: st.success(f"🏆 Lower model-risk indicator: {clean(winner['name'])} — {winner['prob']*100:.1f}%")
            else: st.info("⚖️ Equal model-risk indicator.")
            x,y=st.columns(2)
            with x: st.plotly_chart(gauge(ra["prob"],"Fund A"),use_container_width=True)
            with y: st.plotly_chart(gauge(rb["prob"],"Fund B"),use_container_width=True)
            comp=pd.DataFrame({
                "Metric":["Category","AMC","NAV","1M Return","1Y Return","3Y CAGR","5Y CAGR","Volatility","Max Drawdown","Sharpe","Sortino","Composite Score","Model Risk"],
                "Fund A":[ra["category"],ra["amc"],money(ra["nav"]),pc(ra["ret1m"]),pc(ra["ret1y"]),pc(ra["ret3y"]) if np.isfinite(ra["ret3y"]) else "N/A",pc(ra["ret5y"]) if np.isfinite(ra["ret5y"]) else "N/A",f"{ra['vol']:.2f}%",f"{ra['dd']:.2f}%",f"{ra['sharpe']:.2f}",f"{ra['sortino']:.2f}",f"{ra['score']:.2f}",f"{ra['prob']*100:.1f}%"],
                "Fund B":[rb["category"],rb["amc"],money(rb["nav"]),pc(rb["ret1m"]),pc(rb["ret1y"]),pc(rb["ret3y"]) if np.isfinite(rb["ret3y"]) else "N/A",pc(rb["ret5y"]) if np.isfinite(rb["ret5y"]) else "N/A",f"{rb['vol']:.2f}%",f"{rb['dd']:.2f}%",f"{rb['sharpe']:.2f}",f"{rb['sortino']:.2f}",f"{rb['score']:.2f}",f"{rb['prob']*100:.1f}%"]})
            st.dataframe(comp,use_container_width=True,hide_index=True)
            z=pd.concat([ra["navdf"].nav.rename("Fund A")/ra["navdf"].nav.iloc[0]*100,
                         rb["navdf"].nav.rename("Fund B")/rb["navdf"].nav.iloc[0]*100],axis=1)
            st.plotly_chart(px.line(z,title="Normalized Growth of ₹100"),use_container_width=True)
            q,w=st.columns(2)
            with q: st.markdown(f"**{clean(ra['name'])}**"); explain(ra)
            with w: st.markdown(f"**{clean(rb['name'])}**"); explain(rb)
        except Exception as e: st.error(str(e))
            
# ============================ SINGLE ============================
elif page=="🔍 Fund Intelligence":
    st.subheader("🔍 Single Scheme Intelligence")
    code,name=selector("Select scheme","single")
    if st.button("🚀 Analyze Live Scheme",type="primary",use_container_width=True):
        try:
            with st.spinner("Analyzing NAV history..."): r=analyze(code,rf)
            st.markdown(f"**{clean(r['name'])}** · {r['category']} · {r['amc']} · NAV {money(r['nav'])}")
            st.success(status(r["prob"])+" — "+f"model indicator {r['prob']*100:.1f}%") if r["prob"]<.3 else st.warning(status(r["prob"])+" — "+f"model indicator {r['prob']*100:.1f}%") if r["prob"]<.6 else st.error(status(r["prob"])+" — "+f"model indicator {r['prob']*100:.1f}%")
            a,b,c,d,e=st.columns(5)
            a.metric("1M",pc(r["ret1m"]));b.metric("1Y",pc(r["ret1y"]));c.metric("3Y CAGR",pc(r["ret3y"]) if np.isfinite(r["ret3y"]) else "N/A");d.metric("Sharpe",f"{r['sharpe']:.2f}");e.metric("Max DD",f"{r['dd']:.1f}%")
            t1,t2,t3,t4=st.tabs(["📈 Performance","🛡️ Risk","🤖 AI Score","📄 Factsheet"])
            with t1:
                st.plotly_chart(px.line(r["navdf"],x=r["navdf"].index,y="nav",title="Historical NAV"),use_container_width=True)
                dd=(r["navdf"].nav/r["navdf"].nav.cummax()-1)*100
                st.plotly_chart(px.area(x=dd.index,y=dd.values,title="Full-history Drawdown"),use_container_width=True)
            with t2:
                x,y,z=st.columns(3);x.metric("Annualized Volatility",f"{r['vol']:.2f}%");y.metric("30D Volatility",f"{r['vol30']:.2f}%");z.metric("Sortino",f"{r['sortino']:.2f}")
                explain(r)
            with t3:
                x,y=st.columns([1,1.5])
                with x:st.plotly_chart(gauge(r["prob"]),use_container_width=True)
                with y:
                    st.metric("Composite Health Score",f"{r['score']:.2f}")
                    st.dataframe(pd.DataFrame({"Feature":list(r["features"]),"Value":list(r["features"].values())}),use_container_width=True,hide_index=True)
                    st.caption("The model indicator is a research signal, not a guarantee of fund distress/failure.")
            with t4:
                html=f"""<html><body><h1>AlphaShield Factsheet</h1><h2>{clean(r['name'])}</h2>
                <p>{r['category']} | {r['amc']} | NAV {money(r['nav'])}</p>
                <table border=1 cellpadding=8><tr><th>Metric</th><th>Value</th></tr>
                <tr><td>1Y Return</td><td>{pc(r['ret1y'])}</td></tr><tr><td>3Y CAGR</td><td>{pc(r['ret3y']) if np.isfinite(r['ret3y']) else 'N/A'}</td></tr>
                <tr><td>Volatility</td><td>{r['vol']:.2f}%</td></tr><tr><td>Max Drawdown</td><td>{r['dd']:.2f}%</td></tr>
                <tr><td>Sharpe</td><td>{r['sharpe']:.2f}</td></tr><tr><td>Sortino</td><td>{r['sortino']:.2f}</td></tr>
                <tr><td>Composite Score</td><td>{r['score']:.2f}</td></tr><tr><td>Model Risk</td><td>{r['prob']*100:.1f}%</td></tr></table>
                <p>This is a research document, not investment advice.</p></body></html>"""
                st.download_button("📥 Download HTML Factsheet",html,f"AlphaShield_{code}.html","text/html",type="primary")
        except Exception as e:
            st.error(str(e))

# ============================ ADVANCED ============================
elif page=="📊 Advanced Analytics":
    st.subheader("📊 Advanced Analytics")
    code,name=selector("Select scheme","advanced")
    if st.button("Run Advanced Analytics",type="primary",use_container_width=True):
        try:
            r=analyze(code,rf); nav=r["navdf"].nav; ret=nav.pct_change().dropna()
            a,b,c,d=st.columns(4);a.metric("Observations",f"{len(nav):,}");b.metric("Full-history CAGR",pc(((nav.iloc[-1]/nav.iloc[0])**(365.25/((nav.index[-1]-nav.index[0]).days))-1)*100));c.metric("Best day",pc(ret.max()*100));d.metric("Worst day",pc(ret.min()*100))
            x,y,z=st.tabs(["Rolling Returns","Rolling Risk","Distribution"])
            with x:
                q=pd.DataFrame({"1Y":(nav/nav.shift(252)-1)*100,"3Y":(nav/nav.shift(756)-1)*100})
                st.plotly_chart(px.line(q,title="Rolling Returns"),use_container_width=True)
            with y:
                q=ret.rolling(63).std()*np.sqrt(252)*100
                st.plotly_chart(px.line(q,title="63D Annualized Volatility"),use_container_width=True)
                dd=(nav/nav.cummax()-1)*100
                st.plotly_chart(px.area(x=dd.index,y=dd.values,title="Drawdown"),use_container_width=True)
            with z:
                st.plotly_chart(px.histogram(ret*100,nbins=60,title="Daily Return Distribution"),use_container_width=True)
                st.dataframe(ret.mul(100).quantile([.01,.05,.25,.5,.75,.95,.99]).rename("Return %").to_frame(),use_container_width=True)
        except Exception as e:st.error(str(e))

# ============================ STRESS ============================
elif page=="⚡ Stress Lab":
    st.subheader("⚡ Crisis & Macro Shock Stress Lab")
    code,name=selector("Select scheme","stress")
    try:
        r=analyze(code,rf)
        shock=st.slider("Hypothetical market shock (%)",-60,-5,-20,5)
        sensitivity=st.slider("Sensitivity multiplier",.5,2.,1.,.05)
        loss=shock*(r["vol"]/15)*sensitivity; snav=max(0,r["nav"]*(1+loss/100))
        a,b,c=st.columns(3);a.metric("Current NAV",money(r["nav"]));b.metric("Scenario loss",f"{loss:.2f}%");c.metric("Scenario NAV",money(snav))
        nav=r["navdf"].nav
        covid=nav.loc["2020-01-01":"2020-05-31"];rate=nav.loc["2022-01-01":"2022-06-30"]
        a,b,c=st.columns(3)
        a.metric("COVID window",f"{(covid.min()/covid.max()-1)*100:.1f}%" if len(covid)>10 else "N/A")
        b.metric("2022 rate window",f"{(rate.min()/rate.max()-1)*100:.1f}%" if len(rate)>10 else "N/A")
        c.metric("Current 1Y max DD",f"{r['dd']:.1f}%")
        days=np.arange(31);curve=r["nav"]*(1+loss/100*days/30)
        st.plotly_chart(px.line(pd.DataFrame({"Day":days,"Scenario NAV":curve}),x="Day",y="Scenario NAV",title="30-Day Shock Illustration"),use_container_width=True)
        st.warning("Scenario output is a sensitivity exercise, not a forecast.")
    except Exception as e:st.error(str(e))

# ============================ PAPER PORTFOLIO ============================
elif page=="💼 Paper Portfolio":
    st.subheader("💼 Paper Trading & AI Proof Ledger")
    rows=[];total_cost=total_value=0
    for p in st.session_state.portfolio:
        try:
            r=analyze(p["code"],rf);value=p["units"]*r["nav"]
            total_cost+=p["invested"];total_value+=value
            rows.append({"Scheme":clean(r["name"]),"Category":p["category"],"Invested":p["invested"],"Current Value":value,"P&L":value-p["invested"],"P&L %":(value/p["invested"]-1)*100,"Units":p["units"],"Buy NAV":p["buy_nav"],"Live NAV":r["nav"],"AI Risk at Buy":p["risk"]*100,"Status":status(r["prob"])})
        except Exception:pass
    cash=st.session_state.cash
    a,b,c,d=st.columns(4);a.metric("Paper Cash",money(cash));b.metric("Invested",money(total_cost));c.metric("Current Value",money(total_value),f"{total_value-total_cost:+,.2f} P&L");d.metric("Net Worth",money(cash+total_value))
    code,name=selector("Scheme to buy","trade")
    amt=st.number_input("Investment amount (₹)",1000.,max(1000.,cash),min(10000.,cash),1000.)
    if st.button("📥 Buy Units at Live NAV",type="primary",use_container_width=True):
        if amt>cash:st.error("Insufficient paper cash.")
        else:
            try:
                r=analyze(code,rf);units=amt/r["nav"]
                st.session_state.portfolio.append({"time":datetime.now().strftime("%d-%b-%Y %H:%M"),"code":code,"name":r["name"],"category":r["category"],"buy_nav":r["nav"],"units":units,"invested":amt,"risk":r["prob"]})
                st.session_state.cash-=amt;st.success(f"Bought {units:.3f} units of {clean(r['name'])}.");st.rerun()
            except Exception as e:st.error(str(e))
    if rows:
        df=pd.DataFrame(rows);st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("📥 Download Ledger CSV",df.to_csv(index=False),"AlphaShield_Portfolio.csv","text/csv")
    else:st.info("No paper positions yet.")
    if st.button("🔄 Reset Paper Portfolio"):st.session_state.cash=100000.;st.session_state.portfolio=[];st.rerun()

# ============================ OVERLAP ============================
elif page=="🧩 Overlap Analyzer":
    st.subheader("🧩 Portfolio Overlap & Hidden Concentration")
    a,b=st.columns(2)
    names=list(HOLDINGS)
    with a:f1=st.selectbox("Fund 1",names,0)
    with b:f2=st.selectbox("Fund 2",names,1)
    h1,h2=HOLDINGS[f1],HOLDINGS[f2]
    common=set(h1)&set(h2);stocks=set(h1)|set(h2)
    overlap=sum(min(h1.get(s,0),h2.get(s,0)) for s in common)
    combo={s:(h1.get(s,0)+h2.get(s,0))/2 for s in stocks}
    a,b,c,d=st.columns(4);a.metric("Overlap",f"{overlap:.1f}%");b.metric("Common companies",len(common));c.metric("Tracked companies",len(stocks));d.metric("Largest effective exposure",f"{max(combo.values()):.1f}%")
    st.warning("High overlap can create an illusion of diversification.") if overlap>35 else st.info("Moderate overlap — review common holdings.") if overlap>15 else st.success("Low tracked overlap.")
    df=pd.DataFrame([{"Company":s,"Fund 1 %":h1.get(s,0),"Fund 2 %":h2.get(s,0),"Effective 50:50 %":combo[s],"Common":"Yes" if s in common else "No"} for s in stocks]).sort_values("Effective 50:50 %",ascending=False)
    critical=df[df["Effective 50:50 %"]>=6]
    if not critical.empty:st.dataframe(critical,use_container_width=True,hide_index=True)
    st.plotly_chart(px.bar(df.head(15),x="Company",y=["Fund 1 %","Fund 2 %"],barmode="group",title="Top Underlying Companies"),use_container_width=True)
    with st.expander("Complete holdings table"):st.dataframe(df,use_container_width=True,hide_index=True)

# ============================ SIP ============================
elif page=="💰 SIP & Goal Planner":
    st.subheader("💰 SIP, Lump-Sum & Goal Planner")
    mode=st.radio("Mode",["SIP corpus","Lump sum","Goal-first"],horizontal=True)
    if mode=="SIP corpus":
        a,b,c=st.columns(3);monthly=a.number_input("Monthly SIP ₹",500.,1e6,10000.,500.);years=b.slider("Years",1,40,10);ret=c.number_input("Expected return %",-10.,30.,12.,.5)
        out=[]
        for rr in [ret-3,ret,ret+3]:
            m=(1+rr/100)**(1/12)-1;n=years*12;fv=monthly*n if m==0 else monthly*((1+m)**n-1)/m*(1+m)
            out.append({"Scenario":f"{rr:.1f}%","Invested":monthly*n,"Corpus":fv,"Gain":fv-monthly*n})
        st.dataframe(pd.DataFrame(out).style.format("{:,.0f}",subset=["Invested","Corpus","Gain"]),use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(pd.DataFrame(out),x="Scenario",y="Corpus",title="Scenario Corpus"),use_container_width=True)
    elif mode=="Lump sum":
        a,b,c=st.columns(3);p=a.number_input("Initial ₹",1000.,1e8,100000.,5000.);years=b.slider("Years",1,40,10);ret=c.number_input("Return %",-10.,30.,12.,.5)
        out=[{"Scenario":f"{rr:.1f}%","Future Value":p*(1+rr/100)**years,"Gain":p*((1+rr/100)**years-1)} for rr in [ret-3,ret,ret+3]]
        st.dataframe(pd.DataFrame(out).style.format("{:,.0f}",subset=["Future Value","Gain"]),use_container_width=True,hide_index=True)
    else:
        goal=st.number_input("Target ₹",10000.,1e8,2500000.,50000.);years=st.slider("Years",1,40,10);ret=st.number_input("Expected return %",-10.,30.,12.,.5)
        n=years*12;m=(1+ret/100)**(1/12)-1;required=goal/n if m==0 else goal/(((1+m)**n-1)/m*(1+m))
        st.metric("Approx. required monthly SIP",money(required))
        out=[]
        for rr in [ret-3,ret,ret+3]:
            mm=(1+rr/100)**(1/12)-1;req=goal/n if mm==0 else goal/(((1+mm)**n-1)/mm*(1+mm));out.append({"Return":f"{rr:.1f}%","Required SIP":req})
        st.dataframe(pd.DataFrame(out).style.format({"Required SIP":"₹{:,.0f}"}),use_container_width=True,hide_index=True)

# ============================ WATCHLIST ============================
elif page=="⭐ Watchlist":
    st.subheader("⭐ Investor Watchlist")
    code,name=selector("Scheme to add","watch")
    if st.button("⭐ Add to Watchlist",type="primary"):
        if code not in [x["code"] for x in st.session_state.watchlist]:
            st.session_state.watchlist.append({"code":code,"name":name});st.success("Added.")
        else:st.info("Already on watchlist.")
    rows=[]
    for p in st.session_state.watchlist:
        try:
            r=analyze(p["code"],rf);rows.append({"Scheme":clean(r["name"]),"Category":r["category"],"NAV":r["nav"],"1Y Return":r["ret1y"],"Volatility":r["vol"],"Max DD":r["dd"],"Sharpe":r["sharpe"],"Model Risk":r["prob"]*100,"Status":status(r["prob"])})
        except Exception:rows.append({"Scheme":clean(p["name"]),"Status":"Unavailable"})
    if rows:st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else:st.info("Watchlist is empty.")
    if st.button("🗑️ Clear Watchlist"):st.session_state.watchlist=[];st.rerun()

# ============================ HISTORICAL ============================
elif page=="📁 Historical Dataset":
    st.subheader("📁 Historical Research Dataset")
    funds=pd.DataFrame(bundle.get("sample_funds",[]))
    if funds.empty:
        st.info("No sample_funds table exists in dashboard_model_bundle.pkl. Live AMFI modules remain available.")
    else:
        name=st.selectbox("Historical fund",funds.Fund_Name.astype(str).tolist());row=funds[funds.Fund_Name==name].iloc[0]
        X=pd.DataFrame([[float(row.get(f,0)) for f in features]],columns=features)
        p=float(model.predict_proba(X)[0,1])
        st.metric("Historical Model Risk",f"{p*100:.1f}%")
        st.dataframe(pd.DataFrame({"Feature":features,"Value":[row.get(f,0) for f in features]}),use_container_width=True,hide_index=True)

# ============================ METHODOLOGY ============================
else:
    st.subheader("ℹ️ Methodology & Responsible Use")
    st.markdown("""
    ### Core metrics
    - **1Y Return:** latest NAV versus roughly 252 observations earlier.
    - **3Y/5Y CAGR:** annualized growth over the available history.
    - **Volatility:** daily-return standard deviation annualized using √252.
    - **Maximum Drawdown:** largest fall from a running NAV peak.
    - **Sharpe:** (1Y return − risk-free rate) / annualized volatility.
    - **Sortino:** return relative to downside deviation.
    - **Composite Score:** bounded dashboard score combining return, volatility, drawdown and Sharpe.
    - **Overlap:** Σ minimum weight of each common company between two portfolios.
    ### Model
    If `dashboard_model_bundle.pkl` exists, its model/features are used. Otherwise a small
    synthetic fallback model keeps the application functional. **The fallback is not a
    production investment model.** A production version should use properly labeled historical
    observations, train/validation/test splits, leakage controls, calibration and out-of-sample
    evaluation.
    ### Investor safeguards
    Expense ratio, exit load, tax, benchmark tracking, fund manager changes, portfolio turnover,
    AUM, liquidity, risk-o-meter and current holdings should be verified separately before investing.
    Historical performance and model scores do not guarantee future returns.
    """)

st.markdown("""<div class="note">🛡️ FinShield is an analytical/research dashboard, not a SEBI-registered investment adviser.
Verify the latest official scheme documents before investing.</div>""",unsafe_allow_html=True)
