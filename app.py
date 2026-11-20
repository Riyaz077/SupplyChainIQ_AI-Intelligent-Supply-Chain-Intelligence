import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config( page_title="SupplyChainIQ AI", page_icon="assets/supplychainiq_favicon.png", layout="wide")
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#42D9F5", "#6EA8FF", "#E4C995", "#52D6A0", "#FF8A8A"]
st.markdown("""
<style>
/* SUPPLYCHAINIQ AI — AURORA SAPPHIRE PREMIUM SYSTEM */
:root {
  --bg: #091321;
  --surface: #101F32;
  --surface-2: #142943;
  --surface-3: #192F49;
  --line: rgba(104, 169, 220, .20);
  --cyan: #42D9F5;
  --blue: #6EA8FF;
  --champagne: #E4C995;
  --text: #F1F6FC;
  --muted: #9CB0C6;
  --danger: #FF7D86;
}
.stApp {
  background:
    radial-gradient(ellipse at 12% 0%, rgba(47,132,196,.18), transparent 34%),
    radial-gradient(ellipse at 88% 8%, rgba(66,217,245,.08), transparent 28%),
    linear-gradient(145deg, #08111E 0%, #0B1727 55%, #0A1422 100%);
  color: var(--text);
}
.block-container {
  max-width: 1480px;
  padding: 2.0rem 2.2rem 3.2rem;
}
h1,h2,h3,h4 { color: var(--text) !important; letter-spacing: -.35px; }
h1 { font-weight: 850 !important; letter-spacing: -1px; }
h2,h3 { font-weight: 750 !important; }
p, label, li { color: #D4E0ED; }
small, .stCaption, [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#10243A 0%,#0D1D30 52%,#0A1727 100%);
  border-right: 1px solid rgba(66,217,245,.18);
}
[data-testid="stSidebar"] > div:first-child { padding: 1.55rem .95rem 1.4rem; }
[data-testid="stSidebar"] h2 {
  color: #F4F8FD !important; font-size: 1.22rem !important;
  font-weight: 850 !important; letter-spacing: 1.7px;
  padding: .2rem .25rem .9rem; border-bottom: 1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
  color: var(--champagne) !important; font-size: .70rem !important;
  font-weight: 800 !important; letter-spacing: 1.8px;
  text-transform: uppercase; margin: .5rem 0 .75rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap: 7px; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
  background: rgba(255,255,255,.025); border: 1px solid rgba(142,184,222,.08);
  border-radius: 11px; padding: 10px 11px; margin: 0;
  transition: all .18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p {
  color: #C5D5E6; font-size: .83rem; font-weight: 560;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
  background: rgba(66,217,245,.08); border-color: rgba(66,217,245,.30);
  transform: translateX(2px);
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(100deg,rgba(66,217,245,.17),rgba(110,168,255,.07));
  border: 1px solid rgba(66,217,245,.48);
  box-shadow: inset 3px 0 0 var(--cyan);
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
  color: var(--cyan) !important; font-weight: 760 !important;
}
[data-testid="stSidebar"] input[type="radio"] { accent-color: var(--cyan); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  color: #8FA7BF !important; font-size: .76rem; line-height: 1.55;
}

/* KPI cards */
[data-testid="stMetric"] {
  background: linear-gradient(145deg,rgba(23,47,73,.96),rgba(15,32,51,.96));
  border: 1px solid rgba(110,168,255,.19); border-radius: 15px;
  padding: 17px 18px; box-shadow: 0 10px 28px rgba(0,0,0,.16);
  min-height: 112px; transition: all .2s ease;
}
[data-testid="stMetric"]:hover {
  border-color: rgba(66,217,245,.48); transform: translateY(-2px);
}
[data-testid="stMetricLabel"] { color: #A9BED4 !important; font-size: .78rem !important; }
[data-testid="stMetricValue"] { color: #F5F9FF !important; font-weight: 820 !important; }
[data-testid="stMetricDelta"] { font-size: .76rem !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(110deg,#42D9F5,#6EA8FF);
  color: #071522 !important; border: 0; border-radius: 10px;
  font-weight: 760; padding: .55rem 1rem; transition: all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: linear-gradient(110deg,#7CE8F8,#9BC0FF);
  box-shadow: 0 7px 20px rgba(66,217,245,.17); transform: translateY(-1px);
}

/* Inputs and select controls */
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
  background: #12263C !important; border-color: rgba(110,168,255,.24) !important;
  border-radius: 10px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div { color: var(--cyan); }

/* Data tables */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid rgba(110,168,255,.20);
  border-radius: 13px; overflow: hidden;
  background: rgba(16,31,50,.72);
}

/* Expanders, alerts, tabs */
[data-testid="stExpander"] {
  background: rgba(16,31,50,.66); border: 1px solid rgba(110,168,255,.18);
  border-radius: 12px;
}
[data-testid="stAlert"] { border-radius: 12px; }
.stTabs [data-baseweb="tab-list"] {
  gap: 8px; border-bottom: 1px solid rgba(110,168,255,.17);
}
.stTabs [data-baseweb="tab"] {
  background: rgba(20,41,67,.72); border-radius: 9px 9px 0 0;
  padding: 10px 16px; color: #AFC1D5;
}
.stTabs [aria-selected="true"] {
  color: var(--cyan) !important; border-bottom: 2px solid var(--cyan) !important;
}
hr { border-color: rgba(110,168,255,.18) !important; }
::-webkit-scrollbar { width: 7px; }
::-webkit-scrollbar-track { background: #091321; }
::-webkit-scrollbar-thumb { background: #294662; border-radius: 9px; }
::-webkit-scrollbar-thumb:hover { background: #42D9F5; }

/* Reusable dashboard section and footer */
.sciq-section {
  background: linear-gradient(145deg,rgba(20,41,67,.82),rgba(13,29,47,.82));
  border: 1px solid rgba(110,168,255,.18); border-radius: 15px;
  padding: 17px 19px; margin: 8px 0 16px;
}
.sciq-footer {
  margin-top: 42px; padding: 30px 24px 20px;
  text-align: center; border-radius: 18px;
  background: linear-gradient(145deg,rgba(16,36,58,.98),rgba(8,22,37,.98));
  border: 1px solid rgba(66,217,245,.22);
  box-shadow: 0 14px 38px rgba(0,0,0,.18);
  position: relative; overflow: hidden;
}
.sciq-footer:before {
  content: ""; position: absolute; top: 0; left: 8%; right: 8%;
  height: 2px; background: linear-gradient(90deg,transparent,#42D9F5,#6EA8FF,#E4C995,transparent);
}
.sciq-footer .project {
  color: var(--cyan); font-size: 1.02rem; font-weight: 800;
  letter-spacing: .15px; margin-bottom: 22px;
}
.sciq-footer .eyebrow {
  color: #9CB9D5; font-size: .75rem; letter-spacing: 1.6px;
  font-weight: 700; margin-bottom: 8px;
}
.sciq-footer .name {
  color: #F5F8FC; font-size: 1.65rem; font-weight: 850; margin-bottom: 8px;
}
.sciq-footer .meta { color: #AFC5DC; font-size: .88rem; line-height: 1.9; }
.sciq-footer .copyright {
  border-top: 1px solid rgba(110,168,255,.16); margin-top: 23px;
  padding-top: 17px; color: #7896B3; font-size: .72rem;
}

/* WORKSPACE HEADER — LARGER TYPOGRAPHY */

.stApp div[style*="WORKSPACE /"] {
    margin-top: 24px !important;
    margin-bottom: 26px !important;
}

.stApp div[style*="WORKSPACE /"] > div:first-child {
    font-size: 0.95rem !important;
    letter-spacing: 2.2px !important;
    font-weight: 850 !important;
}

.stApp div[style*="WORKSPACE /"] > div:last-child {
    font-size: 1.15rem !important;
    line-height: 1.7 !important;
    font-weight: 500 !important;
    margin-top: 10px !important;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data
def build_data(n=1800):
    r=np.random.default_rng(42)
    suppliers=["Apex Components","Northstar Logistics","Vertex Materials","Orion Packaging","Pioneer Supply","Crestline Parts"]
    regions=["South","North","West","East","Central"]
    cats=["Electronics","Industrial","Packaging","Components"]
    demand=r.integers(45,520,n)
    lead=np.clip(r.normal(12,4,n),2,32)
    inventory=np.clip(demand*r.uniform(.35,2.1,n),10,1100)
    ontime=np.clip(r.normal(.91,.07,n),.55,1)
    quality=np.clip(r.normal(.965,.025,n),.82,1)
    disruption=r.choice([0,1],n,p=[.88,.12])
    cost=r.uniform(8,180,n)
    delay=np.clip(lead*(1-ontime)+disruption*r.uniform(3,12,n)+r.normal(0,1.5,n),0,25)
    risk=np.clip(.35*(1-ontime)+.25*(1-quality)+.025*lead+.25*disruption+r.normal(0,.035,n),0,1)
    return pd.DataFrame({"date":pd.date_range("2024-01-01",periods=n,freq="D"),
      "supplier":r.choice(suppliers,n),"region":r.choice(regions,n),"category":r.choice(cats,n),
      "demand_units":demand,"inventory_units":inventory.round().astype(int),"lead_time_days":lead.round(1),
      "unit_cost":cost.round(2),"on_time_rate":ontime.round(3),"quality_rate":quality.round(3),
      "disruption_flag":disruption,"delivery_delay_days":delay.round(1),"supplier_risk_score":risk.round(3)})

df=build_data()
num=["demand_units","inventory_units","lead_time_days","unit_cost","on_time_rate","quality_rate","disruption_flag"]
cat=["supplier","region","category"]
prep=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
X=df[num+cat]; y=df.delivery_delay_days
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42)
model=Pipeline([("prep",prep),("rf",RandomForestRegressor(n_estimators=180,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1))])
model.fit(Xtr,ytr)
pred=model.predict(Xte)
iso=IsolationForest(contamination=.06,n_estimators=160,random_state=42).fit(df[num])
df["anomaly"] = iso.predict(df[num])==-1
df["predicted_delay_days"]=model.predict(X).round(1)
df["inventory_status"]=np.where(df.inventory_units<df.demand_units*.55,"At Risk",np.where(df.inventory_units<df.demand_units,"Watch","Healthy"))
df["risk_band"]=pd.cut(df.supplier_risk_score,[-.01,.25,.5,.75,1.01],labels=["Low","Moderate","High","Critical"])

st.markdown("""
<div class="sciq-section" style="padding:25px 27px 22px;">
  <div style="color:#42D9F5;font-size:.72rem;font-weight:800;letter-spacing:2px;margin-bottom:9px;">ENTERPRISE OPERATIONS INTELLIGENCE</div>
  <div style="font-size:2.05rem;font-weight:850;letter-spacing:-1px;color:#F4F8FD;line-height:1.2;">SupplyChainIQ AI</div>
  <div style="font-size:1rem;color:#C0D2E5;margin-top:9px;font-weight:550;">Supply Chain Intelligence · Supplier Risk · Inventory Analytics</div>
  <div style="font-size:.80rem;color:#8FA9C3;margin-top:12px;">A unified decision-support workspace for operational visibility and smarter planning</div>
</div>
""", unsafe_allow_html=True)
st.caption("PORTFOLIO DEMONSTRATOR  •  SYNTHETIC DATA  •  NO LIVE ERP OR LOGISTICS CONNECTIONS")
with st.sidebar:
    st.markdown("## SUPPLYCHAINIQ")
    page=st.radio("WORKSPACES",["Executive Overview","Demand & Inventory","Supplier Intelligence","Risk & Anomalies","Scenario Planner","Data Explorer","Model Performance","Governance"])
    st.caption("1,800 generated operational records")
with st.expander("Global filters"):
    a,b,c=st.columns(3)
    regions=a.multiselect("Region",sorted(df.region.unique()),default=sorted(df.region.unique()))
    categories=b.multiselect("Category",sorted(df.category.unique()),default=sorted(df.category.unique()))
    suppliers=c.multiselect("Supplier",sorted(df.supplier.unique()),default=sorted(df.supplier.unique()))
f=df[df.region.isin(regions)&df.category.isin(categories)&df.supplier.isin(suppliers)].copy()
if f.empty: st.warning("No records match these filters."); st.stop()

page_descriptions = {
    "Executive Overview": "A high-level view of operational health, delivery performance and supply exposure.",
    "Demand & Inventory": "Explore demand patterns, stock coverage and replenishment priorities.",
    "Supplier Intelligence": "Compare supplier reliability, quality, lead times and indicative performance.",
    "Risk & Anomalies": "Surface unusual operational patterns and prioritize records for human review.",
    "Scenario Planner": "Stress-test demand, inventory and lead-time assumptions with interactive scenarios.",
    "Data Explorer": "Inspect and export the filtered operational dataset.",
    "Model Performance": "Review delivery-delay model performance on a synthetic holdout dataset.",
    "Governance": "Understand data provenance, model boundaries and responsible-use requirements."
}
st.markdown(f"""
<div style="margin:18px 0 22px;padding:2px 0 13px;border-bottom:1px solid rgba(110,168,255,.16);">
  <div style="font-size:.68rem;letter-spacing:1.8px;color:#E4C995;font-weight:800;">WORKSPACE / {page.upper()}</div>
  <div style="font-size:.91rem;color:#AFC1D5;margin-top:7px;">{page_descriptions[page]}</div>
</div>
""",unsafe_allow_html=True)

if page=="Executive Overview":
    st.subheader("Executive command centre")
    a,b,c,d=st.columns(4)
    a.metric("Average delivery delay",f"{f.delivery_delay_days.mean():.1f} days")
    b.metric("At-risk inventory",f"{(f.inventory_status=='At Risk').mean()*100:.1f}%")
    c.metric("Anomaly rate",f"{f.anomaly.mean()*100:.1f}%")
    d.metric("Supplier risk index",f"{f.supplier_risk_score.mean()*100:.0f}/100")
    x,y=st.columns(2)
    x.plotly_chart(px.bar(f.groupby("region",as_index=False).delivery_delay_days.mean(),x="region",y="delivery_delay_days",title="Average delivery delay by region",color_discrete_sequence=["#39D98A"]),use_container_width=True)
    y.plotly_chart(px.pie(f.inventory_status.value_counts().rename_axis("status").reset_index(name="records"),names="status",values="records",hole=.55,title="Inventory health",color_discrete_sequence=["#39D98A","#D6BD8A","#D66B55"]),use_container_width=True)
elif page=="Demand & Inventory":
    st.subheader("Demand and inventory intelligence")
    x,y=st.columns(2)
    x.plotly_chart(px.scatter(f,x="demand_units",y="inventory_units",color="inventory_status",size="unit_cost",hover_data=["supplier","category"],title="Demand vs inventory"),use_container_width=True)
    y.plotly_chart(px.bar(f.groupby("category",as_index=False)[["demand_units","inventory_units"]].sum(),x="category",y=["demand_units","inventory_units"],barmode="group",title="Category demand and stock"),use_container_width=True)
    watch=f.assign(inventory_gap=(f.demand_units-f.inventory_units).clip(lower=0)).sort_values("inventory_gap",ascending=False).head(25)
    st.dataframe(watch[["supplier","region","category","demand_units","inventory_units","inventory_gap","lead_time_days","inventory_status"]],use_container_width=True,hide_index=True)
    st.download_button("Export replenishment watchlist",watch.to_csv(index=False),"replenishment_watchlist.csv")
elif page=="Supplier Intelligence":
    st.subheader("Supplier performance scorecards")
    s=f.groupby("supplier",as_index=False).agg(records=("supplier","size"),on_time=("on_time_rate","mean"),quality=("quality_rate","mean"),lead_time=("lead_time_days","mean"),risk=("supplier_risk_score","mean"),delay=("delivery_delay_days","mean"))
    s["performance_score"]=((.4*s.on_time)+(.35*s.quality)+(.25*(1-s.risk)))*100
    st.plotly_chart(px.scatter(s,x="on_time",y="quality",size="records",color="risk",text="supplier",title="Supplier reliability and quality map",color_continuous_scale=["#39D98A","#D6BD8A","#D66B55"]),use_container_width=True)
    st.dataframe(s.round(3),use_container_width=True,hide_index=True)
    st.caption("Composite score is an illustrative project heuristic, not an industry-standard rating.")
elif page=="Risk & Anomalies":
    st.subheader("Operational risk and anomaly discovery")
    a,b=st.columns(2); a.metric("High / critical risk",int(f.risk_band.isin(["High","Critical"]).sum())); b.metric("Anomalous records",int(f.anomaly.sum()))
    st.plotly_chart(px.scatter(f,x="lead_time_days",y="unit_cost",color="anomaly",size="demand_units",hover_data=["supplier","region"],title="Unusual cost and lead-time patterns",color_discrete_map={True:"#D66B55",False:"#39D98A"}),use_container_width=True)
    st.dataframe(f[f.anomaly][["date","supplier","region","lead_time_days","unit_cost","supplier_risk_score"]].head(40),use_container_width=True,hide_index=True)
elif page=="Scenario Planner":
    st.subheader("What-if scenario planner")
    a,b,c=st.columns(3)
    dc=a.slider("Demand change (%)",-30,60,10,5); ic=b.slider("Inventory uplift (%)",0,100,20,5); lc=c.slider("Lead-time change (%)",-40,80,0,5)
    demand=f.demand_units.mean()*(1+dc/100); stock=f.inventory_units.mean()*(1+ic/100); lead=f.lead_time_days.mean()*(1+lc/100)
    coverage=stock/max(demand,1)
    risk=np.clip(f.supplier_risk_score.mean()+.18*(lead/max(f.lead_time_days.mean(),1)-1)-.12*ic/100,0,1)
    x,y,z=st.columns(3); x.metric("Projected stock coverage",f"{coverage:.2f}x"); y.metric("Projected lead time",f"{lead:.1f} days"); z.metric("Illustrative risk index",f"{risk*100:.0f}/100")
    st.info("Directional sensitivity estimate only. This is not an optimization solver or purchase-order recommendation.")
elif page=="Data Explorer":
    st.subheader("Operational data explorer")
    st.dataframe(f,use_container_width=True,hide_index=True)
    st.download_button("Download filtered records",f.to_csv(index=False),"supplychainiq_data.csv")
elif page=="Model Performance":
    st.subheader("Delivery-delay prediction laboratory")
    a,b,c=st.columns(3); a.metric("MAE",f"{mean_absolute_error(yte,pred):.2f} days"); b.metric("RMSE",f"{mean_squared_error(yte,pred)**.5:.2f} days"); c.metric("R²",f"{r2_score(yte,pred):.3f}")
    st.caption("Held-out evaluation on synthetic data; metrics do not establish real-world model performance.")
    sample=pd.DataFrame({"Actual delay":yte.values[:120],"Predicted delay":pred[:120].round(1)})
    st.plotly_chart(px.scatter(sample,x="Actual delay",y="Predicted delay",title="Actual vs predicted delay",color_discrete_sequence=["#39D98A"]),use_container_width=True)
else:
    st.subheader("Governance and responsible use")
    st.markdown("""
- All operational records are synthetic and generated in-app.
- No live ERP, warehouse, procurement, carrier, IoT or external supplier-risk integration exists.
- Random Forest estimates delivery delay; Isolation Forest flags unusual numeric patterns.
- Anomaly flags are review signals, not proof of supplier misconduct or failure.
- Supplier scoring and scenario calculations are illustrative heuristics.
- No automated purchasing, supplier selection or operational intervention is performed.
- Real deployment requires validated data, security controls, monitoring, governance and human oversight.
""")
    st.code("RandomForestRegressor(n_estimators=180, max_depth=12, min_samples_leaf=2)\nIsolationForest(contamination=0.06, n_estimators=160)\nTrain/test split: 75% / 25%",language="python")
st.divider()


st.markdown("""
<div class="sciq-footer">
  <div class="project">SUPPLYCHAINIQ AI — Enterprise Supply Chain Intelligence, Risk Analytics &amp; Inventory Optimization</div>
  <div class="eyebrow">DESIGNED &amp; DEVELOPED BY</div>
  <div class="name">Mohamed Riyaz S.</div>
  <div class="meta">Integrated M.Tech CSE | Business Analytics<br>VIT Chennai</div>
  <div class="meta" style="margin-top:9px;">Interactive portfolio project demonstrating machine learning, supplier intelligence and inventory risk analytics.</div>
  <div class="copyright">© 2026 Mohamed Riyaz S. | All Rights Reserved.</div>
</div>
""", unsafe_allow_html=True)
