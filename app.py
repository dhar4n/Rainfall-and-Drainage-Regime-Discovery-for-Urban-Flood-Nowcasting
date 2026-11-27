import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Urban Flood Intelligence", page_icon="🌧️", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path(__file__).resolve().parent
RESULTS, PROCESSED = ROOT / "results", ROOT / "processed"
FORECASTS = ROOT / "forecast_results"
EVALUATION = ROOT / "forecast_evaluation"
SIMULATION_SOURCE = PROCESSED / "hydraulic_ml.csv"

@st.cache_data
def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

hyd = read_csv(RESULTS / "hydraulic_kmeans.csv")
rain = read_csv(RESULTS / "rainfall_drainage_kmeans.csv")
dbscan = read_csv(RESULTS / "hydraulic_dbscan_sample.csv")
hyd_profile = read_csv(RESULTS / "hydraulic_cluster_profiles.csv")
rain_profile = read_csv(RESULTS / "rainfall_cluster_profiles.csv")
forecast_5m = read_csv(FORECASTS / "predictions_5min.csv")
forecast_metrics = read_csv(EVALUATION / "evaluation_summary.csv")
simulation_data = read_csv(SIMULATION_SOURCE)
if simulation_data.empty:
    simulation_data = hyd.copy()  # fallback: replay the processed hydraulic clustering output

def parse_time(df):
    if not df.empty and "time" in df:
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.dropna(subset=["time"]).sort_values("time")
    return df

hyd, rain = parse_time(hyd), parse_time(rain)
simulation_data = parse_time(simulation_data).reset_index(drop=True)
if not simulation_data.empty:
    for column in ["overflow_flow_rate", "chamber_water_level", "inflow_flow_rate", "rainfall_mean"]:
        if column not in simulation_data.columns:
            simulation_data[column] = np.nan
    if "sim_index" not in st.session_state:
        st.session_state.sim_index = len(simulation_data) - 1
else:
    st.session_state.sim_index = 0

def val(df, col):
    if col not in df: return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else None

def fmt(x, n=2):
    return "—" if x is None or pd.isna(x) else f"{x:,.{n}f}"

def current_replay_row():
    if simulation_data.empty:
        return None
    idx = int(st.session_state.get("sim_index", len(simulation_data) - 1))
    return simulation_data.iloc[idx % len(simulation_data)]

def card(label, value, note=""):
    st.markdown(f'<div class="kpi"><div class="kl">{label}</div><div class="kv">{value}</div><div class="kn">{note}</div></div>', unsafe_allow_html=True)

def heading(text):
    st.markdown(f'<div class="pt">{text}</div>', unsafe_allow_html=True)

def dark(fig, height=260):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(2,10,48,.35)",
        font=dict(color="#d8ecff", family="Microsoft YaHei, Noto Sans SC, Arial, sans-serif"), margin=dict(l=8,r=8,t=28,b=8),
        legend=dict(orientation="h",y=1.15),
        xaxis=dict(gridcolor="rgba(100,170,220,.13)"),
        yaxis=dict(gridcolor="rgba(100,170,220,.13)"))
    return fig

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Manrope:wght@500;600;700;800&display=swap');
:root{--cyan:#20e8ff;--blue:#176bff;--text:#e8f7ff}
html,body,[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 50% 0%,rgba(18,92,190,.28),transparent 45%),linear-gradient(135deg,#02072d,#061044 52%,#02072d);color:var(--text);font-family:'Inter','Manrope','Segoe UI',Arial,sans-serif}
[data-testid="stHeader"]{background:transparent}.block-container{padding:1rem 1.25rem 2rem;max-width:1800px}#MainMenu,footer{visibility:hidden}
h1,h2,h3{font-family:'Inter','Manrope','Segoe UI',Arial,sans-serif,sans-serif!important;color:#eafaff!important}
.topbar{border-bottom:1px solid rgba(22,234,255,.65);padding:4px 4px 14px;margin-bottom:15px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.brand{font-family:'Manrope','Inter','Segoe UI',Arial,sans-serif;font-size:clamp(17px,2vw,27px);font-weight:800;letter-spacing:.45px;color:#e8fbff;text-shadow:0 0 12px rgba(0,217,255,.28)}
.sub{font-size:12px;color:#91c7e8;letter-spacing:2px}.clock{font-family:'Inter','Manrope','Segoe UI',Arial,sans-serif;color:#19eaff;font-size:17px}
.panel{background:linear-gradient(145deg,rgba(8,28,91,.96),rgba(3,12,57,.96));border:1px solid rgba(20,225,255,.75);box-shadow:0 0 12px rgba(0,205,255,.13),inset 0 0 18px rgba(0,100,255,.08);border-radius:5px;padding:13px 14px;margin-bottom:13px}
.pt{font-family:'Manrope','Inter','Segoe UI',Arial,sans-serif;color:#c8f8ff;font-size:13px;font-weight:800;letter-spacing:1.15px;border-bottom:1px solid rgba(23,220,255,.25);padding-bottom:9px;margin-bottom:12px}
.monitor-title{font-family:'Manrope','Inter','Segoe UI',Arial,sans-serif!important;font-size:14px;font-weight:800;letter-spacing:1.25px;color:#e9fbff;text-transform:uppercase}
.monitor-icon{font-size:28px;line-height:1;margin-bottom:8px}
.forecast{border:1px solid rgba(35,199,255,.38);border-radius:10px;background:linear-gradient(145deg,rgba(15,43,111,.88),rgba(5,20,68,.92));padding:16px 8px;text-align:center;min-height:126px;box-shadow:0 5px 18px rgba(0,0,0,.12);transition:transform .18s ease,border-color .18s ease}
.forecast:hover{transform:translateY(-2px);border-color:rgba(32,232,255,.8)}
.forecast-icon{font-size:31px;line-height:1.2;margin:2px 0 10px}
.forecast-label{font-family:'Inter','Segoe UI',Arial,sans-serif;color:#dff8ff;font-size:11px;font-weight:700;letter-spacing:.8px}

.alert{background:linear-gradient(120deg,rgba(8,41,116,.9),rgba(13,18,91,.9));border:1px solid rgba(0,220,255,.35);padding:15px 10px;text-align:center;border-radius:8px;margin-bottom:12px}
.kpi{background:linear-gradient(135deg,#102b82,#07164b);border:1px solid rgba(28,190,255,.45);border-radius:7px;padding:12px 10px;min-height:82px;margin-bottom:9px}
.kl{color:#9ec8e8;font-size:12px;font-weight:600}.kv{font-family:'Inter','Manrope','Segoe UI',Arial,sans-serif;color:#ffd928;font-size:21px;font-weight:700}.kn{color:#20e8ff;font-size:11px}
.forecast{border:1px solid rgba(35,199,255,.38);border-radius:6px;background:rgba(13,34,100,.72);padding:13px 5px;text-align:center}
[data-testid="stPlotlyChart"]{border-radius:6px}
.stTabs [data-baseweb="tab"]{background:#0b205f;border:1px solid rgba(20,225,255,.3);color:#bdeeff;border-radius:4px}
.stTabs [aria-selected="true"]{background:#0e4a91!important;color:white!important}
.small{font-size:11px;color:#83a9ce}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""<div class="topbar"><div><div class="brand">URBAN FLOOD INTELLIGENCE & DRAINAGE COMMAND</div><div class="sub">RAINFALL • HYDRAULICS • CLUSTER DISCOVERY • ANOMALY MONITORING</div></div><div style="text-align:right"><div class="clock">{datetime.now().strftime("%H:%M:%S")}</div><div class="small">LOCAL DASHBOARD TIME</div></div></div>""", unsafe_allow_html=True)

if hyd.empty:
    st.error("Hydraulic results not found. Expected results/hydraulic_kmeans.csv")
    st.info("Place this app.py in your project root beside the results/ and processed/ folders.")
    st.stop()

hyd_clusters = hyd["kmeans_cluster"].nunique() if "kmeans_cluster" in hyd else 0
rain_clusters = rain["kmeans_cluster"].nunique() if "kmeans_cluster" in rain else 0
noise = int(dbscan["is_anomaly"].astype(str).str.lower().isin(["true","1"]).sum()) if "is_anomaly" in dbscan else 0

left, center, right = st.columns([1.02,2.15,1.02], gap="small")
with left:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    heading("⚠ MONITORING & ALERT CENTRE")
    @st.fragment(run_every="2s" if st.session_state.get("sim_playing", False) else None)
    def render_hydraulic_kpis():
        # Advance the shared historical replay index here. The simulation tab
        # reads the same index, so its chart and these cards stay synchronized.
        if st.session_state.get("sim_playing", False) and not simulation_data.empty:
            st.session_state.sim_index = (int(st.session_state.get("sim_index", 0)) + 1) % len(simulation_data)
        replay_row = current_replay_row()
        if replay_row is not None:
            chamber_value = replay_row.get("chamber_water_level")
            inflow_value = replay_row.get("inflow_flow_rate")
            overflow_value = replay_row.get("overflow_flow_rate")
            rainfall_value = replay_row.get("rainfall_mean")
            timestamp_note = replay_row["time"].strftime("%Y-%m-%d %H:%M UTC") if pd.notna(replay_row.get("time")) else "Historical replay"
            note = "Historical replay • " + timestamp_note
        else:
            chamber_value = val(hyd, "chamber_water_level")
            inflow_value = val(hyd, "inflow_flow_rate")
            overflow_value = val(hyd, "overflow_flow_rate")
            rainfall_value = val(rain, "rainfall_mean")
            note = "Latest processed observation"
        st.markdown(f'<div class="alert"><div class="monitor-icon">🌊</div><div class="monitor-title">HYDRAULIC MONITORING</div><div class="small" style="margin-top:7px">Chamber water level</div><div class="kv" style="font-size:25px;margin:5px 0">{fmt(chamber_value,3)}</div><div style="color:#65e6b5;font-weight:700;font-size:11px;letter-spacing:.7px">● REGIME DISCOVERY ACTIVE</div><div class="small">{note}</div></div>',unsafe_allow_html=True)
        a,b=st.columns(2)
        with a: card("INFLOW RATE",fmt(inflow_value),"Replayed observation")
        with b: card("OVERFLOW RATE",fmt(overflow_value),"Replayed observation")
        c,d=st.columns(2)
        with c: card("RAINFALL MEAN",fmt(rainfall_value,3),"Replayed observation")
        with d: card("DBSCAN OUTLIERS",f"{noise:,}","Sampled records • fixed count")
    render_hydraulic_kpis()
    heading("DATASET HEALTH")
    st.markdown(f'<div class="small">Hydraulic observations</div><div class="kv">{len(hyd):,}</div><div class="small">Rainfall + drainage observations</div><div class="kv">{len(rain):,}</div><div class="small">Missing rainfall values are not treated as zero.</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

with center:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    heading("◈ DATA ANALYSIS & PREDICTION")
    if not rain.empty:
        df=rain.copy()
        if len(df)>7:
            # Seven representative time bins for a compact dashboard view.
            df=df.iloc[::max(1,len(df)//7)].head(7)
        xlabels=[t.strftime("%m/%d") if pd.notna(t) else str(i) for i,t in enumerate(df["time"])]
        flow_series=pd.to_numeric(df.get("overflow_flow_rate",pd.Series(np.zeros(len(df)))),errors="coerce").fillna(0)
        level_series=pd.to_numeric(df.get("inflow_water_level",pd.Series(np.zeros(len(df)))),errors="coerce").fillna(0)
        rain_series=pd.to_numeric(df.get("rainfall_mean",pd.Series(np.zeros(len(df)))),errors="coerce").fillna(0)
        # Actual values only: future-looking yellow bars are not fabricated.
        bar_colors=["#176BFF" if i < len(df) else "#FFD21F" for i in range(len(df))]
        fig=go.Figure()
        fig.add_trace(go.Bar(x=xlabels,y=flow_series,name="Overflow flow",marker_color=bar_colors,opacity=.92))
        fig.add_trace(go.Scatter(x=xlabels,y=level_series,name="Water level",mode="lines+markers",
            line=dict(color="#20E8FF",width=3,shape="spline"),marker=dict(size=6,color="#20E8FF"),yaxis="y2"))
        fig.update_layout(height=285,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(2,10,48,.2)",
            font=dict(color="#d8ecff",family="Microsoft YaHei, Noto Sans SC, Arial, sans-serif"),
            margin=dict(l=10,r=10,t=25,b=5),legend=dict(orientation="h",y=1.15),
            xaxis=dict(gridcolor="rgba(100,170,220,.13)"),
            yaxis=dict(title="Overflow flow",gridcolor="rgba(100,170,220,.13)"),
            yaxis2=dict(title="Water level",overlaying="y",side="right",showgrid=False))
        st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
    else:
        st.warning("Rainfall-linked dataset is unavailable.")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="panel">',unsafe_allow_html=True)
    heading("◈ HISTORICAL HYDRAULIC DATA ANALYSIS")
    df=hyd.iloc[::max(1,len(hyd)//12)].copy()
    xlabels=[t.strftime("%H:%M") if pd.notna(t) else str(i) for i,t in enumerate(df["time"])]
    fig2=go.Figure()
    # Bars are colored by the observed cluster, matching the reference's multicolor indicator bars.
    if "overflow_flow_rate" in df:
        cl=df["kmeans_cluster"] if "kmeans_cluster" in df else pd.Series(np.zeros(len(df)))
        palette={0:"#176BFF",1:"#6D65D9",2:"#FF3D36",3:"#FFD21F"}
        barcols=[palette.get(int(v),"#176BFF") for v in cl.fillna(0)]
        fig2.add_trace(go.Bar(x=xlabels,y=pd.to_numeric(df["overflow_flow_rate"],errors="coerce").fillna(0),
            name="Overflow flow",marker_color=barcols,opacity=.85))
    if "chamber_water_level" in df:
        fig2.add_trace(go.Scatter(x=xlabels,y=df["chamber_water_level"],mode="lines",
            name="Chamber water level",line=dict(color="#20E8FF",width=3,shape="spline"),yaxis="y2"))
    if "inflow_water_level" in df:
        fig2.add_trace(go.Scatter(x=xlabels,y=df["inflow_water_level"],mode="lines",
            name="Inflow water level",line=dict(color="#FF8A22",width=1.5,dash="dot"),yaxis="y2"))
    fig2.update_layout(height=255,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(2,10,48,.2)",
        font=dict(color="#d8ecff",family="Microsoft YaHei, Noto Sans SC, Arial, sans-serif"),
        margin=dict(l=10,r=10,t=25,b=5),legend=dict(orientation="h",y=1.15),
        xaxis=dict(gridcolor="rgba(100,170,220,.13)"),
        yaxis=dict(title="Overflow flow",gridcolor="rgba(100,170,220,.13)"),
        yaxis2=dict(title="Water level",overlaying="y",side="right",showgrid=False))
    st.plotly_chart(fig2,width="stretch",config={"displayModeBar":False})
    st.markdown('</div>',unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    heading("▣ SYSTEM & MODEL STATUS")
    card("DATA PIPELINE","● READY","Processed CSV files loaded")
    card("K-MEANS CLUSTERING","ACTIVE","Unsupervised regime discovery")
    card("HYDRAULIC REGIMES",str(hyd_clusters),"Clusters discovered")
    card("RAIN + DRAINAGE REGIMES",str(rain_clusters),"Matched subset clusters")
    st.markdown('<div class="small">Live CCTV, weather forecasts and sensor telemetry are not connected to this academic prototype.</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    heading("◈ PROJECT OUTLOOK")
    cols=st.columns(3)
    for col,ico,label in zip(cols,["🌤️","☁️","🌧️"],["MODEL REVIEW","DATA CHECK","FUTURE WORK"]):
        with col: st.markdown(f'<div class="forecast"><div class="forecast-icon">{ico}</div><div class="forecast-label">{label}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="small" style="margin-top:10px">Workflow outlook only—not a meteorological forecast.</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

st.markdown('<div class="panel">',unsafe_allow_html=True)
heading("◈ ANALYTICS & CLUSTER EXPLORER")
t1,t2,t3,t4,t5,t6=st.tabs(["Hydraulic regimes","Rainfall regimes","Anomaly sample","Data records","5-min forecast","Live simulation"])
with t1:
    x,y=st.columns(2)
    if "kmeans_cluster" in hyd:
        counts=hyd.kmeans_cluster.value_counts().sort_index().rename_axis("Cluster").reset_index(name="Records")
        with x: st.plotly_chart(dark(px.bar(counts,x="Cluster",y="Records",color="Cluster",text="Records",color_discrete_sequence=["#1b7cff","#20dfff","#ff9a32","#ff4d69"]),260),width="stretch")
    with y:
        if not hyd_profile.empty: st.dataframe(hyd_profile,hide_index=True,width="stretch",height=260)
        else: st.info("Hydraulic cluster profile CSV not found.")
with t2:
    x,y=st.columns(2)
    if "kmeans_cluster" in rain:
        counts=rain.kmeans_cluster.value_counts().sort_index().rename_axis("Cluster").reset_index(name="Records")
        with x: st.plotly_chart(dark(px.bar(counts,x="Cluster",y="Records",color="Cluster",text="Records",color_discrete_sequence=["#1b7cff","#20dfff","#ff9a32","#ff4d69"]),260),width="stretch")
    with y:
        if not rain_profile.empty: st.dataframe(rain_profile,hide_index=True,width="stretch",height=260)
        else: st.info("Rainfall cluster profile CSV not found.")
with t3:
    st.caption("DBSCAN was applied to a 5,000-row hydraulic sample. An outlier is not automatically a flood event.")
    if not dbscan.empty: st.dataframe(dbscan.head(300),width="stretch",height=300)
    else: st.info("DBSCAN result CSV not found.")
with t4: st.dataframe(hyd.tail(200),width="stretch",height=330)
with t5:
    st.caption("Historical holdout evaluation only. This is not a live forecast or operational flood warning.")
    if not forecast_5m.empty and {"time", "actual_overflow", "predicted_overflow", "baseline_prediction"}.issubset(forecast_5m.columns):
        plot = forecast_5m.copy()
        plot["time"] = pd.to_datetime(plot["time"], errors="coerce", utc=True)
        plot = plot.dropna(subset=["time"]).head(1000)
        figf = go.Figure()
        figf.add_trace(go.Scatter(x=plot["time"], y=plot["actual_overflow"], name="Actual overflow", mode="lines", line=dict(color="#20E8FF", width=2)))
        figf.add_trace(go.Scatter(x=plot["time"], y=plot["predicted_overflow"], name="Random Forest estimate", mode="lines", line=dict(color="#FFD21F", width=2)))
        figf.add_trace(go.Scatter(x=plot["time"], y=plot["baseline_prediction"], name="Persistence baseline", mode="lines", line=dict(color="#FF8A22", width=1.5, dash="dot")))
        st.plotly_chart(dark(figf, 330), width="stretch", config={"displayModeBar": False})
        st.caption("Chart shows up to the first 1,000 chronological test observations, not the complete test period.")
    else:
        st.info("Forecast predictions not found. Run forecasting.py and place forecast_results/predictions_5min.csv beside app.py.")
    if not forecast_metrics.empty:
        st.subheader("Forecast evaluation metrics")
        st.dataframe(forecast_metrics, hide_index=True, width="stretch")
    else:
        st.info("Evaluation summary not found. Expected forecast_evaluation/evaluation_summary.csv.")
with t6:
    st.caption("DEMONSTRATION ONLY: historical hydraulic records are replayed as if they arrive live. No physical sensors or live feeds are connected.")
    if simulation_data.empty or "time" not in simulation_data.columns:
        st.warning("Simulation data unavailable. Expected processed/hydraulic_ml.csv or results/hydraulic_kmeans.csv.")
    else:
        # Replay every chronological observation in order; do not downsample.
        # The previous sampling shortcut skipped many records and caused
        # multi-day jumps between consecutive 2-second refreshes.
        sim = simulation_data
        if not sim.empty:
            threshold_source = pd.to_numeric(simulation_data.get("overflow_flow_rate", pd.Series(dtype=float)), errors="coerce").dropna()
            high_threshold = float(threshold_source.quantile(0.95)) if len(threshold_source) else 0.6

            st.markdown("**Historical replay controls**")
            playing = st.checkbox("▶ Play simulated historical feed (updates every 2 seconds)", value=False, key="sim_playing")
            next_col, info_col = st.columns([1, 3])
            with next_col:
                if st.button("Next historical reading", key="sim_next"):
                    st.session_state.sim_index = (st.session_state.sim_index + 1) % len(sim)
                    st.rerun()
            with info_col:
                st.caption("The monitoring KPI cards and this simulation panel read the same replay record and update together.")

            @st.fragment(run_every="2s" if playing else None)
            def render_live_simulation():
                current = sim.iloc[int(st.session_state.sim_index) % len(sim)]
                overflow_now = pd.to_numeric(pd.Series([current.get("overflow_flow_rate")]), errors="coerce").iloc[0]
                level_now = pd.to_numeric(pd.Series([current.get("chamber_water_level")]), errors="coerce").iloc[0]
                if pd.notna(overflow_now) and overflow_now >= high_threshold:
                    status, status_color = "HIGH OVERFLOW — DEMO THRESHOLD EXCEEDED", "#ff4d69"
                elif pd.notna(overflow_now) and overflow_now >= 0.75 * high_threshold:
                    status, status_color = "ELEVATED OVERFLOW — REVIEW", "#ffb32d"
                else:
                    status, status_color = "NO DEMO THRESHOLD EXCEEDED", "#00e676"
                st.markdown(f'<div class="alert"><b style="color:{status_color}">{status}</b><br><span class="small">Simulated status based on a historical-data percentile threshold; not a real flood alert.</span></div>', unsafe_allow_html=True)
                k1,k2,k3,k4=st.columns(4)
                with k1: card("REPLAY TIME", current["time"].strftime("%Y-%m-%d %H:%M UTC"), "Historical timestamp")
                with k2: card("OVERFLOW FLOW", fmt(overflow_now), "Replayed observation")
                with k3: card("CHAMBER LEVEL", fmt(level_now,3), "Replayed observation")
                with k4: card("INFLOW RATE", fmt(current.get("inflow_flow_rate")), "Replayed observation")
                st.caption(f"Demonstration threshold: 95th percentile of historical overflow = {high_threshold:.4f}. This is a statistical display threshold, not an engineering flood-risk limit.")
                idx = int(st.session_state.sim_index)
                window = sim.iloc[max(0, idx-39):idx+1]
                figsim = go.Figure()
                figsim.add_trace(go.Scatter(x=window["time"], y=window["overflow_flow_rate"], name="Historical overflow replay", mode="lines+markers", line=dict(color="#20E8FF", width=2)))
                figsim.add_trace(go.Scatter(x=[current["time"]], y=[overflow_now], name="Current replay point", mode="markers", marker=dict(color="#FFD21F", size=11)))
                st.plotly_chart(dark(figsim, 300), width="stretch", config={"displayModeBar":False})
                st.caption(f"Replay record {idx+1:,} of {len(sim):,}. Turn off Play to pause. The sequence loops to the beginning when it reaches the end.")

            render_live_simulation()
st.markdown('</div>',unsafe_allow_html=True)

d1,d2,d3=st.columns(3)
for col,df,name,label in [(d1,hyd,"hydraulic_kmeans_results.csv","Download hydraulic results"),(d2,rain,"rainfall_drainage_kmeans_results.csv","Download rainfall results"),(d3,dbscan,"hydraulic_dbscan_sample.csv","Download DBSCAN sample")]:
    with col:
        if not df.empty: st.download_button(label,df.to_csv(index=False).encode("utf-8"),name,mime="text/csv",width="stretch")

st.markdown('<div style="text-align:center;color:#668eb8;font-size:11px;margin-top:20px;border-top:1px solid rgba(22,234,255,.2);padding-top:10px">URBAN FLOOD NOWCASTING • UNSUPERVISED MACHINE LEARNING RESEARCH PROTOTYPE<br>Cluster labels are data-driven regimes, not validated flood-risk categories.</div>',unsafe_allow_html=True)
