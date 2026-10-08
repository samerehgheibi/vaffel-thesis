"""
Wind Failure Risk — Interactive Web Application
Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
import io

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wind Failure Risk — Sicily Grid",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F3F7FA; }
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    h1 { color: #065A82; }
    h2, h3 { color: #102A43; }
    .metric-card {
        background: white;
        border-left: 4px solid #065A82;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .risk-low    { border-left-color: #1A7A4A !important; }
    .risk-medium { border-left-color: #F4A300 !important; }
    .risk-high   { border-left-color: #BF4026 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data ────────────────────────────────────────────────────────────────
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.abspath(__file__)))

LINE_NAMES = {
    "1155116743": "Fulgatore – Partanna",
    "338850183":  "Line 338850183",
    "338856348":  "Line 338856348",
    "375907989":  "Line 375907989",
    "375922074":  "Melilli – Ragusa",
    "375916002":  "Priolo – Chiaramonte Gulfi",
    "456822447":  "Sorgente – Villafranca – Scilla",
    "375924539":  "Sorgente – Paternò",
}
LINE_VOLTAGE = {
    "1155116743": 220, "338850183": 220, "338856348": 220, "375907989": 220,
    "375922074":  220, "375916002": 380, "456822447": 380, "375924539": 380,
}
# Approximate midpoint coordinates (lon, lat)
LINE_COORDS = {
    "1155116743": (13.15, 37.62),
    "338850183":  (13.55, 37.48),
    "338856348":  (13.68, 37.42),
    "375907989":  (15.15, 37.58),
    "375922074":  (14.75, 37.12),
    "375916002":  (14.88, 37.22),
    "456822447":  (15.32, 37.76),
    "375924539":  (14.90, 37.65),
}

@st.cache_data
def load_params():
    df = pd.read_csv(f"{BASE}/results/step8_params.csv", dtype={"way_id": str})
    df["name"]    = df["way_id"].map(LINE_NAMES)
    df["voltage"] = df["way_id"].map(LINE_VOLTAGE)
    df["lon"]     = df["way_id"].map(lambda x: LINE_COORDS.get(x, (14.0, 37.5))[0])
    df["lat"]     = df["way_id"].map(lambda x: LINE_COORDS.get(x, (14.0, 37.5))[1])
    return df

def p_failure(wind_speed, mu, sigma, L_km, w_thresh=10.0):
    """P(failure) at a given wind speed."""
    E = L_km * max(wind_speed - w_thresh, 0.0) ** 3
    if E <= 0:
        return 0.0
    return float(norm.cdf((np.log(E) - np.log(mu)) / sigma))

def fragility_curve(mu, sigma, L_km, w_thresh=10.0, n=300):
    ws = np.linspace(w_thresh, 40, n)
    E  = L_km * np.maximum(ws - w_thresh, 0) ** 3
    p  = np.where(E > 0, norm.cdf((np.log(np.maximum(E, 1e-300)) - np.log(mu)) / sigma), 0.0)
    return ws, p

def risk_label(p):
    if p < 0.1:  return "Low",    "#1A7A4A"
    if p < 0.4:  return "Medium", "#F4A300"
    return               "High",  "#BF4026"

# ── Load ────────────────────────────────────────────────────────────────
df = load_params()

# ════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Flag_of_Italy.svg/320px-Flag_of_Italy.svg.png", width=80)
    st.markdown("## ⚡ Wind Failure Risk\n**Sicily Transmission Grid**")
    st.markdown("---")

    wind_speed = st.slider(
        "🌬️  Wind Speed  (m/s)",
        min_value=5.0, max_value=40.0, value=15.0, step=0.5,
        help="Move the slider to see how failure probability changes with wind speed"
    )

    w_thresh = st.slider(
        "🔧  Wind Threshold  (m/s)",
        min_value=5.0, max_value=17.0, value=10.0, step=1.0,
        help="Minimum wind speed that causes stress on the line"
    )

    voltage_filter = st.multiselect(
        "🔌  Voltage Level",
        options=[220, 380], default=[220, 380],
        format_func=lambda v: f"{v} kV"
    )

    st.markdown("---")
    st.markdown("**About**\n\nThis application estimates the probability of wind-induced failure for 8 transmission lines in Sicily using a Bayesian fragility model.")
    st.markdown("*Samereh Gheibi · Politecnico di Torino*")

# ── Filter ──────────────────────────────────────────────────────────────
dff = df[df["voltage"].isin(voltage_filter)].copy()
dff["p_fail"] = dff.apply(
    lambda r: p_failure(wind_speed, r["mu"], r["sigma"], r["L_km"], w_thresh), axis=1)
dff["risk"],  dff["risk_color"] = zip(*dff["p_fail"].map(risk_label))
dff["p_pct"]  = (dff["p_fail"] * 100).round(1)
dff["p_str"]  = dff["p_pct"].map(lambda x: f"{x:.1f} %")

# ════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════
st.title("⚡ Wind-Dependent Failure Risk — Sicily Power Grid")
st.markdown(f"**Wind speed: {wind_speed} m/s  ·  Threshold: {w_thresh} m/s  ·  Showing {len(dff)} lines**")

# ════════════════════════════════════════════════════════════════════════
# ROW 1 — KPI cards
# ════════════════════════════════════════════════════════════════════════
cols = st.columns(len(dff))
for col, (_, row) in zip(cols, dff.sort_values("p_fail", ascending=False).iterrows()):
    risk, color = row["risk"], row["risk_color"]
    col.markdown(f"""
    <div class="metric-card risk-{'low' if risk=='Low' else 'medium' if risk=='Medium' else 'high'}">
        <div style="font-size:11px;color:#6B8594;font-weight:600">{row['voltage']} kV</div>
        <div style="font-size:13px;font-weight:700;color:#102A43">{row['name'][:22]}</div>
        <div style="font-size:24px;font-weight:800;color:{color}">{row['p_str']}</div>
        <div style="font-size:11px;color:{color};font-weight:600">{risk} Risk</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# ROW 2 — Map + Fragility curves
# ════════════════════════════════════════════════════════════════════════
col_map, col_chart = st.columns([1.1, 1.0])

with col_map:
    st.subheader("📍 Line Locations — Sicily")

    fig_map = go.Figure()

    # Sicily outline (approximate)
    sic_lon = [12.43,12.60,12.85,13.10,13.40,13.75,14.05,14.30,14.58,14.90,
               15.15,15.42,15.65,15.72,15.68,15.55,15.40,15.20,15.10,14.98,
               14.75,14.50,14.25,14.00,13.80,13.55,13.30,13.10,12.90,12.70,
               12.55,12.43,12.43]
    sic_lat = [37.80,37.94,38.20,38.28,38.26,38.11,38.08,38.10,38.05,37.98,
               37.94,38.02,37.96,37.82,37.65,37.55,37.35,37.05,36.75,36.65,
               36.68,36.82,36.85,36.82,36.90,37.05,37.10,37.12,37.10,37.18,
               37.35,37.55,37.80]
    fig_map.add_trace(go.Scattergeo(
        lon=sic_lon, lat=sic_lat, mode="lines",
        line=dict(color="#065A82", width=1.5),
        fill="toself", fillcolor="rgba(227,240,248,0.6)",
        name="Sicily", showlegend=False,
    ))

    # Line markers
    for _, row in dff.iterrows():
        risk, color = row["risk"], row["risk_color"]
        fig_map.add_trace(go.Scattergeo(
            lon=[row["lon"]], lat=[row["lat"]],
            mode="markers+text",
            marker=dict(size=18, color=color,
                        line=dict(color="white", width=1.5)),
            text=[f"{row['p_pct']}%"],
            textfont=dict(size=9, color="white", family="Arial Black"),
            textposition="middle center",
            name=row["name"],
            hovertemplate=(
                f"<b>{row['name']}</b><br>"
                f"Voltage: {row['voltage']} kV<br>"
                f"Length: {row['L_km']:.1f} km<br>"
                f"P(failure): {row['p_str']}<br>"
                f"Risk: {risk}<extra></extra>"
            ),
        ))

    fig_map.update_layout(
        geo=dict(
            scope="europe",
            center=dict(lon=14.0, lat=37.55),
            projection_scale=35,
            showland=True, landcolor="#F0F4F0",
            showocean=True,  oceancolor="#A8D8EA",
            showcoastlines=True, coastlinecolor="#065A82",
            showframe=False,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_chart:
    st.subheader("📈 Fragility Curves — All Lines")

    fig_fc = go.Figure()
    colors_fc = px.colors.qualitative.Bold
    for i, (_, row) in enumerate(dff.iterrows()):
        ws, p = fragility_curve(row["mu"], row["sigma"], row["L_km"], w_thresh)
        fig_fc.add_trace(go.Scatter(
            x=ws, y=p * 100,
            mode="lines", name=row["name"][:28],
            line=dict(width=2.5, color=colors_fc[i % len(colors_fc)]),
            hovertemplate=f"<b>{row['name'][:28]}</b><br>Wind: %{{x:.1f}} m/s<br>P(failure): %{{y:.1f}}%<extra></extra>",
        ))

    # vertical line for current wind speed
    fig_fc.add_vline(
        x=wind_speed, line_dash="dash", line_color="#F4A300", line_width=2,
        annotation_text=f"  {wind_speed} m/s",
        annotation_font_color="#F4A300", annotation_font_size=12,
    )

    fig_fc.update_layout(
        xaxis=dict(title="Wind Speed  (m/s)", range=[w_thresh, 40], gridcolor="#E0E8F0"),
        yaxis=dict(title="Probability of Failure  (%)", range=[0, 102], gridcolor="#E0E8F0"),
        legend=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=380,
        plot_bgcolor="#F8FBFD",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# ROW 3 — Risk table + bar chart
# ════════════════════════════════════════════════════════════════════════
st.markdown("---")
col_tbl, col_bar = st.columns([1.0, 1.0])

with col_tbl:
    st.subheader("📋 Risk Table")

    tbl = dff[["name","voltage","L_km","p_str","risk"]].copy()
    tbl.columns = ["Line", "Voltage (kV)", "Length (km)", "P(failure)", "Risk Level"]
    tbl["Length (km)"] = tbl["Length (km)"].round(1)
    tbl = tbl.sort_values("P(failure)", ascending=False).reset_index(drop=True)

    def color_risk(val):
        if val == "High":   return "background-color:#FDDBD5;color:#7B1F1F;font-weight:bold"
        if val == "Medium": return "background-color:#FEF3CD;color:#856004;font-weight:bold"
        return                     "background-color:#D4EDDA;color:#155724;font-weight:bold"

    st.dataframe(
        tbl.style.applymap(color_risk, subset=["Risk Level"]),
        use_container_width=True, hide_index=True,
    )

with col_bar:
    st.subheader(f"📊 Failure Probability at {wind_speed} m/s")

    dff_sorted = dff.sort_values("p_fail", ascending=True)
    fig_bar = go.Figure(go.Bar(
        x=dff_sorted["p_pct"],
        y=dff_sorted["name"].str[:28],
        orientation="h",
        marker_color=dff_sorted["risk_color"].tolist(),
        text=dff_sorted["p_str"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>P(failure): %{x:.1f}%<extra></extra>",
    ))
    fig_bar.update_layout(
        xaxis=dict(title="P(failure)  (%)", range=[0, 105], gridcolor="#E0E8F0"),
        yaxis=dict(title=""),
        margin=dict(l=0, r=60, t=10, b=0),
        height=340,
        plot_bgcolor="#F8FBFD",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# ROW 4 — Download
# ════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("⬇️  Export Data")

col_d1, col_d2, col_d3 = st.columns(3)

# Current snapshot CSV
snapshot = dff[["name","voltage","L_km","mu","sigma","lambda_B","p_pct","risk"]].copy()
snapshot.columns = ["Line","Voltage_kV","Length_km","mu","sigma","lambda_B","P_failure_%","Risk"]
csv1 = snapshot.to_csv(index=False).encode("utf-8")
col_d1.download_button(
    "📥  Current Snapshot (CSV)",
    data=csv1,
    file_name=f"risk_snapshot_wind{int(wind_speed)}ms.csv",
    mime="text/csv",
    use_container_width=True,
)

# Full fragility table (wind 5–40 m/s, step 0.5)
ws_range = np.arange(5, 40.5, 0.5)
rows_full = []
for _, row in df.iterrows():
    for ws in ws_range:
        p = p_failure(ws, row["mu"], row["sigma"], row["L_km"], w_thresh)
        rows_full.append({"Line": row["name"], "Wind_mps": ws, "P_failure": round(p, 6)})
df_full = pd.DataFrame(rows_full)
csv2 = df_full.to_csv(index=False).encode("utf-8")
col_d2.download_button(
    "📥  Full Fragility Table (CSV)",
    data=csv2,
    file_name="fragility_curves_full.csv",
    mime="text/csv",
    use_container_width=True,
)

# Model parameters
csv3 = df[["way_id","name","voltage","L_km","mu","sigma","lambda_B"]].to_csv(index=False).encode("utf-8")
col_d3.download_button(
    "📥  Model Parameters (CSV)",
    data=csv3,
    file_name="model_parameters.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption("Wind Failure Risk Application · Samereh Gheibi · Politecnico di Torino — DENERG · 2024")
