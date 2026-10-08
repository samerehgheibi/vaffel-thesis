"""
MAP_HEATMAP.py — Interactive wind heatmap + transmission lines
- Background: ERA5 max wind speed (2013-2023, 100 m)
- Lines color-coded: included (blue/red) vs excluded (grey)
- Click any line for details
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import pandas as pd
import folium
from folium.plugins import HeatMap, MiniMap
from config import (
    WIND_MAX_CACHE, OSM_CACHE, TOWERS_CSV,
    INCLUDED, EXCLUDED, VOLTAGE_MAP, NAME_MAP,
    SICILY_BBOX, COLOR
)

OUT_HTML = os.path.join(BASE, "results", "map_heatmap.html")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. WIND MAX DATA  →  heatmap points
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading max wind...")
wind = pd.read_parquet(WIND_MAX_CACHE, engine="fastparquet")   # columns: lat, lon, ws
ws_min = float(wind["ws"].min())
ws_max = float(wind["ws"].max())
print(f"  Min: {ws_min:.1f} m/s   Max: {ws_max:.1f} m/s   Points: {len(wind):,}")

# normalize 0-1 for folium HeatMap
wind["intensity"] = (wind["ws"] - ws_min) / (ws_max - ws_min)
heat_data = wind[["lat", "lon", "intensity"]].values.tolist()

# wind grid spacing (0.05 deg) -> nearest-point lookup for per-line max wind
WIND_GRID = {(round(r.lat, 2), round(r.lon, 2)): r.ws for r in wind.itertuples()}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD OSM LINES + merge with thesis CSV metadata
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading OSM lines...")
with open(OSM_CACHE, encoding="utf-8") as f:
    elements = json.load(f)

df_csv = pd.read_csv(TOWERS_CSV)
csv_meta = df_csv.groupby("way_id").agg(
    circuits=("line_circuits", "first"),
    cables  =("line_cables",   "first"),
    length_km=("segment_km",   "sum"),
).reset_index()
csv_meta["way_id"] = csv_meta["way_id"].astype(str)
csv_dict = csv_meta.set_index("way_id").to_dict("index")

# per-line max wind speed (2013-2023): max over all towers along the way,
# matching the "Max Wind Speed (m/s)" column in All_116_OSM_Lines.xlsx
df_csv["lat_g"] = (df_csv["lat"] / 0.05).round(0) * 0.05
df_csv["lon_g"] = (df_csv["lon"] / 0.05).round(0) * 0.05
df_csv["lat_g"] = df_csv["lat_g"].round(2)
df_csv["lon_g"] = df_csv["lon_g"].round(2)
df_csv["ws"] = list(map(WIND_GRID.get, zip(df_csv["lat_g"], df_csv["lon_g"])))
LINE_MAX_WIND = df_csv.groupby("way_id")["ws"].max().to_dict()
LINE_MAX_WIND = {str(k): v for k, v in LINE_MAX_WIND.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. BUILD MAP
# ═══════════════════════════════════════════════════════════════════════════════
cx = (SICILY_BBOX["lat_min"] + SICILY_BBOX["lat_max"]) / 2
cy = (SICILY_BBOX["lon_min"] + SICILY_BBOX["lon_max"]) / 2

m = folium.Map(location=[cx, cy], zoom_start=8,
               tiles="CartoDB positron")

# — wind heatmap layer —
HeatMap(
    heat_data,
    name=f"ERA5 Max Wind Speed  [{ws_min:.1f}–{ws_max:.1f} m/s]",
    min_opacity=0.3,
    max_opacity=0.85,
    radius=18,
    blur=20,
    gradient={
        "0.0": "#ffffcc",
        "0.3": "#fed976",
        "0.5": "#fd8d3c",
        "0.7": "#e31a1c",
        "1.0": "#800026",
    },
).add_to(m)

# wind min/max info box
info_html = f"""
<div style="position:fixed; bottom:30px; left:60px; z-index:1000;
     background:white; padding:10px 14px; border-radius:8px;
     border:1px solid #ccc; font-size:13px; line-height:1.6">
  <b>ERA5 Max Wind Speed (100 m)</b><br>
  Period: 2013–2023<br>
  <span style="color:#ffffcc; background:#555; padding:1px 4px">&#9632;</span> Min: <b>{ws_min:.1f} m/s</b><br>
  <span style="color:#800026">&#9632;</span> Max: <b>{ws_max:.1f} m/s</b>
</div>
"""
m.get_root().html.add_child(folium.Element(info_html))

# — layer groups for lines —
grp_inc_220 = folium.FeatureGroup(name="220 kV — Included", show=True)
grp_inc_380 = folium.FeatureGroup(name="380 kV — Included", show=True)
grp_exc     = folium.FeatureGroup(name="Excluded lines",    show=True)
grp_nodes   = folium.FeatureGroup(name="Substation nodes",  show=True)

for el in elements:
    if el.get("type") != "way":
        continue
    tags = el.get("tags", {}); geom = el.get("geometry", [])
    if len(geom) < 2:
        continue

    volt_raw = tags.get("voltage", "0")
    volts = [int(v) for v in volt_raw.split(";") if v.strip().isdigit()]
    volt  = max(volts) if volts else 0
    if volt not in (220000, 380000):
        continue

    way_id   = str(el["id"])
    csv      = csv_dict.get(way_id, {})
    circuits = int(csv["circuits"]) if csv.get("circuits") else int(tags.get("circuits","1") or 1)
    length   = f"{csv['length_km']:.1f} km" if csv.get("length_km") else "—"
    name     = NAME_MAP.get(way_id, tags.get("name", "—"))

    included = way_id in INCLUDED
    excluded = way_id in EXCLUDED

    if included:
        dc    = circuits >= 2
        color = (COLOR["inc_220_dc"] if volt == 220000 and dc else
                 COLOR["inc_220_sc"] if volt == 220000 else
                 COLOR["inc_380_dc"] if dc else
                 COLOR["inc_380_sc"])
        weight = 5 if dc else 3
        status = "INCLUDED"
    elif excluded:
        color  = COLOR["exc"]
        weight = 2
        status = "EXCLUDED"
    else:
        continue   # not in analysis at all — skip

    coords = [[g["lat"], g["lon"]] for g in geom]

    popup_html = f"""
    <b>{name}</b><br>
    Way ID: {way_id}<br>
    Voltage: <b>{volt//1000} kV</b><br>
    Circuits: {circuits}<br>
    Length: {length}<br>
    Status: <b style="color:{'green' if included else 'grey'}">{status}</b>
    """

    line = folium.PolyLine(
        coords, color=color, weight=weight, opacity=0.9,
        tooltip=f"{volt//1000} kV | {name} | {status}",
        popup=folium.Popup(popup_html, max_width=250),
    )

    if included and volt == 220000:
        grp_inc_220.add_child(line)
    elif included and volt == 380000:
        grp_inc_380.add_child(line)
    else:
        grp_exc.add_child(line)

# — endpoint nodes of analysis lines (thesis CSV geometry, matches MAP_FINAL) —
df_nodes = pd.read_csv(TOWERS_CSV)
df_nodes["way_id"] = df_nodes["way_id"].astype(str)
df_nodes = df_nodes[df_nodes["way_id"].isin(INCLUDED | EXCLUDED)]

for way_id, grp in df_nodes.groupby("way_id", sort=False):
    grp = grp.reset_index(drop=True)
    volt     = int(grp["line_voltage"].iloc[0])
    circuits = int(grp["line_circuits"].iloc[0])
    included = way_id in INCLUDED
    name     = NAME_MAP.get(way_id, way_id)
    node_color = COLOR["node_inc"] if included else COLOR["node_exc"]

    line_max_ws = LINE_MAX_WIND.get(way_id)

    for pos, idx in [("Start", 0), ("End", -1)]:
        lat, lon = grp["lat"].iloc[idx], grp["lon"].iloc[idx]
        ws = line_max_ws
        ws_str = f"{ws:.1f} m/s" if ws is not None else "—"

        popup_html = f"""
        <b>{name}</b><br>
        Way ID: {way_id}<br>
        Node: {pos}<br>
        Voltage: <b>{volt//1000} kV</b><br>
        Circuits: {circuits} ({'double-circuit' if circuits >= 2 else 'single-circuit'})<br>
        Status: <b style="color:{'green' if included else 'grey'}">{'INCLUDED' if included else 'EXCLUDED'}</b><br>
        Max wind speed (2013-2023): <b>{ws_str}</b>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="#1a237e",
            fill=True,
            fill_color=node_color,
            fill_opacity=0.9,
            weight=2,
            tooltip=f"{name} — {pos} ({volt//1000} kV, {ws_str})",
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(grp_nodes)

grp_inc_220.add_to(m)
grp_inc_380.add_to(m)
grp_exc.add_to(m)
grp_nodes.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
MiniMap().add_to(m)

m.save(OUT_HTML)
print(f"Saved: {OUT_HTML}")
