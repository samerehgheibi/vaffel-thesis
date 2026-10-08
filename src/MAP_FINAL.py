"""
MAP_FINAL.py — Static PNG network map, reference style
- Background : ERA5 max wind speed heatmap
- Sicily outline visible
- Lines colored: included (blue 220 / red 380) vs excluded (grey)
- Uses config.py as single source of truth
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker

from config import (
    WIND_MAX_CACHE, OSM_CACHE, TOWERS_CSV, SICILY_CACHE,
    INCLUDED, EXCLUDED, VOLTAGE_MAP, NAME_MAP,
    SICILY_BBOX, COLOR
)

OUT_PNG = os.path.join(BASE, "results", "map_final.png")

B = SICILY_BBOX

# ── 1. WIND MAX RASTER ───────────────────────────────────────────────────────
wind = pd.read_parquet(WIND_MAX_CACHE, engine="fastparquet")
lat_vals = np.sort(wind["lat"].unique())
lon_vals = np.sort(wind["lon"].unique())
Z = wind.pivot(index="lat", columns="lon", values="ws").values
ws_min = float(np.nanmin(Z))
ws_max = float(np.nanmax(Z))
print(f"Wind: {ws_min:.1f} – {ws_max:.1f} m/s  grid {Z.shape}")

# ── 2. SICILY OUTLINE ────────────────────────────────────────────────────────
with open(SICILY_CACHE, encoding="utf-8") as f:
    sicily_geom = json.load(f)

rings = []
for el in sicily_geom:
    if el.get("type") != "relation":
        continue
    for member in el.get("members", []):
        if member.get("role") == "outer" and "geometry" in member:
            coords = [(g["lon"], g["lat"]) for g in member["geometry"]]
            if len(coords) > 30:
                rings.append(np.array(coords))

# ── 3. LINES FROM THESIS CSV (has all coordinates + circuit info) ────────────
ALL_ANALYSIS = INCLUDED | EXCLUDED

df_towers = pd.read_csv(TOWERS_CSV)
df_towers["way_id"] = df_towers["way_id"].astype(str)
df_towers = df_towers[df_towers["way_id"].isin(ALL_ANALYSIS)]

lines = []
for way_id, grp in df_towers.groupby("way_id", sort=False):
    grp = grp.reset_index(drop=True)   # preserve original CSV row order
    lines.append({
        "way_id":   way_id,
        "volt":     int(grp["line_voltage"].iloc[0]),
        "circuits": int(grp["line_circuits"].iloc[0]),
        "included": way_id in INCLUDED,
        "lats":     grp["lat"].tolist(),
        "lons":     grp["lon"].tolist(),
    })

print(f"Analysis lines from CSV: {len(lines)}")
for l in lines:
    status = "INC" if l["included"] else "EXC"
    print(f"  [{status}] {l['way_id']}  {l['volt']//1000}kV  {l['circuits']}c  {len(l['lats'])} nodes")

# ── 3b. NUMBER + NAME THE INCLUDED-LINE ENDPOINT NODES ────────────────────────
included_nodes = {}   # (lat_r, lon_r) -> {"num": int, "names": set}
for row in lines:
    if not row["included"]:
        continue
    name = NAME_MAP.get(row["way_id"], row["way_id"])
    for lat, lon in [(row["lats"][0], row["lons"][0]),
                      (row["lats"][-1], row["lons"][-1])]:
        key = (round(lat, 2), round(lon, 2))
        if key not in included_nodes:
            included_nodes[key] = {"num": len(included_nodes) + 1, "names": set()}
        included_nodes[key]["names"].add(name)

print("\nIncluded substation nodes:")
for (lat, lon), info in included_nodes.items():
    print(f"  [{info['num']:2d}] ({lat:.3f}, {lon:.3f})  {', '.join(sorted(info['names']))}")

# ── 4. DRAW PANEL ────────────────────────────────────────────────────────────
def draw_panel(ax, volt_filter, title):
    # wind background
    im = ax.imshow(Z, origin="lower",
                   extent=[lon_vals.min(), lon_vals.max(),
                           lat_vals.min(), lat_vals.max()],
                   cmap="YlOrBr", aspect="auto",
                   vmin=ws_min, vmax=ws_max,
                   interpolation="bilinear", alpha=0.75, zorder=0)

    # Sicily coastline
    for ring in rings:
        ax.plot(ring[:,0], ring[:,1], color="#555555", lw=0.7, zorder=2)

    # draw excluded first (bottom), then included on top
    for status in ["excluded", "included"]:
        for row in lines:
            if row["volt"] != volt_filter: continue
            if row["included"] != (status == "included"): continue

            if row["included"]:
                dc  = row["circuits"] >= 2
                col = (COLOR["inc_220_dc"] if volt_filter == 220000 and dc else
                       COLOR["inc_220_sc"] if volt_filter == 220000 else
                       COLOR["inc_380_dc"] if dc else
                       COLOR["inc_380_sc"])
                lw  = 2.8 if dc else 1.8
                zo  = 4
            else:
                col = COLOR["exc"]
                lw  = 1.4
                zo  = 3

            ax.plot(row["lons"], row["lats"],
                    color=col, lw=lw, solid_capstyle="round", zorder=zo)

    # substation / junction nodes (line endpoints)
    for row in lines:
        if row["volt"] != volt_filter: continue
        node_col = COLOR["node_inc"] if row["included"] else COLOR["node_exc"]
        for lat, lon in [(row["lats"][0], row["lons"][0]),
                         (row["lats"][-1], row["lons"][-1])]:
            ax.scatter(lon, lat, s=28, color=node_col,
                       edgecolors="#1a237e", linewidths=0.6,
                       zorder=5)

        # one label per line, placed at its midpoint
        if row["included"]:
            mid = len(row["lats"]) // 2
            lat, lon = row["lats"][mid], row["lons"][mid]
            name = NAME_MAP.get(row["way_id"], row["way_id"])
            ax.annotate(name, xy=(lon, lat),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=6.5, fontweight="bold", color="#1a237e",
                        zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.7))

    ax.set_xlim(B["lon_min"], B["lon_max"])
    ax.set_ylim(B["lat_min"], B["lat_max"])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.tick_params(labelsize=8)
    ax.grid(True, color="#cccccc", lw=0.3, zorder=1)

    # north arrow
    ax.annotate("N", xy=(0.06, 0.94), xycoords="axes fraction",
                fontsize=11, fontweight="bold", ha="center", zorder=6)
    ax.annotate("", xy=(0.06, 0.98), xytext=(0.06, 0.94),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5), zorder=6)

    # scale bar ~50 km
    deg50 = 50 / (111.0 * np.cos(np.radians(37.5)))
    bx, by = B["lon_min"] + 0.25, B["lat_min"] + 0.12
    ax.plot([bx, bx+deg50], [by, by], "k-", lw=2, zorder=6)
    ax.text(bx+deg50/2, by+0.06, "50 km",
            ha="center", va="bottom", fontsize=7, zorder=6)

    return im

# ── 5. FIGURE ────────────────────────────────────────────────────────────────
print("Drawing...")
fig, axes = plt.subplots(1, 2, figsize=(17, 8),
                         gridspec_kw={"wspace": 0.08})
fig.patch.set_facecolor("white")
fig.subplots_adjust(top=0.91, bottom=0.24, left=0.06, right=0.91)

im = draw_panel(axes[0], 220000, "220 kV Transmission Lines — Sicily")
draw_panel(axes[1], 380000, "380 kV Transmission Lines — Sicily")

axes[0].set_ylabel("Latitude", fontsize=9)
axes[1].set_ylabel("")
axes[1].tick_params(labelleft=False)

# colorbar
cbar_ax = fig.add_axes([0.925, 0.12, 0.012, 0.76])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label("Max Wind Speed (m/s)", fontsize=9)
cb.ax.tick_params(labelsize=8)
ticks = np.linspace(ws_min, ws_max, 9)
cb.set_ticks(ticks)
cb.set_ticklabels([f"{t:.1f}" for t in ticks])

# legend
leg = [
    mlines.Line2D([], [], color=COLOR["inc_220_dc"], lw=2.8,
                  label="220 kV — included (double-circuit)"),
    mlines.Line2D([], [], color=COLOR["inc_220_sc"], lw=1.8,
                  label="220 kV — included (single-circuit)"),
    mlines.Line2D([], [], color=COLOR["inc_380_dc"], lw=2.8,
                  label="380 kV — included (double-circuit)"),
    mlines.Line2D([], [], color=COLOR["inc_380_sc"], lw=1.8,
                  label="380 kV — included (single-circuit)"),
    mlines.Line2D([], [], color=COLOR["exc"], lw=1.4,
                  label="Excluded from analysis"),
    mlines.Line2D([], [], marker="o", color="none", markerfacecolor=COLOR["node_inc"],
                  markeredgecolor="#1a237e", markersize=7,
                  label="Substation node — included line"),
    mlines.Line2D([], [], marker="o", color="none", markerfacecolor=COLOR["node_exc"],
                  markeredgecolor="#1a237e", markersize=7,
                  label="Substation node — excluded line"),
]
fig.legend(handles=leg, loc="lower center", ncol=4,
           fontsize=8.5, framealpha=0.9,
           bbox_to_anchor=(0.46, 0.0))


fig.suptitle(
    "Selected HV Transmission Lines in Sicily — Max Wind Speed (2013-2023)",
    fontsize=12, color="black", y=0.97)

fig.savefig(OUT_PNG, dpi=200, facecolor="white")
print(f"Saved: {OUT_PNG}")
plt.show()
