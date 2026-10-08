"""
STEP 16 — Wind Speed Boxplots  (same style as archive STEP 12)
==============================================================
Identical to archive STEP 12 EXCEPT:
  Red ▲ peak  →  Green ◆ mean  (mean of hourly max wind per line)

Blue  = included in main analysis (11 lines)
Salmon = excluded from Step 6/7/8  (9 lines)

Outputs:
  results/step16_boxplot_220kv.png
  results/step16_boxplot_380kv.png
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
from collections import Counter

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ── CONFIG ────────────────────────────────────────────────────────────
PARQUET_PATH   = os.path.join(BASE, "data", "merged_towers_wind_new.parquet")
LINE_NAMES_CSV = os.path.join(BASE, "line_names.csv")
LAMBDA_B_CSV   = os.path.join(BASE, "step6_lambda_B.csv")
OUT_DIR        = os.path.join(BASE, "results")

INCLUDED = {
    "375924539", "375916002", "456822447",
    "338856348", "375922074", "338850183",
    "375907989", "1155116743",
}

EXCLUDED = {
    "159113596", "1155116741", "191811525",
    "191811504", "338847619", "375908007", "375922072",
    "56984364",
    "375924538",
}

ALL_WAY_IDS = INCLUDED | EXCLUDED

VOLTAGE_MAP = {
    "375924539": 380, "375916002": 380, "375924538": 380,
    "456822447": 380,
    "338856348": 220, "375922074": 220, "338850183": 220,
    "375907989": 220, "1155116743": 220,
    "159113596": 220, "1155116741": 220, "191811525": 220,
    "191811504": 220, "338847619": 220, "375908007": 220,
    "375922072": 220, "56984364": 220,
}

CIRCUITS_MAP = {
    "375924539": 1, "375916002": 1, "375924538": 1,
    "456822447": 2,
    "338856348": 2, "375922074": 2, "338850183": 2,
    "375907989": 2, "1155116743": 1,
    "159113596": 2, "1155116741": 1, "191811525": 2,
    "191811504": 2, "338847619": 2, "375908007": 1,
    "375922072": 2, "56984364": 2,
}

EXTRA_NAMES = {
    "191811525": "191811525",
    "338847619":  "338847619",
}

WIND_YAXIS_LIM = (0.0, 27.0)

# ── HELPERS ───────────────────────────────────────────────────────────

def load_name_map():
    nm = {}
    if os.path.isfile(LINE_NAMES_CSV):
        df = pd.read_csv(LINE_NAMES_CSV)
        df["way_id"] = df["way_id"].astype(str)
        for _, r in df.iterrows():
            n = r.get("line_name")
            if pd.notna(n) and str(n).strip() and str(n) != "nan":
                nm[str(r["way_id"])] = str(n).strip()
    nm.update(EXTRA_NAMES)
    return nm


def load_km_map():
    km = {}
    if os.path.isfile(LAMBDA_B_CSV):
        df = pd.read_csv(LAMBDA_B_CSV)
        df["way_id"] = df["way_id"].astype(str)
        for _, r in df.iterrows():
            try:
                km[str(r["way_id"])] = float(r["total_km"])
            except Exception:
                pass
    return km


def fetch_hourly_max_wind(way_ids):
    ways_str = ", ".join(f"'{w}'" for w in sorted(way_ids))
    con = duckdb.connect()
    try:
        df = con.execute(f"""
            SELECT
                CAST(way_id AS VARCHAR)      AS way_id,
                time,
                CAST(MAX(line_voltage)/1000 AS INT) AS voltage_kV,
                MAX(wind_speed_100m)         AS wind_max_100m
            FROM '{PARQUET_PATH}'
            WHERE CAST(way_id AS VARCHAR) IN ({ways_str})
            GROUP BY way_id, time
            ORDER BY way_id, time
        """).df()
    finally:
        con.close()
    df["way_id"]       = df["way_id"].astype(str)
    df["wind_max_100m"] = pd.to_numeric(df["wind_max_100m"], errors="coerce")
    return df.dropna(subset=["wind_max_100m"])


def _boxplot_compat(ax, data, labels, **kw):
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


# ── PLOT ──────────────────────────────────────────────────────────────

def make_plot(df_all, voltage, out_path, name_map, km_map):
    sub = df_all[df_all["voltage_kV"] == voltage].copy()
    if sub.empty:
        print(f"  [SKIP] No data for {voltage} kV")
        return

    ways_inc = sorted([w for w in sub["way_id"].unique() if w in INCLUDED])
    ways_exc = sorted([w for w in sub["way_id"].unique() if w in EXCLUDED])
    ways_ordered = ways_inc + ways_exc

    data        = []
    labels      = []
    means       = []          # ← mean instead of peak
    is_excluded = []
    legend_rows = []

    for i, w in enumerate(ways_ordered, start=1):
        vals = sub.loc[sub["way_id"] == w, "wind_max_100m"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        excl      = w in EXCLUDED
        raw_name  = name_map.get(w, w)
        tag       = " [EXCL]" if excl else ""
        mean_val  = float(np.mean(vals))        # ← mean
        km_v      = km_map.get(w, float("nan"))
        km_s      = f"{km_v:.1f} km" if np.isfinite(km_v) else "— km"
        voltage_kV = VOLTAGE_MAP.get(w, "?")
        circ       = CIRCUITS_MAP.get(w)
        circ_s     = {1: "single circuit", 2: "double circuit"}.get(circ, "? circuit")
        label     = f"{raw_name}{tag}\n{w}\n{voltage_kV} kV | {circ_s} | {km_s} | mean {mean_val:.1f} m/s"

        data.append(vals.tolist())
        labels.append(label)
        means.append(mean_val)
        is_excluded.append(excl)

        km_v = km_map.get(w, float("nan"))
        legend_rows.append({
            "i":      i,
            "way_id": w,
            "name":   raw_name + tag,
            "km":     km_v,
            "mean":   mean_val,
            "excl":   excl,
        })

    if not data:
        return

    n      = len(data)
    fig_w  = max(14, n * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, 7))

    bp = _boxplot_compat(
        ax, data, labels,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
        widths=0.55,
        medianprops=dict(color="black",  linewidth=1.4),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black",     linewidth=0.9),
        flierprops=dict(marker="o",
                        markerfacecolor="black", markeredgecolor="black",
                        markersize=4, linestyle="none", alpha=0.55),
    )

    for box, excl in zip(bp["boxes"], is_excluded):
        box.set_facecolor("#F4A582" if excl else "#92C5DE")
        box.set_alpha(0.75)

    # Green ◆ mean marker  (replaces the red ▲ peak from archive STEP 12)
    for i, mean_val in enumerate(means, start=1):
        ax.plot(i, mean_val,
                marker="D", color="#2ca02c", markersize=10,
                markeredgecolor="#1a6e1a", markeredgewidth=0.8,
                linestyle="none", zorder=6, clip_on=False)
        ax.annotate(
            f"{mean_val:.1f}",
            xy=(i, mean_val), xytext=(0, 7),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=7, color="#1a6e1a", fontweight="bold",
        )

    # Separator between included and excluded
    n_inc = sum(1 for r in legend_rows if not r["excl"])
    if 0 < n_inc < n:
        ax.axvline(n_inc + 0.5, color="grey", linestyle="--",
                   linewidth=1.2, alpha=0.6)
        ax.text(n_inc + 0.5, WIND_YAXIS_LIM[1] * 0.99,
                " ← Included  |  Excluded →",
                ha="center", va="top", fontsize=8,
                color="grey", style="italic")

    ax.set_title(
        f"STEP 16 — Wind Speed Boxplot by Line ({voltage} kV)\n"
        f"All failure-database lines (before exclusion filter)\n"
        f"Blue = included in analysis  |  Salmon = excluded from Step 6/7/8",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Line (name · way_id · mean wind)", fontsize=10)
    ax.set_ylabel("Wind speed @ 100 m (m/s)", fontsize=10)
    ax.set_ylim(WIND_YAXIS_LIM)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=45, ha="right", fontsize=8)

    # Legend box (right side)
    legend_lines = [
        "Box plot (Tukey): box = Q1–Q3; black line = median;",
        "whiskers = 1.5×IQR; circles = Tukey outliers.",
        "Green ◆ = mean hourly max wind (100 m) over ERA5 record.",
        "Blue boxes = included in main analysis.",
        "Salmon boxes = originally excluded (data-quality reasons).",
        "",
        "Line   |  way_id  |  name  |  km  |  mean (m/s)",
    ]
    for r in legend_rows:
        km_s = f"{r['km']:.1f}" if np.isfinite(r["km"]) else "—"
        tag  = " [EXCL]" if r["excl"] else ""
        legend_lines.append(
            f"  {r['i']:>2d}{tag}  {r['way_id']}  {r['name'][:28]}"
            f"  {km_s} km  {r['mean']:.1f} m/s"
        )

    fig.text(
        0.86, 0.5, "\n".join(legend_lines),
        va="center", ha="left", fontsize=7.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        family="monospace",
    )
    plt.tight_layout(rect=[0, 0, 0.84, 1])
    plt.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── MAIN ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 16 — Wind Speed Boxplots  (mean marker)")
    print("=" * 70)

    name_map = load_name_map()
    km_map   = load_km_map()

    print(f"  Fetching hourly max wind for {len(ALL_WAY_IDS)} lines from parquet...")
    df = fetch_hourly_max_wind(ALL_WAY_IDS)

    df["voltage_kV"] = df["way_id"].map(VOLTAGE_MAP)
    df = df.dropna(subset=["voltage_kV"])
    df["voltage_kV"] = df["voltage_kV"].astype(int)

    print(f"  Rows loaded: {len(df):,}")

    for voltage, fname in [
        (380, "step16_boxplot_380kv.png"),
        (220, "step16_boxplot_220kv.png"),
    ]:
        make_plot(df, voltage, os.path.join(OUT_DIR, fname), name_map, km_map)

    print()
    print("STEP 16 complete.")


if __name__ == "__main__":
    main()
