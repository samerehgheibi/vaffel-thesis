"""
STEP 10 — P(failure) vs Wind Speed  (all segments per line)
============================================================
For each line:
  - loads every node (= segment) from merged_towers_wind_new.parquet
  - sigma fixed from step8_params.csv  (full-line fit)
  - mu fitted per segment via Brentq   (same lambda_B target)
  - one curve per segment: X = wind speed, Y = P(failure)
  - colour by segment length [km]

Output: results/<way_id>/step10_all_segments.png
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------
PARQUET_PATH = (
    os.path.join(BASE, "data", "merged_towers_wind_new.parquet")
)
RESULTS_DIR           = os.path.join(BASE, "results")
STEP8_PARAMS          = os.path.join(BASE, "results", "step8_params.csv")
ASSIGNED_FAILURES_CSV = os.path.join(BASE, "data", "assigned_failures_updated.csv")

W_THRESH   = 15.0
W_MAX_PLOT = 40.0
K_YEARS    = 10
N_POINTS   = 500

INCLUDED_WAY_IDS = [
    "1155116743",
    "338850183",  "338856348",  "375907989",  "375922074",
    "375916002",  "456822447",  "375924539",
]

LINE_NAMES = {
    "1155116743": "Fulgatore-Partanna",
    "375922074":  "Melilli-Ragusa/1-2",
    "375916002":  "Priolo-Chiaramonte Gulfi",
    "456822447":  "Elettrodotto Sorgente-Villafranca-Scilla-Rizziconi",
    "375924539":  "Elettrodotto Sorgente - Paternò",
}

LINE_VOLTAGES = {
    "1155116743": 220,
    "338850183":  220,
    "338856348":  220,
    "375907989":  220,
    "375922074":  220,
    "375916002":  380,
    "456822447":  380,
    "375924539":  380,
}

LINE_CIRCUITS = {
    "1155116743": 1,
    "338850183":  2,
    "338856348":  2,
    "375907989":  2,
    "375922074":  2,
    "375916002":  1,
    "456822447":  2,
    "375924539":  1,
}

def line_label(wid):
    name     = LINE_NAMES.get(str(wid))
    voltage  = LINE_VOLTAGES.get(str(wid))
    circuits = LINE_CIRCUITS.get(str(wid))
    circ_label = {1: "single circuit", 2: "double circuit"}.get(circuits)
    tags = [t for t in (f"{voltage} kV" if voltage else None, circ_label) if t]
    suffix = f" [{', '.join(tags)}]" if tags else ""
    if name:
        return f"{wid} ({name}){suffix}"
    return f"{wid}{suffix}"


# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------

def exposure_from_wind(w_arr, seg_km):
    w = np.asarray(w_arr, dtype=float)
    return np.where(w >= W_THRESH, seg_km * (w - W_THRESH) ** 3, 0.0)


def lognormal_cdf(E_arr, mu, sigma):
    E = np.asarray(E_arr, dtype=float)
    return np.where(
        E > 0,
        norm.cdf((np.log(np.maximum(E, 1e-300)) - np.log(mu)) / sigma),
        0.0,
    )


def fit_mu(E_arr, lambda_B, k_years, sigma):
    """Brentq: find mu such that sum(p)/k_years = lambda_B, sigma fixed."""
    def obj(log_mu):
        return lognormal_cdf(E_arr, np.exp(log_mu), sigma).sum() / k_years - lambda_B
    try:
        return float(np.exp(brentq(obj, -30.0, 60.0, xtol=1e-10, maxiter=500)))
    except ValueError:
        return None


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 10 -- P(failure) vs Wind Speed  (all segments per line)")
    print("=" * 70)

    df_params = pd.read_csv(STEP8_PARAMS)
    df_params["way_id"] = df_params["way_id"].astype(str)

    df_failures = pd.read_csv(ASSIGNED_FAILURES_CSV)
    df_failures["way_id"] = df_failures["way_id"].astype(str)

    con = duckdb.connect()

    for way_id in INCLUDED_WAY_IDS:
        print(f"\nProcessing line: {line_label(way_id)}")

        p_row = df_params[df_params["way_id"] == way_id]
        if p_row.empty:
            print(f"  SKIP -- not in step8_params.csv")
            continue
        sigma    = float(p_row.iloc[0]["sigma"])
        lambda_B = float(p_row.iloc[0]["lambda_B"])

        # Load all nodes for this line
        df = con.execute(f"""
            SELECT node_id, segment_km, time, wind_speed_100m, grid_lat, grid_lon
            FROM '{PARQUET_PATH}'
            WHERE way_id = '{way_id}'
            ORDER BY node_id, time
        """).df()
        df["wind_speed_100m"] = pd.to_numeric(df["wind_speed_100m"], errors="coerce")
        df["segment_km"]      = pd.to_numeric(df["segment_km"],      errors="coerce")
        df = df.dropna().reset_index(drop=True)

        # One entry per (node_id, segment_km)
        seg_info = (
            df.groupby("node_id")["segment_km"]
            .first()
            .reset_index()
            .sort_values("segment_km")
            .reset_index(drop=True)
        )
        n_segments = len(seg_info)
        total_km = seg_info["segment_km"].sum()
        print(f"  {n_segments} segments  |  sigma = {sigma:.4f}  |  lambda_B = {lambda_B:.6f}  |  total_km = {total_km:.3f}")
        print(seg_info["segment_km"].value_counts())

        km_min   = seg_info["segment_km"].min()
        km_max   = seg_info["segment_km"].max()
        km_range = km_max - km_min if km_max > km_min else 1.0

        # ── Grid point summary ────────────────────────────────────────────
        df_above = df[df["wind_speed_100m"] > W_THRESH].copy()
        seg_km_per_node = df.groupby("node_id")["segment_km"].first()
        node_grid = df[["node_id", "grid_lat", "grid_lon"]].drop_duplicates("node_id")
        node_grid = node_grid.join(seg_km_per_node, on="node_id")
        grid_seg_stats = (
            node_grid.groupby(["grid_lat", "grid_lon"])["segment_km"]
            .agg(seg_km_min="min", seg_km_max="max")
            .reset_index()
        )
        grid_summary = (
            df_above.groupby(["grid_lat", "grid_lon"])
            .agg(
                max_wind    =("wind_speed_100m", "max"),
                mean_excess =("wind_speed_100m", lambda x: (x - W_THRESH).mean()),
                hours_above =("wind_speed_100m", "count"),
                n_segments  =("node_id", "nunique"),
            )
            .reset_index()
            .merge(grid_seg_stats, on=["grid_lat", "grid_lon"], how="left")
            .sort_values("seg_km_min", ascending=True)
            .reset_index(drop=True)
        )

        fig = plt.figure(figsize=(13, 10))
        gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1.8], hspace=0.5)
        ax      = fig.add_subplot(gs[0])
        ax_grid = fig.add_subplot(gs[1])

        w_curve = np.linspace(W_THRESH, W_MAX_PLOT, N_POINTS)

        for _, seg_row in seg_info.iterrows():
            nid    = seg_row["node_id"]
            seg_km = float(seg_row["segment_km"])
            color  = cm.viridis((seg_km - km_min) / km_range)

            lambda_B_seg = lambda_B * (seg_km / total_km)

            w_arr = df[df["node_id"] == nid]["wind_speed_100m"].values
            E_arr = exposure_from_wind(w_arr, seg_km)

            mu = fit_mu(E_arr, lambda_B_seg, K_YEARS, sigma)
            if mu is None:
                continue

            E_curve = exposure_from_wind(w_curve, seg_km)
            p_curve = lognormal_cdf(E_curve, mu, sigma)

            ax.plot(w_curve, p_curve, lw=1.0, alpha=0.7, color=color)

        ax.axvline(W_THRESH, color="red", ls="--", lw=1,
                   label=f"w_thresh = {W_THRESH} m/s")
        ax.set_xlabel("Wind Speed  [m/s]", fontsize=11)
        ax.set_ylabel("Probability of failure", fontsize=11)
        ax.set_title(
            f"Line: {line_label(way_id)}   |   {n_segments} segments\n"
            f"Wind Speed vs P(failure)",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlim(W_THRESH - 1, W_MAX_PLOT)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Table — Grid Point Summary ────────────────────────────────────
        ax_grid.axis("off")

        grid_data = [
            [f"{r['grid_lat']:.2f}, {r['grid_lon']:.2f}",
             f"{r['max_wind']:.2f}",
             f"{r['mean_excess']:.2f}",
             f"{int(r['hours_above'])}",
             f"{int(r['n_segments'])}",
             f"{r['seg_km_min']:.3f} – {r['seg_km_max']:.3f}"]
            for _, r in grid_summary.iterrows()
        ]
        grid_tbl = ax_grid.table(
            cellText=grid_data,
            colLabels=["Grid Point (lat, lon)", "Max Wind [m/s]",
                       "Mean(w−W_thresh) [m/s]", "Hours w>15", "# Segments", "seg_km range [km]"],
            loc="center", cellLoc="center"
        )
        grid_tbl.auto_set_font_size(False)
        grid_tbl.set_fontsize(8)
        grid_tbl.scale(1.0, 1.4)
        for j in range(6):
            grid_tbl[0, j].set_facecolor("#2c5f8a")
            grid_tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(grid_data) + 1):
            for j in range(6):
                grid_tbl[i, j].set_facecolor("#eaf3fb" if i % 2 == 0 else "white")

        line_dir = os.path.join(RESULTS_DIR, way_id)
        os.makedirs(line_dir, exist_ok=True)
        out_path = os.path.join(line_dir, "step10_all_segments.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    con.close()
    print("\n" + "=" * 70)
    print("STEP 10 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()