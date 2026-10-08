"""
STEP 11 — All Lines: Fragility Curves (Segment 3)
==================================================
One figure with all 11 lines plotted together.
  X: Wind Speed [m/s]  (inverse of Eq. 3.12)
  Y: Probability of failure  [Eq. 3.15]
Each line gets a distinct colour.
Table below shows per-line: w_max seg3 and actual failure wind speeds.
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
SEG_BASE = (
    os.path.join(BASE, "data", "_archive_old_step_names", "step7_outputs_segment3_11lines")
)
RESULTS_DIR           = os.path.join(BASE, "results")
STEP8_PARAMS          = os.path.join(BASE, "results", "step8_params.csv")
ASSIGNED_FAILURES_CSV = os.path.join(BASE, "data", "assigned_failures_updated.csv")

W_THRESH   = 15.0
W_MAX_PLOT = 40.0
K_YEARS    = 10
N_POINTS   = 1000

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

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79"
]

# ---------------------------------------------------------------------
# FUNCTIONS  (same as STEP 9)
# ---------------------------------------------------------------------

def exposure_from_wind(w_arr, L_km):
    w = np.asarray(w_arr, dtype=float)
    return np.where(w >= W_THRESH, L_km * (w - W_THRESH) ** 3, 0.0)


def lognormal_cdf(E_arr, mu, sigma):
    E = np.asarray(E_arr, dtype=float)
    return np.where(
        E > 0,
        norm.cdf((np.log(np.maximum(E, 1e-300)) - np.log(mu)) / sigma),
        0.0,
    )


def fit_mu_fixed_sigma(E_arr, lambda_B, k_years, sigma):
    def obj(log_mu):
        return lognormal_cdf(E_arr, np.exp(log_mu), sigma).sum() / k_years - lambda_B
    try:
        return float(np.exp(brentq(obj, -30.0, 60.0, xtol=1e-10, maxiter=500)))
    except ValueError:
        from scipy.optimize import minimize
        res = minimize(
            lambda p: (lognormal_cdf(E_arr, np.exp(p[0]), sigma).sum() / k_years - lambda_B) ** 2,
            x0=[np.log(float(np.median(E_arr[E_arr > 0])) if (E_arr > 0).any() else 1.0)],
            method="L-BFGS-B",
            options={"ftol": 1e-20, "maxiter": 2000},
        )
        return float(np.exp(res.x[0]))


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 11 -- All Lines Fragility (Segment 3)")
    print("=" * 70)

    df_params = pd.read_csv(STEP8_PARAMS)
    df_params["way_id"] = df_params["way_id"].astype(str)

    df_failures = pd.read_csv(ASSIGNED_FAILURES_CSV)
    df_failures["way_id"] = df_failures["way_id"].astype(str)

    w_curve = np.linspace(W_THRESH, W_MAX_PLOT, N_POINTS)

    fig = plt.figure(figsize=(12, 10))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.45)
    ax      = fig.add_subplot(gs[0])
    ax_tbl  = fig.add_subplot(gs[1])

    table_rows = []

    for idx, way_id in enumerate(INCLUDED_WAY_IDS):
        print(f"\nProcessing line: {line_label(way_id)}")
        color = COLORS[idx]

        p_row = df_params[df_params["way_id"] == way_id]
        if p_row.empty:
            print(f"  SKIP -- not in step8_params.csv")
            continue
        sigma    = float(p_row.iloc[0]["sigma"])
        lambda_B = float(p_row.iloc[0]["lambda_B"])

        seg_csv = os.path.join(SEG_BASE, f"way_{way_id}", "step7_segment_timeseries.csv")
        if not os.path.isfile(seg_csv):
            print(f"  SKIP -- segment timeseries not found")
            continue

        df_seg = pd.read_csv(seg_csv)
        df_seg = df_seg[df_seg["segment_number"] == 3].copy()
        df_seg["wind_speed_100m"] = pd.to_numeric(df_seg["wind_speed_100m"], errors="coerce")
        df_seg["wind_exposure"]   = pd.to_numeric(df_seg["wind_exposure"],   errors="coerce")
        df_seg = df_seg.dropna(subset=["wind_speed_100m", "wind_exposure"])

        mask = df_seg["wind_speed_100m"] > W_THRESH
        if mask.sum() == 0:
            print(f"  SKIP -- no wind above threshold in segment 3")
            continue

        cubes    = (df_seg.loc[mask, "wind_speed_100m"] - W_THRESH) ** 3
        exp_vals = df_seg.loc[mask, "wind_exposure"]
        seg3_km  = float(np.median(exp_vals.values / cubes.values))

        df_seg["time"] = pd.to_datetime(df_seg["time"])
        w_arr = df_seg.groupby("time")["wind_speed_100m"].max().values
        w_max_seg3 = float(w_arr.max())

        E_alpha1 = exposure_from_wind(w_arr, seg3_km)
        mu = fit_mu_fixed_sigma(E_alpha1, lambda_B, K_YEARS, sigma)

        # Plot curve: wind speed = inverse of exposure
        E_curve = exposure_from_wind(w_curve, seg3_km)
        p_curve = lognormal_cdf(E_curve, mu, sigma)
        ax.plot(w_curve, p_curve, lw=1.8, color=color, label=line_label(way_id))

        # Table row
        line_fails = df_failures[df_failures["way_id"] == way_id]["failure_wind_speed"].tolist()
        fail_str = ", ".join(f"{v:.2f}" for v in sorted(line_fails)) if line_fails else "—"
        table_rows.append([line_label(way_id), f"{w_max_seg3:.2f}", fail_str])

        print(f"  seg3_km={seg3_km:.4f}  w_max={w_max_seg3:.2f}  mu={mu:.4f}  sigma={sigma:.4f}")

    # ── Plot style ────────────────────────────────────────────────────
    ax.axvline(W_THRESH, color="gray", ls="--", lw=1, alpha=0.6,
               label=f"w_thresh = {W_THRESH} m/s")
    ax.set_xlabel("Wind Speed  [m/s]", fontsize=12)
    ax.set_ylabel("Probability of Failure", fontsize=12)
    ax.set_title("Fragility Curves — All Lines (Segment 3)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(W_THRESH - 1, W_MAX_PLOT)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)

    # ── Table ─────────────────────────────────────────────────────────
    ax_tbl.axis("off")

    col_labels = ["Line ID", "w_max Seg3 [m/s]", "Actual Failure Wind Speeds [m/s]"]
    tbl = ax_tbl.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_rows) + 1):
        row_color = COLORS[i - 1] + "33"  # همون رنگ لاین با شفافیت
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(row_color)

    out_path = os.path.join(RESULTS_DIR, "step11_all_lines_seg3.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    print("=" * 70)
    print("STEP 11 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
