"""
STEP9_NORM_PLOTS_ONLY.py
========================
Reads the same data as STEP9_FRAGILITY_SEG3.py but saves ONLY
the normalized-exposure fragility plot for each line.

Output: results/<way_id>/step9_norm_only.png
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

# ── same paths / constants as STEP9 ────────────────────────────────────
SEG_BASE = (
    os.path.join(BASE, "data", "_archive_old_step_names", "step7_outputs_segment3_11lines")
)
RESULTS_DIR           = os.path.join(BASE, "results")
STEP8_PARAMS          = os.path.join(BASE, "results", "step8_params.csv")

W_THRESH   = 15.0
TARGET     = 5.0
K_YEARS    = 10
N_POINTS   = 1000

INCLUDED_WAY_IDS = [
    "1155116743",
    "338850183", "338856348", "375907989", "375922074",
    "375916002", "456822447", "375924539",
]

LINE_NAMES = {
    "1155116743": "Fulgatore-Partanna",
    "375922074":  "Melilli-Ragusa/1-2",
    "375916002":  "Priolo-Chiaramonte Gulfi",
    "456822447":  "Elettrodotto Sorgente-Villafranca-Scilla-Rizziconi",
    "375924539":  "Elettrodotto Sorgente - Paternò",
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
    circuits = LINE_CIRCUITS.get(str(wid))
    circ_label = {1: "single circuit", 2: "double circuit"}.get(circuits)
    suffix = f" [{circ_label}]" if circ_label else ""
    if name:
        return f"{wid} ({name}){suffix}"
    return f"{wid}{suffix}"


# ── same helper functions (copy, no change) ─────────────────────────────
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
        mu = np.exp(log_mu)
        return lognormal_cdf(E_arr, mu, sigma).sum() / k_years - lambda_B
    try:
        log_mu = brentq(obj, -30.0, 60.0, xtol=1e-10, maxiter=500)
        return float(np.exp(log_mu))
    except ValueError:
        from scipy.optimize import minimize
        res = minimize(
            lambda p: (lognormal_cdf(E_arr, np.exp(p[0]), sigma).sum() / k_years - lambda_B) ** 2,
            x0=[np.log(float(np.median(E_arr[E_arr > 0])) if (E_arr > 0).any() else 1.0)],
            method="L-BFGS-B",
            options={"ftol": 1e-20, "maxiter": 2000},
        )
        return float(np.exp(res.x[0]))


# ── single-line normalized plot ─────────────────────────────────────────
def save_norm_plot(way_id, seg3_km, w_max_seg3, mu_norm, sigma, out_path):
    x3 = np.linspace(0, TARGET, N_POINTS)
    p3 = lognormal_cdf(x3, mu_norm, sigma)

    # table rows: 10 evenly spaced exposure points
    e_keys = np.linspace(TARGET / 10, TARGET, 10)
    p_keys = lognormal_cdf(e_keys, mu_norm, sigma)

    fig = plt.figure(figsize=(8, 9))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.45)
    ax     = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    # ── plot ──────────────────────────────────────────────────────────
    ax.plot(x3, p3, color="seagreen", lw=2.5)
    ax.set_xlabel("Normalized Wind Exposure", fontsize=11)
    ax.set_ylabel("Probability of Failure", fontsize=11)
    ax.set_title(
        f"Normalized Exposure  vs  P(failure)\n"
        f"{line_label(way_id)}   |   Seg3 L = {seg3_km:.4f} km   |   "
        f"w_max (seg3) = {w_max_seg3:.2f} m/s",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlim(0, TARGET)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # ── table ─────────────────────────────────────────────────────────
    ax_tbl.axis("off")

    col_labels = ["Norm. Exposure", "P(failure)", "Risk Level"]
    def risk(p):
        if p < 0.10: return "Low"
        if p < 0.40: return "Moderate"
        if p < 0.70: return "High"
        return "Very High"

    table_data = [
        [f"{e:.2f}", f"{p:.4f}", risk(p)]
        for e, p in zip(e_keys, p_keys)
    ]

    RISK_COLORS = {
        "Low":       "#d4edda",
        "Moderate":  "#fff3cd",
        "High":      "#f8d7da",
        "Very High": "#f5c6cb",
    }

    tbl = ax_tbl.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)

    for j in range(3):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, (_, _, risk_label) in enumerate(table_data, start=1):
        bg_base = "#eaf3fb" if i % 2 == 0 else "white"
        tbl[i, 0].set_facecolor(bg_base)
        tbl[i, 1].set_facecolor(bg_base)
        tbl[i, 2].set_facecolor(RISK_COLORS[risk_label])
        tbl[i, 2].set_text_props(fontweight="bold")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("STEP9_NORM_PLOTS_ONLY — normalized fragility plots per line")
    print("=" * 70)

    df_params = pd.read_csv(STEP8_PARAMS)
    df_params["way_id"] = df_params["way_id"].astype(str)

    for way_id in INCLUDED_WAY_IDS:
        print(f"\nLine: {line_label(way_id)}")

        p_row = df_params[df_params["way_id"] == way_id]
        if p_row.empty:
            print("  SKIP -- not in step8_params.csv")
            continue

        sigma    = float(p_row.iloc[0]["sigma"])
        lambda_B = float(p_row.iloc[0]["lambda_B"])

        seg_csv = os.path.join(SEG_BASE, f"way_{way_id}", "step7_segment_timeseries.csv")
        if not os.path.isfile(seg_csv):
            print("  SKIP -- segment timeseries not found")
            continue

        df_seg = pd.read_csv(seg_csv)
        df_seg = df_seg[df_seg["segment_number"] == 3].copy()
        df_seg["wind_speed_100m"] = pd.to_numeric(df_seg["wind_speed_100m"], errors="coerce")
        df_seg["wind_exposure"]   = pd.to_numeric(df_seg["wind_exposure"],   errors="coerce")
        df_seg = df_seg.dropna(subset=["wind_speed_100m", "wind_exposure"])

        mask = df_seg["wind_speed_100m"] > W_THRESH
        if mask.sum() == 0:
            print("  SKIP -- no wind above threshold in segment 3")
            continue

        cubes    = (df_seg.loc[mask, "wind_speed_100m"] - W_THRESH) ** 3
        exp_vals = df_seg.loc[mask, "wind_exposure"]
        seg3_km  = float(np.median(exp_vals.values / cubes.values))

        df_seg["time"] = pd.to_datetime(df_seg["time"])
        w_arr      = df_seg.groupby("time")["wind_speed_100m"].max().values
        w_max_seg3 = float(w_arr.max())

        # normalized exposure — same formula as STEP9
        cube_ref   = max(w_max_seg3 - W_THRESH, 0.0) ** 3
        alpha_norm = TARGET / (seg3_km * cube_ref) if cube_ref > 0 else 1.0
        E_alpha_norm = np.where(
            w_arr >= W_THRESH,
            alpha_norm * seg3_km * (w_arr - W_THRESH) ** 3,
            0.0,
        )
        mu_norm = fit_mu_fixed_sigma(E_alpha_norm, lambda_B, K_YEARS, sigma)

        print(f"  seg3_km={seg3_km:.4f}  w_max={w_max_seg3:.3f}  sigma={sigma:.4f}  mu_norm={mu_norm:.4f}")

        line_dir = os.path.join(RESULTS_DIR, way_id)
        os.makedirs(line_dir, exist_ok=True)
        out_path = os.path.join(line_dir, "step9_norm_only.png")
        save_norm_plot(way_id, seg3_km, w_max_seg3, mu_norm, sigma, out_path)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
