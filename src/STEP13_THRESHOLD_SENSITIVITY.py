"""
STEP 13 — Threshold Sensitivity: Fragility Curves
==================================================
For each line:
  - One figure saved to results/{way_id}/step13_threshold_sensitivity.png
  - Plot: fragility curves for multiple wind thresholds
  - Table: threshold | mu | w50 | P at actual failures (sorted by threshold)
Also saves a combined grid to results/step13_threshold_sensitivity.png.
sigma fixed from STEP 8; mu re-fitted per threshold via Brentq.
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
RESULTS_DIR           = os.path.join(BASE, "results")
STEP8_PARAMS          = os.path.join(BASE, "results", "step8_params.csv")
ASSIGNED_FAILURES_CSV = os.path.join(BASE, "data", "assigned_failures_updated.csv")

THRESHOLDS  = [5.0, 7.0, 10.0, 12.0, 14.0, 15.0, 17.0]
W_MAX_PLOT  = 40.0
K_YEARS     = 10
N_POINTS    = 1000

THRESH_COLORS = [
    "#e41a1c", "#ff7f00", "#f0c000", "#4daf4a",
    "#00bcd4", "#377eb8", "#984ea3"
]

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

def exposure_from_wind(w_arr, L_km, thresh):
    w = np.asarray(w_arr, dtype=float)
    return np.where(w >= thresh, L_km * (w - thresh) ** 3, 0.0)


def lognormal_cdf(E_arr, mu, sigma):
    E = np.asarray(E_arr, dtype=float)
    return np.where(
        E > 0,
        norm.cdf((np.log(np.maximum(E, 1e-300)) - np.log(mu)) / sigma),
        0.0,
    )


def fit_mu(E_arr, lambda_B, k_years, sigma):
    def obj(log_mu):
        return lognormal_cdf(E_arr, np.exp(log_mu), sigma).sum() / k_years - lambda_B
    try:
        return float(np.exp(brentq(obj, -50.0, 80.0, xtol=1e-10, maxiter=500)))
    except ValueError:
        from scipy.optimize import minimize
        res = minimize(
            lambda p: (lognormal_cdf(E_arr, np.exp(p[0]), sigma).sum() / k_years - lambda_B) ** 2,
            x0=[np.log(float(np.median(E_arr[E_arr > 0])) if (E_arr > 0).any() else 1.0)],
            method="L-BFGS-B",
            options={"ftol": 1e-20, "maxiter": 2000},
        )
        return float(np.exp(res.x[0]))


def make_line_figure(way_id, sigma, lambda_B, L_km, w_arr, fail_winds, out_path):
    """Per-line figure: plot + table sorted by threshold."""

    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1.4], hspace=0.5)
    ax      = fig.add_subplot(gs[0])
    ax_tbl  = fig.add_subplot(gs[1])

    fig.suptitle(
        f"Threshold Sensitivity — {line_label(way_id)}  (L = {L_km:.1f} km)",
        fontsize=12, fontweight="bold"
    )

    fail_winds_str = ", ".join(f"{v:.2f}" for v in sorted(fail_winds)) if fail_winds else "—"
    table_rows = []   # (thresh, color, [row_data])

    for tidx, thresh in enumerate(THRESHOLDS):
        color  = THRESH_COLORS[tidx]
        w_plot = np.linspace(thresh, W_MAX_PLOT, N_POINTS)

        E_arr   = exposure_from_wind(w_arr, L_km, thresh)
        mu      = fit_mu(E_arr, lambda_B, K_YEARS, sigma)
        E_curve = exposure_from_wind(w_plot, L_km, thresh)
        p_curve = lognormal_cdf(E_curve, mu, sigma)

        ax.plot(w_plot, p_curve, lw=2.0, color=color, label=f"thresh = {thresh:.0f} m/s")

        idx_jump = np.searchsorted(p_curve, 0.05)
        w_jump = w_plot[min(idx_jump, len(w_plot) - 1)]

        idx_sat = np.searchsorted(p_curve, 0.99)
        w_sat = w_plot[min(idx_sat, len(w_plot) - 1)]

        table_rows.append((thresh, color, [
            f"{thresh:.0f}",
            f"{w_jump:.2f}",
            f"{w_sat:.2f}",
        ]))

    # actual failure lines
    for fw in fail_winds:
        ax.axvline(fw, color="black", ls=":", lw=1.0, alpha=0.7)

    ax.set_xlabel("Wind Speed  [m/s]", fontsize=11)
    ax.set_ylabel("Probability of Failure", fontsize=11)
    ax.set_xlim(14, 40)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=1, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0, framealpha=0.8,
              title=f"Threshold  (L = {L_km:.1f} km)")
    ax.grid(True, linestyle=":", alpha=0.4)

    # ── Table — sorted by threshold ascending ────────────────────────
    table_rows_sorted = sorted(table_rows, key=lambda x: x[0])
    col_labels = ["Threshold [m/s]", "w jump [m/s]", "w → P=1 [m/s]"]
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=[r[2] for r in table_rows_sorted],
        colLabels=col_labels,
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, (_, color, _) in enumerate(table_rows_sorted):
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(color + "33")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 13 -- Threshold Sensitivity")
    print("=" * 70)

    df_params = pd.read_csv(STEP8_PARAMS)
    df_params["way_id"] = df_params["way_id"].astype(str)

    df_failures = pd.read_csv(ASSIGNED_FAILURES_CSV)
    df_failures["way_id"] = df_failures["way_id"].astype(str)

    for way_id in INCLUDED_WAY_IDS:
        print(f"\nLine: {line_label(way_id)}")

        p_row = df_params[df_params["way_id"] == way_id]
        if p_row.empty:
            print("  SKIP -- not in step8_params.csv")
            continue

        sigma    = float(p_row.iloc[0]["sigma"])
        lambda_B = float(p_row.iloc[0]["lambda_B"])
        L_km     = float(p_row.iloc[0]["L_km"])

        csv_path = os.path.join(RESULTS_DIR, way_id, "step7_exposure.csv")
        if not os.path.isfile(csv_path):
            print("  SKIP -- step7_exposure.csv not found")
            continue
        w_arr = pd.read_csv(csv_path)["wind_max_mps"].values

        fail_winds = df_failures[df_failures["way_id"] == way_id]["failure_wind_speed"].tolist()

        line_dir = os.path.join(RESULTS_DIR, way_id)
        os.makedirs(line_dir, exist_ok=True)
        out_path = os.path.join(line_dir, "step13_threshold_sensitivity.png")
        make_line_figure(way_id, sigma, lambda_B, L_km, w_arr, fail_winds, out_path)
        print(f"  Saved: {out_path}")
    print("=" * 70)
    print("STEP 13 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
