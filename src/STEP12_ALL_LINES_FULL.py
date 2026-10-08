"""
STEP 12 — All Lines: Fragility Curves (Full Line)
==================================================
Same layout as STEP 11 but uses full-line mu/sigma from STEP 8.
  X: Wind Speed [m/s]  (inverse of Eq. 3.12 with full L_km)
  Y: Probability of failure  [Eq. 3.15]
Each line gets a distinct colour.
Table: w_max full line + actual failure wind speeds.
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import norm
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------
RESULTS_DIR           = os.path.join(BASE, "results")
STEP8_PARAMS          = os.path.join(BASE, "results", "step8_params.csv")
ASSIGNED_FAILURES_CSV = os.path.join(BASE, "data", "assigned_failures_updated.csv")

W_THRESH   = 15.0
W_MAX_PLOT = 40.0
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
# FUNCTIONS
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


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 12 -- All Lines Fragility (Full Line)")
    print("=" * 70)

    df_params = pd.read_csv(STEP8_PARAMS)
    df_params["way_id"] = df_params["way_id"].astype(str)

    df_failures = pd.read_csv(ASSIGNED_FAILURES_CSV)
    df_failures["way_id"] = df_failures["way_id"].astype(str)

    w_curve = np.linspace(W_THRESH, W_MAX_PLOT, N_POINTS)

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.5)
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

        mu       = float(p_row.iloc[0]["mu"])
        sigma    = float(p_row.iloc[0]["sigma"])
        L_km     = float(p_row.iloc[0]["L_km"])
        lambda_B = float(p_row.iloc[0]["lambda_B"])

        # w_max from step7_exposure.csv
        csv_path = os.path.join(RESULTS_DIR, way_id, "step7_exposure.csv")
        if not os.path.isfile(csv_path):
            print(f"  SKIP -- step7_exposure.csv not found")
            continue
        w_max = float(pd.read_csv(csv_path)["wind_max_mps"].max())

        # Plot: wind speed = inverse of exposure (full L_km)
        E_curve = exposure_from_wind(w_curve, L_km)
        p_curve = lognormal_cdf(E_curve, mu, sigma)
        ax.plot(w_curve, p_curve, lw=1.8, color=color, label=line_label(way_id))

        # km label — staggered y to avoid overlap
        w50 = W_THRESH + (mu / L_km) ** (1.0 / 3.0)
        p_label = 0.10 + (idx % 8) * 0.10
        idx_lbl = np.argmin(np.abs(p_curve - p_label))
        ax.text(w_curve[idx_lbl], p_label, f" {L_km:.0f} km",
                fontsize=7, color=color, fontweight="bold",
                rotation=90, va="bottom", ha="center")

        # Table row — store w50 for sorting, color_idx for coloring


        line_fails = df_failures[df_failures["way_id"] == way_id]["failure_wind_speed"].tolist()
        fail_str = ", ".join(f"{v:.2f}" for v in sorted(line_fails)) if line_fails else "—"
        table_rows.append((w50, idx, [line_label(way_id), f"{L_km:.1f}", f"{lambda_B:.4f}", f"{w_max:.2f}", fail_str]))

        print(f"  L_km={L_km:.2f}  w_max={w_max:.2f}  mu={mu:.4f}  sigma={sigma:.4f}")

    # ── Plot style ────────────────────────────────────────────────────
    ax.axvline(W_THRESH, color="gray", ls="--", lw=1, alpha=0.6,
               label=f"w_thresh = {W_THRESH} m/s")
    ax.set_xlabel("Wind Speed  [m/s]", fontsize=12)
    ax.set_ylabel("Probability of Failure", fontsize=12)
    ax.set_title("Fragility Curves — All Lines (Full Line)", fontsize=12, fontweight="bold")
    ax.set_xlim(W_THRESH - 1, W_MAX_PLOT)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, framealpha=0.8)
    ax.grid(True, linestyle=":", alpha=0.5)

    # ── Table — sorted left-to-right by w50 ──────────────────────────
    ax_tbl.axis("off")

    table_rows_sorted = sorted(table_rows, key=lambda x: x[0])

    col_labels = ["Line ID", "Length [km]", "λ_B [/yr]", "w_max Full Line [m/s]", "Actual Failure Wind Speeds [m/s]"]
    tbl = ax_tbl.table(
        cellText=[r[2] for r in table_rows_sorted],
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, (_, color_idx, _) in enumerate(table_rows_sorted):
        row_color = COLORS[color_idx] + "33"
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(row_color)

    out_path = os.path.join(RESULTS_DIR, "step12_all_lines_full.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    print("=" * 70)
    print("STEP 12 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
