"""
STEP 14 — Fragility curves with P=1% and P=99% markers
=========================================================
X: Wind Speed [m/s]
Y: Probability of Failure
Marks the 1% and 99% wind speeds on each curve.
Table below: line | w(1%) | w(99%)
Separate figures for 220 kV and 380 kV.
Uses threshold = 15 m/s (from STEP 8).
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
RESULTS_DIR  = os.path.join(BASE, "results")
STEP8_PARAMS = os.path.join(BASE, "results", "step8_params.csv")

W_THRESH   = 15.0
W_MAX_PLOT = 40.0
N_POINTS   = 1000

LINES_220 = [
    "1155116743",
    "338850183",  "338856348",  "375907989", "375922074",
]
LINES_380 = ["375916002", "456822447", "375924539"]

COLORS_220 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd",
]
COLORS_380 = ["#1f77b4", "#ff7f0e", "#2ca02c"]

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


def w_at_p(p, mu, sigma, L_km):
    """Wind speed at a given failure probability."""
    z = norm.ppf(p)
    E = mu * np.exp(z * sigma)
    if E <= 0:
        return W_THRESH
    return W_THRESH + (E / L_km) ** (1.0 / 3.0)


# ---------------------------------------------------------------------
# MAKE FIGURE
# ---------------------------------------------------------------------

def make_figure(way_ids, colors, voltage_label, out_path, df_params):

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.5)
    ax     = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    table_rows = []   # (w1, color, [row_data])

    for idx, way_id in enumerate(way_ids):
        color = colors[idx]

        p_row = df_params[df_params["way_id"] == way_id]
        if p_row.empty:
            continue

        mu    = float(p_row.iloc[0]["mu"])
        sigma = float(p_row.iloc[0]["sigma"])
        L_km  = float(p_row.iloc[0]["L_km"])

        csv_path = os.path.join(RESULTS_DIR, way_id, "step7_exposure.csv")
        w_max = float(pd.read_csv(csv_path)["wind_max_mps"].max()) \
                if os.path.isfile(csv_path) else np.nan

        w_plot  = np.linspace(0, W_MAX_PLOT, N_POINTS)
        E_curve = exposure_from_wind(w_plot, L_km)
        p_curve = lognormal_cdf(E_curve, mu, sigma)

        ax.plot(w_plot, p_curve, lw=2.0, color=color, label=line_label(way_id))

        # 1%, Q1, Q3, 99% markers
        w1  = w_at_p(0.01, mu, sigma, L_km)
        wq1 = w_at_p(0.25, mu, sigma, L_km)
        wq3 = w_at_p(0.75, mu, sigma, L_km)
        w99 = w_at_p(0.99, mu, sigma, L_km)

        ax.plot(w1,  0.01, "o", color=color, ms=6)
        ax.plot(wq1, 0.25, "^", color=color, ms=6)
        ax.plot(wq3, 0.75, "v", color=color, ms=6)
        ax.plot(w99, 0.99, "s", color=color, ms=6)

        # P at w_max
        E_wmax     = float(exposure_from_wind(np.array([w_max]), L_km)[0])
        p_wmax     = float(lognormal_cdf(np.array([E_wmax]), mu, sigma)[0])
        p_wmax_str = f"{p_wmax*100:.1f}%"

        table_rows.append((w1, color, [line_label(way_id), f"{L_km:.2f}",
                           f"{w1:.2f}", f"{wq1:.2f}", f"{wq3:.2f}", f"{w99:.2f}",
                           f"{w_max:.2f}", p_wmax_str]))

    # horizontal reference lines
    for p_val, label, va in [(0.01, "1%", "bottom"), (0.25, "25%", "bottom"),
                              (0.75, "75%", "bottom"), (0.99, "99%", "top")]:
        ax.axhline(p_val, color="gray", ls=":", lw=1, alpha=0.6)
        ax.text(W_MAX_PLOT - 0.5, p_val, label, va=va, ha="right",
                fontsize=8, color="gray")

    ax.set_xlabel("Wind Speed  [m/s]", fontsize=12)
    ax.set_ylabel("Probability of Failure", fontsize=12)
    ax.set_title(f"Fragility Curves — {voltage_label}", fontsize=12, fontweight="bold")
    ax.set_xlim(0, W_MAX_PLOT)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=1, loc="upper left", framealpha=0.8)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Table — sorted left-to-right by w1
    table_rows_sorted = sorted(table_rows, key=lambda x: x[0])
    col_labels = ["Line", "L [km]", "w (1%) [m/s]", "w (25%) [m/s]",
                  "w (75%) [m/s]", "w (99%) [m/s]", "w_max [m/s]", "P at w_max"]
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=[r[2] for r in table_rows_sorted],
        colLabels=col_labels,
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, (_, color, _) in enumerate(table_rows_sorted):
        row_color = color + "33"
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(row_color)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 14 -- Fragility: 1% and 99% markers")
    print("=" * 70)

    df_params = pd.read_csv(STEP8_PARAMS)
    df_params["way_id"] = df_params["way_id"].astype(str)

    make_figure(LINES_220, COLORS_220, "220 kV",
                os.path.join(RESULTS_DIR, "step14_220kv.png"), df_params)

    make_figure(LINES_380, COLORS_380, "380 kV",
                os.path.join(RESULTS_DIR, "step14_380kv.png"), df_params)

    print("\n" + "=" * 70)
    print("STEP 14 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
