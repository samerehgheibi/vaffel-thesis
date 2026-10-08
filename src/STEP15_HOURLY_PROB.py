"""
STEP 15 — Hourly Probabilities of Failure due to Wind 2013–2023
================================================================
For each line and each hour:
    P(t) = lognormal_CDF( exposure_alpha1[t] )

Two output figures (split by voltage):
    results/step15_hourly_prob_220kv.png
    results/step15_hourly_prob_380kv.png

Below each plot: summary table — per year, per line:
  max hourly probability  |  ★ = failure assigned that year
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
from scipy.stats import norm
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ── CONFIG ────────────────────────────────────────────────────────────
RESULTS_DIR           = os.path.join(BASE, "results")
STEP8_PARAMS          = os.path.join(BASE, "results", "step8_params.csv")
ASSIGNED_FAILURES_CSV = os.path.join(BASE, "data", "assigned_failures_updated.csv")

LINES_220 = ["1155116743", "338850183", "338856348", "375907989", "375922074"]
LINES_380 = ["375916002", "456822447", "375924539"]

COLORS_220 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
COLORS_380 = ["#1f77b4", "#ff7f0e", "#2ca02c"]

LINE_NAMES = {
    "1155116743": "Fulgatore-Partanna",
    "338850183":  "Line 22.250/22.251",
    "338856348":  "Line 22.245/22.246",
    "375907989":  "Line 22.239/22.252",
    "375922074":  "Melilli-Ragusa/1-2",
    "375916002":  "Priolo-Chiaramonte Gulfi",
    "456822447":  "Sorgente-Rizziconi",
    "375924539":  "Sorgente - Paternò",
}

SHORT_NAMES = {
    "1155116743": "Fulgatore\nPartanna",
    "338850183":  "338850183\nLine 22.250/22.251",
    "338856348":  "338856348\nLine 22.245/22.246",
    "375907989":  "375907989\nLine 22.239/22.252",
    "375922074":  "Melilli\nRagusa/1-2",
    "375916002":  "Priolo\nChiaramonte",
    "456822447":  "Sorgente\nRizziconi",
    "375924539":  "Sorgente\nPaternò",
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
    name = LINE_NAMES.get(str(wid), str(wid))
    circuits = LINE_CIRCUITS.get(str(wid))
    circ_label = {1: "single circuit", 2: "double circuit"}.get(circuits, "")
    circ_tag = f", {circ_label}" if circ_label else ""
    return f"{name}  [{wid}{circ_tag}]"


def lognormal_cdf(E_arr, mu, sigma):
    E = np.asarray(E_arr, dtype=float)
    return np.where(
        E > 0,
        norm.cdf((np.log(np.maximum(E, 1e-300)) - np.log(mu)) / sigma),
        0.0,
    )


def make_figure(way_ids, colors, voltage_label, out_path, df_params, df_failures):

    # ── Per-line per-year data for the table ──────────────────────────
    # line_data[way_id] = { year: (max_P, peak_date_str) }
    line_data   = {}
    y_max_all   = 0.0
    plot_series = []   # (way_id, color, times, P)

    for idx, way_id in enumerate(way_ids):
        color = colors[idx]
        p_row = df_params[df_params["way_id"] == way_id]
        if p_row.empty:
            continue
        mu    = float(p_row.iloc[0]["mu"])
        sigma = float(p_row.iloc[0]["sigma"])

        csv_path = os.path.join(RESULTS_DIR, way_id, "step7_exposure.csv")
        if not os.path.isfile(csv_path):
            continue

        df_exp = pd.read_csv(csv_path, parse_dates=["time"])
        P      = lognormal_cdf(df_exp["exposure_alpha1"].values, mu, sigma)

        local_max = float(np.nanmax(P))
        y_max_all = max(y_max_all, local_max)
        print(f"  {line_label(way_id)} — peak P = {local_max:.4f}")

        plot_series.append((way_id, color, df_exp["time"], P))

        # Aggregate per year
        df_exp["P"]    = P
        df_exp["year"] = df_exp["time"].dt.year
        yr_data = {}
        for yr, grp in df_exp.groupby("year"):
            idx_max  = grp["P"].idxmax()
            max_p    = float(grp["P"].max())
            peak_dt  = grp.loc[idx_max, "time"].strftime("%d %b")
            yr_data[yr] = (max_p, peak_dt)
        line_data[way_id] = yr_data

    all_years = sorted({yr for yd in line_data.values() for yr in yd})

    # ── Layout ────────────────────────────────────────────────────────
    n_lines   = len(way_ids)
    tbl_h     = max(2.5, 0.36 * len(all_years))
    fig       = plt.figure(figsize=(18, 5 + tbl_h))
    gs        = fig.add_gridspec(2, 1,
                                 height_ratios=[5, tbl_h],
                                 hspace=0.55)
    ax        = fig.add_subplot(gs[0])
    ax_tbl    = fig.add_subplot(gs[1])

    # ── Time-series plot ──────────────────────────────────────────────
    for way_id, color, times, P in plot_series:
        ax.plot(times, P, lw=0.7, color=color, alpha=0.85,
                label=line_label(way_id))

    for idx, way_id in enumerate(way_ids):
        color      = colors[idx]
        line_fails = df_failures[df_failures["way_id"] == way_id]
        for _, row in line_fails.iterrows():
            ax.axvline(pd.to_datetime(row["peak_time"]),
                       color=color, lw=1.3, ls="--", alpha=0.75, zorder=5)

    y_top = min(y_max_all * 1.18, 1.0) if y_max_all > 0 else 0.01
    ax.set_ylim(0, y_top)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.set_xlabel("Date and Time", fontsize=11)
    ax.set_ylabel("Hourly Probability", fontsize=11)
    ax.set_title(
        f"Hourly Probabilities of Failure due to Wind  —  {voltage_label}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8.5, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              framealpha=0.88)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.grid(True, axis="x", linestyle=":", alpha=0.20)

    # ── Summary table ─────────────────────────────────────────────────
    ax_tbl.axis("off")

    col_labels = ["Year\n(★ = failure)"] + [SHORT_NAMES.get(w, w) for w in way_ids]
    col_colors = ["#2c5f8a"] + list(colors[:n_lines])

    cell_text  = []
    cell_color = []

    for yr in all_years:
        row_text  = [str(yr)]
        row_color = ["#f0f0f0"]

        for idx, way_id in enumerate(way_ids):
            line_color = colors[idx]
            yr_info    = line_data.get(way_id, {}).get(yr)

            # Check if there was a failure this year for this line
            has_fail = not df_failures[
                (df_failures["way_id"] == way_id) &
                (df_failures["peak_time"].dt.year == yr)
            ].empty

            if yr_info is None:
                cell_val   = "—"
                cell_bg    = "#ffffff"
            else:
                max_p, peak_dt = yr_info
                if max_p < 0.001:
                    p_str    = "< 0.1%"
                else:
                    p_str    = f"{max_p*100:.1f}%"
                star         = "  ★" if has_fail else ""
                cell_val     = f"{p_str}{star}\n{peak_dt}"
                cell_bg      = (line_color + "40") if has_fail else "#ffffff"

            row_text.append(cell_val)
            row_color.append(cell_bg)

        cell_text.append(row_text)
        cell_color.append(row_color)

    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_color,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.2)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    # Header row styling
    for j, bg in enumerate(col_colors):
        cell = tbl[0, j]
        cell.set_facecolor(bg)
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    # Year column styling
    for i in range(len(all_years)):
        tbl[i + 1, 0].set_facecolor("#e8eef4")
        tbl[i + 1, 0].set_text_props(fontweight="bold", fontsize=9)

    pass

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── MAIN ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 15 — Hourly Probabilities of Failure due to Wind")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_params = pd.read_csv(STEP8_PARAMS)
    df_params["way_id"] = df_params["way_id"].astype(str)

    df_failures = pd.read_csv(ASSIGNED_FAILURES_CSV, parse_dates=["peak_time"])
    df_failures["way_id"] = df_failures["way_id"].astype(str)

    all_included = set(LINES_220 + LINES_380)
    df_failures  = df_failures[df_failures["way_id"].isin(all_included)].copy()

    print(f"\n── 220 kV ({len(LINES_220)} lines) ──")
    make_figure(
        LINES_220, COLORS_220, "220 kV",
        os.path.join(RESULTS_DIR, "step15_hourly_prob_220kv.png"),
        df_params, df_failures,
    )

    print(f"\n── 380 kV ({len(LINES_380)} lines) ──")
    make_figure(
        LINES_380, COLORS_380, "380 kV",
        os.path.join(RESULTS_DIR, "step15_hourly_prob_380kv.png"),
        df_params, df_failures,
    )

    print("\n" + "=" * 70)
    print("STEP 15 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
