"""
STEP 13d - Merge Sobol plots for comparison (line 375924539)
==============================================================
Combines:
  Row 1: STEP13b results (lambda_B as ONE bundled input)
  Row 2: STEP13c results (alpha_post / beta_post as TWO separate inputs)

into a single figure, side by side, for direct visual comparison.
Does not modify either source PNG/CSV.
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = BASE
WAY_ID = "375924539"
RES_DIR = os.path.join(BASE_DIR, "results", WAY_ID)
STEP6_CSV = os.path.join(BASE_DIR, "step6_lambda_B.csv")

CSV_LAMBDA = os.path.join(RES_DIR, "sobol_pilot_narrow", "sobol_indices.csv")
CSV_ALPHA_BETA = os.path.join(RES_DIR, "sobol_pilot_alpha_beta", "sobol_indices_alpha_beta.csv")

OUT_PNG = os.path.join(RES_DIR, "sobol_indices_merged_lambda_vs_alphabeta.png")

P_LEVELS = ["V50", "V90", "V95"]


def plot_row(axes_row, df, row_label):
    for ax, name in zip(axes_row, P_LEVELS):
        sub = df[df["output"] == name]
        x = np.arange(len(sub))
        width = 0.35
        ax.bar(x - width / 2, sub["S1"], width, yerr=sub["S1_conf"], label="S1", color="#1565c0")
        ax.bar(x + width / 2, sub["ST"], width, yerr=sub["ST_conf"], label="ST", color="#ef5350")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["param"], rotation=30, ha="right")
        ax.set_title(name)
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(-0.05, 1.0)
    axes_row[0].set_ylabel("Sobol index")


def main():
    df_lambda = pd.read_csv(CSV_LAMBDA)
    df_alphabeta = pd.read_csv(CSV_ALPHA_BETA)

    df6 = pd.read_csv(STEP6_CSV)
    df6["way_id"] = df6["way_id"].astype(str)
    row = df6[df6["way_id"] == WAY_ID].iloc[0]
    length_km = row["total_km"]
    voltage_kv = row["voltage_kV"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)

    plot_row(axes[0], df_lambda, "lambda_B bundled")
    plot_row(axes[1], df_alphabeta, "alpha/beta split")

    axes[0][0].legend()
    fig.suptitle(
        f"Sobol Sensitivity Comparison - line {WAY_ID} "
        f"({length_km:.2f} km, {voltage_kv:.0f} kV)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
