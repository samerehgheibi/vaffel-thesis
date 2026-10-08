"""
STEP 9 — Fragility Curve  (Segment 3, per-segment mu fit)
==========================================================
sigma fixed from step8_params.csv.  mu fitted per method via Brentq.
w_max used = segment 3's own max wind (not full-line max).

Output per line:
  results/<way_id>/step9_fragility_seg3.png
  Row 1 -- 3 plots (exposure alpha=1 | wind speed | normalized 0-5)
  Row 2 -- table: Exposure E | Wind Speed [m/s] | P(fail|E)
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
TARGET     = 5.0
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


# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------

def exposure_from_wind(w_arr, L_km):
    """E = L * (w - w_thresh)^3  [Eq. 3.12, alpha=1]"""
    w = np.asarray(w_arr, dtype=float)
    return np.where(w >= W_THRESH, L_km * (w - W_THRESH) ** 3, 0.0)


def wind_from_exposure(E_arr, L_km):
    """w = w_thresh + (E / L)^(1/3)  [Eq. 3.12 inverse, alpha=1]"""
    E = np.asarray(E_arr, dtype=float)
    return np.where(E > 0, W_THRESH + (E / L_km) ** (1.0 / 3.0), W_THRESH)


def lognormal_cdf(E_arr, mu, sigma):
    """P(failure | E) = Phi( (ln E - ln mu) / sigma )  [Eq. 3.15]"""
    E = np.asarray(E_arr, dtype=float)
    return np.where(
        E > 0,
        norm.cdf((np.log(np.maximum(E, 1e-300)) - np.log(mu)) / sigma),
        0.0,
    )


def fit_mu_fixed_sigma(E_arr, lambda_B, k_years, sigma):
    """Fit mu with sigma fixed (Eq. 3.19 reduced to 1-D Brentq).
    Target: sum(p_t) / k_years = lambda_B"""
    def obj(log_mu):
        mu = np.exp(log_mu)
        return lognormal_cdf(E_arr, mu, sigma).sum() / k_years - lambda_B

    try:
        log_mu = brentq(obj, -30.0, 60.0, xtol=1e-10, maxiter=500)
        return float(np.exp(log_mu))
    except ValueError:
        # fallback: L-BFGS-B on both if Brentq bracket fails
        from scipy.optimize import minimize
        res = minimize(
            lambda p: (lognormal_cdf(E_arr, np.exp(p[0]), sigma).sum() / k_years - lambda_B) ** 2,
            x0=[np.log(float(np.median(E_arr[E_arr > 0])) if (E_arr > 0).any() else 1.0)],
            method="L-BFGS-B",
            options={"ftol": 1e-20, "maxiter": 2000},
        )
        return float(np.exp(res.x[0]))


def build_table(E_data_max, L_km, mu, sigma, n_rows=8):
    E_vals = np.linspace(E_data_max / n_rows, E_data_max, n_rows)
    w_vals = wind_from_exposure(E_vals, L_km)
    p_vals = lognormal_cdf(E_vals, mu, sigma)
    return E_vals, w_vals, p_vals


def make_figure(way_id, seg3_km, w_max_seg3,
                mu, mu_norm, sigma, out_path, df_fail):

    # E_data_max uses segment 3's own w_max
    E_data_max = float(seg3_km * max(w_max_seg3 - W_THRESH, 0.0) ** 3)

    # Plot 1 — exposure alpha=1
    x1       = np.linspace(0, E_data_max, N_POINTS)
    p1       = lognormal_cdf(x1, mu, sigma)

    # Plot 2 — wind speed
    w_range  = np.linspace(W_THRESH, W_MAX_PLOT, N_POINTS)
    E_from_w = exposure_from_wind(w_range, seg3_km)
    p2       = lognormal_cdf(E_from_w, mu, sigma)

    # Plot 3 — normalized exposure 0 → TARGET
    x3 = np.linspace(0, TARGET, N_POINTS)
    p3 = lognormal_cdf(x3, mu_norm, sigma)

    # Table
    E_tbl, w_tbl, p_tbl = build_table(E_data_max, seg3_km, mu, sigma, n_rows=8)

    title = (
        f"Line: {line_label(way_id)}   |   Seg3 L = {seg3_km:.4f} km   |   "
        f"w_max seg3 = {w_max_seg3:.2f} m/s"
    )

    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.5, wspace=0.35)
    ax1     = fig.add_subplot(gs[0, 0])
    ax2     = fig.add_subplot(gs[0, 1])
    ax3     = fig.add_subplot(gs[0, 2])
    ax_tbl  = fig.add_subplot(gs[1, :])
    ax_fail = fig.add_subplot(gs[2, :])

    fig.suptitle(title, fontsize=11, fontweight="bold")

    ax1.plot(x1, p1, color="crimson", lw=2.5)
    ax1.set_xlabel("Wind Exposure  (alpha = 1)", fontsize=10)
    ax1.set_ylabel("Probability of failure", fontsize=10)
    ax1.set_title("Wind Exposure (alpha=1)  vs  P(failure)", fontsize=9)
    ax1.set_xlim(0, E_data_max)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.plot(w_range, p2, color="steelblue", lw=2.5)
    ax2.axvline(W_THRESH, color="gray", ls="--", lw=1,
                label=f"w_thresh = {W_THRESH} m/s")
    ax2.set_xlabel("Wind Speed  [m/s]", fontsize=10)
    ax2.set_ylabel("Probability of failure", fontsize=10)
    ax2.set_title("Wind Speed  vs  P(failure)", fontsize=9)
    ax2.set_xlim(W_THRESH - 1, W_MAX_PLOT)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3.plot(x3, p3, color="seagreen", lw=2.5)
    ax3.set_xlabel("Normalized Wind Exposure", fontsize=10)
    ax3.set_ylabel("Probability of failure", fontsize=10)
    ax3.set_title("Normalized Exposure  vs  P(failure)", fontsize=9)
    ax3.set_xlim(0, TARGET)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)

    ax_tbl.axis("off")
    col_labels = ["Exposure  E", "Wind Speed  [m/s]", "P(fail|E)"]
    table_data = [
        [f"{e:.4f}", f"{w:.2f}", f"{p:.4f}"]
        for e, w, p in zip(E_tbl, w_tbl, p_tbl)
    ]
    tbl = ax_tbl.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor("#eaf3fb" if i % 2 == 0 else "white")

    # ── Table 2 — Actual Failures ─────────────────────────────────────
    ax_fail.axis("off")
    ax_fail.set_title("Actual Failures (assigned)", fontsize=10, fontweight="bold", loc="left")
    if df_fail is not None and len(df_fail) > 0:
        fail_data = [[str(r["date"]), f"{r['failure_wind_speed']:.2f}"]
                     for _, r in df_fail.sort_values("failure_wind_speed").iterrows()]
        fail_tbl = ax_fail.table(
            cellText=fail_data,
            colLabels=["Date", "Wind Speed [m/s]"],
            loc="center", cellLoc="center"
        )
        fail_tbl.auto_set_font_size(False)
        fail_tbl.set_fontsize(10)
        fail_tbl.scale(0.5, 1.5)
        for j in range(2):
            fail_tbl[0, j].set_facecolor("#2c5f8a")
            fail_tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(fail_data) + 1):
            for j in range(2):
                fail_tbl[i, j].set_facecolor("#eaf3fb" if i % 2 == 0 else "white")
    else:
        ax_fail.text(0.5, 0.5, "No assigned failures for this line",
                     ha="center", va="center", fontsize=10, transform=ax_fail.transAxes)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 9 -- Fragility Curve  (Segment 3, per-segment mu fit)")
    print("=" * 70)

    df_params = pd.read_csv(STEP8_PARAMS)
    df_params["way_id"] = df_params["way_id"].astype(str)

    df_failures = pd.read_csv(ASSIGNED_FAILURES_CSV)
    df_failures["way_id"] = df_failures["way_id"].astype(str)

    for way_id in INCLUDED_WAY_IDS:
        print(f"\nProcessing line: {line_label(way_id)}")

        p_row = df_params[df_params["way_id"] == way_id]
        if p_row.empty:
            print(f"  SKIP -- not found in step8_params.csv")
            continue

        # sigma fixed from STEP 8 full-line fit
        sigma    = float(p_row.iloc[0]["sigma"])
        lambda_B = float(p_row.iloc[0]["lambda_B"])

        # w_max of full line
        step7_csv = os.path.join(RESULTS_DIR, way_id, "step7_exposure.csv")
        if not os.path.isfile(step7_csv):
            print(f"  SKIP -- step7_exposure.csv not found")
            continue
        w_max_line = float(pd.read_csv(step7_csv)["wind_max_mps"].max())

        # Segment 3 data -> derive seg3_km and hourly wind time series
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

        # Hourly max wind per time step for segment 3
        df_seg["time"] = pd.to_datetime(df_seg["time"])
        w_arr = (
            df_seg.groupby("time")["wind_speed_100m"]
            .max()
            .values
        )
        w_max_seg3 = float(w_arr.max())

        # Method 1 — alpha=1
        E_alpha1 = exposure_from_wind(w_arr, seg3_km)
        mu = fit_mu_fixed_sigma(E_alpha1, lambda_B, K_YEARS, sigma)

        # Method 2 — normalized (alpha_norm so max E = TARGET)
        cube_ref   = max(w_max_seg3 - W_THRESH, 0.0) ** 3
        alpha_norm = TARGET / (seg3_km * cube_ref) if cube_ref > 0 else 1.0
        E_alpha_norm = np.where(
            w_arr >= W_THRESH,
            alpha_norm * seg3_km * (w_arr - W_THRESH) ** 3,
            0.0
        )
        mu_norm = fit_mu_fixed_sigma(E_alpha_norm, lambda_B, K_YEARS, sigma)

        print(f"  seg3_km     = {seg3_km:.4f} km")
        print(f"  w_max_seg3  = {w_max_seg3:.3f} m/s")
        print(f"  sigma (S8)  = {sigma:.4f}")
        print(f"  mu (M1)     = {mu:.4f}   mu_norm (M2) = {mu_norm:.4f}")

        line_dir = os.path.join(RESULTS_DIR, way_id)
        os.makedirs(line_dir, exist_ok=True)
        out_path = os.path.join(line_dir, "step9_fragility_seg3.png")

        make_figure(way_id, seg3_km, w_max_seg3,
                    mu, mu_norm, sigma, out_path,
                    df_failures[df_failures["way_id"] == way_id])

    print("\n" + "=" * 70)
    print("STEP 9 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
