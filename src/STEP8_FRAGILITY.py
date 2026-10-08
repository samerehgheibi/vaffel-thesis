"""
STEP 8 — Fragility Curve Fitting  (Eq. 3.14 – 3.19)
=====================================================
Paper equations used:
  Eq. 3.15 : P(failure | E) = Phi( (ln(E) - ln(mu)) / sigma )
  Eq. 3.17 : lambda_B = (1/k) * sum_t( p_t )
  Eq. 3.18 : penalty g(mu, sigma) = ( lambda_B - (1/k)*sum_t(p_t) )^2
  Eq. 3.19 : mu_L, sigma_L = argmin g  using L-BFGS-B

Output per line: one PNG with:
  Row 1 — 2 plots side by side
    Plot 1 — X: wind exposure (alpha=1)        Y: P(failure)
    Plot 2 — X: wind speed 0 to 40 m/s         Y: P(failure)  [no dots]
  Row 2 — table: Exposure E vs Wind Speed [m/s]
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------
STEP6_CSV            = os.path.join(BASE, "step6_lambda_B.csv")
RESULTS_DIR          = os.path.join(BASE, "results")
ASSIGNED_FAILURES_CSV = os.path.join(BASE, "data", "assigned_failures_updated.csv")

W_THRESH   = 15.0
W_MAX_PLOT = 40.0
K_YEARS    = 10
N_POINTS   = 1000

INCLUDED_WAY_IDS = [
    "1155116740", "1155116743", "159158472",  "159158474",
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

def exposure_from_wind(w_arr, alpha_w, L_km):
    """Eq. 3.12 — E = alpha * L * (w - w_thresh)^3"""
    w = np.asarray(w_arr, dtype=float)
    return np.where(w >= W_THRESH, alpha_w * L_km * (w - W_THRESH) ** 3, 0.0)


def wind_from_exposure(E_arr, alpha_w, L_km):
    """Eq. 3.12 inverse — w = w_thresh + (E / (alpha * L))^(1/3)"""
    E = np.asarray(E_arr, dtype=float)
    return np.where(E > 0, W_THRESH + (E / (alpha_w * L_km)) ** (1.0 / 3.0), W_THRESH)


def lognormal_cdf(E_arr, mu, sigma):
    """Eq. 3.15 — P(failure | E) = Phi( (ln(E) - ln(mu)) / sigma )"""
    E = np.asarray(E_arr, dtype=float)
    return np.where(
        E > 0,
        norm.cdf((np.log(np.maximum(E, 1e-300)) - np.log(mu)) / sigma),
        0.0
    )


def penalty(params, E_arr, lambda_B, k_years):
    """Eq. 3.18"""
    mu, sigma = float(params[0]), float(params[1])
    if mu <= 0 or sigma <= 0:
        return 1e12
    mean_p = lognormal_cdf(E_arr, mu, sigma).sum() / k_years
    return (lambda_B - mean_p) ** 2


def fit_fragility(E_arr, lambda_B, k_years):
    """Eq. 3.19 — L-BFGS-B with 5 starting points"""
    pos = E_arr[E_arr > 0]
    mu0 = float(np.median(pos)) if len(pos) > 0 else 1.0

    starts = [
        (mu0,        1.0),
        (mu0 * 0.1,  1.0),
        (mu0 * 10.0, 1.0),
        (mu0,        0.5),
        (mu0,        2.0),
    ]

    best_val, best_mu, best_sigma = 1e18, mu0, 1.0
    for m0, s0 in starts:
        try:
            res = minimize(
                penalty, x0=[m0, s0],
                args=(E_arr, lambda_B, k_years),
                method="L-BFGS-B",
                bounds=[(1e-12, None), (0.3, 10.0)],
                options={"ftol": 1e-20, "gtol": 1e-12, "maxiter": 2000}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val   = res.fun
                best_mu    = float(res.x[0])
                best_sigma = float(res.x[1])
        except Exception:
            pass

    return best_mu, best_sigma


def build_table(E_data_max, alpha1, L_km, n_rows=8):
    """
    Build exposure-to-windspeed table.
    E values: evenly spaced from E_data_max/n_rows to E_data_max.
    """
    E_vals = np.linspace(E_data_max / n_rows, E_data_max, n_rows)
    w_vals = wind_from_exposure(E_vals, alpha1, L_km)
    return E_vals, w_vals


def make_figure(way_id, L_km, w_arr,
                mu, sigma, alpha1,
                E_alpha1, out_path, df_fail):

    avg    = np.mean(w_arr)
    median = np.median(w_arr)
    wmax   = np.max(w_arr)
    E_data_max = float(E_alpha1.max())
    E_at_999   = np.exp(np.log(mu) + norm.ppf(0.999) * sigma)
    x_max      = max(E_data_max, E_at_999)

    title = (
        f"Line: {line_label(way_id)}   |   L = {L_km:.1f} km   |   "
        f"avg = {avg:.1f} m/s   |   median = {median:.1f} m/s   |   max = {wmax:.1f} m/s"
    )

    # --- Curves ---
    # Plot 1: exposure alpha=1  — extend to P(failure) ≈ 0.999
    x1 = np.linspace(0, x_max, N_POINTS)
    p1 = lognormal_cdf(x1, mu, sigma)

    # Plot 2: wind speed W_THRESH to W_MAX_PLOT
    w_range  = np.linspace(W_THRESH, W_MAX_PLOT, N_POINTS)
    E_from_w = exposure_from_wind(w_range, alpha1, L_km)
    p2       = lognormal_cdf(E_from_w, mu, sigma)

    # --- Table data ---
    E_tbl, w_tbl = build_table(x_max, alpha1, L_km, n_rows=8)
    p_tbl = lognormal_cdf(E_tbl, mu, sigma)

    # --- Figure layout: 2 rows ---
    # Row 1: 2 plots  (height ratio 3)
    # Row 2: table    (height ratio 1)
    fig = plt.figure(figsize=(13, 11))
    gs  = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], hspace=0.5, wspace=0.35)

    ax1      = fig.add_subplot(gs[0, 0])
    ax2      = fig.add_subplot(gs[0, 1])
    ax_tbl   = fig.add_subplot(gs[1, :])
    ax_fail  = fig.add_subplot(gs[2, :])

    fig.suptitle(title, fontsize=11, fontweight="bold")

    # ── Plot 1 — Wind Exposure (alpha=1) ──────────────────────────────
    ax1.plot(x1, p1, color="crimson", lw=2.5)
    ax1.set_xlabel("Wind Exposure  (alpha = 1)", fontsize=10)
    ax1.set_ylabel("Probability of failure", fontsize=10)
    ax1.set_title("Wind Exposure (alpha=1)  vs  P(failure)", fontsize=9)
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2 — Wind Speed 0→40 (no dots) ────────────────────────────
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

    # ── Table — Exposure E vs Wind Speed ──────────────────────────────
    ax_tbl.axis("off")

    col_labels = ["Exposure  E", "Wind Speed  [m/s]", "P(fail|E)"]
    table_data = [
        [f"{e:.0f}", f"{w:.2f}", f"{p:.4f}"]
        for e, w, p in zip(E_tbl, w_tbl, p_tbl)
    ]

    tbl = ax_tbl.table(
        cellText  = table_data,
        colLabels = col_labels,
        loc       = "center",
        cellLoc   = "center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Style rows alternating
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
    print("STEP 8 — Fragility Curve Fitting  (Eq. 3.15 – 3.19)")
    print("=" * 70)

    df_s6 = pd.read_csv(STEP6_CSV)
    df_s6["way_id"] = df_s6["way_id"].astype(str)

    df_failures = pd.read_csv(ASSIGNED_FAILURES_CSV)
    df_failures["way_id"] = df_failures["way_id"].astype(str)

    records = []

    for way_id in INCLUDED_WAY_IDS:
        print(f"\nProcessing line: {way_id}")

        line_dir = os.path.join(RESULTS_DIR, way_id)
        csv_path = os.path.join(line_dir, "step7_exposure.csv")

        if not os.path.isfile(csv_path):
            print(f"  SKIP — step7_exposure.csv not found")
            continue

        df  = pd.read_csv(csv_path)
        row = df_s6[df_s6["way_id"] == way_id]
        if row.empty:
            print(f"  SKIP — not found in step6_lambda_B.csv")
            continue

        lambda_B = float(row.iloc[0]["lambda_B"])
        L_km     = float(row.iloc[0]["total_km"])
        w_arr    = df["wind_max_mps"].values

        alpha1   = 1.0
        E_alpha1 = df["exposure_alpha1"].values

        # Fit on E_alpha1 — Eq. 3.19
        print(f"  Fitting ...")
        mu, sigma = fit_fragility(E_alpha1, lambda_B, K_YEARS)

        print(f"  lambda_B = {lambda_B:.6f}")
        print(f"  mu={mu:.4f}  sigma={sigma:.4f}  E_max={E_alpha1.max():.2f}")

        records.append({"way_id": way_id, "mu": mu, "sigma": sigma,
                         "lambda_B": lambda_B, "L_km": L_km})

        out_path = os.path.join(line_dir, "step8_fragility.png")
        make_figure(
            way_id, L_km, w_arr,
            mu, sigma, alpha1,
            E_alpha1, out_path,
            df_failures[df_failures["way_id"] == way_id]
        )

    # Save fitted parameters
    params_path = os.path.join(RESULTS_DIR, "step8_params.csv")
    pd.DataFrame(records).to_csv(params_path, index=False)
    print(f"\nParameters saved: {params_path}")

    print("\n" + "=" * 70)
    print("STEP 8 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
