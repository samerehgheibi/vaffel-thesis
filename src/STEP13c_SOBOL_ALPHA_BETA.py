"""
STEP 13c — Sobol Sensitivity Pilot, alpha/beta split (single line)
====================================================================
Pilot line : 375924539  (Sorgente-Paterno)

Same model as STEP13b, but instead of bundling the Bayesian posterior
into a single "lambda_B" input (sampled via inverse-CDF of a Uniform),
alpha_post and beta_post are treated as TWO SEPARATE Sobol inputs:

  lambda_B = alpha_post / beta_post   (posterior mean, Eq. 3.8 of the
                                        Axen 2024 thesis, FULLTEXT01.pdf)

Inputs (5):
  L_km      : line length [km]                — Uniform(L_nom * 0.9, L_nom * 1.1)
  w_thresh  : wind speed failure threshold     — Uniform(5, 17)  [m/s]
  sigma     : fragility dispersion             — Uniform(0.3, 0.6)
  alpha_post: Gamma posterior shape            — Uniform(alpha0 * 0.8, alpha0 * 1.2)
  beta_post : Gamma posterior rate             — Uniform(beta0  * 0.8, beta0  * 1.2)

Outputs (3): V50, V90, V95 (same as STEP13b)

This script does NOT overwrite the STEP13b ("lambda_B") results — it
writes to its own output folder so both versions can be compared
side by side.

Requires: pip install SALib
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize
from scipy.stats import norm

from SALib.sample import saltelli
from SALib.analyze import sobol

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# INPUTS / PATHS
# ---------------------------------------------------------------------
BASE_DIR     = BASE
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
STEP6_CSV    = os.path.join(BASE_DIR, "step6_lambda_B.csv")

WAY_ID  = "375924539"   # Sorgente-Paterno
K_YEARS = 10

L_REL_PERTURB     = 0.10        # L_km in [L_nom*0.9, L_nom*1.1]
W_THRESH_BOUNDS   = [5.0, 17.0]
SIGMA_BOUNDS      = [0.3, 0.6]
ALPHA_BETA_PERTURB = 0.20       # alpha_post, beta_post in [post*0.8, post*1.2]

P_LEVELS = {"V50": 0.50, "V90": 0.90, "V95": 0.95}

N_SALTELLI = 512   # -> N * (2*5 + 2) = 6144 model evaluations

OUT_DIR = os.path.join(RESULTS_DIR, WAY_ID, "sobol_pilot_alpha_beta")


# ---------------------------------------------------------------------
# MODEL FUNCTIONS  (consistent with STEP8 / STEP13 / STEP13b)
# ---------------------------------------------------------------------
def exposure_from_wind(w_arr, L_km, w_thresh):
    w = np.asarray(w_arr, dtype=float)
    return np.where(w >= w_thresh, L_km * (w - w_thresh) ** 3, 0.0)


def lognormal_cdf(E_arr, mu, sigma):
    E = np.asarray(E_arr, dtype=float)
    return np.where(
        E > 0,
        norm.cdf((np.log(np.maximum(E, 1e-300)) - np.log(mu)) / sigma),
        0.0,
    )


def fit_mu(E_arr, lambda_B, k_years, sigma):
    """Solve for mu such that mean P(failure) over history == lambda_B (Eq. 3.17)."""
    def obj(log_mu):
        return lognormal_cdf(E_arr, np.exp(log_mu), sigma).sum() / k_years - lambda_B
    try:
        return float(np.exp(brentq(obj, -50.0, 80.0, xtol=1e-10, maxiter=500)))
    except ValueError:
        pos = E_arr[E_arr > 0]
        x0 = np.log(float(np.median(pos)) if len(pos) > 0 else 1.0)
        res = minimize(
            lambda p: (lognormal_cdf(E_arr, np.exp(p[0]), sigma).sum() / k_years - lambda_B) ** 2,
            x0=[x0], method="L-BFGS-B", options={"ftol": 1e-20, "maxiter": 2000},
        )
        return float(np.exp(res.x[0]))


def v_p_from_mu(mu, sigma, L_km, w_thresh, p):
    """Invert fragility curve: wind speed at which P(failure) = p."""
    E_p = mu * np.exp(sigma * norm.ppf(p))
    return w_thresh + (E_p / L_km) ** (1.0 / 3.0)


# ---------------------------------------------------------------------
# LOAD PILOT LINE DATA
# ---------------------------------------------------------------------
def load_pilot_inputs():
    exposure_csv = os.path.join(RESULTS_DIR, WAY_ID, "step7_exposure.csv")
    if not os.path.isfile(exposure_csv):
        raise FileNotFoundError(f"Missing {exposure_csv} - run STEP7 first.")
    w_arr = pd.read_csv(exposure_csv)["wind_max_mps"].values

    df6 = pd.read_csv(STEP6_CSV)
    df6["way_id"] = df6["way_id"].astype(str)
    row = df6[df6["way_id"] == WAY_ID]
    if row.empty:
        raise ValueError(f"way_id {WAY_ID} not found in {STEP6_CSV}")

    L_nom      = float(row.iloc[0]["total_km"])
    alpha_post = float(row.iloc[0]["alpha_post"])
    beta_post  = float(row.iloc[0]["beta_post"])

    return w_arr, L_nom, alpha_post, beta_post


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"STEP 13c - Sobol Pilot, alpha/beta split (line {WAY_ID})")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    w_arr, L_nom, alpha0, beta0 = load_pilot_inputs()

    print(f"L_nom = {L_nom:.2f} km")
    print(f"alpha_post = {alpha0:.4f} -> bounds {[alpha0*(1-ALPHA_BETA_PERTURB), alpha0*(1+ALPHA_BETA_PERTURB)]}")
    print(f"beta_post  = {beta0:.4f} -> bounds {[beta0*(1-ALPHA_BETA_PERTURB), beta0*(1+ALPHA_BETA_PERTURB)]}")

    # -------------------------------------------------------------
    # SALib problem - 5 independent Uniform inputs.
    # lambda_B = alpha_post / beta_post (posterior mean)
    # -------------------------------------------------------------
    problem = {
        "num_vars": 5,
        "names": ["L_km", "w_thresh", "sigma", "alpha_post", "beta_post"],
        "bounds": [
            [L_nom * (1 - L_REL_PERTURB), L_nom * (1 + L_REL_PERTURB)],
            W_THRESH_BOUNDS,
            SIGMA_BOUNDS,
            [alpha0 * (1 - ALPHA_BETA_PERTURB), alpha0 * (1 + ALPHA_BETA_PERTURB)],
            [beta0 * (1 - ALPHA_BETA_PERTURB), beta0 * (1 + ALPHA_BETA_PERTURB)],
        ],
    }

    X = saltelli.sample(problem, N_SALTELLI, calc_second_order=False)
    print(f"Total model evaluations: {X.shape[0]}")

    outputs = {name: np.empty(X.shape[0]) for name in P_LEVELS}

    for i, (L_km, w_thresh, sigma, alpha_post, beta_post) in enumerate(X):
        lambda_B = alpha_post / beta_post
        E_arr = exposure_from_wind(w_arr, L_km, w_thresh)
        mu = fit_mu(E_arr, lambda_B, K_YEARS, sigma)

        for name, p in P_LEVELS.items():
            outputs[name][i] = v_p_from_mu(mu, sigma, L_km, w_thresh, p)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{X.shape[0]} evaluated")

    # -------------------------------------------------------------
    # Sobol analysis per output
    # -------------------------------------------------------------
    results_rows = []
    for name in P_LEVELS:
        Si = sobol.analyze(problem, outputs[name], calc_second_order=False, print_to_console=False)
        for j, pname in enumerate(problem["names"]):
            results_rows.append({
                "output": name,
                "param": pname,
                "S1": Si["S1"][j],
                "S1_conf": Si["S1_conf"][j],
                "ST": Si["ST"][j],
                "ST_conf": Si["ST_conf"][j],
            })

    df_results = pd.DataFrame(results_rows)
    csv_path = os.path.join(OUT_DIR, "sobol_indices_alpha_beta.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df_results.to_string(index=False))

    # -------------------------------------------------------------
    # Bar charts - S1 and ST per output
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, len(P_LEVELS), figsize=(15, 5), sharey=True)
    for ax, name in zip(axes, P_LEVELS):
        sub = df_results[df_results["output"] == name]
        x = np.arange(len(sub))
        width = 0.35
        ax.bar(x - width / 2, sub["S1"], width, yerr=sub["S1_conf"], label="S1", color="#1565c0")
        ax.bar(x + width / 2, sub["ST"], width, yerr=sub["ST_conf"], label="ST", color="#ef5350")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["param"], rotation=30, ha="right")
        ax.set_title(name)
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("Sobol index")
    axes[0].legend()
    fig.suptitle(f"Sobol Sensitivity (alpha/beta split) - line {WAY_ID}", fontweight="bold")
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "sobol_indices_alpha_beta.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    print("\n" + "=" * 70)
    print("STEP 13c complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
