"""
STEP 7 — Wind Exposure  (Eq. 3.12)
====================================
Goal: Compute wind exposure for each transmission line using two alpha values.

Paper equations used:
  Eq. 3.12 (forward) : wind exposure
             E(t,i) = alpha_w * L_i * (w(t,i) - w_thresh)^3   if w >= w_thresh
             E(t,i) = 0                                         if w < w_thresh

  Eq. 3.12 (inverse) : wind speed from exposure
             w = w_thresh + (E / (alpha_w * L))^(1/3)
             Used to recover equivalent wind speed from any exposure value.

Two methods:
  Method 1 (alpha=1)    : raw exposure, no normalization
  Method 2 (alpha=norm) : alpha chosen so that max exposure = TARGET = 5.0
                          alpha_norm = TARGET / (L_km * (w_max - w_thresh)^3)

CSV columns saved per line (for use in fragility curve step):
  time                 - timestamp
  wind_max_mps         - hourly max wind speed across all towers [m/s]
  exposure_alpha1      - Eq. 3.12 with alpha=1
  exposure_alpha_norm  - Eq. 3.12 with normalized alpha
  w_equiv_alpha1       - inverse Eq. 3.12: wind equivalent of exposure_alpha1
  w_equiv_alpha_norm   - inverse Eq. 3.12: wind equivalent of exposure_alpha_norm

Output structure:
  results/
    <way_id>/
      step7_exposure.csv
      step7_plot.png
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# INPUTS  (read from archive)
# ---------------------------------------------------------------------
ARCHIVE      = os.path.join(BASE, "data")
PARQUET_PATH = ARCHIVE + r"\merged_towers_wind_new.parquet"

# Step 6 output is already in the working folder (not archive)
STEP6_CSV = os.path.join(BASE, "step6_lambda_B.csv")

# Eq. 3.12 parameters
W_THRESH = 15.0   # minimum wind speed threshold [m/s]
TARGET   = 5.0    # normalization target: max exposure = 5.0 (Method 2)

# 8 lines to process (3 shortest archived)
INCLUDED_WAY_IDS = [
    "1155116743",
    "338850183",
    "338856348",
    "375907989",
    "375922074",
    "375916002",
    "456822447",
    "375924539",
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
# OUTPUT
# ---------------------------------------------------------------------
OUT_DIR = os.path.join(BASE, "results")


# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------

def compute_exposure(w_arr, alpha_w, L_km):
    """
    Eq. 3.12 (forward) — Wind exposure.
    E = alpha_w * L * (w - w_thresh)^3  if w >= w_thresh, else 0
    """
    w = np.asarray(w_arr, dtype=float)
    return np.where(w >= W_THRESH, alpha_w * L_km * (w - W_THRESH) ** 3, 0.0)


def exposure_to_wind(E_arr, alpha_w, L_km):
    """
    Eq. 3.12 (inverse) — Equivalent wind speed from exposure.
    Derived from Eq. 3.12:
      E = alpha_w * L * (w - w_thresh)^3
      w = w_thresh + (E / (alpha_w * L))^(1/3)
    Returns 0 where E = 0 (below threshold).
    """
    E = np.asarray(E_arr, dtype=float)
    return np.where(E > 0, W_THRESH + (E / (alpha_w * L_km)) ** (1.0 / 3.0), 0.0)


def get_line_wind(parquet_path, way_id, con):
    """
    Load hourly max wind speed across all towers for a given line.
    Paper simplification: use the segment with max wind per hour (Section 3.3.4).
    """
    df = con.execute(f"""
        SELECT time, MAX(wind_speed_100m) AS wind_max_mps
        FROM '{parquet_path}'
        WHERE way_id = '{way_id}'
        GROUP BY time
        ORDER BY time
    """).df()
    df["time"] = pd.to_datetime(df["time"])
    df["wind_max_mps"] = pd.to_numeric(df["wind_max_mps"], errors="coerce")
    df = df.dropna(subset=["wind_max_mps"]).reset_index(drop=True)
    return df


def get_line_length(step6_csv, way_id):
    """Load total line length (km) from Step 6 output."""
    df = pd.read_csv(step6_csv)
    df["way_id"] = df["way_id"].astype(str)
    row = df[df["way_id"] == way_id]
    if row.empty:
        raise ValueError(f"way_id {way_id} not found in {step6_csv}")
    return float(row.iloc[0]["total_km"])


def make_plot(way_id, df, alpha_norm, L_km, w_max, out_path):
    """
    Two-panel plot:
      Left  - Eq. 3.12 with alpha=1       (Method 1)
      Right - Eq. 3.12 with alpha_norm    (Method 2, max exposure = 5)
    X axis: wind speed [m/s]
    Y axis: wind exposure E(t)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Step 7 — Wind Exposure  (Eq. 3.12)   |   Line: {line_label(way_id)}   |   "
        f"L = {L_km:.2f} km   |   w_max = {w_max:.1f} m/s",
        fontsize=10
    )

    # --- Left: Method 1 (alpha=1) ---
    ax1.scatter(
        df["wind_max_mps"], df["exposure_alpha1"],
        s=6, alpha=0.3, color="steelblue"
    )
    ax1.set_title(
        "Method 1  —  alpha_w = 1\n"
        "Eq. 3.12: E = 1 * L * (w - w_thresh)^3",
        fontsize=9
    )
    ax1.set_xlabel("Wind Speed [m/s]")
    ax1.set_ylabel("Wind Exposure  E(t)")
    ax1.axvline(W_THRESH, color="gray", ls="--", lw=1, label=f"w_thresh = {W_THRESH} m/s")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Right: Method 2 (normalized alpha) ---
    ax2.scatter(
        df["wind_max_mps"], df["exposure_alpha_norm"],
        s=6, alpha=0.3, color="darkorange"
    )
    ax2.set_title(
        f"Method 2  —  alpha_w = {alpha_norm:.5f}\n"
        f"Eq. 3.12: normalized so that max(E) = {TARGET}",
        fontsize=9
    )
    ax2.set_xlabel("Wind Speed [m/s]")
    ax2.set_ylabel("Wind Exposure  E(t)")
    ax2.axvline(W_THRESH, color="gray", ls="--", lw=1, label=f"w_thresh = {W_THRESH} m/s")
    ax2.axhline(TARGET,   color="red",  ls=":",  lw=1, label=f"target = {TARGET}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 7 — Wind Exposure  (Eq. 3.12)")
    print("=" * 70)
    print(f"W_THRESH = {W_THRESH} m/s")
    print(f"TARGET   = {TARGET}  (max exposure for normalized alpha)")
    print(f"Lines    : {len(INCLUDED_WAY_IDS)}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)
    con = duckdb.connect()

    for way_id in INCLUDED_WAY_IDS:
        print(f"Processing line: {line_label(way_id)}")

        # --- Create output folder for this line ---
        line_dir = os.path.join(OUT_DIR, way_id)
        os.makedirs(line_dir, exist_ok=True)

        # --- Load data ---
        df   = get_line_wind(PARQUET_PATH, way_id, con)
        L_km = get_line_length(STEP6_CSV, way_id)

        w_arr = df["wind_max_mps"].values
        w_max = float(w_arr.max())

        # --- Method 1: Eq. 3.12 (forward) with alpha_w = 1 ---
        alpha1 = 1.0
        df["exposure_alpha1"] = compute_exposure(w_arr, alpha1, L_km)

        # --- Method 2: normalized alpha so that max exposure = TARGET = 5.0
        # alpha_norm = TARGET / (L_km * (w_max - w_thresh)^3)
        cube_ref   = (w_max - W_THRESH) ** 3
        alpha_norm = TARGET / (L_km * cube_ref)
        df["exposure_alpha_norm"] = compute_exposure(w_arr, alpha_norm, L_km)

        # --- Eq. 3.12 (inverse): recover wind speed from exposure ---
        # w = w_thresh + (E / (alpha_w * L))^(1/3)
        # Saved for use in fragility curve step
        df["w_equiv_alpha1"]      = exposure_to_wind(df["exposure_alpha1"].values,      alpha1,      L_km)
        df["w_equiv_alpha_norm"]  = exposure_to_wind(df["exposure_alpha_norm"].values,  alpha_norm,  L_km)

        # --- Print summary ---
        print(f"  L = {L_km:.2f} km  |  w_max = {w_max:.2f} m/s  |  cube_ref = {cube_ref:.2f}")
        print(f"  alpha_w M1 = {alpha1}  |  alpha_w M2 (norm) = {alpha_norm:.6f}")
        print(f"  max exposure M1   = {df['exposure_alpha1'].max():.4f}")
        print(f"  max exposure M2   = {df['exposure_alpha_norm'].max():.4f}  (target = {TARGET})")
        print(f"  max w_equiv M1    = {df['w_equiv_alpha1'].max():.4f} m/s")
        print(f"  max w_equiv M2    = {df['w_equiv_alpha_norm'].max():.4f} m/s")

        # --- Save CSV (all columns for fragility step) ---
        csv_path = os.path.join(line_dir, "step7_exposure.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  CSV saved: {csv_path}")

        # --- Save plot ---
        png_path = os.path.join(line_dir, "step7_plot.png")
        make_plot(way_id, df, alpha_norm, L_km, w_max, png_path)
        print(f"  Plot saved: {png_path}")
        print()

    con.close()
    print("=" * 70)
    print("STEP 7 complete.")
    print(f"Results in: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
