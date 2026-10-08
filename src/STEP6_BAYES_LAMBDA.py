"""
STEP 6 — Bayesian Posterior Failure Rate (lambda_B)
=====================================================
Goal: Compute the Bayesian posterior failure rate for each transmission line.

Paper equations implemented here:
  Eq. 3.3  -> lambda_prior_line   (prior failure rate adjusted for line length)
  Eq. 3.9  -> alpha_post          (updated alpha parameter)
  Eq. 3.10 -> beta_post           (updated beta parameter)
  Eq. 3.11 -> lambda_B            (final Bayesian failure rate)
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, BASE)
import config

# ---------------------------------------------------------------------
# INPUTS  (read from archive)
# ---------------------------------------------------------------------
ARCHIVE = os.path.join(BASE, "data")

PARQUET_PATH         = ARCHIVE + r"\merged_towers_wind_new.parquet"
ASSIGNED_FAILURES_CSV = ARCHIVE + r"\assigned_failures_updated.csv"

# T = observation period in years — used in Eq. 3.10: beta' = 1/lambda_prior + T
T_YEARS = 10

# lambda_prior per 100 km per year — Eq. 3.3: lambda_prior_K = N_K / (L * T)
# These values are computed from the overall network failure statistics
LAMBDA_PRIOR_PER100KM = {
    220: 0.7475,
    380: 0.345,
}

# ---------------------------------------------------------------------
# OUTPUT  (saved next to this script)
# ---------------------------------------------------------------------
OUT_CSV = os.path.join(BASE, "step6_lambda_B.csv")

EXCLUDED_WAY_IDS = {
    "159113596",
    "56984362",
    "1155116741",
    "191811525",
    "191811504",
    "338847619",
    "375908007",
    "375922072",
}


def main():
    print("=" * 78)
    print("STEP 6 — Bayesian Posterior Failure Rate (lambda_B)")
    print("=" * 78)
    print(f"Input failures: {ASSIGNED_FAILURES_CSV}")
    print(f"Input network : {PARQUET_PATH}")
    print(f"T (years)     : {T_YEARS}")

    # ------------------------------------------------------------------
    # 1) Load historical failures
    #    Each row = one failure event on a line (way_id)
    # ------------------------------------------------------------------
    df_fail = pd.read_csv(ASSIGNED_FAILURES_CSV)
    df_fail["way_id"] = df_fail["way_id"].astype(str)
    df_fail = df_fail[~df_fail["way_id"].isin(EXCLUDED_WAY_IDS)].copy()

    if "voltage_kV" not in df_fail.columns:
        raise ValueError("Column 'voltage_kV' not found in assigned_failures.csv")

    # Count failures per line -> this becomes sum(yi) in Eq. 3.9
    df_counts = df_fail.groupby("way_id").size().reset_index(name="n_failures")

    # ------------------------------------------------------------------
    # 2) Load total line length from parquet
    #    Line length L is used in Eq. 3.3: lambda_prior_line = lambda_prior_per100km * (L/100)
    # ------------------------------------------------------------------
    # Union with config.INCLUDED — guarantees lines with zero failures
    # (e.g. 456822447) still get a length/lambda_B row for Step 7+.
    used_ways = sorted(set(df_fail["way_id"].unique().tolist()) | config.INCLUDED)
    ways_str = ", ".join(f"'{w}'" for w in used_ways)

    con = duckdb.connect()
    df_len = con.execute(
        f"""
        WITH unique_segments AS (
            SELECT DISTINCT
                way_id,
                node_id,
                segment_km,
                line_voltage
            FROM '{PARQUET_PATH}'
            WHERE way_id IN ({ways_str})
        )
        SELECT
            way_id,
            CAST(line_voltage / 1000 AS INT) AS voltage_kV,
            SUM(segment_km)                  AS total_km   -- L in Eq. 3.3
        FROM unique_segments
        GROUP BY way_id, line_voltage
        """
    ).df()
    con.close()

    df_len["way_id"] = df_len["way_id"].astype(str)

    # ------------------------------------------------------------------
    # 3) Compute Bayesian failure rate — paper Eq. 3.3, 3.9, 3.10, 3.11
    # ------------------------------------------------------------------
    df = df_len.merge(df_counts, on="way_id", how="left").fillna({"n_failures": 0})
    df["n_failures"] = df["n_failures"].astype(int)

    # Eq. 3.3 -> lambda_prior_line = lambda_prior_per100km * (total_km / 100)
    df["lambda_prior_per100km"] = df["voltage_kV"].map(LAMBDA_PRIOR_PER100KM)
    df["lambda_prior_line"] = df["lambda_prior_per100km"] * (df["total_km"] / 100.0)

    # Eq. 3.9  -> alpha' = alpha + sum(yi)
    # Assumption: alpha = 1 (exponential distribution prior)
    df["alpha_post"] = 1.0 + df["n_failures"]

    # Eq. 3.10 -> beta' = beta + n
    # Assumption: beta = 1 / lambda_prior  (from exponential prior)
    #             n = T_YEARS (observation period)
    df["beta_post"] = (1.0 / df["lambda_prior_line"]) + T_YEARS

    # Eq. 3.11 -> lambda_B = alpha' / beta'
    # This is the final Bayesian failure rate — input to Step 7
    df["lambda_B"] = df["alpha_post"] / df["beta_post"]

    # ------------------------------------------------------------------
    # 4) Save output
    # ------------------------------------------------------------------
    df_out = df[
        [
            "voltage_kV",
            "way_id",
            "total_km",               # L — line length (km)
            "n_failures",             # sum(yi) — observed failures
            "lambda_prior_per100km",  # prior rate per 100 km
            "lambda_prior_line",      # prior rate for full line — Eq. 3.3
            "alpha_post",             # Eq. 3.9
            "beta_post",              # Eq. 3.10
            "lambda_B",               # Eq. 3.11 — main output
        ]
    ].copy()

    df_out = df_out.sort_values(
        ["voltage_kV", "lambda_B"], ascending=[True, False]
    ).reset_index(drop=True)

    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\nTop results (by lambda_B):")
    print(df_out.sort_values("lambda_B", ascending=False).head(12).to_string(index=False))
    print("\nSaved:", OUT_CSV)
    print(f"Rows: {len(df_out)}")
    print("STEP 6 complete.")


if __name__ == "__main__":
    main()
