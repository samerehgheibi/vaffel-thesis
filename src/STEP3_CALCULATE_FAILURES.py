import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb

file_path = os.path.join(BASE, "merged_towers_wind_new.parquet")
con = duckdb.connect()

# -------------------------
# STEP 0: GET LINE LENGTHS FROM DATABASE
# -------------------------
print("=" * 55)
print("LINE LENGTHS FROM DATABASE (Sicily grid)")
print("=" * 55)

lengths = con.execute(f"""
    WITH unique_segments AS (
        SELECT DISTINCT way_id, node_id, segment_km, line_voltage
        FROM '{file_path}'
    )
    SELECT
        line_voltage / 1000  AS voltage_kV,
        COUNT(DISTINCT node_id) AS node_count,
        ROUND(SUM(segment_km), 4) AS total_km
    FROM unique_segments
    GROUP BY line_voltage
    ORDER BY line_voltage
""").df()

print(lengths.to_string(index=False))

L_220 = float(lengths[lengths['voltage_kV'] == 220]['total_km'].values[0])
L_380 = float(lengths[lengths['voltage_kV'] == 380]['total_km'].values[0])

print(f"\nL (220 kV) = {L_220:.2f} km")
print(f"L (380 kV) = {L_380:.2f} km")

# -------------------------
# PARAMETERS
# -------------------------
T            = 10       # years (2013-2023)
weather_pct  = 0.46     # 46% weather-related faults
wind_pct     = 0.25     # 25% wind-related faults (of ALL faults)

lambda_base = {
    220: 6.5,   # faults/year/100 km  (Italy table)
    380: 3.0,   # faults/year/100 km  (Italy table)
}

L = {220: L_220, 380: L_380}

# -------------------------
# CALCULATION  n = λ_wind × (L/100) × T
# -------------------------
print("\n" + "=" * 55)
print("FAILURE CALCULATION  —  n = λ_wind × (L/100) × T")
print("=" * 55)

for v in [220, 380]:
    lam_base    = lambda_base[v]
    lam_weather = round(lam_base * weather_pct, 6)
    lam_wind    = round(lam_weather * wind_pct, 6)   # 25% of weather faults
    n           = round(lam_wind * (L[v] / 100) * T, 4)

    print(f"\nVoltage: {v} kV")
    print(f"  Step 1: Base λ                    = {lam_base} faults/year/100 km")
    print(f"  Step 2: Weather adjustment (46%)  → λ_weather = {lam_base} × 0.46 = {lam_weather}")
    print(f"  Step 3: Wind adjustment (25%)     → λ_wind    = {lam_weather} × 0.25 = {lam_wind}")
    print(f"  Step 4: Final λ_wind              = {lam_wind} faults/year/100 km")
    print(f"  Step 5: n = λ_wind × (L/100) × T")
    print(f"          n = {lam_wind} × ({L[v]:.2f}/100) × {T}")
    print(f"  ➜ Final n = {n:.4f} wind-related failures over {T} years")

# -------------------------
# SUMMARY TABLE
# -------------------------
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"{'Voltage':>10} {'L (km)':>12} {'λ_base':>8} {'λ_wind':>8} {'n':>10}")
print("-" * 55)
for v in [220, 380]:
    lam_wind = round(lambda_base[v] * weather_pct * wind_pct, 6)
    n        = round(lam_wind * (L[v] / 100) * T, 4)
    print(f"{str(v)+' kV':>10} {L[v]:>12.2f} {lambda_base[v]:>8.1f} {lam_wind:>8.4f} {n:>10.4f}")

con.close()
