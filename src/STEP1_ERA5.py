"""
STEP 1 — Download wind_speed_100m for Sicily
Model  : ERA5-Land (Open-Meteo Historical API)
Period : 2013-01-01 → 2023-12-31
Grid   : ~5×5 km  →  0.05° spacing
Output : wind_sicily_2013_2023.parquet
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
import numpy as np
import time
import os

# ─────────────────────────────────────────────
# 1. BUILD THE SICILY GRID
# ─────────────────────────────────────────────
lat_min, lat_max = 36.60, 38.30
lon_min, lon_max = 12.40, 15.70
step = 0.05

latitudes  = np.round(np.arange(lat_min, lat_max + 1e-9, step), 2)
longitudes = np.round(np.arange(lon_min, lon_max + 1e-9, step), 2)

grid = pd.DataFrame(
    [(lat, lon) for lat in latitudes for lon in longitudes],
    columns=["latitude", "longitude"]
)

print(f"Grid size : {len(grid)} points")
print(f"Lat range : {latitudes[0]} to {latitudes[-1]}  ({len(latitudes)} steps)")
print(f"Lon range : {longitudes[0]} to {longitudes[-1]}  ({len(longitudes)} steps)")

# ─────────────────────────────────────────────
# 2. API PARAMETERS
# ─────────────────────────────────────────────
START_DATE = "2013-01-01"
END_DATE   = "2023-12-31"
MODEL      = "era5"
VARIABLE   = "wind_speed_100m"
OUTPUT     = os.path.join(BASE, "wind_sicily_2013_2023.parquet")
BASE_URL   = "https://archive-api.open-meteo.com/v1/archive"

DELAY_OK   = 2.0   # wait between successful requests (seconds)
DELAY_FAIL = 5.0   # wait after failed request

# ─────────────────────────────────────────────
# 3. RESUME SUPPORT
# ─────────────────────────────────────────────
done_points = set()
all_dfs = []

if os.path.exists(OUTPUT):
    existing = pd.read_parquet(OUTPUT)
    all_dfs.append(existing)
    done_points = set(
        zip(existing["latitude"].round(2), existing["longitude"].round(2))
    )
    print(f"Resuming — {len(done_points)} points already downloaded.")

# ─────────────────────────────────────────────
# 4. DOWNLOAD LOOP
# ─────────────────────────────────────────────
total = len(grid)

for idx, row in grid.iterrows():
    lat, lon = row["latitude"], row["longitude"]

    if (round(lat, 2), round(lon, 2)) in done_points:
        continue

    print(f"[{idx+1}/{total}]  ({lat}, {lon})", end="  ", flush=True)

    params = {
        "latitude"        : lat,
        "longitude"       : lon,
        "start_date"      : START_DATE,
        "end_date"        : END_DATE,
        "hourly"          : VARIABLE,
        "models"          : MODEL,
        "wind_speed_unit" : "ms",
        "timezone"        : "UTC",
    }

    # retry up to 5 times with increasing wait
    data = None
    for attempt in range(5):
        try:
            r = requests.get(BASE_URL, params=params, timeout=90)
            r.raise_for_status()
            data = r.json()
            break
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                wait = 60 * (attempt + 1)   # 60s, 120s, 180s, 240s, 300s
                print(f"[429 wait {wait}s]", end="  ", flush=True)
                time.sleep(wait)
            else:
                print(f"[HTTP {r.status_code}]", end="  ")
                break
        except Exception as e:
            print(f"[Error: {e}]", end="  ")
            time.sleep(DELAY_FAIL)
            break

    if data and "hourly" in data:
        df = pd.DataFrame({
            "time"           : pd.to_datetime(data["hourly"]["time"]),
            "wind_speed_100m": data["hourly"][VARIABLE],
        })
        df["latitude"]  = lat
        df["longitude"] = lon
        all_dfs.append(df)
        print("OK", flush=True)
        time.sleep(DELAY_OK)
    else:
        print("SKIPPED", flush=True)
        time.sleep(DELAY_FAIL)

    # checkpoint every 25 points
    if (idx + 1) % 25 == 0 and all_dfs:
        checkpoint = pd.concat(all_dfs, ignore_index=True)
        checkpoint.to_parquet(OUTPUT, index=False)
        print(f"  -- checkpoint saved ({len(checkpoint):,} rows) --", flush=True)

# ─────────────────────────────────────────────
# 5. SAVE FINAL OUTPUT
# ─────────────────────────────────────────────
if all_dfs:
    result = pd.concat(all_dfs, ignore_index=True)
    result = result[["latitude", "longitude", "time", "wind_speed_100m"]]
    result.to_parquet(OUTPUT, index=False)
    print(f"\nDone!  Shape: {result.shape}")
    print(f"Saved to: {OUTPUT}")
    print(f"Date range  : {result['time'].min()} to {result['time'].max()}")
    print(f"Grid points : {result.groupby(['latitude','longitude']).ngroups}")
else:
    print("\nNo data downloaded.")
