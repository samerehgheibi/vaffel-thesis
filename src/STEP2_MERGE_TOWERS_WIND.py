"""
STEP 3 — Merge towers with wind data
For each tower, find nearest ACTUAL grid point and attach hourly wind time series
Output: merged_towers_wind_new.parquet
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
WIND_FILE   = os.path.join(BASE, "wind_sicily_2013_2023.parquet")
TOWER_FILE  = os.path.join(BASE, "about towers", "sicily_towers_final (1).csv")
OUTPUT      = os.path.join(BASE, "merged_towers_wind_new.parquet")

# ─────────────────────────────────────────────
# 1. LOAD TOWERS & WIND
# ─────────────────────────────────────────────
print("Loading towers...")
towers = pd.read_csv(TOWER_FILE)
towers['way_id']  = towers['way_id'].astype(str)
towers['node_id'] = towers['node_id'].astype(str)
print(f"  Towers: {len(towers)} | Ways: {towers['way_id'].nunique()}")

print("\nLoading wind data...")
wind = pd.read_parquet(WIND_FILE)
print(f"  Wind columns: {list(wind.columns)}")
print(f"  Wind shape  : {wind.shape}")

# ─────────────────────────────────────────────
# 2. FIX COLUMN NAMES (step1 uses latitude/longitude)
# ─────────────────────────────────────────────
if 'latitude' in wind.columns and 'longitude' in wind.columns:
    wind = wind.rename(columns={'latitude': 'grid_lat', 'longitude': 'grid_lon'})
    print("  (Renamed: latitude → grid_lat, longitude → grid_lon)")

wind['grid_lat'] = wind['grid_lat'].round(2)
wind['grid_lon'] = wind['grid_lon'].round(2)

# ─────────────────────────────────────────────
# 3. SNAP TOWERS TO ACTUAL WIND GRID (0.05 degree from step1)
# ─────────────────────────────────────────────
print("\nSnapping towers to actual wind grid (0.05 degree)...")

def snap_to_grid(val, step=0.05):
    snapped = round(round(val / step) * step, 2)
    return snapped

towers['grid_lat'] = towers['lat'].apply(lambda x: snap_to_grid(x, 0.05))
towers['grid_lon'] = towers['lon'].apply(lambda x: snap_to_grid(x, 0.05))

unique_grid = towers[['grid_lat', 'grid_lon']].drop_duplicates()
print(f"  Unique grid points needed: {len(unique_grid)}")
print(f"  Available grid points in wind data: {wind[['grid_lat','grid_lon']].drop_duplicates().shape[0]}")

# ─────────────────────────────────────────────
# 4. CHECK DATA QUALITY BEFORE MERGE
# ─────────────────────────────────────────────
print("\nWind data quality check:")
print(f"  Null values: {wind['wind_speed_100m'].isna().sum():,}")
print(f"  Max wind   : {wind['wind_speed_100m'].max():.2f} m/s")
print(f"  Mean wind  : {wind['wind_speed_100m'].mean():.2f} m/s")
print(f"  Date range : {wind['time'].min()} to {wind['time'].max()}")

# ─────────────────────────────────────────────
# 5. FILTER WIND DATA TO NEEDED POINTS (optional, speeds up merge)
# ─────────────────────────────────────────────
grid_needed = unique_grid.copy()
grid_needed['_needed'] = True
wind = wind.merge(grid_needed, on=['grid_lat','grid_lon'], how='inner').drop(columns='_needed')
print(f"\n  Wind after filtering to needed points: {wind.shape}")

# ─────────────────────────────────────────────
# 6. MERGE TOWERS + WIND
# ─────────────────────────────────────────────
print("\nMerging towers with wind data...")
merged = towers.merge(wind, on=['grid_lat','grid_lon'], how='left')

merged = merged[[
    'way_id', 'node_id', 'lat', 'lon',
    'segment_km', 'line_voltage', 'line_circuits', 'line_cables',
    'grid_lat', 'grid_lon', 'time', 'wind_speed_100m'
]]

print(f"  Merged shape: {merged.shape}")
print(f"  Null wind values: {merged['wind_speed_100m'].isna().sum():,}")

# ─────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────
print(f"\nSaving to: {OUTPUT}")
merged.to_parquet(OUTPUT, index=False)
print(f"✓ Done! Shape: {merged.shape}")

# Statistics per way
print("\nSample per way (first 5):")
sample_ways = merged['way_id'].unique()[:5]
for w in sample_ways:
    sub = merged[merged['way_id']==w]
    non_null_wind = sub['wind_speed_100m'].notna().sum()
    if non_null_wind > 0:
        print(f"  way {w}: {sub['node_id'].nunique()} towers | "
              f"wind points {non_null_wind:,} | "
              f"max={sub['wind_speed_100m'].max():.2f} | "
              f"mean={sub['wind_speed_100m'].mean():.2f} m/s")
    else:
        print(f"  way {w}: {sub['node_id'].nunique()} towers | NO WIND DATA MATCHED!")
