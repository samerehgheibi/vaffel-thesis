"""
Extract unique way IDs from sicily_towers_final.csv
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

FILE   = os.path.join(BASE, "about towers", "sicily_towers_final (1).csv")
OUTPUT = os.path.join(BASE, "unique_ways.csv")

print("Loading...")
df = pd.read_csv(FILE)
df['way_id']  = df['way_id'].astype(str)
df['node_id'] = df['node_id'].astype(str)
print(f"  Rows: {len(df):,}")

# One row per way
ways = df.groupby('way_id').agg(
    voltage_kV = ('line_voltage',  lambda x: x.iloc[0] // 1000),
    circuits   = ('line_circuits', 'first'),
    towers     = ('node_id',       'nunique'),
    length_km  = ('segment_km',    'sum'),
    lat_min    = ('lat',           'min'),
    lat_max    = ('lat',           'max'),
    lon_min    = ('lon',           'min'),
    lon_max    = ('lon',           'max'),
).reset_index()

ways = ways.sort_values(['voltage_kV', 'way_id']).reset_index(drop=True)

print(f"\n{'='*50}")
print(f"  Total unique ways : {len(ways)}")
print(f"  220kV ways        : {len(ways[ways['voltage_kV']==220])}")
print(f"  380kV ways        : {len(ways[ways['voltage_kV']==380])}")
print(f"{'='*50}")
print(ways.to_string())

ways.to_csv(OUTPUT, index=False)
print(f"\nSaved to: {OUTPUT}")
