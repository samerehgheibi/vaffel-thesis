"""
STEP17_STORAGE_BENCHMARK.py — Layer 4 trade-off evidence: raw Parquet scan vs.
a restructured/persisted table, for (1) a single-field query and (2) a join query.

Requested by the advisor during the 2026-07-08 architecture review: don't just
claim raw-file access is slower than a structured table — measure it, on the
project's real data, and report actual numbers.

Uses the two files already produced by the pipeline (see config.py):
  - wind_sicily_2013_2023.parquet   : raw ERA5-style wind grid (226M rows, no join)
  - merged_towers_wind_new.parquet  : towers already joined to their nearest
                                       wind grid cell (241M rows, pre-restructured)

Test 1 (single-field query, no join):
  average wind_speed_100m at one grid cell, over the full 2013-2023 record
    (a) raw   : DuckDB scans wind_sicily_2013_2023.parquet directly, filters
    (b) table : same data pre-loaded into a persisted DuckDB table, filters

Test 2 (join query):
  wind time series for one tower
    (a) raw   : join a small towers->grid lookup against the raw wind parquet,
                i.e. the join is computed at query time
    (b) table : query merged_towers_wind_new.parquet — already pre-joined —
                once as a raw-file scan and once loaded into a persisted table

Each timed query runs 3 times; min and mean of the 3 runs are reported
(min approximates a warm cache, mean is closer to a cold one).
"""
import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import json
import duckdb
import config

RESULTS_JSON = config.RESULTS + r"\step17_benchmark_results.json"
DB_FILE      = config.RESULTS + r"\step17_bench.duckdb"

WIND_RAW  = config.PARQUET_WIND    # wind_sicily_2013_2023.parquet
MERGED    = config.PARQUET_MERGED  # merged_towers_wind_new.parquet

N_RUNS = 3


def timeit(label, fn):
    times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"    run {i+1}/{N_RUNS}: {dt:.4f}s")
    print(f"  {label}: min={min(times):.4f}s  mean={sum(times)/len(times):.4f}s  -> {result}")
    return {"label": label, "runs": times, "min": min(times), "mean": sum(times) / len(times), "result": result}


def main():
    con = duckdb.connect()  # in-memory connection, used for raw-file scans
    results = {}

    print("=" * 70)
    print("Picking one grid cell and one tower to query")
    print("=" * 70)
    node_id, grid_lat, grid_lon = con.execute(f"""
        SELECT node_id, grid_lat, grid_lon
        FROM read_parquet('{MERGED}')
        LIMIT 1
    """).fetchone()
    print(f"  tower node_id = {node_id}   grid cell = ({grid_lat}, {grid_lon})")

    # -------------------------------------------------------------------
    # TEST 1 — single-field query: avg wind speed at one grid cell
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 1 — single-field query (avg wind speed at one grid cell, 11 years)")
    print("=" * 70)

    print("\n  (a) raw Parquet scan, filtered:")
    r1a = timeit("single-field / raw Parquet", lambda: con.execute(f"""
        SELECT AVG(wind_speed_100m) FROM read_parquet('{WIND_RAW}')
        WHERE latitude = {grid_lat} AND longitude = {grid_lon}
    """).fetchone())

    print("\n  Loading wind_sicily into a persisted DuckDB table (one-time cost)...")
    tcon = duckdb.connect(DB_FILE)
    t0 = time.perf_counter()
    tcon.execute(f"""
        CREATE OR REPLACE TABLE wind_raw AS
        SELECT * FROM read_parquet('{WIND_RAW}')
    """)
    load_time_wind = time.perf_counter() - t0
    print(f"  one-time load cost: {load_time_wind:.2f}s")

    print("\n  (b) persisted structured table, filtered:")
    r1b = timeit("single-field / structured table", lambda: tcon.execute(f"""
        SELECT AVG(wind_speed_100m) FROM wind_raw
        WHERE latitude = {grid_lat} AND longitude = {grid_lon}
    """).fetchone())

    results["test1_single_field"] = {
        "description": "avg wind_speed_100m at one grid cell, 2013-2023",
        "raw_parquet": r1a,
        "structured_table": r1b,
        "one_time_load_cost_seconds": load_time_wind,
    }

    # -------------------------------------------------------------------
    # TEST 2 — join query: wind time series for one tower
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 2 — join query (wind time series for one tower)")
    print("=" * 70)

    print("\n  Extracting the small towers->grid lookup table (raw geometry, no wind data)...")
    towers_grid = con.execute(f"""
        SELECT DISTINCT node_id, grid_lat, grid_lon
        FROM read_parquet('{MERGED}')
    """).df()
    con.register("towers_grid", towers_grid)
    print(f"  towers_grid: {len(towers_grid)} rows")

    print("\n  (a) join computed at query time, against the raw wind Parquet:")
    r2a = timeit("join / raw Parquet", lambda: con.execute(f"""
        SELECT w.time, w.wind_speed_100m
        FROM towers_grid t
        JOIN read_parquet('{WIND_RAW}') w
          ON t.grid_lat = w.latitude AND t.grid_lon = w.longitude
        WHERE t.node_id = '{node_id}'
    """).df().shape)

    print("\n  (b) query the already pre-joined merged Parquet directly (raw file, no join needed):")
    r2b = timeit("join avoided / pre-joined Parquet", lambda: con.execute(f"""
        SELECT time, wind_speed_100m FROM read_parquet('{MERGED}')
        WHERE node_id = '{node_id}'
    """).df().shape)

    print("\n  Loading the pre-joined merged file into a persisted table (one-time cost)...")
    t0 = time.perf_counter()
    tcon.execute(f"""
        CREATE OR REPLACE TABLE merged AS
        SELECT * FROM read_parquet('{MERGED}')
    """)
    load_time_merged = time.perf_counter() - t0
    print(f"  one-time load cost: {load_time_merged:.2f}s")

    print("\n  (c) query the pre-joined data from a persisted structured table:")
    r2c = timeit("join avoided / structured table", lambda: tcon.execute(f"""
        SELECT time, wind_speed_100m FROM merged WHERE node_id = '{node_id}'
    """).df().shape)

    results["test2_join"] = {
        "description": "wind time series for one tower (node_id=" + node_id + ")",
        "join_at_query_time_raw_parquet": r2a,
        "prejoined_raw_parquet": r2b,
        "prejoined_structured_table": r2c,
        "one_time_load_cost_seconds": load_time_merged,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {RESULTS_JSON}")

    print("\n" + "=" * 70)
    print("SUMMARY (min of 3 runs)")
    print("=" * 70)
    print(f"Test 1 - single field avg, raw Parquet scan   : {r1a['min']:.4f}s")
    print(f"Test 1 - single field avg, structured table   : {r1b['min']:.4f}s")
    print(f"         (one-time load cost to build table   : {load_time_wind:.2f}s)")
    print(f"Test 2 - join at query time, raw Parquet       : {r2a['min']:.4f}s")
    print(f"Test 2 - pre-joined data, raw Parquet          : {r2b['min']:.4f}s")
    print(f"Test 2 - pre-joined data, structured table     : {r2c['min']:.4f}s")
    print(f"         (one-time load cost to build table   : {load_time_merged:.2f}s)")


if __name__ == "__main__":
    main()
