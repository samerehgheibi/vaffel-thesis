"""
config.py — Single source of truth for the Sicily wind-fragility project.
All STEP scripts import from here. Never hardcode these values elsewhere.
"""

import os

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.abspath(__file__)))
ARCH_DIR   = os.path.join(BASE_DIR, "data")
RESULTS    = os.path.join(BASE_DIR, "results")

PARQUET_MERGED = os.path.join(ARCH_DIR, "merged_towers_wind_new.parquet")
PARQUET_WIND   = os.path.join(ARCH_DIR, "wind_sicily_2013_2023.parquet")
TOWERS_CSV     = os.path.join(ARCH_DIR, "about towers", "sicily_towers_final (1).csv")
WAYS_SUMMARY   = os.path.join(ARCH_DIR, "about towers", "sicily_ways_summary.csv")
LINE_NAMES_CSV = os.path.join(BASE_DIR, "line_names.csv")
LAMBDA_B_CSV   = os.path.join(BASE_DIR, "step6_lambda_B.csv")

OSM_CACHE      = os.path.join(RESULTS, "osm_lines_cache2.json")
WIND_MAX_CACHE = os.path.join(RESULTS, "wind_max_cache.parquet")
WIND_MEAN_CACHE= os.path.join(RESULTS, "wind_mean_cache.parquet")
SICILY_CACHE   = os.path.join(RESULTS, "sicily_outline.json")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS LINES
# ═══════════════════════════════════════════════════════════════════════════════

# Lines used in Steps 6-8 (Bayes + fragility)
INCLUDED = {
    "375924539",   # 380 kV — Elettrodotto Sorgente-Paternò
    "375916002",   # 380 kV — Priolo-Chiaramonte Gulfi
    "456822447",   # 380 kV — Sorgente-Villafranca-Scilla-Rizziconi
    "338856348",   # 220 kV
    "375922074",   # 220 kV — Melilli-Ragusa/1-2
    "338850183",   # 220 kV
    "375907989",   # 220 kV
    "1155116743",  # 220 kV — Fulgatore-Partanna
}

# Lines in failure database but excluded from Steps 6-8
EXCLUDED = {
    "159113596",   # 220 kV — Corriolo-Caracoli/Sorgente
    "1155116741",  # 220 kV — Fulgatore-Partanna
    "191811525",   # 220 kV
    "191811504",   # 220 kV — Sorgente-Misterbianco
    "338847619",   # 220 kV
    "375908007",   # 220 kV — Fulgatore-Partanna
    "375922072",   # 220 kV — Misterbianco-Melilli/1-2
    "56984364",    # 220 kV
    "375924538",   # 380 kV — Elettrodotto Sorgente-Paradiso
}

ALL_LINES = INCLUDED | EXCLUDED

# ═══════════════════════════════════════════════════════════════════════════════
# METADATA PER LINE
# ═══════════════════════════════════════════════════════════════════════════════
VOLTAGE_MAP = {
    "375924539": 380, "375916002": 380, "375924538": 380, "456822447": 380,
    "338856348": 220, "375922074": 220, "338850183": 220,
    "375907989": 220, "1155116743": 220,
    "159113596": 220, "1155116741": 220, "191811525": 220,
    "191811504": 220, "338847619": 220,  "375908007": 220,
    "375922072": 220, "56984364": 220,
}

# Friendly names (supplement line_names.csv)
NAME_MAP = {
    "375924539": "Sorgente-Paternò",
    "375916002": "Priolo-Chiaramonte Gulfi",
    "375924538": "Sorgente-Paradiso",
    "456822447": "Sorgente-Rizziconi",
    "338856348": "338856348",
    "375922074": "Melilli-Ragusa/1-2",
    "338850183": "338850183",
    "375907989": "375907989",
    "1155116743": "Fulgatore-Partanna",
    "159113596": "Corriolo-Caracoli/Sorgente",
    "1155116741": "Fulgatore-Partanna (alt)",
    "191811525": "191811525",
    "191811504": "Sorgente-Misterbianco",
    "338847619": "338847619",
    "375908007": "Fulgatore-Partanna (B)",
    "375922072": "Misterbianco-Melilli/1-2",
    "56984364": "56984364",
}

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
WIND_PERIOD   = ("2013-01-01", "2023-12-31")
WIND_VAR      = "wind_speed_100m"
SICILY_BBOX   = dict(lat_min=36.55, lat_max=38.35,
                     lon_min=12.35, lon_max=15.75)

# Fragility model
THRESHOLD_DEFAULT = 15.0   # m/s — wind speed failure threshold
WIND_YAXIS_LIM    = (0.0, 27.0)

# ═══════════════════════════════════════════════════════════════════════════════
# MAP STYLE
# ═══════════════════════════════════════════════════════════════════════════════
COLOR = {
    "inc_220_dc": "#1565c0",   # included 220kV double-circuit
    "inc_220_sc": "#42a5f5",   # included 220kV single-circuit
    "inc_380_dc": "#b71c1c",   # included 380kV double-circuit
    "inc_380_sc": "#ef5350",   # included 380kV single-circuit
    "exc":        "#9e9e9e",   # excluded lines (grey)
    "node_inc":   "#ff5252",   # substation node — included
    "node_exc":   "#bdbdbd",   # substation node — excluded
}
