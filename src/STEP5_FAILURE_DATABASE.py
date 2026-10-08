import os
BASE = os.environ.get("VAFFEL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

file_path = os.path.join(BASE, "merged_towers_wind_new.parquet")

# ─────────────────────────────────────────
# ۲۲ خط نهایی
# 12 خط 220kV اصلی (خوب) + 6 جایگزین + 4 خط 380kV
# ─────────────────────────────────────────
FINAL_22 = [
    # 220kV — 12 خط اصلی خوب
    '1155116740','1155116741','1155116743',
    '159113596','159158472','159158474',
    '191811504','191811525',
    '338847619','338850183','338856348','375907989',
    # 220kV — 6 جایگزین جدید
    '375908007','375922074','57403743',
    '56984364','375922072','338900793',
    # 380kV — 4 خط اصلی (بدون تغییر)
    '375916002','375924538','375924539','456822447',
]

print(f"Total final lines: {len(FINAL_22)}")
print(f"220kV: {len([w for w in FINAL_22 if w not in ['375916002','375924538','375924539','456822447']])}")
print(f"380kV: 4")

# ─────────────────────────────────────────
# STEP 1: بارگذاری و فیلتر — فقط ۲۲ خط
# ─────────────────────────────────────────
con = duckdb.connect()

ways_str = ", ".join(f"'{w}'" for w in FINAL_22)

df = con.execute(f"""
    SELECT
        way_id,
        CAST(line_voltage / 1000 AS INT)              AS voltage_kV,
        CAST(DATE_TRUNC('day', time) AS DATE)          AS date,
        YEAR(time)                                     AS year,
        MAX(wind_speed_100m)                           AS max_wind,
        FIRST(time ORDER BY wind_speed_100m DESC)      AS peak_time
    FROM '{file_path}'
    WHERE wind_speed_100m >= 15
      AND way_id IN ({ways_str})
    GROUP BY way_id, line_voltage, DATE_TRUNC('day', time), YEAR(time)
    ORDER BY max_wind DESC, way_id ASC, date ASC
""").df()

con.close()

print(f"\nPool size after threshold=15 m/s: {len(df):,} records")
print(df.head(5).to_string(index=False))

# ─────────────────────────────────────────
# STEP 2: جداسازی بر اساس ولتاژ
# ─────────────────────────────────────────
pool_220 = df[df['voltage_kV'] == 220].reset_index(drop=True)
pool_380 = df[df['voltage_kV'] == 380].reset_index(drop=True)

print(f"\n220kV pool: {len(pool_220):,}  |  380kV pool: {len(pool_380):,}")

# ─────────────────────────────────────────
# STEP 3: Greedy Assignment
# ─────────────────────────────────────────
def assign_failures(pool, n_target, cap_per_line, max_per_day, label):
    selected      = []
    line_year_cnt = {}
    line_day_set  = set()
    line_total    = {}
    day_cnt       = {}

    for _, row in pool.iterrows():
        if len(selected) >= n_target:
            break

        key_ly  = (row['way_id'], row['year'])
        key_ld  = (row['way_id'], str(row['date']))
        day_key = str(row['date'])

        if key_ld in line_day_set:
            continue
        if line_year_cnt.get(key_ly, 0) >= 2:
            continue
        if day_cnt.get(day_key, 0) >= max_per_day:
            continue
        if line_total.get(row['way_id'], 0) >= cap_per_line:
            continue

        selected.append(row)
        line_year_cnt[key_ly]     = line_year_cnt.get(key_ly, 0) + 1
        line_day_set.add(key_ld)
        day_cnt[day_key]          = day_cnt.get(day_key, 0) + 1
        line_total[row['way_id']] = line_total.get(row['way_id'], 0) + 1

    result = pd.DataFrame(selected).reset_index(drop=True)
    print(f"\n  [{label}] Assigned: {len(result)} / {n_target} target")
    return result

# 220kV: 59 failures, cap=4, max 1 per day
# 380kV: 10 failures, cap=3, max 1 per day
result_220 = assign_failures(pool_220, n_target=59, cap_per_line=4, max_per_day=1, label="220kV")
result_380 = assign_failures(pool_380, n_target=10, cap_per_line=3, max_per_day=1, label="380kV")

# ─────────────────────────────────────────
# STEP 4: نمایش نتایج
# ─────────────────────────────────────────
con_ym = duckdb.connect()
yearly_max = con_ym.execute(f"""
    SELECT way_id, YEAR(time) AS year,
           ROUND(MAX(wind_speed_100m), 2) AS max_wind_in_year
    FROM '{file_path}'
    WHERE way_id IN ({ways_str})
    GROUP BY way_id, YEAR(time)
""").df()
con_ym.close()

for label, res in [("220kV", result_220), ("380kV", result_380)]:
    print(f"\n{'='*70}")
    print(f"  {label}  -  {len(res)} failures assigned")
    print(f"{'='*70}")
    res = res.rename(columns={'max_wind': 'failure_wind_speed'})
    res = res.merge(yearly_max, on=['way_id','year'], how='left')
    out = res[['year','way_id','date','peak_time',
               'failure_wind_speed','max_wind_in_year']].sort_values(
              ['year','failure_wind_speed'], ascending=[True,False])
    print(out.to_string(index=False))
    print(f"\n  By year:")
    print(res.groupby('year')['way_id'].count().to_string())

# ─────────────────────────────────────────
# STEP 5: ذخیره Excel و CSV
# ─────────────────────────────────────────
result_220['voltage_kV'] = 220
result_380['voltage_kV'] = 380

final = pd.concat([result_220, result_380], ignore_index=True)
final = final.rename(columns={'max_wind': 'failure_wind_speed'})

con2 = duckdb.connect()
yearly_max = con2.execute(f"""
    SELECT way_id, YEAR(time) AS year,
           ROUND(MAX(wind_speed_100m), 2) AS max_wind_in_year
    FROM '{file_path}'
    WHERE way_id IN ({ways_str})
    GROUP BY way_id, YEAR(time)
""").df()
con2.close()

final = final.merge(yearly_max, on=['way_id','year'], how='left')
final = final[['voltage_kV','way_id','year','date','peak_time',
               'failure_wind_speed','max_wind_in_year']]
final = final.sort_values(['voltage_kV','year','failure_wind_speed'],
                          ascending=[True,True,False]).reset_index(drop=True)

xlsx_path = os.path.join(BASE, "assigned_failures.xlsx")
csv_path  = os.path.join(BASE, "assigned_failures.csv")

final.to_excel(xlsx_path, index=False)
final.to_csv(csv_path,   index=False, encoding='utf-8-sig')

print(f"\nSaved Excel : {xlsx_path}")
print(f"Saved CSV   : {csv_path}")
print(f"Total: {len(final)} failures  (220kV={len(result_220)}, 380kV={len(result_380)})")

# ─────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────
print("\n" + "="*55)
print("VALIDATION CHECKS")
print("="*55)

v1 = len(result_220) == 59 and len(result_380) == 10
print(f"[{'OK' if v1 else 'FAIL'}] Target count: 220kV={len(result_220)}/59  380kV={len(result_380)}/10")

dup_day = final.duplicated(subset=['way_id','date']).sum()
print(f"[{'OK' if dup_day == 0 else 'FAIL'}] No same line + same day duplicates: {dup_day} found")

max_per_line_year = final.groupby(['way_id','year']).size().max()
print(f"[{'OK' if max_per_line_year <= 2 else 'FAIL'}] Max failures per line per year <= 2: max found = {max_per_line_year}")

below_thresh = (final['failure_wind_speed'] < 15).sum()
print(f"[{'OK' if below_thresh == 0 else 'FAIL'}] All wind speeds >= 15 m/s: {below_thresh} below threshold")

unique_ways = final['way_id'].nunique()
print(f"[{'OK' if unique_ways == 22 else 'FAIL'}] Total unique lines = 22: found = {unique_ways}")