"""
Step 2: Comprehensive Data Quality Audit
- Missing value analysis & relationship with CANCELLED/DIVERTED
- Duplicate analysis (full-row and business key)
- Invalid value detection (times, negative distances/times, logical inconsistencies)
- Inconsistent category checks (airline name vs code, DOT codes, airport codes)
- Outlier detection for ARR_DELAY and DEP_DELAY (IQR, extreme percentiles)
"""

import os
import json
import duckdb
import pandas as pd
import numpy as np

def run_data_quality_audit():
    parquet_path = os.path.join("data", "processed", "flights_2m.parquet")
    con = duckdb.connect()

    print("[*] Running Data Quality Audit on 2,000,000 rows...")

    # 1. Row count and Basic Stats
    total_rows = con.execute(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()[0]

    # 2. Missing Values per Column
    print("  -> Calculating missing values per column...")
    columns_info = con.execute(f"DESCRIBE SELECT * FROM '{parquet_path}'").fetchall()
    col_names = [col[0] for col in columns_info]

    missing_counts = {}
    for col in col_names:
        cnt = con.execute(f"SELECT COUNT(*) FROM '{parquet_path}' WHERE {col} IS NULL").fetchone()[0]
        missing_counts[col] = {
            "count": cnt,
            "pct": round(cnt / total_rows * 100, 3)
        }

    # 3. Missing Value Relationship with CANCELLED and DIVERTED
    print("  -> Analyzing relationship with CANCELLED and DIVERTED...")
    flight_status = con.execute(f"""
        SELECT 
            COUNT(*) AS total_flights,
            SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS cancelled_flights,
            SUM(CASE WHEN DIVERTED = 1.0 THEN 1 ELSE 0 END) AS diverted_flights,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1 ELSE 0 END) AS completed_flights
        FROM '{parquet_path}'
    """).df().to_dict(orient="records")[0]

    cancelled_missing = con.execute(f"""
        SELECT 
            COUNT(*) AS total_cancelled,
            SUM(CASE WHEN DEP_TIME IS NULL THEN 1 ELSE 0 END) AS dep_time_null,
            SUM(CASE WHEN ARR_TIME IS NULL THEN 1 ELSE 0 END) AS arr_time_null,
            SUM(CASE WHEN ARR_DELAY IS NULL THEN 1 ELSE 0 END) AS arr_delay_null,
            SUM(CASE WHEN TAXI_OUT IS NULL THEN 1 ELSE 0 END) AS taxi_out_null,
            SUM(CASE WHEN WHEELS_OFF IS NULL THEN 1 ELSE 0 END) AS wheels_off_null,
            SUM(CASE WHEN AIR_TIME IS NULL THEN 1 ELSE 0 END) AS air_time_null,
            SUM(CASE WHEN CANCELLATION_CODE IS NOT NULL THEN 1 ELSE 0 END) AS has_cancellation_code
        FROM '{parquet_path}'
        WHERE CANCELLED = 1.0
    """).df().to_dict(orient="records")[0]

    diverted_missing = con.execute(f"""
        SELECT 
            COUNT(*) AS total_diverted,
            SUM(CASE WHEN DEP_TIME IS NULL THEN 1 ELSE 0 END) AS dep_time_null,
            SUM(CASE WHEN ARR_TIME IS NULL THEN 1 ELSE 0 END) AS arr_time_null,
            SUM(CASE WHEN ARR_DELAY IS NULL THEN 1 ELSE 0 END) AS arr_delay_null,
            SUM(CASE WHEN WHEELS_ON IS NULL THEN 1 ELSE 0 END) AS wheels_on_null,
            SUM(CASE WHEN AIR_TIME IS NULL THEN 1 ELSE 0 END) AS air_time_null
        FROM '{parquet_path}'
        WHERE DIVERTED = 1.0
    """).df().to_dict(orient="records")[0]

    # Non-cancelled, non-diverted delay causes missing
    delay_causes_missing = con.execute(f"""
        SELECT 
            COUNT(*) AS completed_flights,
            SUM(CASE WHEN ARR_DELAY > 15 THEN 1 ELSE 0 END) AS delayed_over_15,
            SUM(CASE WHEN ARR_DELAY > 15 AND DELAY_DUE_CARRIER IS NOT NULL THEN 1 ELSE 0 END) AS delayed_with_cause,
            SUM(CASE WHEN ARR_DELAY <= 15 AND DELAY_DUE_CARRIER IS NOT NULL THEN 1 ELSE 0 END) AS ontime_with_cause,
            SUM(CASE WHEN DELAY_DUE_CARRIER IS NULL THEN 1 ELSE 0 END) AS total_delay_cause_null
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0
    """).df().to_dict(orient="records")[0]

    # 4. Duplicates
    print("  -> Checking duplicates...")
    # Full row duplicates
    full_dup_count = con.execute(f"""
        SELECT (SELECT COUNT(*) FROM '{parquet_path}') - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM '{parquet_path}')) AS full_duplicates
    """).fetchone()[0]

    # Business key duplicates: FL_DATE + AIRLINE_CODE + FL_NUMBER + ORIGIN
    key_dup_stats = con.execute(f"""
        WITH grouped AS (
            SELECT FL_DATE, AIRLINE_CODE, FL_NUMBER, ORIGIN, COUNT(*) AS cnt
            FROM '{parquet_path}'
            GROUP BY FL_DATE, AIRLINE_CODE, FL_NUMBER, ORIGIN
            HAVING COUNT(*) > 1
        )
        SELECT 
            COUNT(*) AS duplicate_groups,
            COALESCE(SUM(cnt), 0) AS total_rows_in_duplicate_groups,
            COALESCE(MAX(cnt), 0) AS max_duplicate_in_group
        FROM grouped
    """).df().to_dict(orient="records")[0]

    # 5. Invalid values
    print("  -> Checking invalid/impossible values...")
    invalid_checks = con.execute(f"""
        SELECT 
            SUM(CASE WHEN CRS_DEP_TIME < 0 OR CRS_DEP_TIME > 2359 THEN 1 ELSE 0 END) AS invalid_crs_dep_time,
            SUM(CASE WHEN (CRS_DEP_TIME % 100) >= 60 THEN 1 ELSE 0 END) AS invalid_crs_dep_minute,
            SUM(CASE WHEN CRS_ARR_TIME < 0 OR CRS_ARR_TIME > 2359 THEN 1 ELSE 0 END) AS invalid_crs_arr_time,
            SUM(CASE WHEN (CRS_ARR_TIME % 100) >= 60 THEN 1 ELSE 0 END) AS invalid_crs_arr_minute,
            SUM(CASE WHEN DISTANCE <= 0 THEN 1 ELSE 0 END) AS non_positive_distance,
            SUM(CASE WHEN CRS_ELAPSED_TIME <= 0 THEN 1 ELSE 0 END) AS non_positive_crs_elapsed,
            SUM(CASE WHEN TAXI_OUT < 0 THEN 1 ELSE 0 END) AS negative_taxi_out,
            SUM(CASE WHEN TAXI_IN < 0 THEN 1 ELSE 0 END) AS negative_taxi_in,
            SUM(CASE WHEN AIR_TIME < 0 THEN 1 ELSE 0 END) AS negative_air_time,
            SUM(CASE WHEN CANCELLED NOT IN (0.0, 1.0) THEN 1 ELSE 0 END) AS invalid_cancelled_flag,
            SUM(CASE WHEN DIVERTED NOT IN (0.0, 1.0) THEN 1 ELSE 0 END) AS invalid_diverted_flag,
            SUM(CASE WHEN CANCELLED = 1.0 AND DIVERTED = 1.0 THEN 1 ELSE 0 END) AS both_cancelled_and_diverted
        FROM '{parquet_path}'
    """).df().to_dict(orient="records")[0]

    # 6. Inconsistent Categories
    print("  -> Checking category consistency...")
    # Airline mapping
    airline_mapping = con.execute(f"""
        SELECT 
            AIRLINE_CODE,
            AIRLINE,
            DOT_CODE,
            COUNT(*) AS flights
        FROM '{parquet_path}'
        GROUP BY AIRLINE_CODE, AIRLINE, DOT_CODE
        ORDER BY flights DESC
    """).df()

    unique_airlines = con.execute(f"SELECT COUNT(DISTINCT AIRLINE) FROM '{parquet_path}'").fetchone()[0]
    unique_airline_codes = con.execute(f"SELECT COUNT(DISTINCT AIRLINE_CODE) FROM '{parquet_path}'").fetchone()[0]
    unique_dot_codes = con.execute(f"SELECT COUNT(DISTINCT DOT_CODE) FROM '{parquet_path}'").fetchone()[0]

    # Airport codes format
    airport_format = con.execute(f"""
        SELECT 
            COUNT(DISTINCT ORIGIN) AS unique_origins,
            COUNT(DISTINCT DEST) AS unique_dests,
            SUM(CASE WHEN LENGTH(ORIGIN) != 3 OR ORIGIN != UPPER(ORIGIN) THEN 1 ELSE 0 END) AS invalid_origin_codes,
            SUM(CASE WHEN LENGTH(DEST) != 3 OR DEST != UPPER(DEST) THEN 1 ELSE 0 END) AS invalid_dest_codes
        FROM '{parquet_path}'
    """).df().to_dict(orient="records")[0]

    # 7. Outliers in ARR_DELAY and DEP_DELAY
    print("  -> Calculating outlier percentiles and statistics...")
    delay_stats = con.execute(f"""
        SELECT 
            'ARR_DELAY' AS metric,
            COUNT(ARR_DELAY) AS count,
            MIN(ARR_DELAY) AS min_val,
            QUANTILE_CONT(ARR_DELAY, 0.01) AS p01,
            QUANTILE_CONT(ARR_DELAY, 0.05) AS p05,
            QUANTILE_CONT(ARR_DELAY, 0.25) AS q1,
            QUANTILE_CONT(ARR_DELAY, 0.50) AS median_val,
            AVG(ARR_DELAY) AS mean_val,
            STDDEV(ARR_DELAY) AS std_val,
            QUANTILE_CONT(ARR_DELAY, 0.75) AS q3,
            QUANTILE_CONT(ARR_DELAY, 0.95) AS p95,
            QUANTILE_CONT(ARR_DELAY, 0.99) AS p99,
            QUANTILE_CONT(ARR_DELAY, 0.999) AS p999,
            MAX(ARR_DELAY) AS max_val
        FROM '{parquet_path}'
        WHERE ARR_DELAY IS NOT NULL
        UNION ALL
        SELECT 
            'DEP_DELAY' AS metric,
            COUNT(DEP_DELAY) AS count,
            MIN(DEP_DELAY) AS min_val,
            QUANTILE_CONT(DEP_DELAY, 0.01) AS p01,
            QUANTILE_CONT(DEP_DELAY, 0.05) AS p05,
            QUANTILE_CONT(DEP_DELAY, 0.25) AS q1,
            QUANTILE_CONT(DEP_DELAY, 0.50) AS median_val,
            AVG(DEP_DELAY) AS mean_val,
            STDDEV(DEP_DELAY) AS std_val,
            QUANTILE_CONT(DEP_DELAY, 0.75) AS q3,
            QUANTILE_CONT(DEP_DELAY, 0.95) AS p95,
            QUANTILE_CONT(DEP_DELAY, 0.99) AS p99,
            QUANTILE_CONT(DEP_DELAY, 0.999) AS p999,
            MAX(DEP_DELAY) AS max_val
        FROM '{parquet_path}'
        WHERE DEP_DELAY IS NOT NULL
    """).df()

    # Outlier counts using IQR
    outlier_counts = {}
    for idx, row in delay_stats.iterrows():
        metric = row["metric"]
        iqr = row["q3"] - row["q1"]
        lower_fence = row["q1"] - 1.5 * iqr
        upper_fence = row["q3"] + 1.5 * iqr
        ext_upper_fence = row["q3"] + 3.0 * iqr

        cnts = con.execute(f"""
            SELECT 
                SUM(CASE WHEN {metric} < {lower_fence} THEN 1 ELSE 0 END) AS below_lower,
                SUM(CASE WHEN {metric} > {upper_fence} THEN 1 ELSE 0 END) AS above_upper,
                SUM(CASE WHEN {metric} > {ext_upper_fence} THEN 1 ELSE 0 END) AS above_extreme_upper
            FROM '{parquet_path}'
            WHERE {metric} IS NOT NULL
        """).df().to_dict(orient="records")[0]

        outlier_counts[metric] = {
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "extreme_upper_fence": ext_upper_fence,
            "below_lower_count": cnts["below_lower"],
            "above_upper_count": cnts["above_upper"],
            "above_upper_pct": round(cnts["above_upper"] / row["count"] * 100, 2),
            "extreme_upper_count": cnts["above_extreme_upper"],
            "extreme_upper_pct": round(cnts["above_extreme_upper"] / row["count"] * 100, 2)
        }

    results = {
        "total_rows": total_rows,
        "missing_counts": missing_counts,
        "flight_status": flight_status,
        "cancelled_missing": cancelled_missing,
        "diverted_missing": diverted_missing,
        "delay_causes_missing": delay_causes_missing,
        "full_dup_count": full_dup_count,
        "key_dup_stats": key_dup_stats,
        "invalid_checks": invalid_checks,
        "unique_airlines": unique_airlines,
        "unique_airline_codes": unique_airline_codes,
        "unique_dot_codes": unique_dot_codes,
        "airport_format": airport_format,
        "delay_stats": delay_stats.to_dict(orient="records"),
        "outlier_counts": outlier_counts
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/audit_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("[+] Data Quality Audit complete! Saved to reports/audit_results.json")
    con.close()
    return results

if __name__ == "__main__":
    run_data_quality_audit()
