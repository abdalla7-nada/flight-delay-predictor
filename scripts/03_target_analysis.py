"""
Step 3: Target Analysis Script
- Analyzes ARR_DELAY distribution and binary target DELAY (ARR_DELAY > 15)
- Evaluates class balance overall and across years (drift detection)
- Evaluates cancellation rates and CANCELLATION_CODE breakdown
- Compares completed flights vs all scheduled flights
"""

import os
import json
import duckdb
import pandas as pd

def run_target_analysis():
    parquet_path = os.path.join("data", "processed", "flights_2m.parquet")
    con = duckdb.connect()

    print("[*] Running Target Analysis...")

    # 1. Overall Class Balance (Completed flights: CANCELLED=0 AND DIVERTED=0)
    overall_balance = con.execute(f"""
        SELECT 
            COUNT(*) AS completed_flights,
            SUM(CASE WHEN ARR_DELAY > 15 THEN 1 ELSE 0 END) AS delayed_count,
            ROUND(SUM(CASE WHEN ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 3) AS delayed_pct,
            SUM(CASE WHEN ARR_DELAY <= 15 THEN 1 ELSE 0 END) AS ontime_count,
            ROUND(SUM(CASE WHEN ARR_DELAY <= 15 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 3) AS ontime_pct
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0
    """).df().to_dict(orient="records")[0]

    # 2. Overall Flight Outcome (including Cancelled and Diverted)
    flight_outcomes = con.execute(f"""
        SELECT 
            COUNT(*) AS total_flights,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY <= 15 THEN 1 ELSE 0 END) AS on_time_flights,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY <= 15 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 3) AS on_time_pct,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1 ELSE 0 END) AS delayed_flights,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 3) AS delayed_pct,
            SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS cancelled_flights,
            ROUND(SUM(CASE WHEN CANCELLED = 1.0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 3) AS cancelled_pct,
            SUM(CASE WHEN DIVERTED = 1.0 THEN 1 ELSE 0 END) AS diverted_flights,
            ROUND(SUM(CASE WHEN DIVERTED = 1.0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 3) AS diverted_pct
        FROM '{parquet_path}'
    """).df().to_dict(orient="records")[0]

    # 3. Cancellation Code Breakdown
    # Standard FAA/DOT mapping: A: Carrier, B: Weather, C: NAS, D: Security
    cancellation_code_breakdown = con.execute(f"""
        SELECT 
            CANCELLATION_CODE,
            CASE 
                WHEN CANCELLATION_CODE = 'A' THEN 'Carrier'
                WHEN CANCELLATION_CODE = 'B' THEN 'Weather'
                WHEN CANCELLATION_CODE = 'C' THEN 'National Aviation System (NAS)'
                WHEN CANCELLATION_CODE = 'D' THEN 'Security'
                ELSE 'Unknown'
            END AS description,
            COUNT(*) AS count,
            ROUND(COUNT(*) / 52522.0 * 100, 3) AS pct_of_cancelled,
            ROUND(COUNT(*) / 2000000.0 * 100, 3) AS pct_of_total
        FROM '{parquet_path}'
        WHERE CANCELLED = 1.0
        GROUP BY CANCELLATION_CODE
        ORDER BY count DESC
    """).df().to_dict(orient="records")

    # 4. Class Balance and Status per Year (Temporal Drift Analysis)
    yearly_drift = con.execute(f"""
        SELECT 
            EXTRACT(YEAR FROM FL_DATE) AS year,
            COUNT(*) AS total_flights,
            SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS cancelled_flights,
            ROUND(SUM(CASE WHEN CANCELLED = 1.0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS cancellation_rate,
            SUM(CASE WHEN DIVERTED = 1.0 THEN 1 ELSE 0 END) AS diverted_flights,
            ROUND(SUM(CASE WHEN DIVERTED = 1.0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS diverted_rate,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1 ELSE 0 END) AS completed_flights,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1 ELSE 0 END) AS delayed_flights,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / 
                  NULLIF(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1.0 ELSE 0.0 END), 0) * 100, 2) AS delay_rate_completed,
            ROUND(AVG(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN ARR_DELAY ELSE NULL END), 2) AS mean_arr_delay,
            ROUND(MEDIAN(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN ARR_DELAY ELSE NULL END), 2) AS median_arr_delay
        FROM '{parquet_path}'
        GROUP BY EXTRACT(YEAR FROM FL_DATE)
        ORDER BY year ASC
    """).df().to_dict(orient="records")

    # 5. Cancellation Code per Year
    cancellation_yearly = con.execute(f"""
        SELECT 
            EXTRACT(YEAR FROM FL_DATE) AS year,
            CANCELLATION_CODE,
            COUNT(*) AS count
        FROM '{parquet_path}'
        WHERE CANCELLED = 1.0
        GROUP BY EXTRACT(YEAR FROM FL_DATE), CANCELLATION_CODE
        ORDER BY year, CANCELLATION_CODE
    """).df()

    # Pivot cancellation codes by year
    cancellation_pivot = cancellation_yearly.pivot(index="year", columns="CANCELLATION_CODE", values="count").fillna(0).reset_index().to_dict(orient="records")

    # 6. Target distribution percentiles (ARR_DELAY)
    arr_delay_quantiles = con.execute(f"""
        SELECT 
            QUANTILE_CONT(ARR_DELAY, 0.10) AS p10,
            QUANTILE_CONT(ARR_DELAY, 0.20) AS p20,
            QUANTILE_CONT(ARR_DELAY, 0.30) AS p30,
            QUANTILE_CONT(ARR_DELAY, 0.40) AS p40,
            QUANTILE_CONT(ARR_DELAY, 0.50) AS p50,
            QUANTILE_CONT(ARR_DELAY, 0.60) AS p60,
            QUANTILE_CONT(ARR_DELAY, 0.70) AS p70,
            QUANTILE_CONT(ARR_DELAY, 0.80) AS p80,
            QUANTILE_CONT(ARR_DELAY, 0.82) AS p82,
            QUANTILE_CONT(ARR_DELAY, 0.85) AS p85,
            QUANTILE_CONT(ARR_DELAY, 0.90) AS p90,
            QUANTILE_CONT(ARR_DELAY, 0.95) AS p95,
            QUANTILE_CONT(ARR_DELAY, 0.99) AS p99
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0
    """).df().to_dict(orient="records")[0]

    results = {
        "overall_balance": overall_balance,
        "flight_outcomes": flight_outcomes,
        "cancellation_code_breakdown": cancellation_code_breakdown,
        "yearly_drift": yearly_drift,
        "cancellation_pivot": cancellation_pivot,
        "arr_delay_quantiles": arr_delay_quantiles
    }

    with open("reports/target_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("[+] Target Analysis complete! Saved to reports/target_analysis.json")
    con.close()
    return results

if __name__ == "__main__":
    run_target_analysis()
