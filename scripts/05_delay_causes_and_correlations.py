"""
Step 5: Delay Causes, Correlations, Categorical Associations & Leakage Audit
- Analyzes DELAY_DUE_* breakdown by airline and year
- Computes Cramér's V and Mutual Information for categorical features against binary target DELAY
- Evaluates categorical cardinality and encoding strategies
- Constructs formal data leakage classification matrix for all 32 columns
- Calculates precise time-split metrics for candidate train/val/test splits
"""

import os
import json
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

def run_causes_and_correlations():
    parquet_path = os.path.join("data", "processed", "flights_2m.parquet")
    con = duckdb.connect()

    print("[*] Running Delay Causes and Advanced Statistical Association Analysis...")

    # 1. Delay Causes by Airline
    print("  -> Computing delay causes by airline...")
    causes_by_airline = con.execute(f"""
        SELECT 
            AIRLINE_CODE,
            AIRLINE,
            COUNT(*) AS delayed_flights_with_cause,
            ROUND(SUM(DELAY_DUE_CARRIER) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS carrier_share_pct,
            ROUND(SUM(DELAY_DUE_LATE_AIRCRAFT) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS late_aircraft_share_pct,
            ROUND(SUM(DELAY_DUE_NAS) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS nas_share_pct,
            ROUND(SUM(DELAY_DUE_WEATHER) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS weather_share_pct,
            ROUND(SUM(DELAY_DUE_SECURITY) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS security_share_pct
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0 AND DELAY_DUE_CARRIER IS NOT NULL
        GROUP BY AIRLINE_CODE, AIRLINE
        ORDER BY carrier_share_pct DESC
    """).df().to_dict(orient="records")

    # 2. Delay Causes by Year
    print("  -> Computing delay causes by year...")
    causes_by_year = con.execute(f"""
        SELECT 
            EXTRACT(YEAR FROM FL_DATE) AS year,
            COUNT(*) AS delayed_flights_with_cause,
            ROUND(SUM(DELAY_DUE_CARRIER) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS carrier_share_pct,
            ROUND(SUM(DELAY_DUE_LATE_AIRCRAFT) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS late_aircraft_share_pct,
            ROUND(SUM(DELAY_DUE_NAS) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS nas_share_pct,
            ROUND(SUM(DELAY_DUE_WEATHER) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS weather_share_pct,
            ROUND(SUM(DELAY_DUE_SECURITY) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS security_share_pct
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0 AND DELAY_DUE_CARRIER IS NOT NULL
        GROUP BY EXTRACT(YEAR FROM FL_DATE)
        ORDER BY year ASC
    """).df().to_dict(orient="records")

    # 3. Categorical Associations with Target (DELAY > 15)
    print("  -> Computing Cramér's V and Mutual Information for categorical variables...")
    sample_cat_df = con.execute(f"""
        SELECT 
            AIRLINE_CODE,
            ORIGIN,
            DEST,
            ORIGIN || '_' || DEST AS route,
            CAST(DAYOFWEEK(FL_DATE) AS VARCHAR) AS day_of_week,
            CAST(EXTRACT(MONTH FROM FL_DATE) AS VARCHAR) AS month,
            CAST(FLOOR(CRS_DEP_TIME / 100) AS VARCHAR) AS dep_hour,
            CASE WHEN ARR_DELAY > 15 THEN 1 ELSE 0 END AS is_delayed
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0
        USING SAMPLE 200000
    """).df()

    def cramers_v(confusion_matrix):
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        phi2 = chi2 / n
        phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corr = r - ((r - 1)**2) / (n - 1)
        k_corr = k - ((k - 1)**2) / (n - 1)
        return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))

    cat_associations = []
    for col in ["dep_hour", "month", "AIRLINE_CODE", "day_of_week", "ORIGIN", "DEST", "route"]:
        contingency = pd.crosstab(sample_cat_df[col], sample_cat_df["is_delayed"])
        v = cramers_v(contingency)
        mi = mutual_info_score(sample_cat_df[col], sample_cat_df["is_delayed"])
        unique_cnt = sample_cat_df[col].nunique()
        cat_associations.append({
            "feature": col,
            "unique_values_in_sample": unique_cnt,
            "cramers_v": round(float(v), 4),
            "mutual_information": round(float(mi), 5)
        })

    cat_associations.sort(key=lambda x: x["cramers_v"], reverse=True)

    # 4. Cardinality Analysis on Full 2 Million Dataset
    print("  -> Checking full categorical cardinality on 2,000,000 rows...")
    cardinality_stats = con.execute(f"""
        SELECT 
            COUNT(DISTINCT AIRLINE_CODE) AS unique_airlines,
            COUNT(DISTINCT ORIGIN) AS unique_origins,
            COUNT(DISTINCT DEST) AS unique_dests,
            COUNT(DISTINCT ORIGIN || '_' || DEST) AS unique_routes
        FROM '{parquet_path}'
    """).df().to_dict(orient="records")[0]

    route_frequency_stats = con.execute(f"""
        WITH route_counts AS (
            SELECT ORIGIN || '_' || DEST AS route, COUNT(*) AS cnt
            FROM '{parquet_path}'
            GROUP BY ORIGIN || '_' || DEST
        )
        SELECT 
            COUNT(*) AS total_routes,
            SUM(CASE WHEN cnt >= 1000 THEN 1 ELSE 0 END) AS routes_over_1000_flights,
            SUM(CASE WHEN cnt BETWEEN 100 AND 999 THEN 1 ELSE 0 END) AS routes_100_to_999,
            SUM(CASE WHEN cnt BETWEEN 10 AND 99 THEN 1 ELSE 0 END) AS routes_10_to_99,
            SUM(CASE WHEN cnt < 10 THEN 1 ELSE 0 END) AS rare_routes_under_10
        FROM route_counts
    """).df().to_dict(orient="records")[0]

    # 5. Candidate Time-Splits Feasibility
    print("  -> Evaluating Candidate Time Splits...")
    split_options = {}

    # Option 1: Standard Chronological 3-way Split (Train: 2019-2021, Val: 2022, Test: 2023)
    opt1 = con.execute(f"""
        SELECT 
            'Train (2019-2021)' AS split_name,
            COUNT(*) AS total_flights,
            SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS cancelled,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1 ELSE 0 END) AS completed,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / 
                  NULLIF(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1.0 ELSE 0.0 END), 0) * 100, 2) AS delay_rate,
            ROUND(AVG(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN ARR_DELAY ELSE NULL END), 2) AS mean_arr_delay
        FROM '{parquet_path}'
        WHERE EXTRACT(YEAR FROM FL_DATE) BETWEEN 2019 AND 2021
        UNION ALL
        SELECT 
            'Validation (2022)' AS split_name,
            COUNT(*) AS total_flights,
            SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS cancelled,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1 ELSE 0 END) AS completed,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / 
                  NULLIF(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1.0 ELSE 0.0 END), 0) * 100, 2) AS delay_rate,
            ROUND(AVG(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN ARR_DELAY ELSE NULL END), 2) AS mean_arr_delay
        FROM '{parquet_path}'
        WHERE EXTRACT(YEAR FROM FL_DATE) = 2022
        UNION ALL
        SELECT 
            'Test (2023)' AS split_name,
            COUNT(*) AS total_flights,
            SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS cancelled,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1 ELSE 0 END) AS completed,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / 
                  NULLIF(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1.0 ELSE 0.0 END), 0) * 100, 2) AS delay_rate,
            ROUND(AVG(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN ARR_DELAY ELSE NULL END), 2) AS mean_arr_delay
        FROM '{parquet_path}'
        WHERE EXTRACT(YEAR FROM FL_DATE) = 2023
    """).df().to_dict(orient="records")
    split_options["Standard_3way_Split"] = opt1

    # Option 2: COVID-Filtered Split (Train: 2019 + 2021, Val: 2022, Test: 2023)
    opt2 = con.execute(f"""
        SELECT 
            'Train (2019 + 2021 No COVID)' AS split_name,
            COUNT(*) AS total_flights,
            SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS cancelled,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1 ELSE 0 END) AS completed,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / 
                  NULLIF(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1.0 ELSE 0.0 END), 0) * 100, 2) AS delay_rate,
            ROUND(AVG(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN ARR_DELAY ELSE NULL END), 2) AS mean_arr_delay
        FROM '{parquet_path}'
        WHERE EXTRACT(YEAR FROM FL_DATE) IN (2019, 2021)
        UNION ALL
        SELECT 
            'Validation (2022)' AS split_name,
            COUNT(*) AS total_flights,
            SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS cancelled,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1 ELSE 0 END) AS completed,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / 
                  NULLIF(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1.0 ELSE 0.0 END), 0) * 100, 2) AS delay_rate,
            ROUND(AVG(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN ARR_DELAY ELSE NULL END), 2) AS mean_arr_delay
        FROM '{parquet_path}'
        WHERE EXTRACT(YEAR FROM FL_DATE) = 2022
        UNION ALL
        SELECT 
            'Test (2023)' AS split_name,
            COUNT(*) AS total_flights,
            SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS cancelled,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1 ELSE 0 END) AS completed,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / 
                  NULLIF(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1.0 ELSE 0.0 END), 0) * 100, 2) AS delay_rate,
            ROUND(AVG(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN ARR_DELAY ELSE NULL END), 2) AS mean_arr_delay
        FROM '{parquet_path}'
        WHERE EXTRACT(YEAR FROM FL_DATE) = 2023
    """).df().to_dict(orient="records")
    split_options["COVID_Excluded_Split"] = opt2

    # 6. Formal Data Leakage Matrix
    # All 32 columns classified with one-line reason and status
    columns_leakage = [
        {"column": "FL_DATE", "category": "Temporal", "status": "SAFE", "reason": "Known at schedule publication and ticket booking time."},
        {"column": "AIRLINE", "category": "Identifier", "status": "SAFE", "reason": "Known at booking time (redundant with AIRLINE_CODE)."},
        {"column": "AIRLINE_DOT", "category": "Identifier", "status": "SAFE", "reason": "Known at booking time (redundant text format)."},
        {"column": "AIRLINE_CODE", "category": "Identifier", "status": "SAFE", "reason": "Primary 2-letter airline code known at booking time."},
        {"column": "DOT_CODE", "category": "Identifier", "status": "SAFE", "reason": "Numeric carrier DOT code known at booking time."},
        {"column": "FL_NUMBER", "category": "Identifier", "status": "SAFE", "reason": "Flight number assigned at schedule publication."},
        {"column": "ORIGIN", "category": "Spatial", "status": "SAFE", "reason": "Departure airport code known at booking time."},
        {"column": "ORIGIN_CITY", "category": "Spatial", "status": "SAFE", "reason": "Origin municipality known at booking time."},
        {"column": "DEST", "category": "Spatial", "status": "SAFE", "reason": "Destination airport code known at booking time."},
        {"column": "DEST_CITY", "category": "Spatial", "status": "SAFE", "reason": "Destination municipality known at booking time."},
        {"column": "CRS_DEP_TIME", "category": "Temporal", "status": "SAFE", "reason": "Computer Reservation System scheduled departure time (HHMM)."},
        {"column": "DEP_TIME", "category": "Operational", "status": "LEAKAGE", "reason": "Actual gate pushback time; only recorded as flight departs."},
        {"column": "DEP_DELAY", "category": "Operational", "status": "LEAKAGE", "reason": "Actual departure delay; strongly collinear with target (r=0.96) and unknown pre-departure."},
        {"column": "TAXI_OUT", "category": "Operational", "status": "LEAKAGE", "reason": "Taxiing duration from gate to runway; occurs after departure."},
        {"column": "WHEELS_OFF", "category": "Operational", "status": "LEAKAGE", "reason": "Time aircraft wheels leave ground; occurs post-departure."},
        {"column": "WHEELS_ON", "category": "Operational", "status": "LEAKAGE", "reason": "Touchdown time at destination; occurs post-flight."},
        {"column": "TAXI_IN", "category": "Operational", "status": "LEAKAGE", "reason": "Taxi duration from runway to arrival gate; occurs post-flight."},
        {"column": "CRS_ARR_TIME", "category": "Temporal", "status": "SAFE", "reason": "Computer Reservation System scheduled arrival time (HHMM)."},
        {"column": "ARR_TIME", "category": "Operational", "status": "LEAKAGE", "reason": "Actual gate arrival time; directly reveals arrival delay."},
        {"column": "ARR_DELAY", "category": "Target", "status": "TARGET", "reason": "Ground truth continuous arrival delay in minutes; used to create binary target."},
        {"column": "CANCELLED", "category": "Target / Filter", "status": "LEAKAGE / SECONDARY TARGET", "reason": "Recorded upon flight cancellation; cannot be used as an input feature for predicting ARR_DELAY."},
        {"column": "CANCELLATION_CODE", "category": "Target / Filter", "status": "LEAKAGE", "reason": "Reason code (A, B, C, D) entered upon cancellation; post-schedule operational outcome."},
        {"column": "DIVERTED", "category": "Filter", "status": "LEAKAGE", "reason": "Diversion flag recorded mid-flight or upon diversion landing."},
        {"column": "CRS_ELAPSED_TIME", "category": "Operational", "status": "SAFE", "reason": "Scheduled gate-to-gate duration; published in timetable."},
        {"column": "ELAPSED_TIME", "category": "Operational", "status": "LEAKAGE", "reason": "Actual gate-to-gate flight duration; only known after gate arrival."},
        {"column": "AIR_TIME", "category": "Operational", "status": "LEAKAGE", "reason": "Airborne flight duration (wheels-off to wheels-on); known only after flight."},
        {"column": "DISTANCE", "category": "Spatial", "status": "SAFE", "reason": "Direct nautical/statute mileage between airport coordinates."},
        {"column": "DELAY_DUE_CARRIER", "category": "Delay Cause", "status": "LEAKAGE", "reason": "DOT post-flight delay attribution; populated only after arrival."},
        {"column": "DELAY_DUE_WEATHER", "category": "Delay Cause", "status": "LEAKAGE", "reason": "DOT post-flight delay attribution; populated only after arrival."},
        {"column": "DELAY_DUE_NAS", "category": "Delay Cause", "status": "LEAKAGE", "reason": "DOT post-flight delay attribution; populated only after arrival."},
        {"column": "DELAY_DUE_SECURITY", "category": "Delay Cause", "status": "LEAKAGE", "reason": "DOT post-flight delay attribution; populated only after arrival."},
        {"column": "DELAY_DUE_LATE_AIRCRAFT", "category": "Delay Cause", "status": "LEAKAGE", "reason": "DOT post-flight delay attribution; populated only after arrival."}
    ]

    results = {
        "causes_by_airline": causes_by_airline,
        "causes_by_year": causes_by_year,
        "cat_associations": cat_associations,
        "cardinality_stats": cardinality_stats,
        "route_frequency_stats": route_frequency_stats,
        "split_options": split_options,
        "columns_leakage": columns_leakage
    }

    with open("reports/causes_and_associations.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("[+] Step 5 complete! Saved to reports/causes_and_associations.json")
    con.close()
    return results

if __name__ == "__main__":
    run_causes_and_correlations()
