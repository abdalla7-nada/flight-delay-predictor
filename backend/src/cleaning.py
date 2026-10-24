"""
src/cleaning.py
Phase 1: Flight Data Cleaning Pipeline
- Filters completed flights (drops CANCELLED == 1 and DIVERTED == 1)
- Excludes the anomalous 2020 COVID shock year
- Fixes time boundary anomalies (CRS_ARR_TIME == 2400 -> 0)
- Computes route median CRS_ELAPSED_TIME strictly on TRAIN (2019 + 2021)
  and imputes null/zero durations (fallback: DISTANCE / 7.5 + 30)
- Constructs binary target DELAY = (ARR_DELAY > 15).astype(int)
- Retains strictly the pre-departure feature whitelist + FL_DATE + DELAY
- Enforces strict data leakage assertions preventing post-departure features
"""

import os
import joblib
import duckdb
import pandas as pd
import numpy as np

# Whitelist of approved pre-departure columns
FEATURE_WHITELIST = [
    "FL_DATE",
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "DELAY"
]

# Strict list of prohibited post-departure / leakage columns
LEAKAGE_COLUMNS = [
    "DEP_TIME",
    "DEP_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "ARR_TIME",
    "ARR_DELAY",
    "ELAPSED_TIME",
    "AIR_TIME",
    "CANCELLED",
    "CANCELLATION_CODE",
    "DIVERTED",
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT"
]

def clean_flight_data(
    input_parquet: str = "data/processed/flights_2m.parquet",
    output_parquet: str = "data/processed/cleaned_flights.parquet",
    artifacts_dir: str = "artifacts"
) -> pd.DataFrame:
    """
    Cleans raw 2M flight dataset according to project specifications.
    Route elapsed time medians are fit strictly on training years (2019, 2021).
    """
    print(f"[*] Starting Data Cleaning Pipeline on {input_parquet}...")
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    con = duckdb.connect()

    # Step 1: Compute Route Medians for CRS_ELAPSED_TIME strictly on TRAIN (2019 + 2021)
    print("  -> Computing route median CRS_ELAPSED_TIME strictly on TRAIN (2019 + 2021)...")
    route_medians_query = f"""
        SELECT 
            ORIGIN || '_' || DEST AS route,
            MEDIAN(CRS_ELAPSED_TIME) AS median_elapsed_time
        FROM '{input_parquet}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0
          AND EXTRACT(YEAR FROM FL_DATE) IN (2019, 2021)
          AND CRS_ELAPSED_TIME IS NOT NULL
          AND CRS_ELAPSED_TIME > 0
        GROUP BY ORIGIN || '_' || DEST
    """
    route_medians_df = con.execute(route_medians_query).df()
    route_elapsed_dict = dict(zip(route_medians_df["route"], route_medians_df["median_elapsed_time"]))

    medians_path = os.path.join(artifacts_dir, "route_elapsed_medians.joblib")
    joblib.dump(route_elapsed_dict, medians_path)
    print(f"  -> Saved {len(route_elapsed_dict):,} route elapsed medians to {medians_path}")

    # Register medians table in DuckDB for vectorized imputation
    con.register("train_route_medians", route_medians_df)

    # Step 2: Clean full non-2020 completed flights
    print("  -> Filtering completed flights, dropping 2020, fixing boundary times and imputing durations...")
    cleaning_query = f"""
        WITH filtered AS (
            SELECT 
                FL_DATE,
                AIRLINE_CODE,
                ORIGIN,
                DEST,
                CRS_DEP_TIME,
                CASE 
                    WHEN CRS_ARR_TIME = 2400 THEN 0 
                    ELSE CRS_ARR_TIME 
                END AS CRS_ARR_TIME,
                CRS_ELAPSED_TIME,
                DISTANCE,
                ARR_DELAY,
                ORIGIN || '_' || DEST AS route
            FROM '{input_parquet}'
            WHERE CANCELLED = 0.0 
              AND DIVERTED = 0.0 
              AND EXTRACT(YEAR FROM FL_DATE) != 2020
        )
        SELECT 
            f.FL_DATE,
            f.AIRLINE_CODE,
            f.ORIGIN,
            f.DEST,
            f.CRS_DEP_TIME,
            f.CRS_ARR_TIME,
            COALESCE(
                NULLIF(f.CRS_ELAPSED_TIME, 0.0),
                m.median_elapsed_time,
                ROUND(f.DISTANCE / 7.5 + 30.0, 1)
            ) AS CRS_ELAPSED_TIME,
            f.DISTANCE,
            CAST(CASE WHEN f.ARR_DELAY > 15 THEN 1 ELSE 0 END AS BIGINT) AS DELAY
        FROM filtered f
        LEFT JOIN train_route_medians m ON f.route = m.route
        ORDER BY f.FL_DATE, f.CRS_DEP_TIME
    """

    cleaned_df = con.execute(cleaning_query).df()

    # Step 3: Enforce strict leakage assertions
    print("  -> Performing strict data leakage and schema integrity checks...")
    for col in LEAKAGE_COLUMNS:
        assert col not in cleaned_df.columns, f"CRITICAL LEAKAGE ERROR: Prohibited column '{col}' found in cleaned features!"

    assert set(cleaned_df.columns) == set(FEATURE_WHITELIST), (
        f"Column mismatch! Expected {FEATURE_WHITELIST}, got {list(cleaned_df.columns)}"
    )

    # Check nulls
    null_counts = cleaned_df.isnull().sum()
    assert null_counts.sum() == 0, f"Unexpected null values found in cleaned data:\n{null_counts}"

    # Verify no 2020 dates
    years_present = pd.to_datetime(cleaned_df["FL_DATE"]).dt.year.unique()
    assert 2020 not in years_present, "CRITICAL ERROR: 2020 data was not fully excluded!"

    # Save to Parquet
    cleaned_df.to_parquet(output_parquet, index=False, engine="pyarrow", compression="zstd")
    print(f"[+] Cleaning complete! Saved {len(cleaned_df):,} rows to {output_parquet}")

    con.close()
    return cleaned_df

if __name__ == "__main__":
    df = clean_flight_data()
    print("Cleaned DataFrame sample:")
    print(df.head())
    print("\nShape:", df.shape)
    print("Target distribution:\n", df["DELAY"].value_counts(normalize=True))
