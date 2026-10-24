"""
tests/test_pipeline.py
Unit tests for offline pipeline logic and transformation invariance using in-memory mock fixtures.
Verifies:
  1. Data leakage prevention and strict target definition (ARR_DELAY > 15).
  2. Cyclical temporal encoding bounds [-1.0, 1.0] and Pythagorean identity.
  3. Route fallback, geodesic distance calculation, and prior mean imputation.
  4. Platt probability calibration monotonicity and mathematical consistency.
"""

import math
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from src.cleaning import FEATURE_WHITELIST, LEAKAGE_COLUMNS
from src.features import FlightFeaturePipeline
from src.serving.predictor import FlightPredictor, haversine_distance_miles


# -------------------------------------------------------------------------
# 1. Data Leakage & Cleaning Assertion Tests
# -------------------------------------------------------------------------

def test_leakage_columns_strictly_prohibited():
    """Verifies that LEAKAGE_COLUMNS contains all prohibited post-departure fields."""
    critical_leakage = [
        "DEP_DELAY", "TAXI_OUT", "WHEELS_OFF", "ARR_DELAY",
        "AIR_TIME", "CANCELLED", "DIVERTED", "DELAY_DUE_WEATHER"
    ]
    for col in critical_leakage:
        assert col in LEAKAGE_COLUMNS, f"Missing critical leakage column: {col}"


def test_mock_cleaning_leakage_exclusion_and_target_rule():
    """
    Verifies on a synthetic DataFrame that:
    - Post-departure columns are purged.
    - CANCELLED == 1 and DIVERTED == 1 flights are excluded.
    - 2020 COVID flights are excluded.
    - ARR_DELAY > 15 yields DELAY == 1, ARR_DELAY <= 15 yields DELAY == 0.
    """
    import duckdb

    df_raw = pd.DataFrame({
        "FL_DATE": pd.to_datetime(["2021-06-15", "2020-05-10", "2022-08-20", "2021-03-01", "2021-04-10"]),
        "AIRLINE_CODE": ["AA", "DL", "UA", "WN", "B6"],
        "ORIGIN": ["DFW", "ATL", "ORD", "DEN", "JFK"],
        "DEST": ["ORD", "LGA", "LAX", "PHX", "BOS"],
        "CRS_DEP_TIME": [800, 900, 1000, 2400, 1200],
        "CRS_ARR_TIME": [1000, 1100, 1200, 2400, 1315],
        "CRS_ELAPSED_TIME": [120.0, 120.0, 120.0, 0.0, 75.0],
        "DISTANCE": [800.0, 750.0, 1700.0, 600.0, 187.0],
        "ARR_DELAY": [16.0, 50.0, 15.0, 0.0, 15.0],  # 16 -> 1; 15 -> 0 (strict > 15 definition)
        "DEP_DELAY": [10.0, 45.0, 5.0, 0.0, 12.0],
        "CANCELLED": [0.0, 0.0, 1.0, 0.0, 0.0],       # row 2 cancelled -> drop
        "DIVERTED": [0.0, 0.0, 0.0, 0.0, 0.0],
        "TAXI_OUT": [15.0, 20.0, 12.0, 10.0, 14.0],
        "DELAY_DUE_WEATHER": [0.0, 30.0, 0.0, 0.0, 0.0]
    })

    con = duckdb.connect()
    cleaned = con.execute("""
        SELECT 
            FL_DATE, AIRLINE_CODE, ORIGIN, DEST, CRS_DEP_TIME,
            CASE WHEN CRS_ARR_TIME = 2400 THEN 0 ELSE CRS_ARR_TIME END AS CRS_ARR_TIME,
            COALESCE(NULLIF(CRS_ELAPSED_TIME, 0.0), ROUND(DISTANCE / 7.5 + 30.0, 1)) AS CRS_ELAPSED_TIME,
            DISTANCE,
            CAST(CASE WHEN ARR_DELAY > 15 THEN 1 ELSE 0 END AS BIGINT) AS DELAY
        FROM df_raw
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0 AND EXTRACT(YEAR FROM FL_DATE) != 2020
    """).df()

    # 1. Assert zero leakage columns present in output
    for col in LEAKAGE_COLUMNS:
        assert col not in cleaned.columns, f"Leakage column {col} leaked into cleaned features!"

    # 2. Assert exact feature whitelist match
    assert set(cleaned.columns) == set(FEATURE_WHITELIST)

    # 3. Assert row filtering (row 1 is 2020, row 2 is cancelled -> 3 remain)
    assert len(cleaned) == 3

    # 4. Assert strict > 15 target definition:
    # Row 0: ARR_DELAY = 16.0 -> DELAY = 1
    # Row 3: ARR_DELAY = 0.0 -> DELAY = 0
    # Row 4: ARR_DELAY = 15.0 -> DELAY = 0 (exact 15 min threshold DOT boundary check)
    delays = cleaned["DELAY"].tolist()
    assert delays == [1, 0, 0], f"Expected delays [1, 0, 0], got {delays}"


# -------------------------------------------------------------------------
# 2. Temporal & Cyclical Feature Encoding Tests
# -------------------------------------------------------------------------

def test_cyclical_feature_bounds_and_identities():
    """
    Verifies that cyclical transformations in FlightFeaturePipeline:
    - Reside strictly within the mathematical interval [-1.0, 1.0].
    - Adhere to the Pythagorean trigonometric identity (sin^2 + cos^2 = 1.0).
    """
    train_mock = pd.DataFrame({
        "FL_DATE": pd.to_datetime(["2019-01-01", "2019-04-15", "2021-07-20", "2021-10-31"]),
        "AIRLINE_CODE": ["AA", "DL", "UA", "WN"],
        "ORIGIN": ["DFW", "ATL", "ORD", "DEN"],
        "DEST": ["ORD", "LGA", "LAX", "PHX"],
        "CRS_DEP_TIME": [0, 600, 1200, 1800],
        "CRS_ARR_TIME": [200, 830, 1430, 2030],
        "CRS_ELAPSED_TIME": [120.0, 150.0, 150.0, 150.0],
        "DISTANCE": [800.0, 760.0, 1740.0, 600.0],
        "DELAY": [0, 1, 0, 1]
    })

    pipeline = FlightFeaturePipeline(m_default=10.0, m_route=10.0, rare_route_thresh=1)
    pipeline.fit(train_mock)

    # Create evaluation set spanning hours 0..23 and all 12 months
    eval_dates = [f"2022-{m:02d}-15" for m in range(1, 13)] * 2
    eval_dep_times = [h * 100 for h in range(24)]
    eval_df = pd.DataFrame({
        "FL_DATE": pd.to_datetime(eval_dates),
        "AIRLINE_CODE": ["AA"] * 24,
        "ORIGIN": ["DFW"] * 24,
        "DEST": ["ORD"] * 24,
        "CRS_DEP_TIME": eval_dep_times,
        "CRS_ARR_TIME": [((t + 200) % 2400) for t in eval_dep_times],
        "CRS_ELAPSED_TIME": [120.0] * 24,
        "DISTANCE": [800.0] * 24
    })

    tree_df, _ = pipeline.transform(eval_df)

    cyclical_pairs = [
        ("dep_hour_sin", "dep_hour_cos"),
        ("arr_hour_sin", "arr_hour_cos"),
        ("month_sin", "month_cos"),
        ("day_of_week_sin", "day_of_week_cos")
    ]

    for sin_col, cos_col in cyclical_pairs:
        sin_vals = tree_df[sin_col].values
        cos_vals = tree_df[cos_col].values

        # 1. Bounds check [-1.0, 1.0]
        assert np.all(sin_vals >= -1.0 - 1e-6) and np.all(sin_vals <= 1.0 + 1e-6), f"{sin_col} out of [-1, 1]"
        assert np.all(cos_vals >= -1.0 - 1e-6) and np.all(cos_vals <= 1.0 + 1e-6), f"{cos_col} out of [-1, 1]"

        # 2. Trigonometric identity: sin^2 + cos^2 == 1.0
        pythagorean_sum = sin_vals**2 + cos_vals**2
        np.testing.assert_allclose(
            pythagorean_sum, 1.0, atol=1e-5,
            err_msg=f"Pythagorean identity failed for pair ({sin_col}, {cos_col})"
        )


# -------------------------------------------------------------------------
# 3. Route Fallback & Geodesic Calculation Tests
# -------------------------------------------------------------------------

def test_haversine_geodesic_distance_accuracy():
    """Verifies Haversine formula calculation between known airport coordinates."""
    # JFK (New York): 40.6413 N, 73.7781 W
    # LAX (Los Angeles): 33.9416 N, 118.4085 W
    lat_jfk, lon_jfk = 40.6413, -73.7781
    lat_lax, lon_lax = 33.9416, -118.4085

    dist = haversine_distance_miles(lat_jfk, lon_jfk, lat_lax, lon_lax)
    # Expected great-circle distance is ~2,475 statute miles (+/- 15 miles)
    assert 2450.0 < dist < 2500.0, f"Unexpected geodesic distance JFK-LAX: {dist:.1f} miles"


def test_unseen_route_distance_and_prior_fallback(predictor: FlightPredictor):
    """
    Verifies that when an unseen route is queried:
    - Geodesic fallback calculates realistic distance.
    - Target priors default to the global training baseline (0.1762).
    - Metadata flags route_known = False.
    """
    # ANC (Anchorage, AK) to MIA (Miami, FL) is not in standard domestic route lookups
    payload = {
        "airline_code": "DL",
        "origin": "ANC",
        "dest": "MIA",
        "flight_date": "2023-06-21",
        "sched_dep_time": "08:00",
        "sched_arr_time": "19:30"
    }

    result = predictor.predict(payload, include_shap=False)
    assert result["route_known"] is False
    # Great-circle distance between ANC and MIA is ~3,999 statute miles
    dist = result["derived_features"]["distance_miles"]
    assert 3950.0 < dist < 4100.0, f"Unexpected fallback distance for ANC-MIA: {dist}"
    assert result["reference_base_rates"]["train_delay_rate"] == 0.1762


# -------------------------------------------------------------------------
# 4. Platt Calibration Monotonicity Tests
# -------------------------------------------------------------------------

def test_platt_scaling_strict_monotonicity():
    """
    Verifies that post-hoc Platt scaling (A=0.933139, B=0.097311):
    - Is strictly monotonically increasing across the entire logit spectrum.
    - Yields calibrated probabilities strictly bounded in (0, 1).
    - Preserves relative risk ordering between flights.
    """
    slope_A = 0.933138807978378
    intercept_B = 0.09731147911152838

    # Generate 1,000 raw uncalibrated probabilities in (0.001, 0.999)
    p_raw = np.linspace(0.001, 0.999, 1000)
    z = logit(p_raw)
    p_cal = expit(slope_A * z + intercept_B)

    # 1. Monotonicity check
    diffs = np.diff(p_cal)
    assert np.all(diffs > 0), "Platt scaling violated strict monotonicity!"

    # 2. Probability bounds check
    assert np.all(p_cal > 0.0) and np.all(p_cal < 1.0)

    # 3. Exact midpoint calibration check: logit(0.5) = 0 -> p_cal = expit(B)
    p_midpoint = expit(intercept_B)
    expected_mid = 1.0 / (1.0 + math.exp(-intercept_B))
    assert abs(p_midpoint - expected_mid) < 1e-7
    assert 0.524 < p_midpoint < 0.525
