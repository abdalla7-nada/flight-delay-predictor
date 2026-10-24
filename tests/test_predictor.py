"""
tests/test_predictor.py
Test suite for FlightPredictor serving core.
Verifies:
  - Predictor vs batch pipeline numerical equivalence (< 1e-6)
  - Payload without DELAY column
  - Valid probability range (0, 1)
  - Monotonic decile mapping
  - Unseen route handling & coordinate fallback
  - Input validation (unknown airline/airport, same origin/dest, invalid times/dates)
  - Batch limit assertion
  - Graceful skips when local data is missing
"""

import pytest
import numpy as np
import pandas as pd
from scipy.special import logit

from src.serving.predictor import FlightPredictor, InputValidationError

def test_predictor_loaded_and_metadata_present(predictor):
    """Verifies predictor loads frozen assets and serving metadata successfully."""
    assert predictor.model is not None
    assert predictor.pipeline is not None
    assert predictor.calibrator is not None
    assert len(predictor.decile_edges) == 11
    assert predictor.model_version == "v2.0"

def test_payload_without_delay_key(predictor, valid_flight_payload):
    """Verifies that pipe.transform works on dictionary payloads without DELAY key."""
    assert "DELAY" not in valid_flight_payload
    result = predictor.predict(valid_flight_payload, include_shap=False)
    
    assert "calibrated_probability" in result
    assert "risk_decile" in result
    assert "risk_band" in result
    assert 0.0 < result["calibrated_probability"] < 1.0
    assert 1 <= result["risk_decile"] <= 10

def test_probabilities_in_valid_range(predictor, valid_flight_payload):
    """Verifies that calibrated and raw probabilities are strictly bounded in (0, 1)."""
    result = predictor.predict(valid_flight_payload, include_shap=True)
    p_cal = result["calibrated_probability"]
    p_raw = result["raw_probability"]
    
    assert 0.0 < p_cal < 1.0, f"Calibrated probability {p_cal} out of bounds!"
    assert 0.0 < p_raw < 1.0, f"Raw probability {p_raw} out of bounds!"
    assert result["risk_band"] in ["Low", "Below average", "Average", "Above average", "High"]

def test_decile_mapping_monotonic(predictor):
    """Verifies that the decile mapping function is strictly monotonic non-decreasing."""
    test_probs = np.linspace(0.01, 0.99, 100)
    deciles = [predictor._determine_risk_tier(p)[0] for p in test_probs]
    
    # Check monotonicity
    for i in range(len(deciles) - 1):
        assert deciles[i] <= deciles[i + 1], f"Monotonicity inversion at prob {test_probs[i]}: {deciles[i]} > {deciles[i+1]}"

    # Verify boundary edge cases
    d_low, b_low = predictor._determine_risk_tier(0.01)
    d_high, b_high = predictor._determine_risk_tier(0.95)
    assert d_low == 1 and b_low == "Low"
    assert d_high == 10 and b_high == "High"

def test_unseen_route_handling(predictor, unseen_route_payload):
    """Verifies that an unseen route is handled gracefully with fallback distance and warnings."""
    result = predictor.predict(unseen_route_payload, include_shap=False)
    
    assert result["route_known"] is False
    assert len(result["warnings"]) >= 1
    assert any("not in the historical training route table" in w for w in result["warnings"])
    assert result["derived_features"]["distance_miles"] > 1000.0
    assert 0.0 < result["calibrated_probability"] < 1.0

def test_invalid_airline_raises_422(predictor, valid_flight_payload):
    """Verifies that an unknown airline raises InputValidationError with clear message."""
    bad_payload = {**valid_flight_payload, "airline_code": "UNKNOWN_AIRLINE"}
    with pytest.raises(InputValidationError) as exc_info:
        predictor.predict(bad_payload)
    assert "Unknown airline_code" in str(exc_info.value)

def test_invalid_origin_raises_422(predictor, valid_flight_payload):
    """Verifies that an unknown origin airport raises InputValidationError."""
    bad_payload = {**valid_flight_payload, "origin": "XYZ_FAKE"}
    with pytest.raises(InputValidationError) as exc_info:
        predictor.predict(bad_payload)
    assert "Unknown origin airport" in str(exc_info.value)

def test_invalid_dest_raises_422(predictor, valid_flight_payload):
    """Verifies that an unknown destination airport raises InputValidationError."""
    bad_payload = {**valid_flight_payload, "dest": "ZZZ_FAKE"}
    with pytest.raises(InputValidationError) as exc_info:
        predictor.predict(bad_payload)
    assert "Unknown destination airport" in str(exc_info.value)

def test_same_origin_and_dest_raises_422(predictor, valid_flight_payload):
    """Verifies that identical origin and destination airport raises InputValidationError."""
    bad_payload = {**valid_flight_payload, "origin": "DFW", "dest": "DFW"}
    with pytest.raises(InputValidationError) as exc_info:
        predictor.predict(bad_payload)
    assert "cannot be the same airport" in str(exc_info.value)

def test_invalid_departure_time_raises_422(predictor, valid_flight_payload):
    """Verifies that an invalid scheduled departure time raises InputValidationError."""
    bad_payload = {**valid_flight_payload, "sched_dep_time": "25:70"}
    with pytest.raises(InputValidationError) as exc_info:
        predictor.predict(bad_payload)
    assert "outside valid 24-hour range" in str(exc_info.value)

def test_invalid_date_raises_422(predictor, valid_flight_payload):
    """Verifies that an unparseable flight date raises InputValidationError."""
    bad_payload = {**valid_flight_payload, "flight_date": "not-a-date"}
    with pytest.raises(InputValidationError) as exc_info:
        predictor.predict(bad_payload)
    assert "Invalid flight_date format" in str(exc_info.value)

def test_batch_prediction_limit(predictor, valid_flight_payload):
    """Verifies that submitting over 100 items in batch mode raises InputValidationError."""
    over_limit_batch = [valid_flight_payload] * 101
    with pytest.raises(InputValidationError) as exc_info:
        predictor.predict_batch(over_limit_batch, max_batch=100)
    assert "exceeds maximum allowed size of 100" in str(exc_info.value)

def test_predictor_vs_batch_equivalence(predictor, repo_root):
    """
    Evaluates 200 validation rows through predictor (with true values) vs batch pipeline.
    Max absolute difference must be below 1e-6.
    Skips with a clear message if validation data is missing.
    """
    val_path = repo_root / "data" / "processed" / "val.parquet"
    if not val_path.exists():
        pytest.skip(f"Validation parquet file missing at {val_path}; skipping batch equivalence test.")

    val_df = pd.read_parquet(val_path)
    sample_200 = val_df.sample(n=200, random_state=123).copy().reset_index(drop=True)

    # 1. Batch pipeline execution
    tree_200, _ = predictor.pipeline.transform(sample_200)
    p_raw_batch = predictor.model.predict_proba(tree_200[predictor.features])[:, 1]
    z_batch = logit(np.clip(p_raw_batch, 1e-12, 1.0 - 1e-12)).reshape(-1, 1)
    p_cal_batch = predictor.calibrator["model"].predict_proba(z_batch)[:, 1]

    # 2. Predictor execution with true elapsed/distance
    p_cal_predictor = []
    for _, row in sample_200.iterrows():
        res = predictor.predict({
            "airline_code": row["AIRLINE_CODE"],
            "origin": row["ORIGIN"],
            "dest": row["DEST"],
            "flight_date": str(row["FL_DATE"])[:10],
            "sched_dep_time": int(row["CRS_DEP_TIME"]),
            "sched_arr_time": int(row["CRS_ARR_TIME"]),
            "crs_elapsed_time": float(row["CRS_ELAPSED_TIME"]),
            "distance": float(row["DISTANCE"])
        }, include_shap=False)
        p_cal_predictor.append(res["calibrated_probability"])

    max_diff = float(np.max(np.abs(np.array(p_cal_predictor) - p_cal_batch)))
    assert max_diff < 1e-6, f"Batch vs predictor max difference {max_diff} exceeds 1e-6 tolerance!"
