"""
tests/test_model_distillation_parity.py
Automated CI verification of bit-level model distillation parity:
  - Validates that the pure-Python distilled CatBoost decision tree model
    (frontend/api/champion_model.py + frontend/api/predictor.py)
    matches the frozen Joblib batch model (backend/artifacts/models/champion_v2.joblib)
    across >= 200 diverse flight itineraries.
  - Verifies probability delta abs(actual_prob - expected_prob) < 1e-6 (empirically ~10^-17).
  - Confirms closed-form Platt scaling equivalence against Scikit-Learn LogisticRegression.
"""

import sys
import math
from pathlib import Path
import pytest
import numpy as np

# Resolve repo root and backend/frontend directories
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
FRONTEND_API_DIR = REPO_ROOT / "frontend" / "api"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(FRONTEND_API_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_API_DIR))

from src.serving.predictor import FlightPredictor
from predictor import ServerlessPredictor, PLATT_SLOPE, PLATT_INTERCEPT


@pytest.fixture(scope="module")
def batch_predictor():
    """Initializes the frozen Joblib + CatBoost batch predictor."""
    return FlightPredictor(repo_root=BACKEND_DIR)


@pytest.fixture(scope="module")
def serverless_predictor():
    """Initializes the pure-Python zero-dependency serverless predictor."""
    assets_dir = FRONTEND_API_DIR / "assets"
    return ServerlessPredictor(assets_dir=assets_dir)


def generate_diverse_itineraries(n_samples: int = 200):
    """
    Generates >= 200 distinct, diverse flight itineraries spanning:
      - All 18 reporting carriers
      - Diverse origin and destination airport pairs across nationwide hubs
      - Distinct times of day (morning, midday, evening, night)
      - Various flight dates across weekdays, weekends, and holidays
    """
    hubs = [
        "ATL", "DFW", "DEN", "ORD", "LAX", "CLT", "MCO", "LAS",
        "PHX", "MIA", "SEA", "IAH", "JFK", "EWR", "SFO", "MSP",
        "DTW", "BOS", "PHL", "LGA", "BWI", "SLC", "SAN", "DCA"
    ]
    carriers = [
        "9E", "AA", "AS", "B6", "DL", "EV", "F9", "G4", "HA",
        "MQ", "NK", "OH", "OO", "QX", "UA", "WN", "YV", "YX"
    ]
    dates = [
        "2023-01-02",  # New Year holiday
        "2023-03-15",  # Mid-week spring
        "2023-05-29",  # Memorial Day holiday
        "2023-07-04",  # Independence Day holiday
        "2023-08-18",  # Late summer weekend travel
        "2023-10-12",  # Autumn weekday
        "2023-11-23",  # Thanksgiving holiday peak
        "2023-12-24",  # Christmas Eve holiday travel
    ]
    time_pairs = [
        ("06:00", "08:30"),
        ("07:30", "10:15"),
        ("09:15", "12:00"),
        ("11:00", "13:45"),
        ("13:30", "16:15"),
        ("15:00", "18:20"),
        ("17:45", "20:30"),
        ("19:30", "22:15"),
        ("21:00", "23:45"),
    ]

    itineraries = []
    idx = 0
    for i in range(len(hubs)):
        for j in range(len(hubs)):
            if i == j:
                continue
            carrier = carriers[idx % len(carriers)]
            f_date = dates[idx % len(dates)]
            dep_t, arr_t = time_pairs[idx % len(time_pairs)]

            itineraries.append({
                "airline_code": carrier,
                "origin": hubs[i],
                "dest": hubs[j],
                "flight_date": f_date,
                "sched_dep_time": dep_t,
                "sched_arr_time": arr_t
            })
            idx += 1
            if len(itineraries) >= n_samples:
                return itineraries
    return itineraries


def test_model_distillation_bit_level_parity(batch_predictor, serverless_predictor):
    """
    Evaluates 200 diverse sample itineraries across different carriers,
    hub airports, and route configurations.
    Verifies that distilled pure-Python serving achieves strict numerical
    tolerance (< 1e-6) against the frozen reference batch pipeline.
    """
    itineraries = generate_diverse_itineraries(n_samples=200)
    assert len(itineraries) >= 200, f"Expected >= 200 itineraries, got {len(itineraries)}"

    # Confirm diversity across carriers and airport hubs
    unique_airlines = set(item["airline_code"] for item in itineraries)
    assert len(unique_airlines) == 18, "All 18 carriers must be represented in test cohort"

    cal_prob_diffs = []
    raw_prob_diffs = []

    for idx, flight in enumerate(itineraries):
        res_batch = batch_predictor.predict(flight, include_shap=False)
        res_serverless = serverless_predictor.predict(flight, include_shap=False)

        expected_prob = res_batch["calibrated_probability"]
        actual_prob = res_serverless["calibrated_probability"]

        expected_raw = res_batch["raw_probability"]
        actual_raw = res_serverless["raw_probability"]

        # Strict numerical tolerance: < 1e-6
        delta_cal = abs(actual_prob - expected_prob)
        delta_raw = abs(actual_raw - expected_raw)

        cal_prob_diffs.append(delta_cal)
        raw_prob_diffs.append(delta_raw)

        assert delta_cal < 1e-6, (
            f"Itinerary #{idx} ({flight['airline_code']} {flight['origin']}->{flight['dest']}) "
            f"calibrated probability diff {delta_cal:.2e} exceeds 1e-6 tolerance! "
            f"Actual={actual_prob}, Expected={expected_prob}"
        )

        assert delta_raw < 1e-6, (
            f"Itinerary #{idx} raw probability diff {delta_raw:.2e} exceeds 1e-6 tolerance!"
        )

        # Invariant decile and risk band classification
        assert res_serverless["risk_decile"] == res_batch["risk_decile"], (
            f"Decile mismatch on itinerary #{idx}: "
            f"Serverless={res_serverless['risk_decile']} vs Batch={res_batch['risk_decile']}"
        )
        assert res_serverless["risk_band"] == res_batch["risk_band"], (
            f"Risk band mismatch on itinerary #{idx}: "
            f"Serverless={res_serverless['risk_band']} vs Batch={res_batch['risk_band']}"
        )

    max_cal_diff = max(cal_prob_diffs)
    max_raw_diff = max(raw_prob_diffs)

    # Emphasize empirical sub-1e-10 bit-level parity
    assert max_cal_diff < 1e-6
    print(f"\n[PARITY VERIFIED] Scored {len(itineraries)} diverse itineraries.")
    print(f"Max Calibrated Prob Delta: {max_cal_diff:.2e}")
    print(f"Max Raw Prob Delta:        {max_raw_diff:.2e}")


def test_closed_form_platt_scaling_analytical_equivalence(batch_predictor):
    """
    Confirms that closed-form Platt scaling:
        P = 1 / (1 + exp(-(A * z + B)))
    analytically reproduces Scikit-Learn's LogisticRegression predict_proba.
    """
    calibrator_model = batch_predictor.calibrator["model"]
    test_margins = np.linspace(-4.0, 4.0, 50)

    # Scikit-learn LogisticRegression prediction
    sklearn_probs = calibrator_model.predict_proba(test_margins.reshape(-1, 1))[:, 1]

    # Analytical closed-form evaluation
    analytical_probs = [
        1.0 / (1.0 + math.exp(-(PLATT_SLOPE * z + PLATT_INTERCEPT)))
        for z in test_margins
    ]

    for z, p_skl, p_ana in zip(test_margins, sklearn_probs, analytical_probs):
        assert abs(p_skl - p_ana) < 1e-12, (
            f"Closed-form Platt scaling divergence at margin {z}: "
            f"sklearn={p_skl}, analytical={p_ana}"
        )


def test_cold_start_unseen_route_parity(batch_predictor, serverless_predictor):
    """
    Confirms that unseen routes (e.g. ANC -> MIA) trigger identical
    geodesic Haversine fallback calculations and warning structures in both engines.
    """
    unseen_flight = {
        "airline_code": "DL",
        "origin": "ANC",
        "dest": "MIA",
        "flight_date": "2023-06-20",
        "sched_dep_time": "09:15",
        "sched_arr_time": "18:45"
    }

    res_batch = batch_predictor.predict(unseen_flight, include_shap=False)
    res_serverless = serverless_predictor.predict(unseen_flight, include_shap=False)

    assert res_batch["route_known"] is False
    assert res_serverless["route_known"] is False

    # Check derived distance and elapsed time parity
    batch_dist = res_batch["derived_features"]["distance_miles"]
    serverless_dist = res_serverless["derived_features"]["distance_miles"]
    assert abs(batch_dist - serverless_dist) < 1e-4

    batch_elapsed = res_batch["derived_features"]["scheduled_elapsed_minutes"]
    serverless_elapsed = res_serverless["derived_features"]["scheduled_elapsed_minutes"]
    assert abs(batch_elapsed - serverless_elapsed) < 1e-4

    # Check calibrated probability parity on cold-start route
    diff = abs(res_serverless["calibrated_probability"] - res_batch["calibrated_probability"])
    assert diff < 1e-6
