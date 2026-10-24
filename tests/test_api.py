"""
tests/test_api.py
Test suite for FastAPI endpoints (HTTP contracts & status codes).
"""

import pytest
from fastapi.testclient import TestClient

def test_api_health_endpoint(api_client: TestClient):
    """Verifies GET /health returns 200 with healthy status."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "v2.0"
    assert data["artifacts_verified"] is True
    assert "timestamp" in data

def test_api_metadata_endpoint(api_client: TestClient):
    """Verifies GET /metadata returns complete serving configuration and metrics."""
    response = api_client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "base_rates" in data
    assert "decile_edges_validation" in data
    assert "frozen_artifacts_sha256" in data
    assert "validation_metrics" in data
    assert "test_metrics" in data

def test_api_options_endpoint(api_client: TestClient):
    """Verifies GET /options returns supported airlines and airports."""
    response = api_client.get("/options")
    assert response.status_code == 200
    data = response.json()
    assert "airlines" in data
    assert "airports" in data
    assert len(data["airlines"]) >= 17
    assert len(data["airports"]) >= 375
    assert "AA" in data["airlines"]
    assert "DFW" in data["airports"]

def test_api_predict_single_success(api_client: TestClient, valid_flight_payload):
    """Verifies POST /predict successfully scores a flight and returns valid schema."""
    response = api_client.post("/predict", json=valid_flight_payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "calibrated_probability" in data
    assert 0.0 < data["calibrated_probability"] < 1.0
    assert 1 <= data["risk_decile"] <= 10
    assert data["risk_band"] in ["Low", "Below average", "Average", "Above average", "High"]
    assert data["route_known"] is True
    assert data["derived_features"]["distance_miles"] > 0
    assert data["derived_features"]["scheduled_elapsed_minutes"] > 0
    assert "reference_base_rates" in data
    assert "disclaimer" in data
    assert len(data["top_drivers"]) <= 3

def test_api_predict_unseen_route(api_client: TestClient, unseen_route_payload):
    """Verifies POST /predict handles unseen route with fallback and warning."""
    response = api_client.post("/predict", json=unseen_route_payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["route_known"] is False
    assert len(data["warnings"]) >= 1
    assert data["derived_features"]["distance_miles"] > 1000.0

def test_api_predict_unknown_airline_422(api_client: TestClient, valid_flight_payload):
    """Verifies POST /predict returns 422 for unsupported airline."""
    bad_payload = {**valid_flight_payload, "airline_code": "INVALID_AIRLINE"}
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "Unknown airline_code" in data["detail"]

def test_api_predict_unknown_airport_422(api_client: TestClient, valid_flight_payload):
    """Verifies POST /predict returns 422 for unsupported airport code."""
    bad_payload = {**valid_flight_payload, "origin": "XXX_UNKNOWN"}
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "Unknown origin airport" in data["detail"]

def test_api_predict_same_origin_dest_422(api_client: TestClient, valid_flight_payload):
    """Verifies POST /predict returns 422 when origin and destination match."""
    bad_payload = {**valid_flight_payload, "dest": "DFW"}
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "cannot be the same airport" in data["detail"]

def test_api_batch_predict_success(api_client: TestClient, valid_flight_payload, unseen_route_payload):
    """Verifies POST /predict/batch scores multiple flights."""
    batch_payload = {"flights": [valid_flight_payload, unseen_route_payload]}
    response = api_client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["route_known"] is True
    assert data[1]["route_known"] is False

def test_api_batch_predict_over_limit_422(api_client: TestClient, valid_flight_payload):
    """Verifies POST /predict/batch returns 422 when exceeding 100 items."""
    batch_payload = {"flights": [valid_flight_payload] * 101}
    response = api_client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 422

def test_api_logging_privacy(api_client: TestClient, valid_flight_payload):
    """Verifies that requests.jsonl records NEVER contain client IP, user_agent, or personal identifiers."""
    import json
    from pathlib import Path
    log_path = Path("api/logs/requests.jsonl")
    
    # Make a prediction with custom headers
    response = api_client.post(
        "/predict",
        json=valid_flight_payload,
        headers={"User-Agent": "TestPrivacyAuditor/1.0", "X-Forwarded-For": "192.168.1.100"}
    )
    assert response.status_code == 200
    
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    latest_record = json.loads(lines[-1])
    
    # Verify strict privacy constraints: no IP, no user agent
    assert "ip" not in latest_record
    assert "client_ip" not in latest_record
    assert "user_agent" not in latest_record
    assert "User-Agent" not in latest_record
    assert "headers" not in latest_record
    assert set(latest_record.keys()) == {"timestamp", "endpoint", "status_code", "duration_ms", "input", "output"}


def test_api_cors_preflight(api_client: TestClient):
    """Verifies CORS preflight headers for localhost:3000 and Vercel domains."""
    # Test localhost:3000
    headers_local = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    resp_local = api_client.options("/predict", headers=headers_local)
    assert resp_local.status_code == 200
    assert resp_local.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "POST" in resp_local.headers.get("access-control-allow-methods", "")

    # Test Vercel regex domain
    headers_vercel = {
        "Origin": "https://flight-delay-predictor-app.vercel.app",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    resp_vercel = api_client.options("/predict", headers=headers_vercel)
    assert resp_vercel.status_code == 200
    assert resp_vercel.headers.get("access-control-allow-origin") == "https://flight-delay-predictor-app.vercel.app"

