"""
tests/conftest.py
Pytest fixtures for Flight Delay Serving test suite.
"""

import os
from pathlib import Path
import pytest

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.serving.predictor import FlightPredictor
from api.main import app

@pytest.fixture(scope="session")
def repo_root():
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def predictor(repo_root):
    return FlightPredictor(repo_root=repo_root)

@pytest.fixture(scope="session")
def api_client():
    return TestClient(app)

@pytest.fixture
def valid_flight_payload():
    return {
        "airline_code": "AA",
        "origin": "DFW",
        "dest": "ORD",
        "flight_date": "2023-05-15",
        "sched_dep_time": "08:30",
        "sched_arr_time": "11:00"
    }

@pytest.fixture
def unseen_route_payload():
    return {
        "airline_code": "DL",
        "origin": "ANC",
        "dest": "MIA",
        "flight_date": "2023-06-20",
        "sched_dep_time": "09:15",
        "sched_arr_time": "18:45"
    }
