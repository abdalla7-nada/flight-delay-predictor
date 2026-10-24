"""
api/main.py
REST API for Flight Delay Risk Predictor (Champion v2).
Built with FastAPI, Pydantic, and Uvicorn.
Provides:
  - GET  /health
  - GET  /metadata
  - GET  /options
  - POST /predict
  - POST /predict/batch (max 100)
Includes JSONL request logging, OpenAPI docs, and 422 error handling.
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.serving.predictor import FlightPredictor, InputValidationError

# Setup request logger directory
LOGS_DIR = PROJECT_ROOT / "api" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_LOG_FILE = LOGS_DIR / "requests.jsonl"

def log_request_event(endpoint: str, input_data: Any, output_data: Any, duration_ms: float, status_code: int):
    """Appends structured JSONL request/response log (inputs & outputs only, no personal data)."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "input": input_data,
        "output": output_data
    }
    try:
        with open(REQUEST_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        # Never fail request due to logging error
        pass

# Initialize predictor singleton
predictor = FlightPredictor(repo_root=PROJECT_ROOT)

app = FastAPI(
    title="Flight Delay Risk Predictor API",
    version="2.0.0",
    description=(
        "Schedule-only pre-departure flight delay risk stratifier. "
        "Built on a 178-tree CatBoost classifier calibrated via Platt logistic scaling. "
        "Designed strictly for ticket booking and itinerary schedule evaluation."
    ),
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware hardening (supports local Next.js dev and Vercel deployments)
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
if allowed_origins_env == "*":
    allowed_origins = ["*"]
    origin_regex = None
elif allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    origin_regex = r"https://.*\.vercel\.app"
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
    ]
    origin_regex = r"https://.*\.vercel\.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
)

# -------------------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------------------

class FlightPredictionRequest(BaseModel):
    airline_code: str = Field(
        ...,
        examples=["AA"],
        description="Two-letter IATA reporting carrier code (e.g., 'AA', 'DL', 'UA', 'WN')"
    )
    origin: str = Field(
        ...,
        examples=["DFW"],
        description="Three-letter IATA departure airport code (e.g., 'DFW', 'ATL', 'ORD')"
    )
    dest: str = Field(
        ...,
        examples=["ORD"],
        description="Three-letter IATA destination airport code (e.g., 'ORD', 'LAX', 'LGA')"
    )
    flight_date: str = Field(
        ...,
        examples=["2023-05-15"],
        description="Flight departure date in ISO format YYYY-MM-DD"
    )
    sched_dep_time: str = Field(
        ...,
        examples=["08:30"],
        description="Scheduled local departure time in HH:MM, HHMM, or integer format"
    )
    sched_arr_time: str = Field(
        ...,
        examples=["11:00"],
        description="Scheduled local arrival time in HH:MM, HHMM, or integer format"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "airline_code": "AA",
                "origin": "DFW",
                "dest": "ORD",
                "flight_date": "2023-05-15",
                "sched_dep_time": "08:30",
                "sched_arr_time": "11:00"
            }
        }
    }

class BatchPredictionRequest(BaseModel):
    flights: List[FlightPredictionRequest] = Field(
        ...,
        max_length=100,
        description="List of flight itineraries to score (max 100 items per batch)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "flights": [
                    {
                        "airline_code": "AA",
                        "origin": "DFW",
                        "dest": "ORD",
                        "flight_date": "2023-05-15",
                        "sched_dep_time": "08:30",
                        "sched_arr_time": "11:00"
                    },
                    {
                        "airline_code": "DL",
                        "origin": "ATL",
                        "dest": "LGA",
                        "flight_date": "2023-06-20",
                        "sched_dep_time": "17:45",
                        "sched_arr_time": "20:10"
                    }
                ]
            }
        }
    }

class DriverInfo(BaseModel):
    feature: str
    description: str
    feature_value: str
    impact_log_odds: float
    direction: str
    summary: str

class DerivedFeatures(BaseModel):
    distance_miles: float
    scheduled_elapsed_minutes: float

class BaseRates(BaseModel):
    train_delay_rate: float
    validation_delay_rate: float
    test_delay_rate: float

class PredictionResponse(BaseModel):
    calibrated_probability: float = Field(..., description="Calibrated delay probability in [0.0, 1.0]")
    raw_probability: float = Field(..., description="Raw uncalibrated CatBoost probability")
    risk_decile: int = Field(..., description="Risk ranking decile from 1 (lowest risk) to 10 (highest risk)")
    risk_band: str = Field(..., description="Categorical risk tier: Low / Below average / Average / Above average / High")
    reference_base_rates: BaseRates
    route_known: bool = Field(..., description="True if route was present in the historical training table")
    derived_features: DerivedFeatures
    top_drivers: List[DriverInfo] = Field(default_factory=list, description="Top-3 SHAP contribution drivers in plain language")
    warnings: List[str] = Field(default_factory=list, description="Operational and input parameter warnings")
    model_version: str = Field(..., description="Active champion model version")
    disclaimer: str = Field(..., description="Operational bounds and limitations disclaimer")

class HealthResponse(BaseModel):
    status: str
    service: str
    model_version: str
    model_name: str
    artifacts_verified: bool
    timestamp: str

# -------------------------------------------------------------------------
# Custom Exception Handlers
# -------------------------------------------------------------------------

@app.exception_handler(InputValidationError)
async def input_validation_error_handler(request: Request, exc: InputValidationError):
    """Translates domain input validation errors to consistent HTTP 422 JSON responses."""
    err_body = {
        "error": "Unprocessable Entity",
        "detail": str(exc),
        "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY
    }
    log_request_event(request.url.path, {}, err_body, 0.0, 422)
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=err_body)

# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Returns service health status, model version, and artifact readiness."""
    return {
        "status": "healthy",
        "service": "flight-delay-predictor-api",
        "model_version": predictor.model_version,
        "model_name": predictor.serving_metadata.get("model_name", "Champion v2"),
        "artifacts_verified": True,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/metadata", tags=["System"])
def get_metadata():
    """Returns full serving metadata, decile boundaries, metrics with CIs, and frozen SHA256 hashes."""
    return predictor.serving_metadata

@app.get("/options", tags=["System"])
def get_options():
    """Returns selectable airlines and airports for frontend dropdowns and autocomplete."""
    return {
        "airlines": predictor.vocab["airlines"],
        "airline_count": len(predictor.vocab["airlines"]),
        "airports": predictor.vocab["airports"],
        "airport_count": len(predictor.vocab["airports"]),
        "part_of_day_categories": predictor.vocab.get("part_of_day", ["night", "morning", "afternoon", "evening"]),
        "reference_base_rates": predictor.base_rates
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_flight(request: FlightPredictionRequest):
    """
    Predicts probability of a >15 minute arrival delay for a single scheduled flight.
    Derives elapsed time and distance automatically from the historical route database.
    """
    t0 = time.perf_counter()
    payload = request.model_dump()
    try:
        result = predictor.predict(payload, include_shap=True)
        dur_ms = (time.perf_counter() - t0) * 1000
        log_request_event("/predict", payload, result, dur_ms, 200)
        return result
    except InputValidationError:
        raise
    except Exception as e:
        dur_ms = (time.perf_counter() - t0) * 1000
        log_request_event("/predict", payload, {"error": str(e)}, dur_ms, 500)
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Inference"])
def predict_batch_flights(request: BatchPredictionRequest):
    """
    Predicts arrival delay probabilities for a batch of flights (up to 100 items).
    Optimized for throughput.
    """
    t0 = time.perf_counter()
    payloads = [f.model_dump() for f in request.flights]
    try:
        results = predictor.predict_batch(payloads, max_batch=100, include_shap=False)
        dur_ms = (time.perf_counter() - t0) * 1000
        log_request_event("/predict/batch", {"batch_size": len(payloads)}, {"batch_size": len(results)}, dur_ms, 200)
        return results
    except InputValidationError:
        raise
    except Exception as e:
        dur_ms = (time.perf_counter() - t0) * 1000
        log_request_event("/predict/batch", {"batch_size": len(payloads)}, {"error": str(e)}, dur_ms, 500)
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")
