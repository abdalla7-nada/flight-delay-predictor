"""
api/index.py
FastAPI Serverless Application for Flight Delay Risk Predictor (Champion v2).
Optimized for Vercel Serverless Functions with zero third-party ML runtime dependencies.
Zero heavy ML libraries required.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Union

# Ensure api directory is on sys.path
API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator, field_validator

from predictor import ServerlessPredictor, InputValidationError

# Initialize singleton predictor with local assets
ASSETS_DIR = API_DIR / "assets"
predictor = ServerlessPredictor(assets_dir=ASSETS_DIR)

app = FastAPI(
    title="Flight Delay Risk Predictor Serverless API",
    version="2.0.0",
    description=(
        "Schedule-only pre-departure flight delay risk stratifier running on Vercel Serverless. "
        "Built on a 178-tree CatBoost classifier calibrated via Platt logistic scaling. "
        "Engineered with zero third-party ML runtime dependencies."
    ),
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for Next.js frontend and external client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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

    @model_validator(mode="after")
    def check_airports_distinct(self):
        if self.origin and self.dest and self.origin.strip().upper() == self.dest.strip().upper():
            raise ValueError("Origin and destination airports cannot be identical.")
        return self

    @field_validator("flight_date")
    @classmethod
    def validate_flight_date(cls, v: str) -> str:
        s = str(v).strip()
        try:
            datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid flight_date format '{v}'; must be ISO format YYYY-MM-DD.")
        return s

    @field_validator("sched_dep_time", "sched_arr_time")
    @classmethod
    def validate_time(cls, v: str) -> str:
        s = str(v).strip()
        try:
            if ":" in s:
                parts = s.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Invalid time format '{v}'; expected HH:MM.")
                h, m = int(parts[0]), int(parts[1])
            else:
                if not s.isdigit():
                    raise ValueError(f"Non-numeric time value '{v}'.")
                if len(s) in (1, 2):
                    h, m = int(s), 0
                elif len(s) in (3, 4):
                    h, m = int(s[:-2]), int(s[-2:])
                else:
                    raise ValueError(f"Time string '{v}' has invalid length.")

            if (h == 24 and m == 0) or (h == 0 and m == 0):
                return s
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError(f"Time '{v}' is outside valid military time range (00:00 - 23:59).")
        except ValueError as e:
            raise ValueError(str(e))
        return s

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
    top_drivers: List[DriverInfo] = Field(default_factory=list, description="Top-3 risk attribution drivers")
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
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=err_body)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Translates Pydantic schema validation errors to consistent HTTP 422 JSON responses."""
    messages = []
    for err in exc.errors():
        msg = err.get("msg", "")
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        loc = ": ".join(str(l) for l in err.get("loc", []) if l != "body")
        messages.append(f"{loc}: {msg}" if loc else msg)
    detail_str = " | ".join(messages) if messages else "Invalid request payload."
    err_body = {
        "error": "Unprocessable Entity",
        "detail": detail_str,
        "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY
    }
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=err_body)

# -------------------------------------------------------------------------
# Core Helper Logic
# -------------------------------------------------------------------------

def _health_payload():
    return {
        "status": "healthy",
        "service": "flight-delay-predictor-serverless",
        "model_version": predictor.model_version,
        "model_name": predictor.serving_metadata.get("model_name", "Champion v2 (Serverless Pure Python)"),
        "artifacts_verified": True,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def _options_payload():
    return {
        "airlines": predictor.vocab.get("airlines", []),
        "airline_count": len(predictor.vocab.get("airlines", [])),
        "airports": predictor.vocab.get("airports", []),
        "airport_count": len(predictor.vocab.get("airports", [])),
        "part_of_day_categories": predictor.vocab.get("part_of_day", ["night", "morning", "afternoon", "evening"]),
        "reference_base_rates": predictor.base_rates
    }

def _predict_payload(payload: dict, include_shap: bool = True):
    try:
        return predictor.predict(payload, include_shap=include_shap)
    except InputValidationError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

def _predict_batch_payload(payloads: List[dict], include_shap: bool = False):
    try:
        return predictor.predict_batch(payloads, max_batch=100, include_shap=include_shap)
    except InputValidationError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")

# -------------------------------------------------------------------------
# Dual Route Endpoints (Supports both root paths and /api/* prefixes)
# -------------------------------------------------------------------------

@app.get("/", tags=["System"])
@app.get("/api", tags=["System"])
def get_root():
    """Root info check."""
    return _health_payload()

@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Returns service health status, model version, and artifact readiness."""
    return _health_payload()

@app.get("/metadata", tags=["System"])
@app.get("/api/metadata", tags=["System"])
def get_metadata():
    """Returns full serving metadata, decile boundaries, metrics, and parameters."""
    return predictor.serving_metadata

@app.get("/options", tags=["System"])
@app.get("/api/options", tags=["System"])
def get_options():
    """Returns selectable airlines and airports for frontend dropdowns."""
    return _options_payload()

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_flight(request: FlightPredictionRequest, include_shap: bool = True):
    """
    Predicts probability of a >15 minute arrival delay for a single scheduled flight.
    Derives elapsed time and distance automatically from the historical route database.
    """
    return _predict_payload(request.model_dump(), include_shap=include_shap)

@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Inference"])
@app.post("/api/predict/batch", response_model=List[PredictionResponse], tags=["Inference"])
def predict_batch_flights(
    request: Union[BatchPredictionRequest, List[FlightPredictionRequest]],
    include_shap: bool = False
):
    """
    Predicts arrival delay probabilities for a batch of flights (up to 100 items).
    Accepts either `{'flights': [...]}` or a raw list `[...]`.
    """
    if isinstance(request, BatchPredictionRequest):
        payloads = [f.model_dump() for f in request.flights]
    elif isinstance(request, list):
        payloads = [f.model_dump() for f in request]
    else:
        payloads = []
    return _predict_batch_payload(payloads, include_shap=include_shap)
