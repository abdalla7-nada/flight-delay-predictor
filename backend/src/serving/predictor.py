"""
src/serving/predictor.py
Production Flight Delay Risk Predictor (Champion v2).
Loads frozen artifacts once, validates payloads, derives elapsed time & distance,
and returns calibrated delay probabilities with risk tiers, drivers, and disclaimers.
"""

import os
import sys
import json
import math
import time
import logging
import hashlib
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from scipy.special import logit
from catboost import Pool

from src.features import FlightFeaturePipeline

logger = logging.getLogger("flight_delay_serving")
logging.basicConfig(level=logging.INFO)

DISCLAIMER_TEXT = (
    "Schedule-only pre-departure risk estimate based on historical timetable patterns. "
    "Model discrimination is modest (ROC-AUC ~0.65) and represents a relative risk ranking, "
    "not an absolute certainty. Not intended for operational dispatch or safety-critical decisions."
)

FEATURE_DESCRIPTIONS = {
    "prior_route": "Historical delay rate for this specific origin-destination route",
    "prior_origin": "Historical delay rate at the departure airport",
    "prior_dest": "Historical delay rate at the arrival airport",
    "prior_airline": "Historical delay rate of the operating airline carrier",
    "prior_origin_hour": "Historical delay rate at the departure airport for this time of day",
    "dep_hour": "Scheduled departure hour (afternoon/evening flights accumulate higher delay risk)",
    "dep_hour_sin": "Cyclical time-of-day departure effect",
    "dep_hour_cos": "Cyclical time-of-day departure effect",
    "part_of_day": "Scheduled departure time window (morning vs evening)",
    "day_of_week": "Day-of-week travel demand pattern",
    "day_of_week_sin": "Cyclical day-of-week travel pattern",
    "day_of_week_cos": "Cyclical day-of-week travel pattern",
    "is_weekend": "Weekend vs weekday flight schedule effect",
    "is_holiday_period": "High-volume holiday travel congestion period",
    "month": "Seasonal flight traffic and weather pattern",
    "month_sin": "Cyclical seasonal travel pattern",
    "month_cos": "Cyclical seasonal travel pattern",
    "scheduled_speed": "Ratio of distance to scheduled duration (schedule buffer/compression)",
    "origin_freq": "Operating flight frequency and congestion at departure airport",
    "dest_freq": "Operating flight frequency and congestion at arrival airport",
    "route_freq": "Frequency of service on this specific route",
    "DISTANCE": "Flight distance between city pairs",
    "log_distance": "Log-transformed route distance",
    "CRS_ELAPSED_TIME": "Scheduled gate-to-gate flight duration",
    "AIRLINE_CODE": "Airline carrier operational profile",
    "ORIGIN": "Origin airport congestion and airspace profile",
    "DEST": "Destination airport congestion and airspace profile",
    "route": "Directional airport-pair network route"
}

def compute_sha256(filepath: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def haversine_distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great circle distance between two coordinates in statute miles."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

class InputValidationError(ValueError):
    """Exception raised when an inference payload violates schema or vocabulary rules (HTTP 422)."""
    pass

class FlightPredictor:
    """
    Singleton-style predictor for Flight Delay Risk Model (Champion v2).
    Loads model, calibrator, pipeline, and serving lookup tables once.
    """
    def __init__(self, repo_root: Optional[Union[str, Path]] = None):
        self.repo_root = Path(repo_root) if repo_root else PROJECT_ROOT
        self.artifacts_dir = self.repo_root / "artifacts"
        self.serving_dir = self.artifacts_dir / "serving"
        
        # 1. Load serving metadata & vocabulary
        meta_path = self.serving_dir / "serving_metadata.json"
        vocab_path = self.serving_dir / "vocab.json"
        route_path = self.serving_dir / "route_lookup.json"
        
        if not meta_path.exists() or not vocab_path.exists() or not route_path.exists():
            raise FileNotFoundError(
                f"Serving assets missing in {self.serving_dir}. "
                f"Run 'python scripts/build_serving_assets.py' first."
            )
            
        self.serving_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        self.vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.route_lookup = json.loads(route_path.read_text(encoding="utf-8"))
        
        self.known_airlines = set(self.vocab["airlines"])
        self.known_airports = set(self.vocab["airports"])
        self.decile_edges = self.serving_metadata["decile_edges_validation"]
        self.base_rates = self.serving_metadata["base_rates"]
        self.model_version = self.serving_metadata.get("model_version", "v2.0")

        # 2. Verify SHA256 of frozen artifacts against freeze manifest
        self._verify_frozen_hashes()

        # 3. Load ML model artifacts
        pipe_path = self.artifacts_dir / "feature_pipeline_v2.joblib"
        model_path = self.artifacts_dir / "models" / "champion_v2.joblib"
        cal_path = self.artifacts_dir / "models" / "calibrator_v2.joblib"
        meta_v2_path = self.artifacts_dir / "models" / "model_metadata_v2.json"

        logger.info("Loading frozen feature pipeline...")
        self.pipeline: FlightFeaturePipeline = joblib.load(pipe_path)
        logger.info("Loading frozen champion CatBoost model...")
        self.model = joblib.load(model_path)
        logger.info("Loading frozen Platt calibrator...")
        self.calibrator = joblib.load(cal_path)
        
        meta_v2 = json.loads(meta_v2_path.read_text(encoding="utf-8"))
        self.features = meta_v2["features"]
        self.cat_cols = meta_v2["categoricals"]
        self.cat_indices = [self.features.index(c) for c in self.cat_cols]

        # 4. Initialize airports DB for unseen route distance fallback
        self.airports_db = None
        try:
            import airportsdata
            self.airports_db = airportsdata.load("IATA")
            logger.info("airportsdata database loaded successfully.")
        except ImportError:
            logger.warning("airportsdata package not available; using geometric default distance for unseen routes.")

        # Warmup model once
        self._warmup()
        logger.info("FlightPredictor initialized and ready for inference.")

    def _verify_frozen_hashes(self):
        """Verifies SHA256 hashes of frozen artifacts and logs warning on mismatch."""
        manifest_path = self.repo_root / "reports" / "freeze_manifest.json"
        if not manifest_path.exists():
            logger.warning("freeze_manifest.json not found; skipping startup hash verification.")
            return

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for asset_rel, info in manifest.items():
            if asset_rel.startswith("artifacts/"):
                actual_path = self.repo_root / asset_rel
                if actual_path.exists():
                    actual_sha = compute_sha256(actual_path)
                    if actual_sha != info["sha256"]:
                        logger.warning(
                            f"SECURITY / INTEGRITY MISMATCH: {asset_rel} actual SHA256 "
                            f"{actual_sha} does not match frozen manifest {info['sha256']}!"
                        )
                    else:
                        logger.debug(f"Integrity verified for {asset_rel}.")
                else:
                    logger.warning(f"Frozen asset {actual_path} not found!")

    def _warmup(self):
        """Runs a synthetic inference to warm up JIT, cache, and internal structures."""
        sample = {
            "airline_code": "AA",
            "origin": "DFW",
            "dest": "ORD",
            "flight_date": "2023-05-15",
            "sched_dep_time": "08:30",
            "sched_arr_time": "11:00"
        }
        try:
            self.predict(sample, include_shap=False)
        except Exception as e:
            logger.warning(f"Predictor warmup encountered: {e}")

    @staticmethod
    def parse_time_to_int(time_val: Union[int, str]) -> int:
        """
        Parses time in HH:MM, HHMM, or int format to HHMM integer.
        Maps 2400 or 24:00 to 0 (per cleaning rules).
        """
        if isinstance(time_val, int):
            if time_val == 2400:
                return 0
            if 0 <= time_val <= 2359:
                h = time_val // 100
                m = time_val % 100
                if m >= 60:
                    raise InputValidationError(f"Invalid minute component in time {time_val}.")
                return time_val
            raise InputValidationError(f"Time integer {time_val} out of valid range (0 - 2359).")
        
        s = str(time_val).strip()
        if ":" in s:
            parts = s.split(":")
            if len(parts) != 2:
                raise InputValidationError(f"Invalid time format '{time_val}'; expected HH:MM.")
            try:
                h = int(parts[0])
                m = int(parts[1])
            except ValueError:
                raise InputValidationError(f"Non-numeric values in time '{time_val}'.")
        else:
            if not s.isdigit():
                raise InputValidationError(f"Non-numeric time value '{time_val}'.")
            if len(s) in (1, 2):
                h = int(s)
                m = 0
            elif len(s) in (3, 4):
                h = int(s[:-2])
                m = int(s[-2:])
            else:
                raise InputValidationError(f"Time string '{time_val}' has invalid length.")

        if (h == 24 and m == 0) or (h == 0 and m == 0):
            return 0
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise InputValidationError(f"Time {h:02d}:{m:02d} is outside valid 24-hour range (00:00 - 23:59).")
        return h * 100 + m

    @staticmethod
    def parse_date(date_val: Union[str, date, datetime]) -> str:
        """Parses and validates date in ISO format YYYY-MM-DD."""
        if isinstance(date_val, (date, datetime)):
            return date_val.strftime("%Y-%m-%d")
        s = str(date_val).strip()
        try:
            dt = datetime.strptime(s, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            raise InputValidationError(f"Invalid flight_date format '{date_val}'; must be YYYY-MM-DD.")

    def _determine_risk_tier(self, prob: float) -> Tuple[int, str]:
        """Maps probability to validation-derived decile (1-10) and risk band."""
        decile = 10
        for i in range(1, 10):
            if prob < self.decile_edges[i]:
                decile = i
                break
        
        if decile <= 2:
            band = "Low"
        elif decile <= 4:
            band = "Below average"
        elif decile <= 6:
            band = "Average"
        elif decile <= 8:
            band = "Above average"
        else:
            band = "High"

        return decile, band

    def _derive_elapsed_and_distance(
        self, origin: str, dest: str, crs_dep: int, crs_arr: int
    ) -> Tuple[float, float, bool, List[str]]:
        """
        Derives CRS_ELAPSED_TIME and DISTANCE from route_lookup.json,
        or computes geodesic distance via airportsdata and cleaning fallback.
        """
        warnings = []
        route_key = f"{origin}_{dest}"

        if route_key in self.route_lookup:
            info = self.route_lookup[route_key]
            elapsed = float(info["elapsed"])
            distance = float(info["distance"])
            route_known = True
        else:
            route_known = False
            # Geodesic fallback using airportsdata
            dist_est = 1000.0
            if self.airports_db and origin in self.airports_db and dest in self.airports_db:
                o_lat = self.airports_db[origin]["lat"]
                o_lon = self.airports_db[origin]["lon"]
                d_lat = self.airports_db[dest]["lat"]
                d_lon = self.airports_db[dest]["lon"]
                dist_est = round(haversine_distance_miles(o_lat, o_lon, d_lat, d_lon), 1)
            
            # Cleaning fallback: distance / 7.5 + 30.0
            elapsed_est = round(dist_est / 7.5 + 30.0, 1)
            distance = dist_est
            elapsed = elapsed_est
            warnings.append(
                f"Route {route_key} is not in the historical training route table; "
                f"distance estimated as {distance:.1f} mi and elapsed time as {elapsed:.1f} min."
            )

        # Plausibility checks on elapsed duration
        if elapsed <= 15.0 or elapsed > 900.0:
            warnings.append(f"Derived elapsed duration ({elapsed:.1f} min) is atypical for a domestic flight.")
        
        speed = distance / (elapsed / 60.0) if elapsed > 0 else 0
        if speed > 700.0 or speed < 50.0:
            warnings.append(f"Implausible scheduled speed ({speed:.1f} mph) detected.")

        return elapsed, distance, route_known, warnings

    def _compute_shap_drivers(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Computes top-3 SHAP contribution drivers for the flight in plain language."""
        try:
            pool = Pool(X, cat_features=self.cat_indices)
            shap_matrix = self.model.get_feature_importance(pool, type="ShapValues")
            shap_vals = shap_matrix[0, :-1]  # Exclude expected value offset
            
            # Rank by absolute magnitude of contribution
            ranked_indices = (-np.abs(shap_vals)).argsort()[:3]
            drivers = []
            for idx in ranked_indices:
                feat_name = self.features[idx]
                val = X.iloc[0, idx]
                impact = float(shap_vals[idx])
                direction = "increased delay risk" if impact > 0 else "reduced delay risk"
                
                drivers.append({
                    "feature": feat_name,
                    "description": FEATURE_DESCRIPTIONS.get(feat_name, f"Flight feature: {feat_name}"),
                    "feature_value": str(val),
                    "impact_log_odds": round(impact, 4),
                    "direction": direction,
                    "summary": f"{direction.capitalize()} ({impact:+.2f} log-odds): {FEATURE_DESCRIPTIONS.get(feat_name, feat_name)}"
                })
            return drivers
        except Exception as e:
            logger.debug(f"SHAP computation bypassed: {e}")
            return []

    def validate_and_prepare_input(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Validates payload schema, airline/airport vocabs, times, and dates."""
        warnings = []

        # 1. Required keys
        req_keys = ["airline_code", "origin", "dest", "flight_date", "sched_dep_time", "sched_arr_time"]
        for k in req_keys:
            if k not in payload or payload[k] is None:
                raise InputValidationError(f"Missing required input parameter: '{k}'.")

        airline = str(payload["airline_code"]).strip().upper()
        origin = str(payload["origin"]).strip().upper()
        dest = str(payload["dest"]).strip().upper()

        # 2. Vocabulary checks (clear 422 error on unknown)
        if airline not in self.known_airlines:
            raise InputValidationError(
                f"Unknown airline_code '{airline}'. Supported carriers: {sorted(list(self.known_airlines))}."
            )
        if origin not in self.known_airports:
            raise InputValidationError(
                f"Unknown origin airport '{origin}'. Airport must be one of {len(self.known_airports)} supported IATA codes."
            )
        if dest not in self.known_airports:
            raise InputValidationError(
                f"Unknown destination airport '{dest}'. Airport must be one of {len(self.known_airports)} supported IATA codes."
            )
        if origin == dest:
            raise InputValidationError(f"Origin and destination cannot be the same airport ('{origin}').")

        # 3. Time & Date parsing
        dep_time_int = self.parse_time_to_int(payload["sched_dep_time"])
        arr_time_int = self.parse_time_to_int(payload["sched_arr_time"])
        date_str = self.parse_date(payload["flight_date"])

        # Year warning
        year = int(date_str.split("-")[0])
        if year < 2019 or year > 2023:
            warnings.append(
                f"Flight date {date_str} falls outside the baseline historical training window (2019-2023); "
                f"underlying airspace congestion baselines may differ."
            )

        # 4. Elapsed and Distance derivation
        if "crs_elapsed_time" in payload and payload["crs_elapsed_time"] is not None and "distance" in payload and payload["distance"] is not None:
            elapsed = float(payload["crs_elapsed_time"])
            distance = float(payload["distance"])
            route_known = True
        elif "CRS_ELAPSED_TIME" in payload and payload["CRS_ELAPSED_TIME"] is not None and "DISTANCE" in payload and payload["DISTANCE"] is not None:
            elapsed = float(payload["CRS_ELAPSED_TIME"])
            distance = float(payload["DISTANCE"])
            route_known = True
        else:
            elapsed, distance, route_known, route_warn = self._derive_elapsed_and_distance(
                origin, dest, dep_time_int, arr_time_int
            )
            warnings.extend(route_warn)

        cleaned_row = {
            "FL_DATE": date_str,
            "AIRLINE_CODE": airline,
            "ORIGIN": origin,
            "DEST": dest,
            "CRS_DEP_TIME": dep_time_int,
            "CRS_ARR_TIME": arr_time_int,
            "CRS_ELAPSED_TIME": elapsed,
            "DISTANCE": distance,
            "_route_known": route_known
        }
        return cleaned_row, warnings

    def predict(
        self, payload: Dict[str, Any], include_shap: bool = True
    ) -> Dict[str, Any]:
        """
        Executes single-flight inference.
        Returns calibrated probability, risk decile, risk band, drivers, and disclaimers.
        """
        cleaned_row, warnings = self.validate_and_prepare_input(payload)
        route_known = cleaned_row.pop("_route_known")

        # Transform using frozen pipeline (accepts dict without DELAY)
        tree_df, _ = self.pipeline.transform(cleaned_row)
        X = tree_df[self.features]

        # Raw CatBoost probability
        p_raw = float(self.model.predict_proba(X)[0, 1])

        # Platt scaling probability calibration
        z = logit(np.clip(p_raw, 1e-12, 1.0 - 1e-12)).reshape(-1, 1)
        p_cal = float(self.calibrator["model"].predict_proba(z)[0, 1])

        # Decile and risk band
        decile, risk_band = self._determine_risk_tier(p_cal)

        # Optional SHAP drivers
        drivers = []
        if include_shap:
            drivers = self._compute_shap_drivers(X)

        return {
            "calibrated_probability": float(p_cal),
            "raw_probability": float(p_raw),
            "risk_decile": decile,
            "risk_band": risk_band,
            "reference_base_rates": {
                "train_delay_rate": self.base_rates["train"],
                "validation_delay_rate": self.base_rates["validation"],
                "test_delay_rate": self.base_rates["test"]
            },
            "route_known": route_known,
            "derived_features": {
                "distance_miles": cleaned_row["DISTANCE"],
                "scheduled_elapsed_minutes": cleaned_row["CRS_ELAPSED_TIME"]
            },
            "top_drivers": drivers,
            "warnings": warnings,
            "model_version": self.model_version,
            "disclaimer": DISCLAIMER_TEXT
        }

    def predict_batch(
        self, payloads: List[Dict[str, Any]], max_batch: int = 100, include_shap: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Executes batch inference up to max_batch items.
        Optimized for multi-row throughput.
        """
        if len(payloads) > max_batch:
            raise InputValidationError(f"Batch payload exceeds maximum allowed size of {max_batch} items.")
        if len(payloads) == 0:
            return []

        cleaned_rows = []
        warnings_list = []
        routes_known = []

        for p in payloads:
            c_row, w = self.validate_and_prepare_input(p)
            routes_known.append(c_row.pop("_route_known"))
            cleaned_rows.append(c_row)
            warnings_list.append(w)

        batch_df = pd.DataFrame(cleaned_rows)
        tree_batch, _ = self.pipeline.transform(batch_df)
        X_batch = tree_batch[self.features]

        p_raw_batch = self.model.predict_proba(X_batch)[:, 1]
        z_batch = logit(np.clip(p_raw_batch, 1e-12, 1.0 - 1e-12)).reshape(-1, 1)
        p_cal_batch = self.calibrator["model"].predict_proba(z_batch)[:, 1]

        results = []
        for i in range(len(payloads)):
            p_cal = float(p_cal_batch[i])
            p_raw = float(p_raw_batch[i])
            decile, risk_band = self._determine_risk_tier(p_cal)
            
            drivers = []
            if include_shap:
                drivers = self._compute_shap_drivers(X_batch.iloc[[i]])

            results.append({
                "calibrated_probability": float(p_cal),
                "raw_probability": float(p_raw),
                "risk_decile": decile,
                "risk_band": risk_band,
                "reference_base_rates": {
                    "train_delay_rate": self.base_rates["train"],
                    "validation_delay_rate": self.base_rates["validation"],
                    "test_delay_rate": self.base_rates["test"]
                },
                "route_known": routes_known[i],
                "derived_features": {
                    "distance_miles": cleaned_rows[i]["DISTANCE"],
                    "scheduled_elapsed_minutes": cleaned_rows[i]["CRS_ELAPSED_TIME"]
                },
                "top_drivers": drivers,
                "warnings": warnings_list[i],
                "model_version": self.model_version,
                "disclaimer": DISCLAIMER_TEXT
            })
        return results

# Convenience types and aliases
FlightInput = Dict[str, Any]
PredictionResult = Dict[str, Any]
FlightDelayPredictor = FlightPredictor
