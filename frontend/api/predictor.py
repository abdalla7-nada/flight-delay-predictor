"""
api/predictor.py
Zero-dependency, high-performance pure Python inference engine for Flight Delay Risk Predictor (Champion v2).
Engineered specifically for Vercel Serverless Functions with zero third-party ML dependencies.
Requires strictly standard library (math, json, datetime, pathlib).
Zero external ML dependencies (no numpy, scipy, pandas, or catboost).
"""

import os
import sys
import json
import math
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Import pure Python standalone CatBoost model
try:
    from api.champion_model import apply_catboost_model
except ImportError:
    try:
        from champion_model import apply_catboost_model
    except ImportError:
        # Fallback for relative imports in diverse serverless runtimes
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from champion_model import apply_catboost_model

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
    "prior_origin_hour": "Historical delay rate at departure airport for this scheduled hour",
    "dep_hour": "Scheduled departure hour (afternoon/evening flights accumulate higher delay risk)",
    "is_holiday_period": "High-volume holiday travel congestion period",
    "scheduled_speed": "Ratio of distance to scheduled duration (schedule buffer/compression)",
    "DISTANCE": "Flight distance between city pairs",
    "CRS_ELAPSED_TIME": "Scheduled gate-to-gate flight duration",
    "AIRLINE_CODE": "Airline carrier operational profile",
    "ORIGIN": "Origin airport congestion and airspace profile",
    "DEST": "Destination airport congestion and airspace profile",
    "route": "Directional airport-pair network route"
}

# Platt Scaling Logistic Calibrator frozen parameters (Champion v2)
PLATT_SLOPE = 0.933138807978378
PLATT_INTERCEPT = 0.09731147911152838

def haversine_distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great circle distance between two coordinates in statute miles."""
    r_miles = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r_miles * c

def logit(p: float) -> float:
    """Computes log-odds of probability clamped safely away from 0 and 1."""
    p_clamped = max(1e-12, min(1.0 - 1e-12, p))
    return math.log(p_clamped / (1.0 - p_clamped))


class InputValidationError(ValueError):
    """Exception raised when an inference payload violates schema or vocabulary rules (HTTP 422)."""
    pass


class ServerlessPredictor:
    """
    Lightweight, dependency-free predictor for Vercel Serverless Functions.
    Loads lookup dictionaries once into memory and scores flights in ~1ms.
    """
    def __init__(self, assets_dir: Optional[Union[str, Path]] = None):
        if assets_dir:
            self.assets_dir = Path(assets_dir)
        else:
            self.assets_dir = Path(__file__).resolve().parent / "assets"

        # 1. Load vocabulary & serving metadata
        meta_path = self.assets_dir / "serving_metadata.json"
        vocab_path = self.assets_dir / "vocab.json"
        route_path = self.assets_dir / "route_lookup.json"
        lookups_path = self.assets_dir / "model_lookups.json"
        coords_path = self.assets_dir / "airport_coords.json"
        holidays_path = self.assets_dir / "holiday_dates.json"

        if not (meta_path.exists() and vocab_path.exists() and lookups_path.exists()):
            raise FileNotFoundError(f"Required JSON assets missing in {self.assets_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.serving_metadata = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(route_path, "r", encoding="utf-8") as f:
            self.route_lookup = json.load(f)
        with open(lookups_path, "r", encoding="utf-8") as f:
            self.model_lookups = json.load(f)

        self.airport_coords = {}
        if coords_path.exists():
            with open(coords_path, "r", encoding="utf-8") as f:
                self.airport_coords = json.load(f)

        self.holiday_dates = set()
        if holidays_path.exists():
            with open(holidays_path, "r", encoding="utf-8") as f:
                self.holiday_dates = set(json.load(f))

        # Unpack lookups
        self.known_airlines = set(self.vocab.get("airlines", []))
        self.known_airports = set(self.vocab.get("airports", []))
        self.decile_edges = self.serving_metadata.get("decile_edges_validation", [])
        self.base_rates = self.serving_metadata.get("base_rates", {})
        self.model_version = self.serving_metadata.get("model_version", "v2.0")

        self.global_delay_mean = float(self.model_lookups.get("global_delay_mean", 0.1762))
        self.frequent_routes = set(self.model_lookups.get("frequent_routes", []))
        self.airline_priors = self.model_lookups.get("airline_priors", {})
        self.origin_priors = self.model_lookups.get("origin_priors", {})
        self.dest_priors = self.model_lookups.get("dest_priors", {})
        self.route_priors = self.model_lookups.get("route_priors", {})
        self.origin_hour_priors = self.model_lookups.get("origin_hour_priors", {})
        self.origin_frequencies = self.model_lookups.get("origin_frequencies", {})
        self.dest_frequencies = self.model_lookups.get("dest_frequencies", {})
        self.route_frequencies = self.model_lookups.get("route_frequencies", {})

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
        """Maps calibrated probability to decile (1-10) and categorical risk band."""
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
        or computes geodesic distance via airport_coords fallback.
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
            dist_est = 1000.0
            if origin in self.airport_coords and dest in self.airport_coords:
                o_c = self.airport_coords[origin]
                d_c = self.airport_coords[dest]
                dist_est = round(haversine_distance_miles(o_c["lat"], o_c["lon"], d_c["lat"], d_c["lon"]), 1)

            elapsed_est = round(dist_est / 7.5 + 30.0, 1)
            distance = dist_est
            elapsed = elapsed_est
            warnings.append(
                f"Route {route_key} is not in the historical training route table; "
                f"distance estimated as {distance:.1f} mi and elapsed time as {elapsed:.1f} min."
            )

        if elapsed <= 15.0 or elapsed > 900.0:
            warnings.append(f"Derived elapsed duration ({elapsed:.1f} min) is atypical for a domestic flight.")

        speed = distance / (elapsed / 60.0) if elapsed > 0 else 0
        if speed > 700.0 or speed < 50.0:
            warnings.append(f"Implausible scheduled speed ({speed:.1f} mph) detected.")

        return elapsed, distance, route_known, warnings

    def validate_and_prepare_input(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Validates payload schema, airline/airport vocabs, times, and dates."""
        warnings = []
        req_keys = ["airline_code", "origin", "dest", "flight_date", "sched_dep_time", "sched_arr_time"]
        for k in req_keys:
            if k not in payload or payload[k] is None:
                raise InputValidationError(f"Missing required input parameter: '{k}'.")

        airline = str(payload["airline_code"]).strip().upper()
        origin = str(payload["origin"]).strip().upper()
        dest = str(payload["dest"]).strip().upper()

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
            raise InputValidationError("Origin and destination airports cannot be identical.")

        dep_time_int = self.parse_time_to_int(payload["sched_dep_time"])
        arr_time_int = self.parse_time_to_int(payload["sched_arr_time"])
        date_str = self.parse_date(payload["flight_date"])

        year = int(date_str.split("-")[0])
        if year < 2019 or year > 2023:
            warnings.append(
                f"Flight date {date_str} falls outside the baseline historical training window (2019-2023); "
                f"underlying airspace congestion baselines may differ."
            )

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

    def _extract_model_features(self, cleaned_row: Dict[str, Any]) -> Tuple[List[float], List[str], Dict[str, Any]]:
        """Transforms validated row into exact float_features (28) and cat_features (5) required by CatBoost."""
        date_str = cleaned_row["FL_DATE"]
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        month = dt.month
        day_of_week = dt.weekday()  # 0=Monday .. 6=Sunday
        is_weekend = 1 if day_of_week >= 5 else 0
        is_holiday_period = 1 if date_str in self.holiday_dates else 0

        crs_dep = cleaned_row["CRS_DEP_TIME"]
        crs_arr = cleaned_row["CRS_ARR_TIME"]
        dep_hour = crs_dep // 100
        arr_hour = crs_arr // 100

        # Cyclical sines & cosines
        dep_hour_sin = math.sin(2.0 * math.pi * dep_hour / 24.0)
        dep_hour_cos = math.cos(2.0 * math.pi * dep_hour / 24.0)
        arr_hour_sin = math.sin(2.0 * math.pi * arr_hour / 24.0)
        arr_hour_cos = math.cos(2.0 * math.pi * arr_hour / 24.0)
        month_sin = math.sin(2.0 * math.pi * (month - 1) / 12.0)
        month_cos = math.cos(2.0 * math.pi * (month - 1) / 12.0)
        day_of_week_sin = math.sin(2.0 * math.pi * day_of_week / 7.0)
        day_of_week_cos = math.cos(2.0 * math.pi * day_of_week / 7.0)

        # Part of day
        if dep_hour <= 5:
            part_of_day = "night"
        elif dep_hour <= 11:
            part_of_day = "morning"
        elif dep_hour <= 17:
            part_of_day = "afternoon"
        else:
            part_of_day = "evening"

        airline = cleaned_row["AIRLINE_CODE"]
        origin = cleaned_row["ORIGIN"]
        dest = cleaned_row["DEST"]
        raw_route = f"{origin}_{dest}"
        mapped_route = raw_route if raw_route in self.frequent_routes else "OTHER_RARE_ROUTE"

        distance = float(cleaned_row["DISTANCE"])
        elapsed = float(cleaned_row["CRS_ELAPSED_TIME"])
        guard_elapsed = elapsed if elapsed > 0 else (distance / 7.5 + 30.0)
        log_distance = math.log1p(distance)
        scheduled_speed = distance / (guard_elapsed / 60.0)

        origin_hour_key = f"{origin}_{dep_hour}"
        prior_airline = float(self.airline_priors.get(airline, self.global_delay_mean))
        prior_origin = float(self.origin_priors.get(origin, self.global_delay_mean))
        prior_dest = float(self.dest_priors.get(dest, self.global_delay_mean))
        prior_route = float(self.route_priors.get(mapped_route, self.global_delay_mean))
        prior_origin_hour = float(self.origin_hour_priors.get(origin_hour_key, self.global_delay_mean))

        origin_freq = float(self.origin_frequencies.get(origin, 0))
        dest_freq = float(self.dest_frequencies.get(dest, 0))
        route_freq = float(self.route_frequencies.get(mapped_route, 0))

        # 28 float features in exact training order
        float_features = [
            float(month),
            float(day_of_week),
            float(is_weekend),
            float(is_holiday_period),
            float(dep_hour),
            float(arr_hour),
            float(dep_hour_sin),
            float(dep_hour_cos),
            float(arr_hour_sin),
            float(arr_hour_cos),
            float(month_sin),
            float(month_cos),
            float(day_of_week_sin),
            float(day_of_week_cos),
            float(crs_dep),
            float(crs_arr),
            float(guard_elapsed),
            float(distance),
            float(log_distance),
            float(scheduled_speed),
            float(prior_airline),
            float(prior_origin),
            float(prior_dest),
            float(prior_route),
            float(prior_origin_hour),
            float(origin_freq),
            float(dest_freq),
            float(route_freq),
        ]

        # 5 categorical features in exact training order
        cat_features = [
            str(part_of_day),
            str(airline),
            str(origin),
            str(dest),
            str(mapped_route),
        ]

        feature_context = {
            "prior_route": prior_route,
            "prior_origin": prior_origin,
            "prior_dest": prior_dest,
            "prior_airline": prior_airline,
            "prior_origin_hour": prior_origin_hour,
            "dep_hour": dep_hour,
            "is_holiday_period": is_holiday_period,
            "part_of_day": part_of_day,
            "airline": airline,
            "origin": origin,
            "dest": dest,
            "mapped_route": mapped_route,
            "scheduled_speed": scheduled_speed,
        }

        return float_features, cat_features, feature_context

    def _compute_risk_drivers(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Computes top-3 risk attribution drivers using log-odds contrast against
        the baseline historical training delay rate (0.1762).
        """
        baseline_logit = logit(self.global_delay_mean)
        candidates = []

        # 1. Route prior
        p_route = ctx["prior_route"]
        route_delta = logit(p_route) - baseline_logit
        candidates.append((
            abs(route_delta),
            {
                "feature": "prior_route",
                "description": FEATURE_DESCRIPTIONS["prior_route"],
                "feature_value": f"{p_route * 100:.1f}%",
                "impact_log_odds": round(route_delta, 4),
                "direction": "increased delay risk" if route_delta > 0 else "reduced delay risk",
                "summary": f"{'Increased' if route_delta > 0 else 'Reduced'} delay risk ({route_delta:+.2f} log-odds): Historical route delay rate is {p_route * 100:.1f}% vs baseline {self.global_delay_mean * 100:.1f}%"
            }
        ))

        # 2. Origin airport prior
        p_orig = ctx["prior_origin"]
        orig_delta = logit(p_orig) - baseline_logit
        candidates.append((
            abs(orig_delta),
            {
                "feature": "prior_origin",
                "description": FEATURE_DESCRIPTIONS["prior_origin"],
                "feature_value": f"{p_orig * 100:.1f}%",
                "impact_log_odds": round(orig_delta, 4),
                "direction": "increased delay risk" if orig_delta > 0 else "reduced delay risk",
                "summary": f"{'Increased' if orig_delta > 0 else 'Reduced'} delay risk ({orig_delta:+.2f} log-odds): Origin {ctx['origin']} historical delay rate is {p_orig * 100:.1f}%"
            }
        ))

        # 3. Destination airport prior
        p_dest = ctx["prior_dest"]
        dest_delta = logit(p_dest) - baseline_logit
        candidates.append((
            abs(dest_delta),
            {
                "feature": "prior_dest",
                "description": FEATURE_DESCRIPTIONS["prior_dest"],
                "feature_value": f"{p_dest * 100:.1f}%",
                "impact_log_odds": round(dest_delta, 4),
                "direction": "increased delay risk" if dest_delta > 0 else "reduced delay risk",
                "summary": f"{'Increased' if dest_delta > 0 else 'Reduced'} delay risk ({dest_delta:+.2f} log-odds): Destination {ctx['dest']} historical delay rate is {p_dest * 100:.1f}%"
            }
        ))

        # 4. Airline carrier prior
        p_air = ctx["prior_airline"]
        air_delta = logit(p_air) - baseline_logit
        candidates.append((
            abs(air_delta),
            {
                "feature": "prior_airline",
                "description": FEATURE_DESCRIPTIONS["prior_airline"],
                "feature_value": f"{p_air * 100:.1f}%",
                "impact_log_odds": round(air_delta, 4),
                "direction": "increased delay risk" if air_delta > 0 else "reduced delay risk",
                "summary": f"{'Increased' if air_delta > 0 else 'Reduced'} delay risk ({air_delta:+.2f} log-odds): Carrier {ctx['airline']} historical delay rate is {p_air * 100:.1f}%"
            }
        ))

        # 5. Departure hour / Time-of-day prior
        p_hour = ctx["prior_origin_hour"]
        hour_delta = logit(p_hour) - baseline_logit
        candidates.append((
            abs(hour_delta),
            {
                "feature": "prior_origin_hour",
                "description": FEATURE_DESCRIPTIONS["prior_origin_hour"],
                "feature_value": f"{p_hour * 100:.1f}%",
                "impact_log_odds": round(hour_delta, 4),
                "direction": "increased delay risk" if hour_delta > 0 else "reduced delay risk",
                "summary": f"{'Increased' if hour_delta > 0 else 'Reduced'} delay risk ({hour_delta:+.2f} log-odds): {ctx['origin']} departure window ({ctx['dep_hour']:02d}:00) delay rate is {p_hour * 100:.1f}%"
            }
        ))

        # 6. Holiday period impact
        if ctx["is_holiday_period"]:
            holiday_delta = 0.285
            candidates.append((
                abs(holiday_delta),
                {
                    "feature": "is_holiday_period",
                    "description": FEATURE_DESCRIPTIONS["is_holiday_period"],
                    "feature_value": "Holiday Period",
                    "impact_log_odds": round(holiday_delta, 4),
                    "direction": "increased delay risk",
                    "summary": f"Increased delay risk (+{holiday_delta:.2f} log-odds): High-volume holiday travel congestion period"
                }
            ))

        # Sort by absolute log-odds magnitude and return top-3
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:3]]

    def predict(self, payload: Dict[str, Any], include_shap: bool = True) -> Dict[str, Any]:
        """
        Executes single flight inference with zero ML dependencies.
        Returns calibrated probability, risk decile, risk band, drivers, and disclaimers.
        """
        cleaned_row, warnings = self.validate_and_prepare_input(payload)
        route_known = cleaned_row.pop("_route_known")

        float_features, cat_features, ctx = self._extract_model_features(cleaned_row)

        # 1. Standalone CatBoost model evaluation
        raw_score = apply_catboost_model(float_features, cat_features)
        p_raw = 1.0 / (1.0 + math.exp(-raw_score))

        # 2. Platt scaling logistic calibration
        # Formula: 1.0 / (1.0 + exp(-(PLATT_SLOPE * raw_score + PLATT_INTERCEPT)))
        p_cal = 1.0 / (1.0 + math.exp(-(PLATT_SLOPE * raw_score + PLATT_INTERCEPT)))

        # 3. Decile & Risk Band
        decile, risk_band = self._determine_risk_tier(p_cal)

        # 4. Top drivers
        drivers = self._compute_risk_drivers(ctx) if include_shap else []

        return {
            "calibrated_probability": float(p_cal),
            "raw_probability": float(p_raw),
            "risk_decile": decile,
            "risk_band": risk_band,
            "reference_base_rates": {
                "train_delay_rate": self.base_rates.get("train", 0.1762),
                "validation_delay_rate": self.base_rates.get("validation", 0.2032),
                "test_delay_rate": self.base_rates.get("test", 0.2224)
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
        """
        if len(payloads) > max_batch:
            raise InputValidationError(f"Batch payload exceeds maximum allowed size of {max_batch} items.")
        if len(payloads) == 0:
            return []

        results = []
        for p in payloads:
            results.append(self.predict(p, include_shap=include_shap))
        return results
