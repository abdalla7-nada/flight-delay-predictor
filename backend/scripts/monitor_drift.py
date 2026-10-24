#!/usr/bin/env python3
"""
scripts/monitor_drift.py
Production CLI tool for monitoring inference input and prediction drift.
Calculates Population Stability Index (PSI) against baseline validation deciles
based on docs/MONITORING_AND_MAINTENANCE.md.

Usage:
    python scripts/monitor_drift.py [--log-file PATH] [--json] [--generate-synthetic]
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor flight delay prediction request drift and calculate PSI."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="api/logs/requests.jsonl",
        help="Path to request JSONL log file (default: api/logs/requests.jsonl)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="artifacts/serving/serving_metadata.json",
        help="Path to serving metadata JSON (default: artifacts/serving/serving_metadata.json)"
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic inference requests to test drift calculation."
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=1000,
        help="Number of synthetic requests to generate (default: 1000)"
    )
    parser.add_argument(
        "--synthetic-scenario",
        choices=["stable", "moderate", "critical"],
        default="stable",
        help="Synthetic scenario type: 'stable' (PSI < 0.1), 'moderate' (0.1 <= PSI < 0.25), or 'critical' (PSI >= 0.25)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit results in machine-readable JSON format."
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with non-zero code if critical drift (PSI >= 0.25) is detected."
    )
    return parser.parse_args()


def load_serving_metadata(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Serving metadata not found at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_predictions_from_log(log_path: Path) -> Tuple[List[float], Dict[str, Any]]:
    """Extracts calibrated probabilities and runtime telemetry from JSONL log."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return [], {"total_requests": 0, "route_misses": 0, "durations_ms": []}

    probabilities = []
    durations = []
    total_events = 0
    route_misses = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_events += 1
            if "duration_ms" in entry:
                durations.append(entry["duration_ms"])

            output = entry.get("output")
            if not output:
                continue

            # Check single prediction vs batch output
            if isinstance(output, dict) and "calibrated_probability" in output:
                probabilities.append(float(output["calibrated_probability"]))
                if output.get("route_known") is False:
                    route_misses += 1
            elif isinstance(output, list):
                for item in output:
                    if isinstance(item, dict) and "calibrated_probability" in item:
                        probabilities.append(float(item["calibrated_probability"]))
                        if item.get("route_known") is False:
                            route_misses += 1

    telemetry = {
        "total_requests": total_events,
        "total_predictions": len(probabilities),
        "route_misses": route_misses,
        "durations_ms": durations
    }
    return probabilities, telemetry


def generate_synthetic_cohort(scenario: str, n: int, decile_edges: List[float]) -> Tuple[List[float], Dict[str, Any]]:
    """Generates synthetic predictions according to pre-defined drift regimes."""
    np.random.seed(42)
    probs = []

    if scenario == "stable":
        # Sample uniformly within baseline decile bins
        for i in range(10):
            low = 0.0 if i == 0 else decile_edges[i]
            high = 1.0 if i == 9 else decile_edges[i + 1]
            probs.extend(np.random.uniform(low, high, n // 10))
    elif scenario == "moderate":
        # Slight right skew (mild travel surge)
        probs = list(np.random.beta(2.5, 9.0, n))
    else:  # critical
        # Severe right skew (major weather disruption / congestion)
        probs = list(np.random.beta(4.0, 5.0, n))

    telemetry = {
        "total_requests": n,
        "total_predictions": len(probs),
        "route_misses": int(n * 0.015),
        "durations_ms": list(np.random.normal(5.5, 1.2, min(n, 100)))
    }
    return probs, telemetry


def calculate_psi(
    probabilities: List[float],
    decile_edges: List[float],
    eps: float = 1e-4
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes Population Stability Index (PSI) across 10 deciles:
    PSI = sum((Actual_i - Expected_i) * ln(Actual_i / Expected_i))
    """
    n = len(probabilities)
    if n == 0:
        return 0.0, []

    # Bin counts across the 10 deciles defined by 11 edges
    counts, _ = np.histogram(probabilities, bins=decile_edges)

    # Accumulate probabilities outside the interior edges
    underflow = sum(1 for p in probabilities if p < decile_edges[0])
    overflow = sum(1 for p in probabilities if p > decile_edges[-1])
    counts[0] += underflow
    counts[-1] += overflow

    actual_proportions = counts / n
    expected_proportions = np.array([0.10] * 10)

    # Smooth zero bins to avoid division by zero
    smoothed_actual = np.clip(actual_proportions, eps, 1.0)
    smoothed_actual = smoothed_actual / smoothed_actual.sum()

    per_decile_breakdown = []
    total_psi = 0.0

    for i in range(10):
        low = decile_edges[i]
        high = decile_edges[i + 1]
        act_p = float(actual_proportions[i])
        act_smooth = float(smoothed_actual[i])
        exp_p = float(expected_proportions[i])

        bin_psi = (act_smooth - exp_p) * math.log(act_smooth / exp_p)
        total_psi += bin_psi

        per_decile_breakdown.append({
            "decile": i + 1,
            "range": f"[{low:.4f}, {high:.4f}]",
            "count": int(counts[i]),
            "actual_pct": round(act_p * 100.0, 2),
            "expected_pct": round(exp_p * 100.0, 2),
            "delta_pct": round((act_p - exp_p) * 100.0, 2),
            "psi_contribution": round(bin_psi, 5)
        })

    return float(total_psi), per_decile_breakdown


def get_drift_status(psi: float) -> Tuple[str, str, str]:
    """Returns (status_label, severity_level, action_recommendation)."""
    if psi < 0.10:
        return (
            "HEALTHY / STABLE",
            "LOW",
            "Inference distribution matches baseline. No operational intervention needed."
        )
    elif psi < 0.25:
        return (
            "MODERATE DRIFT (WARNING)",
            "MEDIUM",
            "Heightened surveillance. Inspect carrier/route volume mix. Prepare for intercept recalibration."
        )
    else:
        return (
            "SIGNIFICANT DRIFT (CRITICAL)",
            "CRITICAL",
            "Trigger Protocol A (Platt intercept recalibration) or initiate feature priority audit."
        )


def main():
    args = parse_arguments()
    repo_root = Path(__file__).resolve().parent.parent

    meta_path = repo_root / args.metadata
    log_path = repo_root / args.log_file

    try:
        metadata = load_serving_metadata(meta_path)
    except Exception as e:
        print(f"[ERROR] Failed to load metadata: {e}", file=sys.stderr)
        sys.exit(2)

    decile_edges = metadata.get("decile_edges_validation", [])
    if len(decile_edges) != 11:
        print(f"[ERROR] Expected 11 decile edges, found {len(decile_edges)}", file=sys.stderr)
        sys.exit(2)

    base_rates = metadata.get("base_rates", {})
    val_base_rate = base_rates.get("validation", 0.2032)
    train_base_rate = base_rates.get("train", 0.1762)

    # 1. Fetch probabilities (either from real log or synthetic simulation)
    if args.generate_synthetic:
        probs, telemetry = generate_synthetic_cohort(
            scenario=args.synthetic_scenario,
            n=args.synthetic_samples,
            decile_edges=decile_edges
        )
        data_source = f"Synthetic Simulation ({args.synthetic_scenario.upper()}, N={args.synthetic_samples})"
    else:
        probs, telemetry = extract_predictions_from_log(log_path)
        data_source = f"Request Log ({log_path})"

    if len(probs) == 0:
        if args.json_output:
            print(json.dumps({
                "status": "EMPTY",
                "message": "No prediction records found in log file.",
                "log_file": str(log_path)
            }))
        else:
            print("=" * 76)
            print("  FLIGHT DELAY RISK PREDICTOR: DRIFT MONITORING REPORT")
            print("=" * 76)
            print(f"  Source: {data_source}")
            print(f"  [!] No inference records found in: {log_path}")
            print("      To test with synthetic requests, run with: --generate-synthetic")
            print("=" * 76)
        sys.exit(0)

    # 2. Compute Drift & Telemetry Metrics
    psi_score, decile_table = calculate_psi(probs, decile_edges)
    status_label, severity, action = get_drift_status(psi_score)

    mean_predicted_prob = float(np.mean(probs))
    calib_ratio = mean_predicted_prob / val_base_rate
    miss_rate_pct = (telemetry["route_misses"] / len(probs)) * 100.0 if probs else 0.0

    durations = telemetry.get("durations_ms", [])
    p50_latency = float(np.percentile(durations, 50)) if durations else 0.0
    p95_latency = float(np.percentile(durations, 95)) if durations else 0.0

    report_payload = {
        "status": status_label,
        "severity": severity,
        "psi": round(psi_score, 4),
        "sample_size": len(probs),
        "mean_predicted_probability": round(mean_predicted_prob, 4),
        "validation_base_rate": val_base_rate,
        "train_base_rate": train_base_rate,
        "calibration_ratio_vs_val": round(calib_ratio, 3),
        "route_miss_rate_pct": round(miss_rate_pct, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "action_required": action,
        "decile_breakdown": decile_table
    }

    # 3. Output Display
    if args.json_output:
        print(json.dumps(report_payload, indent=2))
    else:
        print("\n" + "=" * 76)
        print("  FLIGHT DELAY RISK PREDICTOR: PRODUCTION DRIFT REPORT")
        print("=" * 76)
        print(f"  Data Source        : {data_source}")
        print(f"  Sample Cohort Size : {len(probs):,} predictions")
        print(f"  Mean Delay Risk    : {mean_predicted_prob * 100:.2f}% (Baseline Val: {val_base_rate * 100:.2f}%)")
        print(f"  Calibration Ratio  : {calib_ratio:.3f}x")
        print(f"  Route Miss Rate    : {miss_rate_pct:.2f}% (Unseen route geodesic fallbacks)")
        if durations:
            print(f"  Latency Telemetry  : P50={p50_latency:.1f}ms | P95={p95_latency:.1f}ms")
        print("-" * 76)
        print(f"  POPULATION STABILITY INDEX (PSI): {psi_score:.4f}")
        print(f"  STATUS   : {status_label}")
        print(f"  SEVERITY : {severity}")
        print(f"  ACTION   : {action}")
        print("-" * 76)
        print("  Decile Distribution Breakdown vs Baseline (Expected 10% per bin):")
        print(f"  {'Decile':<8} {'Prob Range':<20} {'Count':<8} {'Actual %':<10} {'Exp %':<8} {'Delta %':<10} {'PSI Contrib':<10}")
        print("  " + "-" * 72)
        for row in decile_table:
            delta_str = f"{row['delta_pct']:+.2f}%"
            print(f"  {row['decile']:<8} {row['range']:<20} {row['count']:<8} {row['actual_pct']:<10.2f} {row['expected_pct']:<8.2f} {delta_str:<10} {row['psi_contribution']:<10.5f}")
        print("=" * 76 + "\n")

    if args.fail_on_drift and psi_score >= 0.25:
        sys.exit(1)


if __name__ == "__main__":
    main()
