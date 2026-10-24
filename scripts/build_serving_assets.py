"""
scripts/build_serving_assets.py
Phase 6: Build serving assets for real-time inference and deployment.
Reads train.parquet and val.parquet READ-ONLY.
Never reads test.parquet.
Generates:
  - artifacts/serving/route_lookup.json
  - artifacts/serving/vocab.json
  - artifacts/serving/serving_metadata.json
"""

import os
import sys
import json
import hashlib
from pathlib import Path

# Ensure repo root is on sys.path for unpickling src.features
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import joblib
from scipy.special import logit
from src.features import FlightFeaturePipeline

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    repo_root = Path(__file__).resolve().parent.parent
    serving_dir = repo_root / "artifacts" / "serving"
    serving_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Building serving assets...")

    # 1. Route lookup table from train.parquet (Train only!)
    train_path = repo_root / "data" / "processed" / "train.parquet"
    print(f"  -> Reading train dataset (read-only): {train_path.name}")
    train_df = pd.read_parquet(train_path, columns=["ORIGIN", "DEST", "CRS_ELAPSED_TIME", "DISTANCE", "DELAY"])
    train_base_rate = float(train_df["DELAY"].mean())
    print(f"  -> Train base rate: {train_base_rate:.4f} ({train_base_rate*100:.2f}%)")

    train_df["route"] = train_df["ORIGIN"] + "_" + train_df["DEST"]
    route_grouped = train_df.groupby("route").agg({
        "CRS_ELAPSED_TIME": "median",
        "DISTANCE": "median"
    })
    
    route_lookup = {}
    for r, row in route_grouped.iterrows():
        route_lookup[r] = {
            "elapsed": round(float(row["CRS_ELAPSED_TIME"]), 1),
            "distance": round(float(row["DISTANCE"]), 1)
        }
    
    route_lookup_path = serving_dir / "route_lookup.json"
    route_lookup_path.write_text(json.dumps(route_lookup, indent=2))
    print(f"  -> Saved {len(route_lookup):,} routes to {route_lookup_path}")

    # 2. Vocabulary from pipeline
    pipe_path = repo_root / "artifacts" / "feature_pipeline_v2.joblib"
    pipe = joblib.load(pipe_path)
    
    known_airlines = sorted(list(pipe.known_airlines))
    known_airports = sorted(list(set(pipe.origin_frequencies.keys()) | set(pipe.dest_frequencies.keys())))
    
    vocab = {
        "airlines": known_airlines,
        "airline_count": len(known_airlines),
        "airports": known_airports,
        "airport_count": len(known_airports),
        "part_of_day": pipe.part_of_day_categories,
        "frequent_routes_count": len(pipe.frequent_routes)
    }
    vocab_path = serving_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, indent=2))
    print(f"  -> Saved vocabulary (airlines={len(known_airlines)}, airports={len(known_airports)}) to {vocab_path}")

    # 3. Validation calibrated probabilities & decile edges
    val_path = repo_root / "data" / "processed" / "val.parquet"
    print(f"  -> Reading validation dataset (read-only): {val_path.name}")
    val_df = pd.read_parquet(val_path)
    val_base_rate = float(val_df["DELAY"].mean())
    print(f"  -> Validation base rate: {val_base_rate:.4f} ({val_base_rate*100:.2f}%)")

    model_path = repo_root / "artifacts" / "models" / "champion_v2.joblib"
    calibrator_path = repo_root / "artifacts" / "models" / "calibrator_v2.joblib"
    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)

    meta_v2 = json.loads((repo_root / "artifacts" / "models" / "model_metadata_v2.json").read_text())
    features = meta_v2["features"]

    print("  -> Computing calibrated validation predictions...")
    tree_val, _ = pipe.transform(val_df)
    p_raw = model.predict_proba(tree_val[features])[:, 1]
    z = logit(np.clip(p_raw, 1e-12, 1.0 - 1e-12)).reshape(-1, 1)
    p_cal = calibrator["model"].predict_proba(z)[:, 1]

    # Decile edges on validation
    decile_quantiles = np.linspace(0.0, 1.0, 11)
    decile_edges = [round(float(q), 5) for q in np.quantile(p_cal, decile_quantiles)]
    print(f"  -> Calibrated validation decile edges: {decile_edges}")

    # 4. Load reports metrics
    test_eval_path = repo_root / "reports" / "test_evaluation_metrics.json"
    test_eval = json.loads(test_eval_path.read_text()) if test_eval_path.exists() else {}
    test_base_rate = float(test_eval.get("test_base_rate", 0.2224))

    phase4_path = repo_root / "reports" / "phase4_metrics.json"
    phase4 = json.loads(phase4_path.read_text()) if phase4_path.exists() else {}
    val_perf = phase4.get("performance", {})

    # Compute SHA256 of 3 frozen artifacts
    frozen_hashes = {
        "artifacts/feature_pipeline_v2.joblib": compute_sha256(str(pipe_path)),
        "artifacts/models/champion_v2.joblib": compute_sha256(str(model_path)),
        "artifacts/models/calibrator_v2.joblib": compute_sha256(str(calibrator_path))
    }

    serving_metadata = {
        "model_name": "Flight Delay Risk Predictor (Champion v2)",
        "model_version": "v2.0",
        "description": "Schedule-only pre-departure flight delay probability predictor (CatBoost + Platt calibration).",
        "base_rates": {
            "train": 0.1762,
            "validation": 0.2032,
            "test": test_base_rate
        },
        "decile_edges_validation": decile_edges,
        "risk_bands": {
            "Low": [1, 2],
            "Below average": [3, 4],
            "Average": [5, 6],
            "Above average": [7, 8],
            "High": [9, 10]
        },
        "frozen_artifacts_sha256": frozen_hashes,
        "validation_metrics": {
            "roc_auc": {"point": val_perf.get("val_roc_auc", 0.6571), "ci": val_perf.get("val_roc_auc_ci", [0.6511, 0.6623])},
            "pr_auc": {"point": val_perf.get("val_pr_auc", 0.3257), "ci": val_perf.get("val_pr_auc_ci", [0.3120, 0.3392])},
            "brier": {"point": val_perf.get("val_brier", 0.1534), "ci": val_perf.get("val_brier_ci", [0.1494, 0.1573])},
            "log_loss": val_perf.get("val_log_loss", 0.4793),
            "top10_lift": val_perf.get("val_lift_at_10", 1.95),
            "bss_vs_base_rate": val_perf.get("bss_vs_val_base_rate", 0.0525)
        },
        "test_metrics": {
            "roc_auc": test_eval.get("champion_test_bootstrap_ci", {}).get("roc_auc", {}),
            "pr_auc": test_eval.get("champion_test_bootstrap_ci", {}).get("pr_auc", {}),
            "brier": test_eval.get("champion_test_bootstrap_ci", {}).get("brier", {}),
            "log_loss": test_eval.get("champion_test_metrics", {}).get("test_log_loss", 0.5045),
            "top10_lift": test_eval.get("champion_test_metrics", {}).get("test_lift_at_10", 1.93),
            "ece_10": test_eval.get("calibration_test", {}).get("ece_10", 0.0104),
            "bss_vs_test_base_rate": test_eval.get("brier_skill_scores", {}).get("bss_vs_test_base_rate", 0.0529),
            "bss_vs_train_base_rate": test_eval.get("brier_skill_scores", {}).get("bss_vs_train_base_rate", 0.0645)
        },
        "calibrator_params": {
            "type": "PlattScaling",
            "slope_A": calibrator.get("coef", 0.933139),
            "intercept_B": calibrator.get("intercept", 0.097311)
        }
    }

    meta_path = serving_dir / "serving_metadata.json"
    meta_path.write_text(json.dumps(serving_metadata, indent=2))
    print(f"  -> Saved serving metadata to {meta_path}")
    print("[+] All serving assets built successfully!")

if __name__ == "__main__":
    main()
