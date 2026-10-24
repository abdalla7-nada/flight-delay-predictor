"""
scripts/evaluate_serving_approximation.py
Phase 6 Part 3: Evaluate approximation error of the serving inputs.
Validation data only (val.parquet READ-ONLY). Never loads test.parquet.
1. Samples 5,000 random validation rows (seed 42).
2. Compares probability computed with TRUE scheduled elapsed/distance vs route-lookup values.
3. Computes mean & max absolute difference, Spearman correlation, and share of rows whose risk decile changes.
4. Runs 200 validation rows through FlightPredictor vs batch pipeline and verifies max absolute difference < 1e-6.
5. Saves results to reports/serving_approximation_metrics.json.
"""

import os
import sys
import json
from pathlib import Path

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from scipy.special import logit
from scipy.stats import spearmanr

from src.features import FlightFeaturePipeline
from src.serving.predictor import FlightPredictor

def main():
    print("[*] Evaluating serving approximation error on validation set (seed=42)...")
    val_path = PROJECT_ROOT / "data" / "processed" / "val.parquet"
    assert val_path.exists(), f"Validation file {val_path} not found!"
    
    val_df = pd.read_parquet(val_path)
    print(f"  -> Total validation flights available: {len(val_df):,}")

    # Initialize predictor
    predictor = FlightPredictor(repo_root=PROJECT_ROOT)
    route_lookup = predictor.route_lookup
    pipe = predictor.pipeline
    model = predictor.model
    calibrator = predictor.calibrator
    features = predictor.features

    # 1. Sample 5,000 random validation rows with seed 42
    n_sample = 5000
    sample_df = val_df.sample(n=n_sample, random_state=42).copy().reset_index(drop=True)
    print(f"  -> Sampled {n_sample} validation flights (seed 42).")

    # 2. Probability with TRUE scheduled elapsed / distance
    print("  -> Computing probabilities with TRUE scheduled elapsed and distance...")
    tree_true, _ = pipe.transform(sample_df)
    p_raw_true = model.predict_proba(tree_true[features])[:, 1]
    z_true = logit(np.clip(p_raw_true, 1e-12, 1.0 - 1e-12)).reshape(-1, 1)
    p_cal_true = calibrator["model"].predict_proba(z_true)[:, 1]

    # 3. Probability with ROUTE-LOOKUP elapsed / distance
    print("  -> Computing probabilities with ROUTE-LOOKUP median elapsed and distance...")
    sample_approx_df = sample_df.copy()
    
    approx_elapsed = []
    approx_distance = []
    unseen_routes_list = []
    
    for i, row in sample_approx_df.iterrows():
        el, dist, is_known, _ = predictor._derive_elapsed_and_distance(
            row["ORIGIN"], row["DEST"], int(row["CRS_DEP_TIME"]), int(row["CRS_ARR_TIME"])
        )
        approx_elapsed.append(el)
        approx_distance.append(dist)
        if not is_known:
            unseen_routes_list.append(f"{row['ORIGIN']}_{row['DEST']}")

    sample_approx_df["CRS_ELAPSED_TIME"] = approx_elapsed
    sample_approx_df["DISTANCE"] = approx_distance
    
    unseen_routes_unique = sorted(list(set(unseen_routes_list)))
    print(f"  -> Unseen route flights in sample: {len(unseen_routes_list)} ({len(unseen_routes_list)/n_sample*100:.2f}%)")
    print(f"  -> Distinct unseen routes in sample: {len(unseen_routes_unique)}: {unseen_routes_unique}")

    tree_approx, _ = pipe.transform(sample_approx_df)
    p_raw_approx = model.predict_proba(tree_approx[features])[:, 1]
    z_approx = logit(np.clip(p_raw_approx, 1e-12, 1.0 - 1e-12)).reshape(-1, 1)
    p_cal_approx = calibrator["model"].predict_proba(z_approx)[:, 1]

    # 4. Compute metrics
    abs_diff = np.abs(p_cal_true - p_cal_approx)
    mean_abs_diff = float(np.mean(abs_diff))
    max_abs_diff = float(np.max(abs_diff))
    p95_abs_diff = float(np.percentile(abs_diff, 95))
    
    corr_res = spearmanr(p_cal_true, p_cal_approx)
    spearman_corr = float(corr_res.statistic if hasattr(corr_res, "statistic") else corr_res.correlation)

    # Decile assignment
    deciles_true = np.array([predictor._determine_risk_tier(p)[0] for p in p_cal_true])
    deciles_approx = np.array([predictor._determine_risk_tier(p)[0] for p in p_cal_approx])
    decile_changes = (deciles_true != deciles_approx)
    share_decile_changed = float(np.mean(decile_changes))
    share_decile_within_1 = float(np.mean(np.abs(deciles_true - deciles_approx) <= 1))

    print(f"  -> Mean absolute difference: {mean_abs_diff:.6f} ({mean_abs_diff*100:.4f} pp)")
    print(f"  -> Max absolute difference:  {max_abs_diff:.6f} ({max_abs_diff*100:.4f} pp)")
    print(f"  -> 95th pct absolute diff:  {p95_abs_diff:.6f} ({p95_abs_diff*100:.4f} pp)")
    print(f"  -> Spearman rank correlation: {spearman_corr:.6f}")
    print(f"  -> Share of rows whose decile changed: {share_decile_changed:.4f} ({share_decile_changed*100:.2f}%)")
    print(f"  -> Share within +/- 1 decile: {share_decile_within_1:.4f} ({share_decile_within_1*100:.2f}%)")

    # 5. Run 200 validation rows through predictor (true values) vs batch pipeline
    print("  -> Testing 200 validation rows through predictor (true values) vs batch pipeline...")
    sample_200 = val_df.sample(n=200, random_state=123).copy().reset_index(drop=True)
    
    # Batch pipeline probabilities
    tree_200, _ = pipe.transform(sample_200)
    p_raw_batch = model.predict_proba(tree_200[features])[:, 1]
    z_batch = logit(np.clip(p_raw_batch, 1e-12, 1.0 - 1e-12)).reshape(-1, 1)
    p_cal_batch = calibrator["model"].predict_proba(z_batch)[:, 1]

    # Predictor single-row probabilities (passing exact true elapsed/distance)
    p_cal_single = []
    for i, row in sample_200.iterrows():
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
        p_cal_single.append(res["calibrated_probability"])

    diff_predictor_batch = float(np.max(np.abs(np.array(p_cal_single) - p_cal_batch)))
    print(f"  -> Max abs diff between predictor (true values) and batch pipeline: {diff_predictor_batch:.10e}")
    assert diff_predictor_batch < 1e-6, f"Predictor vs batch pipeline diff {diff_predictor_batch} exceeds 1e-6!"

    # Save results
    metrics = {
        "sample_size": n_sample,
        "seed": 42,
        "mean_absolute_difference": round(mean_abs_diff, 6),
        "mean_absolute_difference_pp": round(mean_abs_diff * 100, 4),
        "max_absolute_difference": round(max_abs_diff, 6),
        "max_absolute_difference_pp": round(max_abs_diff * 100, 4),
        "p95_absolute_difference": round(p95_abs_diff, 6),
        "spearman_correlation": round(spearman_corr, 6),
        "share_decile_changed": round(share_decile_changed, 6),
        "share_decile_changed_pct": round(share_decile_changed * 100, 2),
        "share_decile_within_1_pct": round(share_decile_within_1 * 100, 2),
        "unseen_flights_in_sample": len(unseen_routes_list),
        "unseen_flights_in_sample_pct": round(len(unseen_routes_list) / n_sample * 100, 2),
        "distinct_unseen_routes_in_sample": len(unseen_routes_unique),
        "verification_200_rows_max_abs_diff": float(diff_predictor_batch),
        "status": "PASSED"
    }

    out_path = PROJECT_ROOT / "reports" / "serving_approximation_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[+] Saved metrics to {out_path}")

if __name__ == "__main__":
    main()
