"""
scripts/preregister_phase5.py
Phase 5 - Step 1: Artifact Freezing, Validation Jan-Aug Benchmarking, and Pre-registration

Computes:
1. SHA256 hashes of frozen artifacts and datasets.
2. Champion metrics restricted to Jan-Aug 2022 (same 8 months as 2023 test set) with 250 day-clustered bootstrap CIs.
3. 2023 test set base rate from labels ONLY (without touching features or predictions).
4. Saves reports/freeze_manifest.json.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss
)

from src.research.stats import bootstrap_day_ci
from src.research.metrics import evaluate_predictions

SEED = 42

def compute_sha256(filepath: str) -> str:
    """Computes SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("PHASE 5 - STEP 1: FREEZE PROOF & PRE-REGISTRATION")
    print("=" * 80)
    
    # 1. Freeze Proof Hashes
    files_to_hash = [
        "artifacts/feature_pipeline_v2.joblib",
        "artifacts/models/champion_v2.joblib",
        "artifacts/models/calibrator_v2.joblib",
        "data/processed/train.parquet",
        "data/processed/val.parquet",
        "data/processed/test.parquet"
    ]
    
    freeze_manifest = {}
    print("[*] Computing SHA256 hashes of frozen assets...")
    for fp in files_to_hash:
        assert os.path.exists(fp), f"Missing required file for freeze: {fp}"
        sha = compute_sha256(fp)
        size_bytes = os.path.getsize(fp)
        freeze_manifest[fp] = {
            "sha256": sha,
            "size_bytes": size_bytes
        }
        print(f"  - {fp:<40}: {sha[:16]}... ({size_bytes:,} bytes)")
        
    os.makedirs("reports", exist_ok=True)
    with open("reports/freeze_manifest.json", "w") as f:
        json.dump(freeze_manifest, f, indent=2)
    print("[+] Saved freeze manifest to reports/freeze_manifest.json")
    
    # 2. Validation Metrics Restricted to Jan-Aug 2022
    print("\n[*] Evaluating Champion on Validation Restricted to Jan-Aug 2022...")
    val_raw = pd.read_parquet("data/processed/val.parquet")
    val_tree = pd.read_parquet("data/processed/features_tree_val.parquet")
    
    val_dates_dt = pd.to_datetime(val_raw["FL_DATE"])
    jan_aug_mask = (val_dates_dt.dt.month <= 8).values
    
    val_raw_sub = val_raw[jan_aug_mask].copy()
    val_tree_sub = val_tree[jan_aug_mask].copy()
    
    y_val_sub = val_raw_sub["DELAY"].values.astype(int)
    val_dates_sub = val_raw_sub["FL_DATE"].values
    n_jan_aug = len(y_val_sub)
    n_days_sub = len(np.unique(val_dates_sub))
    base_rate_jan_aug = float(np.mean(y_val_sub))
    
    print(f"  -> Jan-Aug 2022 Flights: {n_jan_aug:,} across {n_days_sub} unique days.")
    print(f"  -> Jan-Aug 2022 Base Rate: {base_rate_jan_aug:.4f} ({base_rate_jan_aug*100:.2f}%)")
    
    # Load Champion Model & Calibrator
    cb2 = joblib.load("artifacts/models/champion_v2.joblib")
    calibrator_payload = joblib.load("artifacts/models/calibrator_v2.joblib")
    
    with open("artifacts/models/model_metadata_v2.json", "r") as f:
        meta_v2 = json.load(f)
    champ_features = meta_v2["features"]
    
    p_raw_sub = cb2.predict_proba(val_tree_sub[champ_features])[:, 1]
    z_sub = logit(np.clip(p_raw_sub, 1e-12, 1.0 - 1e-12)).reshape(-1, 1)
    p_cal_sub = calibrator_payload["model"].predict_proba(z_sub)[:, 1]
    
    # Day-clustered bootstrap CIs on Jan-Aug 2022
    boot_ci_sub = bootstrap_day_ci(y_val_sub, p_cal_sub, val_dates_sub, n_resamples=250, seed=SEED)
    metrics_sub = evaluate_predictions(y_val_sub, p_cal_sub, split_name="val_jan_aug")
    
    # Reference Brier & BSS on Jan-Aug
    brier_ref_val = base_rate_jan_aug * (1.0 - base_rate_jan_aug)
    brier_ref_train = float(brier_score_loss(y_val_sub, np.full_like(y_val_sub, 0.1762, dtype=float)))
    bss_val = 1.0 - (metrics_sub["val_jan_aug_brier"] / brier_ref_val)
    bss_train = 1.0 - (metrics_sub["val_jan_aug_brier"] / brier_ref_train)
    
    # Lift at top 10%
    k10 = max(1, int(round(0.10 * n_jan_aug)))
    top10_idx = np.argsort(p_cal_sub)[-k10:]
    prec10 = float(np.mean(y_val_sub[top10_idx]))
    lift10 = float(prec10 / base_rate_jan_aug)
    
    print(f"  -> PR-AUC  : {metrics_sub['val_jan_aug_pr_auc']:.4f} (95% CI: [{boot_ci_sub['pr_auc']['ci_lower']:.4f}, {boot_ci_sub['pr_auc']['ci_upper']:.4f}])")
    print(f"  -> ROC-AUC : {metrics_sub['val_jan_aug_roc_auc']:.4f} (95% CI: [{boot_ci_sub['roc_auc']['ci_lower']:.4f}, {boot_ci_sub['roc_auc']['ci_upper']:.4f}])")
    print(f"  -> Brier   : {metrics_sub['val_jan_aug_brier']:.4f} (95% CI: [{boot_ci_sub['brier']['ci_lower']:.4f}, {boot_ci_sub['brier']['ci_upper']:.4f}])")
    print(f"  -> BSS (vs base rate) : {bss_val:+.4f} ({bss_val*100:+.2f}%)")
    print(f"  -> Top 10% Lift       : {lift10:.2f}x (Precision = {prec10*100:.2f}%)")
    
    # 3. Compute 2023 Test Set Base Rate from LABELS ONLY
    print("\n[*] Reading 2023 Test Set LABELS ONLY to determine base rate...")
    test_df_labels = pd.read_parquet("data/processed/test.parquet", columns=["FL_DATE", "DELAY"])
    n_test = len(test_df_labels)
    test_dates = test_df_labels["FL_DATE"]
    test_months = pd.to_datetime(test_dates).dt.month.values
    test_base_rate = float(test_df_labels["DELAY"].mean())
    n_test_days = test_df_labels["FL_DATE"].nunique()
    
    print(f"  -> 2023 Test Completed Flights: {n_test:,} across {n_test_days} calendar days.")
    print(f"  -> 2023 Test Months Span: Month {test_months.min()} to Month {test_months.max()} (Jan to Aug).")
    print(f"  -> 2023 Test Ground-Truth Base Delay Rate: {test_base_rate:.4f} ({test_base_rate*100:.2f}%)")
    
    # Compute reference Brier on Test
    brier_ref_test = test_base_rate * (1.0 - test_base_rate)
    print(f"  -> Reference Brier p*(1-p) on Test: {brier_ref_test:.4f}")
    
    prereg_data = {
        "val_jan_aug_benchmark": {
            "flight_count": n_jan_aug,
            "unique_days": n_days_sub,
            "base_rate": round(base_rate_jan_aug, 4),
            "pr_auc": metrics_sub["val_jan_aug_pr_auc"],
            "pr_auc_ci": [boot_ci_sub["pr_auc"]["ci_lower"], boot_ci_sub["pr_auc"]["ci_upper"]],
            "roc_auc": metrics_sub["val_jan_aug_roc_auc"],
            "roc_auc_ci": [boot_ci_sub["roc_auc"]["ci_lower"], boot_ci_sub["roc_auc"]["ci_upper"]],
            "brier": metrics_sub["val_jan_aug_brier"],
            "brier_ci": [boot_ci_sub["brier"]["ci_lower"], boot_ci_sub["brier"]["ci_upper"]],
            "bss_vs_base_rate": round(bss_val, 4),
            "top_10_lift": round(lift10, 2),
            "top_10_precision": round(prec10, 4)
        },
        "test_ground_truth_prevalence": {
            "flight_count": n_test,
            "unique_days": n_test_days,
            "months_span": [int(test_months.min()), int(test_months.max())],
            "base_rate": round(test_base_rate, 4),
            "reference_brier": round(brier_ref_test, 4)
        }
    }
    
    with open("reports/preregistered_benchmarks.json", "w") as f:
        json.dump(prereg_data, f, indent=2)
    print("[+] Saved preregistered benchmarks to reports/preregistered_benchmarks.json")

if __name__ == "__main__":
    main()
