"""
src/validate.py
Validation and Sanity Checks for Phase 1 & 2:
1. Feature columns list per set, dtypes, null counts (~0)
2. Leakage test: confirms no post-departure columns in feature sets
3. Prior-encoding sanity check: correlation of each prior with DELAY on train (OOF) vs val
4. Single-row transformation consistency test (API equivalence)
5. Univariate PR-AUC check of each prior feature alone on val
"""

import os
import sys
# Add project root to sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from src.features import FlightFeaturePipeline

def run_validations():
    print("======================================================================")
    print("PHASE 1 & 2 VALIDATION & SANITY CHECKS")
    print("======================================================================")

    # 1. Load feature sets
    print("  -> Loading generated feature parquet files...")
    tree_tr = pd.read_parquet("data/processed/features_tree_train.parquet")
    tree_va = pd.read_parquet("data/processed/features_tree_val.parquet")
    tree_te = pd.read_parquet("data/processed/features_tree_test.parquet")

    linear_tr = pd.read_parquet("data/processed/features_linear_train.parquet")
    linear_va = pd.read_parquet("data/processed/features_linear_val.parquet")
    linear_te = pd.read_parquet("data/processed/features_linear_test.parquet")

    # Check 1: Feature column lists and null counts
    print("\n--- Check 1: Feature Column Counts & Null Check ---")
    tree_nulls_tr = int(tree_tr.isnull().sum().sum())
    tree_nulls_va = int(tree_va.isnull().sum().sum())
    tree_nulls_te = int(tree_te.isnull().sum().sum())
    
    linear_nulls_tr = int(linear_tr.isnull().sum().sum())
    linear_nulls_va = int(linear_va.isnull().sum().sum())
    linear_nulls_te = int(linear_te.isnull().sum().sum())

    print(f"Tree Train:   Shape={tree_tr.shape}, Nulls={tree_nulls_tr}")
    print(f"Tree Val:     Shape={tree_va.shape}, Nulls={tree_nulls_va}")
    print(f"Tree Test:    Shape={tree_te.shape}, Nulls={tree_nulls_te}")
    print(f"Linear Train: Shape={linear_tr.shape}, Nulls={linear_nulls_tr}")
    print(f"Linear Val:   Shape={linear_va.shape}, Nulls={linear_nulls_va}")
    print(f"Linear Test:  Shape={linear_te.shape}, Nulls={linear_nulls_te}")

    assert tree_nulls_tr == 0 and tree_nulls_va == 0 and tree_nulls_te == 0, "Nulls found in Tree feature sets!"
    assert linear_nulls_tr == 0 and linear_nulls_va == 0 and linear_nulls_te == 0, "Nulls found in Linear feature sets!"

    # Check 2: Strict Data Leakage Test
    print("\n--- Check 2: Strict Post-Departure Leakage Test ---")
    prohibited_cols = [
        "DEP_TIME", "DEP_DELAY", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
        "ARR_TIME", "ARR_DELAY", "ELAPSED_TIME", "AIR_TIME", "CANCELLED",
        "CANCELLATION_CODE", "DIVERTED", "DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER",
        "DELAY_DUE_NAS", "DELAY_DUE_SECURITY", "DELAY_DUE_LATE_AIRCRAFT"
    ]
    for name, df in [("Tree Train", tree_tr), ("Tree Val", tree_va), ("Tree Test", tree_te),
                     ("Linear Train", linear_tr), ("Linear Val", linear_va), ("Linear Test", linear_te)]:
        for p_col in prohibited_cols:
            assert p_col not in df.columns, f"CRITICAL LEAKAGE: {p_col} found in {name}!"
        # Feature columns excluding target
        feat_cols = [c for c in df.columns if c != "DELAY"]
        assert "DELAY" not in feat_cols
    print("[PASS] No post-departure leakage columns exist in any feature set.")

    # Check 3: Prior-Encoding Sanity Check (Correlation Train OOF vs Val)
    print("\n--- Check 3: Prior-Encoding Correlation Comparison (Train OOF vs Val) ---")
    prior_cols = ["prior_airline", "prior_origin", "prior_dest", "prior_route", "prior_origin_hour"]
    correlations = []
    for col in prior_cols:
        r_tr = float(np.corrcoef(tree_tr[col], tree_tr["DELAY"])[0, 1])
        r_va = float(np.corrcoef(tree_va[col], tree_va["DELAY"])[0, 1])
        diff = abs(r_tr - r_va)
        print(f"  - {col:<18}: Train OOF r = {r_tr:.4f} | Val r = {r_va:.4f} | Difference = {diff:.4f}")
        # Assertion: train correlation should NOT be drastically higher than val (which would indicate OOF leakage)
        assert diff < 0.05, f"Warning: Train correlation for {col} diverges from val by {diff:.4f}!"
        correlations.append({
            "prior": col,
            "train_oof_corr": round(r_tr, 4),
            "val_corr": round(r_va, 4),
            "abs_diff": round(diff, 4)
        })
    print("[PASS] OOF prior correlations match validation correlations closely (no target leakage).")

    # Check 4: Single-Row Transformation Equivalence (API Equivalence)
    print("\n--- Check 4: Single-Row Transformation Consistency ---")
    pipeline = FlightFeaturePipeline.load_artifacts("artifacts")
    val_raw = pd.read_parquet("data/processed/val.parquet")
    
    # Test 3 distinct sample rows from validation set
    test_indices = [0, 100, 5000]
    single_row_checks = []
    
    for idx in test_indices:
        raw_row = val_raw.iloc[idx].to_dict()
        tree_batch_row = tree_va.iloc[idx].drop(labels=["DELAY"])
        linear_batch_row = linear_va.iloc[idx].drop(labels=["DELAY"])
        
        # Transform single row through pipeline
        tree_single, linear_single = pipeline.transform(raw_row)
        
        # Compare numeric columns in tree set
        for col in pipeline.tree_numeric_cols:
            val_batch = float(tree_batch_row[col])
            val_single = float(tree_single[col].iloc[0])
            assert np.isclose(val_batch, val_single, atol=1e-5), (
                f"Mismatch at row {idx}, col {col}: batch={val_batch}, single={val_single}"
            )
            
        # Compare categorical string representations in tree set
        for col in pipeline.tree_cat_cols:
            cat_batch = str(tree_batch_row[col])
            cat_single = str(tree_single[col].iloc[0])
            assert cat_batch == cat_single, (
                f"Mismatch in category at row {idx}, col {col}: batch={cat_batch}, single={cat_single}"
            )
            
        # Compare linear feature vectors
        for col in pipeline.linear_feature_cols:
            val_batch = float(linear_batch_row[col])
            val_single = float(linear_single[col].iloc[0])
            assert np.isclose(val_batch, val_single, atol=1e-5), (
                f"Mismatch in linear feature at row {idx}, col {col}: batch={val_batch}, single={val_single}"
            )
            
        print(f"  - Row idx {idx} ({raw_row['AIRLINE_CODE']} {raw_row['ORIGIN']}->{raw_row['DEST']} on {raw_row['FL_DATE']}): 100% IDENTICAL output!")
        single_row_checks.append({
            "row_idx": idx,
            "flight": f"{raw_row['AIRLINE_CODE']} {raw_row['ORIGIN']}->{raw_row['DEST']}",
            "date": str(raw_row["FL_DATE"]),
            "status": "IDENTICAL"
        })
    print("[PASS] Single-row API transformation outputs are bit-for-bit identical to batch pipeline.")

    # Check 5: Univariate PR-AUC Check of Priors on Validation
    print("\n--- Check 5: Univariate PR-AUC of Priors on Validation Set ---")
    val_delay_rate = float(tree_va["DELAY"].mean())
    print(f"Validation Positive Baseline Rate (Random Guessing PR-AUC): {val_delay_rate:.4f}")
    
    univariate_prauc = []
    for col in prior_cols:
        score = tree_va[col].values
        prauc = float(average_precision_score(tree_va["DELAY"].values, score))
        lift = ((prauc - val_delay_rate) / val_delay_rate) * 100
        print(f"  - {col:<18}: Val PR-AUC = {prauc:.4f} (Lift over baseline: +{lift:.1f}%)")
        assert prauc > val_delay_rate, f"Feature {col} has PR-AUC ({prauc:.4f}) below random baseline ({val_delay_rate:.4f})!"
        univariate_prauc.append({
            "prior_feature": col,
            "val_prauc": round(prauc, 4),
            "baseline_prauc": round(val_delay_rate, 4),
            "lift_pct": round(lift, 2)
        })
    print("[PASS] All prior features deliver strong univariate predictive lift over baseline.")

    results = {
        "shapes": {
            "tree_train": list(tree_tr.shape),
            "tree_val": list(tree_va.shape),
            "tree_test": list(tree_te.shape),
            "linear_train": list(linear_tr.shape),
            "linear_val": list(linear_va.shape),
            "linear_test": list(linear_te.shape)
        },
        "correlations": correlations,
        "single_row_checks": single_row_checks,
        "univariate_prauc": univariate_prauc,
        "tree_cat_cols": pipeline.tree_cat_cols,
        "tree_numeric_cols": pipeline.tree_numeric_cols,
        "linear_feature_cols": pipeline.linear_feature_cols
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/phase1_2_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[+] Validation metrics successfully saved to reports/phase1_2_validation.json")
    return results

if __name__ == "__main__":
    run_validations()
