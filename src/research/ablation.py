"""
Feature Ablation Study with LightGBM (Phase 3 - Part 2)

Evaluates 7 distinct feature subsets using identical LightGBM hyperparameter settings:
  Variant A: Calendar/schedule only (16 features)
  Variant B: A + AIRLINE_CODE, ORIGIN, DEST (categorical) + distance/physics features (25 features)
  Variant C: B + historical priors (30 features)
  Variant D: C + frequencies + raw route categorical (Full default tree set, 34 features)
  Variant E: D minus day_of_year (33 features)
  Variant F: E minus raw route categorical (keep prior_route and route_freq) (32 features)
  Variant G: F minus raw ORIGIN/DEST categoricals (keep priors + freq) (30 features)

Hard Rules Followed:
- Validation set ONLY for model selection and scoring. Test set is strictly untouched.
- Same LightGBM hyperparameters and early stopping across all runs.
- Fixed random seed = 42 logged everywhere.
- Real computed metrics saved to reports/ablation_metrics.json.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from typing import Dict, Any, List

import lightgbm as lgb
from src.research.metrics import evaluate_predictions

SEED = 42

def load_data():
    """Loads tree feature datasets from data/processed/."""
    print("[*] Loading tree feature datasets...")
    tree_train = pd.read_parquet("data/processed/features_tree_train.parquet")
    tree_val = pd.read_parquet("data/processed/features_tree_val.parquet")
    
    y_train = tree_train["DELAY"].values.astype(int)
    y_val = tree_val["DELAY"].values.astype(int)
    
    X_train = tree_train.drop(columns=["DELAY"])
    X_val = tree_val.drop(columns=["DELAY"])
    
    # Ensure categorical dtypes
    raw_cat_cols = ["AIRLINE_CODE", "ORIGIN", "DEST", "route", "part_of_day"]
    for col in raw_cat_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")
            
    print(f"[+] Loaded {len(y_train):,} train rows and {len(y_val):,} val rows.")
    return X_train, y_train, X_val, y_val

def define_ablation_variants() -> List[Dict[str, Any]]:
    """Defines the 7 feature ablation configurations."""
    # Base Calendar / Schedule features (Variant A)
    cal_features = [
        "month", "day_of_week", "day_of_year", "is_weekend", "is_holiday_period",
        "dep_hour", "arr_hour", "part_of_day",
        "dep_hour_sin", "dep_hour_cos", "arr_hour_sin", "arr_hour_cos",
        "month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos"
    ]
    
    # Physics & distance features
    physics_features = [
        "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE",
        "log_distance", "scheduled_speed"
    ]
    
    # Basic categoricals
    basic_cats = ["AIRLINE_CODE", "ORIGIN", "DEST"]
    
    # Priors
    priors = ["prior_airline", "prior_origin", "prior_dest", "prior_route", "prior_origin_hour"]
    
    # Frequencies
    frequencies = ["origin_freq", "dest_freq", "route_freq"]
    
    variants = [
        {
            "variant_id": "A",
            "name": "Calendar / Schedule Only",
            "description": "Diurnal hour, day of week, month, holidays, part of day, cyclical encodings",
            "features": cal_features,
            "cats": ["part_of_day"]
        },
        {
            "variant_id": "B",
            "name": "Calendar + Physics + Basic Categoricals",
            "description": "Variant A + AIRLINE_CODE, ORIGIN, DEST + scheduled speed, elapsed time, distance",
            "features": cal_features + basic_cats + physics_features,
            "cats": ["part_of_day", "AIRLINE_CODE", "ORIGIN", "DEST"]
        },
        {
            "variant_id": "C",
            "name": "Variant B + Historical Priors",
            "description": "Variant B + 5 smoothed empirical Bayes priors (airline, origin, dest, route, origin_hour)",
            "features": cal_features + basic_cats + physics_features + priors,
            "cats": ["part_of_day", "AIRLINE_CODE", "ORIGIN", "DEST"]
        },
        {
            "variant_id": "D",
            "name": "Full Tree Set (Current Default)",
            "description": "Variant C + operational frequencies + raw route categorical",
            "features": cal_features + basic_cats + physics_features + priors + frequencies + ["route"],
            "cats": ["part_of_day", "AIRLINE_CODE", "ORIGIN", "DEST", "route"]
        },
        {
            "variant_id": "E",
            "name": "Full Set Minus day_of_year",
            "description": "Variant D minus day_of_year to eliminate potential seasonal train-val drift",
            "features": [f for f in (cal_features + basic_cats + physics_features + priors + frequencies + ["route"]) if f != "day_of_year"],
            "cats": ["part_of_day", "AIRLINE_CODE", "ORIGIN", "DEST", "route"]
        },
        {
            "variant_id": "F",
            "name": "Minus day_of_year & Minus raw route",
            "description": "Variant E minus high-cardinality raw route (keep prior_route and route_freq)",
            "features": [f for f in (cal_features + basic_cats + physics_features + priors + frequencies) if f != "day_of_year"],
            "cats": ["part_of_day", "AIRLINE_CODE", "ORIGIN", "DEST"]
        },
        {
            "variant_id": "G",
            "name": "Priors & Frequencies Only (No raw ORIGIN/DEST/route)",
            "description": "Variant F minus raw ORIGIN and DEST (only AIRLINE_CODE and part_of_day as raw categoricals)",
            "features": [f for f in (cal_features + ["AIRLINE_CODE"] + physics_features + priors + frequencies) if f != "day_of_year"],
            "cats": ["part_of_day", "AIRLINE_CODE"]
        }
    ]
    
    return variants

def train_and_eval_variant(
    variant: Dict[str, Any],
    X_train_full: pd.DataFrame,
    y_train: np.ndarray,
    X_val_full: pd.DataFrame,
    y_val: np.ndarray
) -> Dict[str, Any]:
    """Trains LightGBM on a specific feature subset and computes train & val metrics."""
    vid = variant["variant_id"]
    features = variant["features"]
    cats = variant["cats"]
    
    print("\n" + "=" * 70)
    print(f"RUNNING ABLATION VARIANT {vid}: {variant['name']}")
    print(f"Features ({len(features)}): {features}")
    print(f"Categoricals ({len(cats)}): {cats}")
    print("=" * 70)
    
    X_tr = X_train_full[features].copy()
    X_v = X_val_full[features].copy()
    
    start = time.time()
    
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        cat_smooth=10.0,
        min_data_per_group=100,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)
    ]
    
    model.fit(
        X_tr,
        y_train,
        eval_set=[(X_v, y_val)],
        eval_metric=["average_precision", "binary_logloss"],
        callbacks=callbacks
    )
    
    train_time = time.time() - start
    best_iter = int(model.best_iteration_) if hasattr(model, "best_iteration_") else 2000
    
    y_prob_train = model.predict_proba(X_tr)[:, 1]
    y_prob_val = model.predict_proba(X_v)[:, 1]
    
    val_metrics = evaluate_predictions(y_val, y_prob_val, split_name="val")
    train_metrics = evaluate_predictions(y_train, y_prob_train, split_name="train")
    
    pr_auc_gap = round(train_metrics["train_pr_auc"] - val_metrics["val_pr_auc"], 4)
    roc_auc_gap = round(train_metrics["train_roc_auc"] - val_metrics["val_roc_auc"], 4)
    
    print(f"  [+] Finished Variant {vid} in {train_time:.2f}s | Best Iteration: {best_iter}")
    print(f"      Val PR-AUC: {val_metrics['val_pr_auc']:.4f} (Train: {train_metrics['train_pr_auc']:.4f}, Gap: {pr_auc_gap:+.4f})")
    print(f"      Val ROC-AUC: {val_metrics['val_roc_auc']:.4f} (Train: {train_metrics['train_roc_auc']:.4f}, Gap: {roc_auc_gap:+.4f})")
    print(f"      Val Brier: {val_metrics['val_brier']:.4f} | Val Lift@10%: {val_metrics['val_lift_at_10']:.2f}x")
    
    return {
        "variant_id": vid,
        "variant_name": variant["name"],
        "description": variant["description"],
        "num_features": len(features),
        "best_iteration": best_iter,
        "train_time_sec": round(train_time, 2),
        "pr_auc_gap": pr_auc_gap,
        "roc_auc_gap": roc_auc_gap,
        "features": features,
        "categoricals": cats,
        **val_metrics,
        **train_metrics
    }

def main():
    print("*" * 75)
    print("   PHASE 3 - PART 2: LIGHTGBM FEATURE ABLATION STUDY   ")
    print(f"   Fixed Random Seed: {SEED}")
    print("*" * 75)
    
    X_train_full, y_train, X_val_full, y_val = load_data()
    variants = define_ablation_variants()
    
    ablation_results = []
    
    for var in variants:
        res = train_and_eval_variant(var, X_train_full, y_train, X_val_full, y_val)
        ablation_results.append(res)
        
    # Build comparison summary DataFrame
    df_summary = pd.DataFrame(ablation_results)[[
        "variant_id", "variant_name", "num_features", "best_iteration",
        "val_pr_auc", "val_roc_auc", "val_brier", "val_best_f1",
        "val_lift_at_10", "pr_auc_gap", "roc_auc_gap", "train_time_sec"
    ]]
    
    print("\n" + "=" * 115)
    print("FEATURE ABLATION STUDY RESULTS (EVALUATION ON VALIDATION SET: 444,554 FLIGHTS)")
    print("=" * 115)
    print(df_summary.to_string(index=False))
    print("-" * 115)
    
    # Save to reports/ablation_metrics.json
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/ablation_metrics.json"
    with open(out_path, "w") as f:
        json.dump({
            "seed": SEED,
            "validation_delay_prevalence": float(np.mean(y_val)),
            "variants": ablation_results
        }, f, indent=2)
    print(f"\n[+] Ablation metrics saved to {out_path}")

if __name__ == "__main__":
    main()
