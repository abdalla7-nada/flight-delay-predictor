"""
Baseline Models Training & Evaluation (Phase 3 - Part 1)

Models Evaluated:
1. Dummy Classifier (Constant train delay rate = 0.1762)
2. Logistic Regression (L2 regularized, class_weight=None) on Linear Set
3. Logistic Regression (L2 regularized, class_weight='balanced') on Linear Set
4. LightGBM Default (n_estimators=2000, lr=0.05, num_leaves=63, early_stopping=50) on Tree Set

Hard Rules Followed:
- Validation set ONLY for model selection and scoring. Test set is strictly untouched.
- FL_DATE and year are not features.
- Fixed random seed = 42 logged everywhere.
- Real computed metrics saved to reports/baseline_metrics.json.
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
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from src.research.metrics import evaluate_predictions

SEED = 42

def load_data():
    """Loads train and validation sets for tree and linear feature representations."""
    print("[*] Loading feature datasets from data/processed/...")
    start = time.time()
    
    # 1. Tree set
    tree_train = pd.read_parquet("data/processed/features_tree_train.parquet")
    tree_val = pd.read_parquet("data/processed/features_tree_val.parquet")
    
    # 2. Linear set
    linear_train = pd.read_parquet("data/processed/features_linear_train.parquet")
    linear_val = pd.read_parquet("data/processed/features_linear_val.parquet")
    
    # Target vectors
    y_train = tree_train["DELAY"].values.astype(int)
    y_val = tree_val["DELAY"].values.astype(int)
    
    # Feature matrices
    X_tree_train = tree_train.drop(columns=["DELAY"])
    X_tree_val = tree_val.drop(columns=["DELAY"])
    
    X_linear_train = linear_train.drop(columns=["DELAY"])
    X_linear_val = linear_val.drop(columns=["DELAY"])
    
    # Ensure categorical columns in tree set are proper pandas category dtype
    cat_cols = ["AIRLINE_CODE", "ORIGIN", "DEST", "route", "part_of_day"]
    for col in cat_cols:
        X_tree_train[col] = X_tree_train[col].astype("category")
        X_tree_val[col] = X_tree_val[col].astype("category")
        
    print(f"[+] Loaded datasets in {time.time() - start:.2f}s:")
    print(f"    Train: {len(y_train):,} rows (Delay Rate: {y_train.mean():.4f})")
    print(f"    Val:   {len(y_val):,} rows (Delay Rate: {y_val.mean():.4f})")
    print(f"    Tree Features:   {X_tree_train.shape[1]} columns")
    print(f"    Linear Features: {X_linear_train.shape[1]} columns")
    
    return (
        X_tree_train, y_train, X_tree_val, y_val,
        X_linear_train, X_linear_val, cat_cols
    )

def run_dummy_baseline(y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """Evaluates constant prediction equal to train delay prevalence."""
    print("\n" + "=" * 60)
    print("1. DUMMY BASELINE (Constant Train Delay Rate)")
    print("=" * 60)
    start = time.time()
    
    const_pred = float(np.mean(y_train))
    y_pred_train = np.full(len(y_train), const_pred, dtype=float)
    y_pred_val = np.full(len(y_val), const_pred, dtype=float)
    
    val_metrics = evaluate_predictions(y_val, y_pred_val, split_name="val")
    train_metrics = evaluate_predictions(y_train, y_pred_train, split_name="train")
    
    elapsed = time.time() - start
    print(f"  Train PR-AUC: {train_metrics['train_pr_auc']:.4f} | Val PR-AUC: {val_metrics['val_pr_auc']:.4f}")
    print(f"  Train ROC-AUC: {train_metrics['train_roc_auc']:.4f} | Val ROC-AUC: {val_metrics['val_roc_auc']:.4f}")
    print(f"  Val Brier: {val_metrics['val_brier']:.4f} | Val Lift@10%: {val_metrics['val_lift_at_10']:.2f}x")
    
    return {
        "model_name": "Dummy (Constant Mean)",
        "train_time_sec": round(elapsed, 2),
        **val_metrics,
        **train_metrics
    }

def run_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    class_weight: Any = None
) -> Dict[str, Any]:
    """Trains and evaluates L2 regularized Logistic Regression on linear features."""
    name_suffix = "None" if class_weight is None else "balanced"
    model_name = f"Logistic Regression (class_weight={name_suffix})"
    print("\n" + "=" * 60)
    print(f"2/3. {model_name.upper()}")
    print("=" * 60)
    print(f"  Parameters: C=1.0, penalty=l2, solver=lbfgs, max_iter=1000, random_state={SEED}")
    start = time.time()
    
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED,
        class_weight=class_weight,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    fit_time = time.time() - start
    print(f"  [+] Fit completed in {fit_time:.2f}s (Iterations: {model.n_iter_[0]})")
    
    # Predict probabilities
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val = model.predict_proba(X_val)[:, 1]
    
    val_metrics = evaluate_predictions(y_val, y_prob_val, split_name="val")
    train_metrics = evaluate_predictions(y_train, y_prob_train, split_name="train")
    
    print(f"  Train PR-AUC: {train_metrics['train_pr_auc']:.4f} | Val PR-AUC: {val_metrics['val_pr_auc']:.4f}")
    print(f"  Train ROC-AUC: {train_metrics['train_roc_auc']:.4f} | Val ROC-AUC: {val_metrics['val_roc_auc']:.4f}")
    print(f"  Val Brier: {val_metrics['val_brier']:.4f} | Val Lift@10%: {val_metrics['val_lift_at_10']:.2f}x | Best F1: {val_metrics['val_best_f1']:.4f} (thresh={val_metrics['val_best_threshold']})")
    
    return {
        "model_name": model_name,
        "train_time_sec": round(fit_time, 2),
        "iterations": int(model.n_iter_[0]),
        **val_metrics,
        **train_metrics
    }

def run_lightgbm_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list
) -> Tuple[Dict[str, Any], lgb.LGBMClassifier]:
    """Trains and evaluates LightGBM with project-specified defaults on tree set."""
    print("\n" + "=" * 60)
    print("4. LIGHTGBM DEFAULT BASELINE")
    print("=" * 60)
    print(
        f"  Parameters: n_estimators=2000, lr=0.05, num_leaves=63, min_child_samples=200,\n"
        f"              subsample=0.8, colsample=0.8, cat_smooth=10.0, min_data_per_group=100,\n"
        f"              early_stopping=50, random_state={SEED}"
    )
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
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["average_precision", "binary_logloss"],
        callbacks=callbacks
    )
    
    fit_time = time.time() - start
    best_iter = model.best_iteration_ if hasattr(model, "best_iteration_") else 2000
    print(f"  [+] Fit completed in {fit_time:.2f}s (Best Iteration: {best_iter})")
    
    # Predict probabilities
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val = model.predict_proba(X_val)[:, 1]
    
    val_metrics = evaluate_predictions(y_val, y_prob_val, split_name="val")
    train_metrics = evaluate_predictions(y_train, y_prob_train, split_name="train")
    
    print(f"  Train PR-AUC: {train_metrics['train_pr_auc']:.4f} | Val PR-AUC: {val_metrics['val_pr_auc']:.4f}")
    print(f"  Train ROC-AUC: {train_metrics['train_roc_auc']:.4f} | Val ROC-AUC: {val_metrics['val_roc_auc']:.4f}")
    print(f"  Val Brier: {val_metrics['val_brier']:.4f} | Val Lift@10%: {val_metrics['val_lift_at_10']:.2f}x | Best F1: {val_metrics['val_best_f1']:.4f} (thresh={val_metrics['val_best_threshold']})")
    
    result_dict = {
        "model_name": "LightGBM Default",
        "train_time_sec": round(fit_time, 2),
        "best_iteration": int(best_iter),
        **val_metrics,
        **train_metrics
    }
    
    return result_dict, model

def main():
    print("*" * 65)
    print("   PHASE 3 - PART 1: BASELINE MODELS EVALUATION   ")
    print(f"   Fixed Random Seed: {SEED}")
    print("*" * 65)
    
    X_tree_tr, y_tr, X_tree_val, y_val, X_lin_tr, X_lin_val, cat_cols = load_data()
    
    results = []
    
    # Model 1: Dummy
    res_dummy = run_dummy_baseline(y_tr, y_val)
    results.append(res_dummy)
    
    # Model 2: Logistic Regression (class_weight=None)
    res_lr_none = run_logistic_regression(X_lin_tr, y_tr, X_lin_val, y_val, class_weight=None)
    results.append(res_lr_none)
    
    # Model 3: Logistic Regression (class_weight='balanced')
    res_lr_bal = run_logistic_regression(X_lin_tr, y_tr, X_lin_val, y_val, class_weight="balanced")
    results.append(res_lr_bal)
    
    # Model 4: LightGBM Default
    res_lgbm, lgbm_model = run_lightgbm_baseline(X_tree_tr, y_tr, X_tree_val, y_val, cat_cols)
    results.append(res_lgbm)
    
    # Summary Table
    print("\n" + "=" * 90)
    print("BASELINE COMPARISON TABLE (VALIDATION SET: 444,554 FLIGHTS)")
    print("=" * 90)
    summary_df = pd.DataFrame(results)[[
        "model_name", "val_pr_auc", "val_roc_auc", "val_brier",
        "val_log_loss", "val_best_f1", "val_best_threshold",
        "val_prec_at_10", "val_rec_at_10", "val_lift_at_10",
        "train_pr_auc", "train_roc_auc", "train_time_sec"
    ]]
    
    print(summary_df.to_string(index=False))
    print("-" * 90)
    print("Reference Baseline: Best Single Prior (prior_origin_hour) Val PR-AUC = 0.2845")
    print("Validation Delay Prevalence Baseline: 0.2032")
    print("=" * 90)
    
    # Save to reports/baseline_metrics.json
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/baseline_metrics.json"
    with open(out_path, "w") as f:
        json.dump({
            "seed": SEED,
            "reference_prior_origin_hour_pr_auc": 0.2845,
            "val_prevalence": float(np.mean(y_val)),
            "train_prevalence": float(np.mean(y_tr)),
            "models": results
        }, f, indent=2)
    print(f"\n[+] Baseline metrics saved to {out_path}")

if __name__ == "__main__":
    main()
