"""
src/train_alternatives.py
Phase 4 - Part 3: Alternative Models and Ensembles

Evaluates:
1. CatBoost (Version 1 - Priors Only):
   Trained on Variant G feature set (AIRLINE_CODE, part_of_day as categoricals).
2. CatBoost (Version 2 - Native Categoricals):
   Trained with raw ORIGIN, DEST, and route as native categorical features.
3. Simple Blend 1:
   Average probabilities of LightGBM + Logistic Regression (0.5 * p_lgbm + 0.5 * p_lr).
4. Simple Blend 2:
   Average probabilities of LightGBM + CatBoost (0.5 * p_lgbm + 0.5 * p_catboost).

Paired cluster-bootstrap testing (resampling by DAY, 250 iterations)
against the LightGBM baseline champion.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

from src.research.stats import paired_bootstrap_difference, bootstrap_day_ci
from src.research.metrics import evaluate_predictions

SEED = 42

def run_alternatives():
    print("=" * 80)
    print("PHASE 4 - PART 3: ALTERNATIVE MODELS & BLENDS")
    print("=" * 80)
    
    # 1. Load Data
    print("[*] Loading tree and linear feature datasets...")
    tree_tr = pd.read_parquet("data/processed/features_tree_train.parquet")
    tree_va = pd.read_parquet("data/processed/features_tree_val.parquet")
    
    lin_tr = pd.read_parquet("data/processed/features_linear_train.parquet")
    lin_va = pd.read_parquet("data/processed/features_linear_val.parquet")
    
    val_raw = pd.read_parquet("data/processed/val.parquet")
    val_dates = val_raw["FL_DATE"].values
    
    y_train = tree_tr["DELAY"].values.astype(int)
    y_val = tree_va["DELAY"].values.astype(int)
    
    # Load feature metadata for Variant G
    with open("artifacts/models/model_metadata.json", "r") as f:
        meta = json.load(f)
    feat_g = meta["features"]
    cat_g = meta["categoricals"]
    
    # Ensure string / categorical types
    for c in ["AIRLINE_CODE", "part_of_day", "ORIGIN", "DEST", "route"]:
        if c in tree_tr.columns:
            tree_tr[c] = tree_tr[c].astype(str)
            tree_va[c] = tree_va[c].astype(str)
            
    # Load champion LightGBM model
    print("[*] Loading Champion LightGBM model...")
    lgbm_model = joblib.load("artifacts/models/best_baseline_lgbm.joblib")
    
    # LightGBM requires categorical dtypes
    X_va_lgbm = tree_va[feat_g].copy()
    for c in cat_g:
        X_va_lgbm[c] = X_va_lgbm[c].astype("category")
    p_lgbm = lgbm_model.predict_proba(X_va_lgbm)[:, 1]
    m_lgbm = evaluate_predictions(y_val, p_lgbm, split_name="val")
    print(f"  -> Champion LightGBM Val PR-AUC: {m_lgbm['val_pr_auc']:.4f} | ROC-AUC: {m_lgbm['val_roc_auc']:.4f} | Brier: {m_lgbm['val_brier']:.4f}")
    
    results = {}
    
    # =========================================================================
    # 2. CatBoost Version 1: Priors Only (Variant G feature set)
    # =========================================================================
    print("\n--- Model 1: CatBoost (Version 1 - Priors Only) ---")
    cb_params_v1 = {
        "iterations": 1000,
        "learning_rate": 0.08,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": SEED,
        "early_stopping_rounds": 40,
        "thread_count": -1,
        "verbose": 100
    }
    
    X_tr_cb1 = tree_tr[feat_g]
    X_va_cb1 = tree_va[feat_g]
    
    pool_tr1 = Pool(X_tr_cb1, y_train, cat_features=cat_g)
    pool_va1 = Pool(X_va_cb1, y_val, cat_features=cat_g)
    
    t0_cb1 = time.time()
    cb_v1 = CatBoostClassifier(**cb_params_v1)
    cb_v1.fit(pool_tr1, eval_set=pool_va1, use_best_model=True)
    time_cb1 = time.time() - t0_cb1
    
    p_cb1 = cb_v1.predict_proba(pool_va1)[:, 1]
    m_cb1 = evaluate_predictions(y_val, p_cb1, split_name="val")
    pb_cb1 = paired_bootstrap_difference(y_val, p_lgbm, p_cb1, val_dates, n_resamples=250, seed=SEED)
    
    print(f"  CatBoost (Priors Only) Fit Time: {time_cb1:.1f}s | Best Iter: {cb_v1.get_best_iteration()}")
    print(f"  Val PR-AUC: {m_cb1['val_pr_auc']:.4f} | ROC-AUC: {m_cb1['val_roc_auc']:.4f} | Brier: {m_cb1['val_brier']:.4f}")
    print(f"  Delta vs LightGBM: {pb_cb1['delta_pr_auc']['point']:+.5f} (95% CI: [{pb_cb1['delta_pr_auc']['ci_lower']:+.5f}, {pb_cb1['delta_pr_auc']['ci_upper']:+.5f}])")
    print(f"  Verdict: {pb_cb1['verdict'].upper()} (p-value: {pb_cb1['delta_pr_auc']['p_value_two_tailed']:.4f})")
    
    results["catboost_priors_only"] = {
        "metrics": m_cb1,
        "best_iteration": cb_v1.get_best_iteration(),
        "fit_time_sec": round(time_cb1, 2),
        "paired_bootstrap_vs_lgbm": pb_cb1
    }
    joblib.dump(cb_v1, "artifacts/models/catboost_priors_v2.joblib")
    
    # =========================================================================
    # 3. CatBoost Version 2: Native Categoricals (ORIGIN, DEST, route)
    # =========================================================================
    print("\n--- Model 2: CatBoost (Version 2 - Native High-Cardinality Categoricals) ---")
    feat_v2 = [f for f in feat_g] + ["ORIGIN", "DEST", "route"]
    cat_v2 = ["AIRLINE_CODE", "part_of_day", "ORIGIN", "DEST", "route"]
    
    X_tr_cb2 = tree_tr[feat_v2]
    X_va_cb2 = tree_va[feat_v2]
    
    pool_tr2 = Pool(X_tr_cb2, y_train, cat_features=cat_v2)
    pool_va2 = Pool(X_va_cb2, y_val, cat_features=cat_v2)
    
    cb_params_v2 = {
        "iterations": 1000,
        "learning_rate": 0.08,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": SEED,
        "early_stopping_rounds": 40,
        "thread_count": -1,
        "verbose": 100
    }
    
    t0_cb2 = time.time()
    cb_v2 = CatBoostClassifier(**cb_params_v2)
    cb_v2.fit(pool_tr2, eval_set=pool_va2, use_best_model=True)
    time_cb2 = time.time() - t0_cb2
    
    p_cb2 = cb_v2.predict_proba(pool_va2)[:, 1]
    m_cb2 = evaluate_predictions(y_val, p_cb2, split_name="val")
    pb_cb2 = paired_bootstrap_difference(y_val, p_lgbm, p_cb2, val_dates, n_resamples=250, seed=SEED)
    
    print(f"  CatBoost (Native Cats) Fit Time: {time_cb2:.1f}s | Best Iter: {cb_v2.get_best_iteration()}")
    print(f"  Val PR-AUC: {m_cb2['val_pr_auc']:.4f} | ROC-AUC: {m_cb2['val_roc_auc']:.4f} | Brier: {m_cb2['val_brier']:.4f}")
    print(f"  Delta vs LightGBM: {pb_cb2['delta_pr_auc']['point']:+.5f} (95% CI: [{pb_cb2['delta_pr_auc']['ci_lower']:+.5f}, {pb_cb2['delta_pr_auc']['ci_upper']:+.5f}])")
    print(f"  Verdict: {pb_cb2['verdict'].upper()} (p-value: {pb_cb2['delta_pr_auc']['p_value_two_tailed']:.4f})")
    
    results["catboost_native_cats"] = {
        "metrics": m_cb2,
        "best_iteration": cb_v2.get_best_iteration(),
        "fit_time_sec": round(time_cb2, 2),
        "paired_bootstrap_vs_lgbm": pb_cb2
    }
    joblib.dump(cb_v2, "artifacts/models/catboost_native_cats_v2.joblib")
    
    # =========================================================================
    # 4. Logistic Regression Training & Probabilities
    # =========================================================================
    print("\n--- Training Logistic Regression for Ensembling ---")
    lin_feat_cols = [c for c in lin_tr.columns if c != "DELAY"]
    
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, n_jobs=-1)
    t0_lr = time.time()
    lr.fit(lin_tr[lin_feat_cols], y_train)
    time_lr = time.time() - t0_lr
    p_lr = lr.predict_proba(lin_va[lin_feat_cols])[:, 1]
    m_lr = evaluate_predictions(y_val, p_lr, split_name="val")
    print(f"  Logistic Regression Fit Time: {time_lr:.1f}s | Val PR-AUC: {m_lr['val_pr_auc']:.4f} | ROC-AUC: {m_lr['val_roc_auc']:.4f}")
    
    # =========================================================================
    # 5. Simple Blend 1: LightGBM + Logistic Regression (50/50)
    # =========================================================================
    print("\n--- Simple Blend 1: LightGBM + Logistic Regression (50/50) ---")
    p_blend_lr = 0.5 * p_lgbm + 0.5 * p_lr
    m_blend_lr = evaluate_predictions(y_val, p_blend_lr, split_name="val")
    pb_blend_lr = paired_bootstrap_difference(y_val, p_lgbm, p_blend_lr, val_dates, n_resamples=250, seed=SEED)
    
    print(f"  Blend (LGBM + LR) Val PR-AUC: {m_blend_lr['val_pr_auc']:.4f} | ROC-AUC: {m_blend_lr['val_roc_auc']:.4f} | Brier: {m_blend_lr['val_brier']:.4f}")
    print(f"  Delta vs LightGBM: {pb_blend_lr['delta_pr_auc']['point']:+.5f} (95% CI: [{pb_blend_lr['delta_pr_auc']['ci_lower']:+.5f}, {pb_blend_lr['delta_pr_auc']['ci_upper']:+.5f}])")
    print(f"  Verdict: {pb_blend_lr['verdict'].upper()} (p-value: {pb_blend_lr['delta_pr_auc']['p_value_two_tailed']:.4f})")
    
    results["blend_lgbm_lr"] = {
        "metrics": m_blend_lr,
        "weights": {"lgbm": 0.5, "lr": 0.5},
        "paired_bootstrap_vs_lgbm": pb_blend_lr
    }
    
    # =========================================================================
    # 6. Simple Blend 2: LightGBM + CatBoost (50/50)
    # =========================================================================
    print("\n--- Simple Blend 2: LightGBM + CatBoost Priors (50/50) ---")
    p_blend_cb = 0.5 * p_lgbm + 0.5 * p_cb1
    m_blend_cb = evaluate_predictions(y_val, p_blend_cb, split_name="val")
    pb_blend_cb = paired_bootstrap_difference(y_val, p_lgbm, p_blend_cb, val_dates, n_resamples=250, seed=SEED)
    
    print(f"  Blend (LGBM + CatBoost) Val PR-AUC: {m_blend_cb['val_pr_auc']:.4f} | ROC-AUC: {m_blend_cb['val_roc_auc']:.4f} | Brier: {m_blend_cb['val_brier']:.4f}")
    print(f"  Delta vs LightGBM: {pb_blend_cb['delta_pr_auc']['point']:+.5f} (95% CI: [{pb_blend_cb['delta_pr_auc']['ci_lower']:+.5f}, {pb_blend_cb['delta_pr_auc']['ci_upper']:+.5f}])")
    print(f"  Verdict: {pb_blend_cb['verdict'].upper()} (p-value: {pb_blend_cb['delta_pr_auc']['p_value_two_tailed']:.4f})")
    
    results["blend_lgbm_catboost"] = {
        "metrics": m_blend_cb,
        "weights": {"lgbm": 0.5, "catboost": 0.5},
        "paired_bootstrap_vs_lgbm": pb_blend_cb
    }
    
    # Save overall alternatives summary
    output_summary = {
        "champion_lgbm_baseline": {
            "metrics": m_lgbm
        },
        "alternatives": results
    }
    
    with open("reports/alternatives_metrics.json", "w") as f:
        json.dump(output_summary, f, indent=2)
    print("\n[+] Saved alternatives comparison to reports/alternatives_metrics.json")
    return output_summary

if __name__ == "__main__":
    run_alternatives()
