"""
src/tune_lgbm.py
Phase 4 - Part 2: Hyperparameter Tuning of LightGBM with Optuna

Tuning Setup:
- 40 trials Bayesian optimization using Optuna.
- Explores hyperparameter search space from Phase 3 report:
  num_leaves: [31, 63, 127]
  max_depth: [-1, 6, 8, 10]
  min_child_samples: [100, 200, 500, 1000]
  learning_rate: [0.02, 0.08]
  colsample_bytree: [0.6, 1.0]
  subsample: [0.7, 1.0]
  reg_alpha: [0.0, 5.0]
  reg_lambda: [0.0, 5.0]
- Search executed on 50% training subsample (447,388 flights).
- Early stopping on validation (50 rounds).
- Objective: validation PR-AUC.
- Refits best configuration on 100% of train.
- Evaluates statistical significance vs baseline using paired cluster-bootstrap by day.
- Saves study history and top-5 configurations to reports/.
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
import optuna
from sklearn.metrics import average_precision_score
import lightgbm as lgb

from src.research.stats import paired_bootstrap_difference, bootstrap_day_ci
from src.research.metrics import evaluate_predictions

# Suppress optuna logging clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42

def run_optuna_tuning():
    print("=" * 80)
    print("PHASE 4 - PART 2: LIGHTGBM HYPERPARAMETER TUNING (OPTUNA, 40 TRIALS)")
    print("=" * 80)
    
    # 1. Load Data & Baseline
    print("[*] Loading tree features and baseline model...")
    tree_tr = pd.read_parquet("data/processed/features_tree_train.parquet")
    tree_va = pd.read_parquet("data/processed/features_tree_val.parquet")
    val_raw = pd.read_parquet("data/processed/val.parquet")
    
    with open("artifacts/models/model_metadata.json", "r") as f:
        meta = json.load(f)
    features = meta["features"]
    cat_features = meta["categoricals"]
    
    for c in cat_features:
        tree_tr[c] = tree_tr[c].astype("category")
        tree_va[c] = tree_va[c].astype("category")
        
    y_train = tree_tr["DELAY"].values.astype(int)
    y_val = tree_va["DELAY"].values.astype(int)
    val_dates = val_raw["FL_DATE"].values
    
    # Baseline predictions
    base_model = joblib.load("artifacts/models/best_baseline_lgbm.joblib")
    base_val_prob = base_model.predict_proba(tree_va[features])[:, 1]
    base_metrics = evaluate_predictions(y_val, base_val_prob, split_name="val")
    print(f"  -> Baseline Variant G Val PR-AUC: {base_metrics['val_pr_auc']:.4f}")
    
    # 2. 50% Subsample of Train for Fast Exploration
    print("[*] Drawing 50% random training subsample (seed=42)...")
    np.random.seed(SEED)
    train_indices = np.arange(len(tree_tr))
    subsample_idx = np.random.choice(train_indices, size=int(0.5 * len(train_indices)), replace=False)
    
    X_tr_sub = tree_tr.iloc[subsample_idx][features]
    y_tr_sub = y_train[subsample_idx]
    X_va = tree_va[features]
    print(f"[+] Subsample size: {len(X_tr_sub):,} rows | Validation size: {len(X_va):,} rows")
    
    # 3. Define Optuna Objective
    def objective(trial: optuna.Trial) -> float:
        num_leaves = trial.suggest_categorical("num_leaves", [31, 63, 127])
        max_depth = trial.suggest_categorical("max_depth", [-1, 6, 8, 10])
        min_child_samples = trial.suggest_categorical("min_child_samples", [100, 200, 500, 1000])
        learning_rate = trial.suggest_float("learning_rate", 0.02, 0.08, step=0.01)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1)
        subsample = trial.suggest_float("subsample", 0.7, 1.0, step=0.1)
        reg_alpha = trial.suggest_float("reg_alpha", 0.0, 5.0, step=0.5)
        reg_lambda = trial.suggest_float("reg_lambda", 0.0, 5.0, step=0.5)
        
        params = {
            "objective": "binary",
            "n_estimators": 2000,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "subsample_freq": 1,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "cat_smooth": 10.0,
            "min_data_per_group": 100,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr_sub, y_tr_sub,
            eval_set=[(X_va, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        val_preds = model.predict_proba(X_va)[:, 1]
        score = float(average_precision_score(y_val, val_preds))
        
        trial.set_user_attr("best_iteration", model.best_iteration_)
        return score
        
    print("\n[*] Starting Optuna 40-trial search...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    t0_optuna = time.time()
    for t_idx in range(40):
        t0_trial = time.time()
        trial = study.ask()
        val_score = objective(trial)
        study.tell(trial, val_score)
        t_elapsed = time.time() - t0_trial
        print(f"  Trial {t_idx+1:02d}/40 | Score: {val_score:.5f} | Best so far: {study.best_value:.5f} | ({t_elapsed:.1f}s)")
        
    optuna_total_time = time.time() - t0_optuna
    print(f"\n[+] Optuna optimization completed in {optuna_total_time:.1f}s!")
    print(f"    Best Validation PR-AUC on 50% train subsample: {study.best_value:.5f}")
    print(f"    Best Parameters: {json.dumps(study.best_params, indent=2)}")
    
    # 4. Extract Top 5 Configurations
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    top_5 = []
    for rank, t in enumerate(sorted_trials[:5], 1):
        top_5.append({
            "rank": rank,
            "trial_number": t.number,
            "val_pr_auc_subsample": round(t.value, 5),
            "best_iteration": t.user_attrs.get("best_iteration", None),
            "params": t.params
        })
    print("\n--- Top 5 Hyperparameter Configurations ---")
    for item in top_5:
        print(f"  Rank {item['rank']} | Trial {item['trial_number']:02d} | Val PR-AUC: {item['val_pr_auc_subsample']:.5f} | Params: {item['params']}")
        
    # 5. Refit Best Config on 100% of Train Data
    print("\n[*] Refitting Best Configuration on 100% of Train Data (894,775 rows)...")
    best_params = study.best_params.copy()
    refit_params = {
        "objective": "binary",
        "n_estimators": 2000,
        "learning_rate": best_params["learning_rate"],
        "num_leaves": best_params["num_leaves"],
        "max_depth": best_params["max_depth"],
        "min_child_samples": best_params["min_child_samples"],
        "subsample": best_params["subsample"],
        "subsample_freq": 1,
        "colsample_bytree": best_params["colsample_bytree"],
        "reg_alpha": best_params["reg_alpha"],
        "reg_lambda": best_params["reg_lambda"],
        "cat_smooth": 10.0,
        "min_data_per_group": 100,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1
    }
    
    t0_refit = time.time()
    best_model = lgb.LGBMClassifier(**refit_params)
    best_model.fit(
        tree_tr[features], y_train,
        eval_set=[(tree_va[features], y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    refit_time = time.time() - t0_refit
    best_iter = best_model.best_iteration_
    
    # 6. Evaluate Refitted Model & Compare with Baseline
    tuned_val_prob = best_model.predict_proba(tree_va[features])[:, 1]
    tuned_metrics = evaluate_predictions(y_val, tuned_val_prob, split_name="val")
    tuned_boot_ci = bootstrap_day_ci(y_val, tuned_val_prob, val_dates, n_resamples=250, seed=SEED)
    
    pb = paired_bootstrap_difference(y_val, base_val_prob, tuned_val_prob, val_dates, n_resamples=250, seed=SEED)
    
    print("\n--- Model Performance Comparison (Tuned vs Baseline Variant G) ---")
    print(f"  Baseline Variant G : Val PR-AUC = {base_metrics['val_pr_auc']:.4f} | ROC-AUC = {base_metrics['val_roc_auc']:.4f} | Brier = {base_metrics['val_brier']:.4f}")
    print(f"  Tuned LightGBM     : Val PR-AUC = {tuned_metrics['val_pr_auc']:.4f} | ROC-AUC = {tuned_metrics['val_roc_auc']:.4f} | Brier = {tuned_metrics['val_brier']:.4f}")
    print(f"  Delta PR-AUC       : {pb['delta_pr_auc']['point']:+.5f} (95% CI: [{pb['delta_pr_auc']['ci_lower']:+.5f}, {pb['delta_pr_auc']['ci_upper']:+.5f}])")
    print(f"  Delta ROC-AUC      : {pb['delta_roc_auc']['point']:+.5f} (95% CI: [{pb['delta_roc_auc']['ci_lower']:+.5f}, {pb['delta_roc_auc']['ci_upper']:+.5f}])")
    print(f"  Delta Brier        : {pb['delta_brier']['point']:+.5f} (95% CI: [{pb['delta_brier']['ci_lower']:+.5f}, {pb['delta_brier']['ci_upper']:+.5f}])")
    print(f"  Verdict            : {pb['verdict'].upper()} (Excludes 0: {pb['delta_pr_auc']['excludes_zero']}, p-value: {pb['delta_pr_auc']['p_value_two_tailed']:.4f})")
    
    # Save artifacts and reports
    os.makedirs("reports", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)
    
    joblib.dump(best_model, "artifacts/models/tuned_lgbm_v2.joblib")
    print("[+] Saved tuned model to artifacts/models/tuned_lgbm_v2.joblib")
    
    study_records = []
    for t in study.trials:
        study_records.append({
            "trial_number": t.number,
            "value": round(t.value, 5),
            "best_iteration": t.user_attrs.get("best_iteration", None),
            "params": t.params,
            "datetime": t.datetime_start.isoformat() if t.datetime_start else None
        })
        
    with open("reports/optuna_study.json", "w") as f:
        json.dump(study_records, f, indent=2)
        
    with open("reports/top5_lgbm_configs.json", "w") as f:
        json.dump(top_5, f, indent=2)
        
    tuning_summary = {
        "best_params": best_params,
        "best_iteration": best_iter,
        "refit_time_sec": round(refit_time, 2),
        "optuna_total_time_sec": round(optuna_total_time, 2),
        "baseline_metrics": base_metrics,
        "tuned_metrics": tuned_metrics,
        "tuned_bootstrap_ci": tuned_boot_ci,
        "paired_bootstrap_vs_baseline": pb
    }
    
    with open("reports/lgbm_tuning_summary.json", "w") as f:
        json.dump(tuning_summary, f, indent=2)
        
    print("[+] Saved tuning records and summaries to reports/")
    return tuning_summary

if __name__ == "__main__":
    run_optuna_tuning()
