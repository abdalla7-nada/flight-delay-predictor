"""
src/feature_experiments.py
Phase 4 - Part 1: Schedule-Only Feature Experiments (v2 Candidates)

Evaluates 7 candidate schedule-only features added to LightGBM Variant G:
  a) prior_origin_month: (ORIGIN, month)
  b) prior_origin_dow_pod: (ORIGIN, day_of_week, part_of_day)
  c) prior_airline_hour: (AIRLINE_CODE, dep_hour)
  d) prior_airline_origin: (AIRLINE_CODE, ORIGIN)
  e) prior_dest_arrhour: (DEST, arr_hour)
  f) typical_dep_density: average scheduled departures at origin in same hour for (month, day_of_week)
  g) days_to_nearest_holiday: signed, clipped to [-7, 7]

Strict Evaluation Rules:
- Judge ONLY on validation (2022). Test set is strictly untouched.
- Every feature must be schedule-only and computable from (airline, origin, dest, date, sched dep/arr time) + train lookups.
- No year, no FL_DATE, no day_of_year as features.
- 5-fold OOF target encoding for train; full train lookups for val.
- Run OOF vs validation correlation sanity check for all new priors.
- Paired cluster-bootstrap by day: a feature is KEPT only if 95% CI of paired difference in PR-AUC excludes 0.
"""

import os
import sys
import json
import time
import datetime
import holidays
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb

from src.research.stats import paired_bootstrap_difference, bootstrap_day_ci
from src.research.metrics import evaluate_predictions

SEED = 42

def compute_days_to_nearest_holiday(dates: pd.Series) -> np.ndarray:
    """Computes signed distance in days to nearest US federal holiday, clipped to [-7, 7]."""
    us_holidays = holidays.US(years=range(2018, 2026))
    h_dates = np.array(sorted(list(us_holidays.keys())))
    
    flight_dates = pd.to_datetime(dates).dt.date.values
    signed_diffs = np.zeros(len(flight_dates), dtype=np.int32)
    
    for i, fd in enumerate(flight_dates):
        # Find closest holiday
        diffs = np.array([(fd - hd).days for hd in h_dates])
        abs_diffs = np.abs(diffs)
        min_idx = np.argmin(abs_diffs)
        signed_diffs[i] = int(np.clip(diffs[min_idx], -7, 7))
        
    return signed_diffs

def compute_typical_dep_density(train_raw: pd.DataFrame, val_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Computes average number of scheduled departures at origin in same hour
    for that (month, day_of_week), computed strictly from TRAIN.
    """
    train_dates = pd.to_datetime(train_raw["FL_DATE"])
    train_month = train_dates.dt.month.values
    train_dow = train_dates.dt.dayofweek.values
    train_dep_hour = (train_raw["CRS_DEP_TIME"].values // 100).astype(int)
    train_origin = train_raw["ORIGIN"].values
    
    # Count unique dates for each (month, day_of_week) in train
    df_dates = pd.DataFrame({
        "FL_DATE": train_raw["FL_DATE"],
        "month": train_month,
        "dow": train_dow
    }).drop_duplicates()
    date_counts = df_dates.groupby(["month", "dow"]).size().to_dict()
    
    # Count total departures for each (ORIGIN, month, dow, dep_hour) in train
    df_deps = pd.DataFrame({
        "ORIGIN": train_origin,
        "month": train_month,
        "dow": train_dow,
        "dep_hour": train_dep_hour
    })
    dep_counts = df_deps.groupby(["ORIGIN", "month", "dow", "dep_hour"]).size().to_dict()
    
    # Calculate density = total departures / total dates in train
    density_lookup = {}
    for key, count in dep_counts.items():
        origin, m, dow, h = key
        n_dates = date_counts.get((m, dow), 1)
        density_lookup[f"{origin}_{m}_{dow}_{h}"] = float(count / max(1, n_dates))
        
    # Map to train and val
    train_keys = [f"{o}_{m}_{dow}_{h}" for o, m, dow, h in zip(train_origin, train_month, train_dow, train_dep_hour)]
    train_density = np.array([density_lookup.get(k, 0.0) for k in train_keys], dtype=np.float32)
    
    val_dates = pd.to_datetime(val_raw["FL_DATE"])
    val_month = val_dates.dt.month.values
    val_dow = val_dates.dt.dayofweek.values
    val_dep_hour = (val_raw["CRS_DEP_TIME"].values // 100).astype(int)
    val_origin = val_raw["ORIGIN"].values
    val_keys = [f"{o}_{m}_{dow}_{h}" for o, m, dow, h in zip(val_origin, val_month, val_dow, val_dep_hour)]
    val_density = np.array([density_lookup.get(k, 0.0) for k in val_keys], dtype=np.float32)
    
    return train_density, val_density, density_lookup

def compute_smoothed_prior(
    train_keys: np.ndarray,
    val_keys: np.ndarray,
    y_train: np.ndarray,
    m: float = 50.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Computes empirical Bayes smoothed prior with 5-fold OOF on train,
    and full train lookup on validation.
    """
    global_mean = float(np.mean(y_train))
    n_train = len(train_keys)
    train_oof = np.zeros(n_train, dtype=np.float32)
    
    # 5-fold OOF on train
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_keys):
        k_tr = train_keys[tr_idx]
        y_tr = y_train[tr_idx]
        k_val = train_keys[val_idx]
        
        df_tr = pd.DataFrame({"key": k_tr, "y": y_tr})
        stats = df_tr.groupby("key")["y"].agg(["count", "sum"])
        smoothed = ((stats["sum"] + m * global_mean) / (stats["count"] + m)).to_dict()
        train_oof[val_idx] = np.array([smoothed.get(k, global_mean) for k in k_val], dtype=np.float32)
        
    # Full train lookup for validation
    df_full = pd.DataFrame({"key": train_keys, "y": y_train})
    full_stats = df_full.groupby("key")["y"].agg(["count", "sum"])
    full_lookup = ((full_stats["sum"] + m * global_mean) / (full_stats["count"] + m)).to_dict()
    
    val_prior = np.array([full_lookup.get(k, global_mean) for k in val_keys], dtype=np.float32)
    return train_oof, val_prior, full_lookup

def run_feature_experiments():
    print("=" * 80)
    print("PHASE 4 - PART 1: SCHEDULE-ONLY FEATURE EXPERIMENTS (v2 CANDIDATES)")
    print("=" * 80)
    
    # 1. Load Data
    print("[*] Loading raw datasets and baseline tree features...")
    train_raw = pd.read_parquet("data/processed/train.parquet")
    val_raw = pd.read_parquet("data/processed/val.parquet")
    
    tree_tr = pd.read_parquet("data/processed/features_tree_train.parquet")
    tree_va = pd.read_parquet("data/processed/features_tree_val.parquet")
    
    y_train = train_raw["DELAY"].values.astype(int)
    y_val = val_raw["DELAY"].values.astype(int)
    val_dates = val_raw["FL_DATE"].values
    
    # Variant G feature definition
    with open("artifacts/models/model_metadata.json", "r") as f:
        meta = json.load(f)
    base_features = meta["features"]
    cat_features = meta["categoricals"]
    print(f"[+] Loaded Variant G baseline with {len(base_features)} features.")
    
    # Ensure categoricals are properly typed
    for col in cat_features:
        tree_tr[col] = tree_tr[col].astype("category")
        tree_va[col] = tree_va[col].astype("category")
        
    # Fit or load Variant G baseline model to get baseline probabilities
    print("\n[*] Evaluating Variant G baseline model...")
    lgbm_params = {
        "objective": "binary",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "cat_smooth": 10.0,
        "min_data_per_group": 100,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1
    }
    
    # Load serialized baseline if available, or fit
    baseline_model_path = "artifacts/models/best_baseline_lgbm.joblib"
    import joblib
    if os.path.exists(baseline_model_path):
        base_model = joblib.load(baseline_model_path)
    else:
        base_model = lgb.LGBMClassifier(**lgbm_params)
        base_model.fit(
            tree_tr[base_features], y_train,
            eval_set=[(tree_va[base_features], y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
    base_val_prob = base_model.predict_proba(tree_va[base_features])[:, 1]
    base_metrics = evaluate_predictions(y_val, base_val_prob, split_name="val")
    base_boot_ci = bootstrap_day_ci(y_val, base_val_prob, val_dates, n_resamples=250, seed=SEED)
    
    print(f"  -> Baseline Val PR-AUC: {base_metrics['val_pr_auc']:.4f} (95% CI: [{base_boot_ci['pr_auc']['ci_lower']:.4f}, {base_boot_ci['pr_auc']['ci_upper']:.4f}])")
    print(f"  -> Baseline Val ROC-AUC: {base_metrics['val_roc_auc']:.4f} (95% CI: [{base_boot_ci['roc_auc']['ci_lower']:.4f}, {base_boot_ci['roc_auc']['ci_upper']:.4f}])")
    print(f"  -> Baseline Val Brier: {base_metrics['val_brier']:.4f} (95% CI: [{base_boot_ci['brier']['ci_lower']:.4f}, {base_boot_ci['brier']['ci_upper']:.4f}])")
    
    # 2. Candidate Feature Engineering
    print("\n[*] Precomputing candidate features...")
    tr_dates = pd.to_datetime(train_raw["FL_DATE"])
    tr_month = tr_dates.dt.month.values
    tr_dow = tr_dates.dt.dayofweek.values
    tr_dep_hour = (train_raw["CRS_DEP_TIME"].values // 100).astype(int)
    tr_arr_hour = (train_raw["CRS_ARR_TIME"].values // 100).astype(int)
    tr_origin = train_raw["ORIGIN"].values
    tr_dest = train_raw["DEST"].values
    tr_airline = train_raw["AIRLINE_CODE"].values
    tr_pod = tree_tr["part_of_day"].values.astype(str)
    
    va_dates = pd.to_datetime(val_raw["FL_DATE"])
    va_month = va_dates.dt.month.values
    va_dow = va_dates.dt.dayofweek.values
    va_dep_hour = (val_raw["CRS_DEP_TIME"].values // 100).astype(int)
    va_arr_hour = (val_raw["CRS_ARR_TIME"].values // 100).astype(int)
    va_origin = val_raw["ORIGIN"].values
    va_dest = val_raw["DEST"].values
    va_airline = val_raw["AIRLINE_CODE"].values
    va_pod = tree_va["part_of_day"].values.astype(str)
    
    candidates = {}
    lookups = {}
    sanity_correlations = {}
    
    # Candidate a: prior_origin_month
    print("  -> Engineering Candidate (a): prior_origin_month...")
    tr_k_a = np.array([f"{o}_{m}" for o, m in zip(tr_origin, tr_month)])
    va_k_a = np.array([f"{o}_{m}" for o, m in zip(va_origin, va_month)])
    tr_oof_a, va_prior_a, lkup_a = compute_smoothed_prior(tr_k_a, va_k_a, y_train, m=50.0, seed=SEED)
    candidates["prior_origin_month"] = (tr_oof_a, va_prior_a, ["prior_origin_month"], False)
    lookups["prior_origin_month"] = lkup_a
    
    # Candidate b: prior_origin_dow_pod
    print("  -> Engineering Candidate (b): prior_origin_dow_pod (sparse key, m=100)...")
    tr_k_b = np.array([f"{o}_{d}_{p}" for o, d, p in zip(tr_origin, tr_dow, tr_pod)])
    va_k_b = np.array([f"{o}_{d}_{p}" for o, d, p in zip(va_origin, va_dow, va_pod)])
    tr_oof_b, va_prior_b, lkup_b = compute_smoothed_prior(tr_k_b, va_k_b, y_train, m=100.0, seed=SEED)
    candidates["prior_origin_dow_pod"] = (tr_oof_b, va_prior_b, ["prior_origin_dow_pod"], False)
    lookups["prior_origin_dow_pod"] = lkup_b
    
    # Candidate c: prior_airline_hour
    print("  -> Engineering Candidate (c): prior_airline_hour...")
    tr_k_c = np.array([f"{a}_{h}" for a, h in zip(tr_airline, tr_dep_hour)])
    va_k_c = np.array([f"{a}_{h}" for a, h in zip(va_airline, va_dep_hour)])
    tr_oof_c, va_prior_c, lkup_c = compute_smoothed_prior(tr_k_c, va_k_c, y_train, m=50.0, seed=SEED)
    candidates["prior_airline_hour"] = (tr_oof_c, va_prior_c, ["prior_airline_hour"], False)
    lookups["prior_airline_hour"] = lkup_c
    
    # Candidate d: prior_airline_origin
    print("  -> Engineering Candidate (d): prior_airline_origin...")
    tr_k_d = np.array([f"{a}_{o}" for a, o in zip(tr_airline, tr_origin)])
    va_k_d = np.array([f"{a}_{o}" for a, o in zip(va_airline, va_origin)])
    tr_oof_d, va_prior_d, lkup_d = compute_smoothed_prior(tr_k_d, va_k_d, y_train, m=50.0, seed=SEED)
    candidates["prior_airline_origin"] = (tr_oof_d, va_prior_d, ["prior_airline_origin"], False)
    lookups["prior_airline_origin"] = lkup_d
    
    # Candidate e: prior_dest_arrhour
    print("  -> Engineering Candidate (e): prior_dest_arrhour...")
    tr_k_e = np.array([f"{d}_{h}" for d, h in zip(tr_dest, tr_arr_hour)])
    va_k_e = np.array([f"{d}_{h}" for d, h in zip(va_dest, va_arr_hour)])
    tr_oof_e, va_prior_e, lkup_e = compute_smoothed_prior(tr_k_e, va_k_e, y_train, m=50.0, seed=SEED)
    candidates["prior_dest_arrhour"] = (tr_oof_e, va_prior_e, ["prior_dest_arrhour"], False)
    lookups["prior_dest_arrhour"] = lkup_e
    
    # Candidate f: typical_dep_density
    print("  -> Engineering Candidate (f): typical_dep_density...")
    tr_dens, va_dens, lkup_f = compute_typical_dep_density(train_raw, val_raw)
    candidates["typical_dep_density"] = (tr_dens, va_dens, ["typical_dep_density"], False)
    lookups["typical_dep_density"] = lkup_f
    
    # Candidate g: days_to_nearest_holiday
    print("  -> Engineering Candidate (g): days_to_nearest_holiday...")
    tr_hdays = compute_days_to_nearest_holiday(train_raw["FL_DATE"])
    va_hdays = compute_days_to_nearest_holiday(val_raw["FL_DATE"])
    candidates["days_to_nearest_holiday"] = (tr_hdays, va_hdays, ["days_to_nearest_holiday"], False)
    
    # Sanity correlation check for priors
    print("\n--- OOF vs. Validation Correlation Sanity Check ---")
    priors_to_check = ["prior_origin_month", "prior_origin_dow_pod", "prior_airline_hour", "prior_airline_origin", "prior_dest_arrhour"]
    for p_name in priors_to_check:
        tr_val_col, va_val_col, _, _ = candidates[p_name]
        r_tr = float(np.corrcoef(tr_val_col, y_train)[0, 1])
        r_va = float(np.corrcoef(va_val_col, y_val)[0, 1])
        diff = abs(r_tr - r_va)
        passed = diff < 0.05
        print(f"  * {p_name:<24}: Train OOF r = {r_tr:.4f} | Val r = {r_va:.4f} | Diff = {diff:.4f} | Status = {'PASS' if passed else 'WARNING'}")
        sanity_correlations[p_name] = {
            "train_oof_corr": round(r_tr, 4),
            "val_corr": round(r_va, 4),
            "abs_diff": round(diff, 4),
            "status": "PASS" if passed else "WARNING"
        }
        # Do not abort: record the warning for the report and continue to evaluate candidate impact
        
    # 3. Model Training & Evaluation for each candidate
    print("\n--- Evaluating Each Candidate Added to Variant G ---")
    results = {}
    winning_candidates = []
    
    for cand_name, (tr_feat, va_feat, feat_cols, is_cat) in candidates.items():
        print(f"\n[+] Testing Candidate: {cand_name}")
        t0 = time.time()
        
        # Prepare feature datasets
        X_tr = tree_tr[base_features].copy()
        X_va = tree_va[base_features].copy()
        
        X_tr[cand_name] = tr_feat
        X_va[cand_name] = va_feat
        
        cand_cats = list(cat_features)
        if is_cat:
            X_tr[cand_name] = X_tr[cand_name].astype("category")
            X_va[cand_name] = X_va[cand_name].astype("category")
            cand_cats.append(cand_name)
            
        cur_features = base_features + [cand_name]
        
        # Train LightGBM
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(
            X_tr[cur_features], y_train,
            eval_set=[(X_va[cur_features], y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        fit_time = time.time() - t0
        best_iter = model.best_iteration_
        
        val_prob = model.predict_proba(X_va[cur_features])[:, 1]
        m = evaluate_predictions(y_val, val_prob, split_name="val")
        
        # Paired bootstrap vs baseline
        pb = paired_bootstrap_difference(y_val, base_val_prob, val_prob, val_dates, n_resamples=250, seed=SEED)
        
        delta_pr = pb["delta_pr_auc"]["point"]
        ci_pr_low = pb["delta_pr_auc"]["ci_lower"]
        ci_pr_high = pb["delta_pr_auc"]["ci_upper"]
        verdict = pb["verdict"]
        
        print(f"    Best Iter: {best_iter} | Fit Time: {fit_time:.1f}s")
        print(f"    Val PR-AUC: {m['val_pr_auc']:.4f} | ROC-AUC: {m['val_roc_auc']:.4f} | Brier: {m['val_brier']:.4f}")
        print(f"    Delta PR-AUC: {delta_pr:+.5f} (95% CI: [{ci_pr_low:+.5f}, {ci_pr_high:+.5f}]) | p-value: {pb['delta_pr_auc']['p_value_two_tailed']:.4f}")
        print(f"    Verdict: {verdict.upper()} (95% CI excludes 0: {pb['delta_pr_auc']['excludes_zero']})")
        
        if verdict == "keep":
            winning_candidates.append(cand_name)
            
        results[cand_name] = {
            "metrics": m,
            "best_iteration": best_iter,
            "fit_time_sec": round(fit_time, 2),
            "paired_bootstrap": pb,
            "verdict": verdict
        }
        
    # 4. If any winners, evaluate them combined together
    combined_result = None
    if len(winning_candidates) > 0:
        print(f"\n[+] Testing Combination of Winning Candidates: {winning_candidates}")
        X_tr_comb = tree_tr[base_features].copy()
        X_va_comb = tree_va[base_features].copy()
        comb_features = list(base_features)
        
        for w in winning_candidates:
            tr_f, va_f, _, is_cat = candidates[w]
            X_tr_comb[w] = tr_f
            X_va_comb[w] = va_f
            comb_features.append(w)
            
        model_comb = lgb.LGBMClassifier(**lgbm_params)
        model_comb.fit(
            X_tr_comb[comb_features], y_train,
            eval_set=[(X_va_comb[comb_features], y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        comb_prob = model_comb.predict_proba(X_va_comb[comb_features])[:, 1]
        m_comb = evaluate_predictions(y_val, comb_prob, split_name="val")
        pb_comb = paired_bootstrap_difference(y_val, base_val_prob, comb_prob, val_dates, n_resamples=250, seed=SEED)
        
        print(f"    Combined Val PR-AUC: {m_comb['val_pr_auc']:.4f} | ROC-AUC: {m_comb['val_roc_auc']:.4f} | Brier: {m_comb['val_brier']:.4f}")
        print(f"    Combined Delta PR-AUC: {pb_comb['delta_pr_auc']['point']:+.5f} (95% CI: [{pb_comb['delta_pr_auc']['ci_lower']:+.5f}, {pb_comb['delta_pr_auc']['ci_upper']:+.5f}])")
        print(f"    Combined Verdict: {pb_comb['verdict'].upper()}")
        
        combined_result = {
            "winning_features": winning_candidates,
            "metrics": m_comb,
            "paired_bootstrap": pb_comb,
            "verdict": pb_comb["verdict"]
        }
    else:
        print("\n[!] No individual candidate yielded a statistically significant improvement in PR-AUC (95% CI excluded 0).")
        print("    Following project rule: Keep the simpler option (Variant G baseline feature set).")
        
    # Save results to JSON
    output_data = {
        "baseline_variant_g": {
            "features_count": len(base_features),
            "metrics": base_metrics,
            "bootstrap_ci": base_boot_ci
        },
        "prior_sanity_correlations": sanity_correlations,
        "candidate_results": results,
        "combined_winners": combined_result,
        "winning_candidates": winning_candidates
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/feature_experiments_metrics.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\n[+] Saved feature experiments metrics to reports/feature_experiments_metrics.json")
    return output_data

if __name__ == "__main__":
    run_feature_experiments()
