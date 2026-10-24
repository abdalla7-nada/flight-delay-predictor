"""
src/final_test_eval.py
Phase 5 - Step 2 & 3: One-Time Final Evaluation on the 2023 Test Set

Strict Execution Rules:
- Scored exactly ONCE by this single script.
- No tuning, no feature changes, no re-fitting, no threshold search on test.
- Transform with FlightFeaturePipeline v2, score with Champion CatBoost v2, calibrate with Platt Calibrator v2.
- 250 day-clustered bootstrap CIs by test FL_DATE.
- Compare with Baselines: Constant Train Rate Dummy, Logistic Regression, LightGBM Variant G (v1), Champion Raw (Uncalibrated).
- Generate decile risk table, slice diagnostics (>= 3,000 flights), drift analysis, oracle calibration diagnostic.
- Check unseen categories in test and their flight counts.
- Save predictions to artifacts/test_predictions.parquet.
- Save comprehensive test metrics to reports/test_evaluation_metrics.json.
- Generate high-resolution figures in reports/figures/:
  * fig15_test_reliability_curve.png
  * fig16_test_decile_lift.png
  * fig17_test_roc_pr_curves.png
  * fig18_test_monthly_performance.png
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
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    roc_curve,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression

from src.features import FlightFeaturePipeline
from src.research.stats import paired_bootstrap_difference, bootstrap_day_ci
from src.research.metrics import evaluate_predictions
from src.calibrate import compute_ece, PlattCalibrator, InterceptShiftCalibrator

SEED = 42

def compute_decile_table(y_true: np.ndarray, y_prob_cal: np.ndarray) -> pd.DataFrame:
    """Constructs 10 risk deciles sorted from lowest to highest predicted risk."""
    df = pd.DataFrame({"y": y_true, "prob": y_prob_cal})
    df["decile"] = pd.qcut(df["prob"], q=10, labels=False) + 1
    
    base_rate = float(np.mean(y_true))
    deciles = []
    
    for d in range(1, 11):
        sub = df[df["decile"] == d]
        n_flights = len(sub)
        mean_prob = float(sub["prob"].mean())
        obs_rate = float(sub["y"].mean())
        lift = float(obs_rate / base_rate) if base_rate > 0 else 1.0
        p_min = float(sub["prob"].min())
        p_max = float(sub["prob"].max())
        cal_ratio = float(mean_prob / obs_rate) if obs_rate > 0 else 1.0
        
        deciles.append({
            "decile": d,
            "flight_count": n_flights,
            "prob_min": round(p_min, 4),
            "prob_max": round(p_max, 4),
            "mean_pred_prob": round(mean_prob, 4),
            "observed_delay_rate": round(obs_rate, 4),
            "lift": round(lift, 2),
            "calibration_ratio": round(cal_ratio, 3)
        })
        
    return pd.DataFrame(deciles)

def run_test_evaluation():
    print("=" * 80)
    print("PHASE 5: ONE-TIME FINAL EVALUATION ON 2023 TEST SET")
    print("=" * 80)
    
    t0_all = time.time()
    
    # 1. Load Datasets & Pipeline
    print("[*] Ingesting 2023 test set and loading frozen Feature Pipeline v2...")
    test_raw = pd.read_parquet("data/processed/test.parquet")
    n_test = len(test_raw)
    y_test = test_raw["DELAY"].values.astype(int)
    test_dates = test_raw["FL_DATE"].values
    test_base_rate = float(np.mean(y_test))
    
    print(f"[+] Loaded {n_test:,} completed test flights across {len(np.unique(test_dates))} days (Jan–Aug 2023).")
    print(f"    Observed Test Base Delay Rate: {test_base_rate:.4f} ({test_base_rate*100:.2f}%)")
    
    pipeline = joblib.load("artifacts/feature_pipeline_v2.joblib")
    print("[*] Transforming test dataset through Feature Pipeline v2...")
    tree_test, lin_test = pipeline.transform(test_raw)
    
    # 2. Check for Unseen Categories in Test Set
    print("\n[*] Auditing Unseen Categories in Test Set against Training...")
    train_raw = pd.read_parquet("data/processed/train.parquet")
    known_airports = set(train_raw["ORIGIN"].unique()).union(set(train_raw["DEST"].unique()))
    known_airlines = set(train_raw["AIRLINE_CODE"].unique())
    known_routes = set((train_raw["ORIGIN"] + "_" + train_raw["DEST"]).unique())
    
    test_origins = set(test_raw["ORIGIN"].unique())
    test_dests = set(test_raw["DEST"].unique())
    test_airports = test_origins.union(test_dests)
    test_airlines = set(test_raw["AIRLINE_CODE"].unique())
    test_routes = set((test_raw["ORIGIN"] + "_" + test_raw["DEST"]).unique())
    
    unseen_airports = sorted(list(test_airports - known_airports))
    unseen_airlines = sorted(list(test_airlines - known_airlines))
    unseen_routes = sorted(list(test_routes - known_routes))
    
    # Count flights affected
    flights_unseen_orig = int(test_raw["ORIGIN"].isin(unseen_airports).sum())
    flights_unseen_dest = int(test_raw["DEST"].isin(unseen_airports).sum())
    flights_unseen_airline = int(test_raw["AIRLINE_CODE"].isin(unseen_airlines).sum())
    test_route_series = test_raw["ORIGIN"] + "_" + test_raw["DEST"]
    flights_unseen_route = int(test_route_series.isin(unseen_routes).sum())
    
    print(f"  - Unseen Airlines : {len(unseen_airlines)} ({flights_unseen_airline:,} flights, {flights_unseen_airline/n_test*100:.3f}%)")
    print(f"  - Unseen Airports : {len(unseen_airports)} (Orig: {flights_unseen_orig:,}, Dest: {flights_unseen_dest:,})")
    print(f"  - Unseen Routes   : {len(unseen_routes):,} routes ({flights_unseen_route:,} flights, {flights_unseen_route/n_test*100:.2f}%)")
    
    unseen_audit = {
        "unseen_airlines_count": len(unseen_airlines),
        "unseen_airlines_list": unseen_airlines,
        "flights_unseen_airline": flights_unseen_airline,
        "unseen_airports_count": len(unseen_airports),
        "unseen_airports_list": unseen_airports,
        "flights_unseen_origin": flights_unseen_orig,
        "flights_unseen_dest": flights_unseen_dest,
        "unseen_routes_count": len(unseen_routes),
        "flights_unseen_route": flights_unseen_route,
        "flights_unseen_route_pct": round(flights_unseen_route / n_test * 100, 2)
    }
    
    # 3. Score with Frozen Champion Model & Calibrator
    print("\n[*] Scoring with Champion Model (CatBoost v2 Native Cats)...")
    champion_model = joblib.load("artifacts/models/champion_v2.joblib")
    calibrator_payload = joblib.load("artifacts/models/calibrator_v2.joblib")
    
    with open("artifacts/models/model_metadata_v2.json", "r") as f:
        meta_v2 = json.load(f)
    champ_features = meta_v2["features"]
    
    p_champ_raw = champion_model.predict_proba(tree_test[champ_features])[:, 1]
    
    # Apply Platt Calibrator
    print("[*] Applying Frozen Platt Calibrator...")
    z_test = logit(np.clip(p_champ_raw, 1e-12, 1.0 - 1e-12)).reshape(-1, 1)
    p_champ_cal = calibrator_payload["model"].predict_proba(z_test)[:, 1]
    
    # Sanity Leakage & Anomaly Check
    test_roc = float(roc_auc_score(y_test, p_champ_cal))
    print(f"\n[+] Champion Test ROC-AUC = {test_roc:.4f}")
    assert test_roc >= 0.62, f"ANOMALY WARNING: Test ROC-AUC {test_roc:.4f} < 0.62 indicates potential pipeline bug!"
    assert test_roc <= 0.75, f"LEAKAGE WARNING: Test ROC-AUC {test_roc:.4f} > 0.75 indicates potential post-departure leakage!"
    print("    [PASS] Verification confirmed: Performance is within realistic schedule-only range (0.62 - 0.75).")
    
    # Save test predictions to parquet
    print("\n[*] Saving test predictions to artifacts/test_predictions.parquet...")
    pred_df = pd.DataFrame({
        "FL_DATE": test_dates,
        "AIRLINE_CODE": test_raw["AIRLINE_CODE"],
        "ORIGIN": test_raw["ORIGIN"],
        "DEST": test_raw["DEST"],
        "CRS_DEP_TIME": test_raw["CRS_DEP_TIME"],
        "CRS_ARR_TIME": test_raw["CRS_ARR_TIME"],
        "DELAY": y_test,
        "prob_raw": np.round(p_champ_raw, 5),
        "prob_calibrated": np.round(p_champ_cal, 5)
    })
    pred_df.to_parquet("artifacts/test_predictions.parquet", index=False)
    print(f"[+] Saved {len(pred_df):,} test predictions to artifacts/test_predictions.parquet")
    
    # 4. Compute Test Metrics with 250 Day-Clustered Bootstrap CIs
    print("\n[*] Computing 250 Day-Clustered Bootstrap CIs for Champion (Calibrated)...")
    champ_boot_ci = bootstrap_day_ci(y_test, p_champ_cal, test_dates, n_resamples=250, seed=SEED)
    champ_metrics = evaluate_predictions(y_test, p_champ_cal, split_name="test")
    
    # Calibration metrics on test
    ece10, curve10 = compute_ece(y_test, p_champ_cal, n_bins=10)
    ece20, _ = compute_ece(y_test, p_champ_cal, n_bins=20)
    mean_pred_p = float(np.mean(p_champ_cal))
    cal_bias = float(mean_pred_p - test_base_rate)
    
    # Brier Skill Scores
    brier_ref_test = test_base_rate * (1.0 - test_base_rate) # 0.1729
    brier_ref_train = float(brier_score_loss(y_test, np.full_like(y_test, 0.1762, dtype=float)))
    bss_test = 1.0 - (champ_metrics["test_brier"] / brier_ref_test)
    bss_train = 1.0 - (champ_metrics["test_brier"] / brier_ref_train)
    
    print("\n" + "=" * 50)
    print("CHAMPION MODEL PERFORMANCE ON 2023 TEST SET")
    print("=" * 50)
    print(f"  PR-AUC               : {champ_metrics['test_pr_auc']:.4f} (95% CI: [{champ_boot_ci['pr_auc']['ci_lower']:.4f}, {champ_boot_ci['pr_auc']['ci_upper']:.4f}])")
    print(f"  ROC-AUC              : {champ_metrics['test_roc_auc']:.4f} (95% CI: [{champ_boot_ci['roc_auc']['ci_lower']:.4f}, {champ_boot_ci['roc_auc']['ci_upper']:.4f}])")
    print(f"  Brier Score          : {champ_metrics['test_brier']:.4f} (95% CI: [{champ_boot_ci['brier']['ci_lower']:.4f}, {champ_boot_ci['brier']['ci_upper']:.4f}])")
    print(f"  Binary Log Loss      : {champ_metrics['test_log_loss']:.4f}")
    print(f"  BSS (vs Test Rate)   : {bss_test:+.4f} ({bss_test*100:+.2f}%) [Reference Brier = {brier_ref_test:.4f}]")
    print(f"  BSS (vs Train Rate)  : {bss_train:+.4f} ({bss_train*100:+.2f}%) [Reference Brier = {brier_ref_train:.4f}]")
    print(f"  ECE (10 Bins)        : {ece10:.4f}")
    print(f"  ECE (20 Bins)        : {ece20:.4f}")
    print(f"  Mean Predicted Prob  : {mean_pred_p:.4f} (Observed Base Rate: {test_base_rate:.4f}, Bias: {cal_bias:+.4f})")
    print(f"  Top 10% Precision    : {champ_metrics['test_prec_at_10']*100:.2f}% (Lift: {champ_metrics['test_lift_at_10']:.2f}x)")
    
    # 5. Evaluate Baselines on Exact Same Test Flights
    print("\n[*] Evaluating Comparative Baselines on 2023 Test Set...")
    # Baseline 1: Constant Train Rate Dummy
    p_dummy = np.full(n_test, 0.1762, dtype=float)
    m_dummy = evaluate_predictions(y_test, p_dummy, split_name="test_dummy")
    pb_dummy = paired_bootstrap_difference(y_test, p_dummy, p_champ_cal, test_dates, n_resamples=250, seed=SEED)
    
    # Baseline 2: Logistic Regression (trained on train linear set)
    print("  -> Training / Loading Logistic Regression baseline...")
    lin_tr = pd.read_parquet("data/processed/features_linear_train.parquet")
    lin_cols = [c for c in lin_tr.columns if c != "DELAY"]
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    lr_model.fit(lin_tr[lin_cols], train_raw["DELAY"].values.astype(int))
    p_lr = lr_model.predict_proba(lin_test[lin_cols])[:, 1]
    m_lr = evaluate_predictions(y_test, p_lr, split_name="test_lr")
    pb_lr = paired_bootstrap_difference(y_test, p_lr, p_champ_cal, test_dates, n_resamples=250, seed=SEED)
    
    # Baseline 3: LightGBM Variant G (v1 Artifact)
    print("  -> Loading LightGBM Variant G (v1 artifact)...")
    lgbm_v1 = joblib.load("artifacts/models/best_baseline_lgbm.joblib")
    with open("artifacts/models/model_metadata.json", "r") as f:
        meta_v1 = json.load(f)
    feat_g = meta_v1["features"]
    cat_g = meta_v1["categoricals"]
    tree_test_lgbm = tree_test[feat_g].copy()
    for c in cat_g:
        tree_test_lgbm[c] = tree_test_lgbm[c].astype("category")
    p_lgbm_v1 = lgbm_v1.predict_proba(tree_test_lgbm)[:, 1]
    m_lgbm_v1 = evaluate_predictions(y_test, p_lgbm_v1, split_name="test_lgbm_v1")
    pb_lgbm_v1 = paired_bootstrap_difference(y_test, p_lgbm_v1, p_champ_cal, test_dates, n_resamples=250, seed=SEED)
    
    # Baseline 4: Champion Before Calibration (Raw CatBoost)
    m_raw = evaluate_predictions(y_test, p_champ_raw, split_name="test_champ_raw")
    pb_raw = paired_bootstrap_difference(y_test, p_champ_raw, p_champ_cal, test_dates, n_resamples=250, seed=SEED)
    
    baselines_comparison = {
        "Dummy_Constant_Train_Rate": {
            "metrics": m_dummy,
            "paired_diff_vs_champion": pb_dummy
        },
        "Logistic_Regression": {
            "metrics": m_lr,
            "paired_diff_vs_champion": pb_lr
        },
        "LightGBM_Variant_G_v1": {
            "metrics": m_lgbm_v1,
            "paired_diff_vs_champion": pb_lgbm_v1
        },
        "Champion_Uncalibrated_Raw": {
            "metrics": m_raw,
            "paired_diff_vs_champion": pb_raw
        }
    }
    
    print("\n--- Comparative Models Summary on 2023 Test Set ---")
    print(f"  Dummy Constant Rate  : PR-AUC = {m_dummy['test_dummy_pr_auc']:.4f} | ROC-AUC = {m_dummy['test_dummy_roc_auc']:.4f} | Brier = {m_dummy['test_dummy_brier']:.4f}")
    print(f"  Logistic Regression  : PR-AUC = {m_lr['test_lr_pr_auc']:.4f} | ROC-AUC = {m_lr['test_lr_roc_auc']:.4f} | Brier = {m_lr['test_lr_brier']:.4f}")
    print(f"  LightGBM Variant G   : PR-AUC = {m_lgbm_v1['test_lgbm_v1_pr_auc']:.4f} | ROC-AUC = {m_lgbm_v1['test_lgbm_v1_roc_auc']:.4f} | Brier = {m_lgbm_v1['test_lgbm_v1_brier']:.4f}")
    print(f"  Champion Raw (Uncal) : PR-AUC = {m_raw['test_champ_raw_pr_auc']:.4f} | ROC-AUC = {m_raw['test_champ_raw_roc_auc']:.4f} | Brier = {m_raw['test_champ_raw_brier']:.4f}")
    print(f"  Champion (Calibrated): PR-AUC = {champ_metrics['test_pr_auc']:.4f} | ROC-AUC = {champ_metrics['test_roc_auc']:.4f} | Brier = {champ_metrics['test_brier']:.4f}")
    print(f"  Delta PR-AUC (Champ vs LightGBM v1): {pb_lgbm_v1['delta_pr_auc']['point']:+.5f} (95% CI: [{pb_lgbm_v1['delta_pr_auc']['ci_lower']:+.5f}, {pb_lgbm_v1['delta_pr_auc']['ci_upper']:+.5f}], p={pb_lgbm_v1['delta_pr_auc']['p_value_two_tailed']:.4f})")
    
    # 6. Decile Risk Table on Test
    print("\n[*] Generating Decile Risk Table on 2023 Test Set...")
    decile_df_test = compute_decile_table(y_test, p_champ_cal)
    print(decile_df_test.to_string(index=False))
    decile_df_test.to_json("reports/decile_table_test.json", orient="records", indent=2)
    
    # 7. Drift & Oracle Calibration Diagnostic (Upper Bound Diagnostic Only)
    print("\n[*] Running Oracle Calibration Diagnostic on Test Set...")
    # How much would fitting Platt scaling or intercept shift ON TEST improve ECE/Brier?
    oracle_shift = InterceptShiftCalibrator().fit(p_champ_raw, y_test)
    p_oracle_shift = oracle_shift.predict_proba(p_champ_raw)
    
    oracle_platt = PlattCalibrator().fit(p_champ_raw, y_test)
    p_oracle_platt = oracle_platt.predict_proba(p_champ_raw)
    
    brier_oracle_shift = float(brier_score_loss(y_test, p_oracle_shift))
    ece10_oracle_shift, _ = compute_ece(y_test, p_oracle_shift, n_bins=10)
    
    brier_oracle_platt = float(brier_score_loss(y_test, p_oracle_platt))
    ece10_oracle_platt, _ = compute_ece(y_test, p_oracle_platt, n_bins=10)
    
    oracle_diag = {
        "uncalibrated_raw": {
            "mean_pred_prob": round(float(np.mean(p_champ_raw)), 4),
            "brier": round(float(brier_score_loss(y_test, p_champ_raw)), 4),
            "ece_10": round(float(compute_ece(y_test, p_champ_raw, 10)[0]), 4)
        },
        "frozen_platt_2022_calibrator": {
            "mean_pred_prob": round(mean_pred_p, 4),
            "brier": round(champ_metrics["test_brier"], 4),
            "ece_10": round(ece10, 4)
        },
        "oracle_intercept_shift_upper_bound": {
            "mean_pred_prob": round(float(np.mean(p_oracle_shift)), 4),
            "brier": round(brier_oracle_shift, 4),
            "ece_10": round(ece10_oracle_shift, 4),
            "delta_brier": round(brier_oracle_shift - champ_metrics["test_brier"], 5)
        },
        "oracle_platt_upper_bound": {
            "mean_pred_prob": round(float(np.mean(p_oracle_platt)), 4),
            "brier": round(brier_oracle_platt, 4),
            "ece_10": round(ece10_oracle_platt, 4),
            "delta_brier": round(brier_oracle_platt - champ_metrics["test_brier"], 5)
        }
    }
    print(f"  Uncalibrated Raw Test      : Mean P = {oracle_diag['uncalibrated_raw']['mean_pred_prob']:.4f} | ECE-10 = {oracle_diag['uncalibrated_raw']['ece_10']:.4f} | Brier = {oracle_diag['uncalibrated_raw']['brier']:.4f}")
    print(f"  Frozen Platt 2022 (Actual) : Mean P = {oracle_diag['frozen_platt_2022_calibrator']['mean_pred_prob']:.4f} | ECE-10 = {oracle_diag['frozen_platt_2022_calibrator']['ece_10']:.4f} | Brier = {oracle_diag['frozen_platt_2022_calibrator']['brier']:.4f}")
    print(f"  Oracle Platt (Theoretical) : Mean P = {oracle_diag['oracle_platt_upper_bound']['mean_pred_prob']:.4f} | ECE-10 = {oracle_diag['oracle_platt_upper_bound']['ece_10']:.4f} | Brier = {oracle_diag['oracle_platt_upper_bound']['brier']:.4f}")
    
    # 8. Monthly Performance Analysis on Test Set (Jan–Aug 2023)
    print("\n[*] Computing Monthly Diagnostics on 2023 Test Set...")
    test_months = pd.DatetimeIndex(test_dates).month
    monthly_test = []
    
    for m in range(1, 9):
        mask = (test_months == m)
        n_m = int(np.sum(mask))
        y_m = y_test[mask]
        p_m = p_champ_cal[mask]
        rate_m = float(np.mean(y_m))
        mean_p_m = float(np.mean(p_m))
        roc_m = float(roc_auc_score(y_m, p_m))
        pr_m = float(average_precision_score(y_m, p_m))
        k10_m = max(1, int(round(0.10 * n_m)))
        top10_idx = np.argsort(p_m)[-k10_m:]
        lift10_m = float(np.mean(y_m[top10_idx]) / rate_m) if rate_m > 0 else 1.0
        cal_ratio_m = float(mean_p_m / rate_m) if rate_m > 0 else 1.0
        
        monthly_test.append({
            "month": m,
            "flight_count": n_m,
            "base_rate": round(rate_m, 4),
            "mean_pred_prob": round(mean_p_m, 4),
            "calibration_ratio": round(cal_ratio_m, 3),
            "roc_auc": round(roc_m, 4),
            "pr_auc": round(pr_m, 4),
            "top_10_lift": round(lift10_m, 2)
        })
        print(f"  Month {m:02d} (N={n_m:,}) | Base Rate: {rate_m:.4f} | Pred: {mean_p_m:.4f} (Ratio: {cal_ratio_m:.3f}) | ROC: {roc_m:.4f} | PR: {pr_m:.4f} | Lift: {lift10_m:.2f}x")
        
    # 9. Slices by Airline, Top-10 Airports, and Part of Day (Min 3,000 flights)
    print("\n[*] Computing Granular Slices on Test Set (Minimum 3,000 flights)...")
    slice_min_thresh = 3000
    test_slices = {}
    
    # Slice A: Airline
    airline_slices = []
    for a in sorted(test_raw["AIRLINE_CODE"].unique()):
        mask = (test_raw["AIRLINE_CODE"] == a).values
        n = int(np.sum(mask))
        if n >= slice_min_thresh:
            y_s = y_test[mask]
            p_s = p_champ_cal[mask]
            rate = float(np.mean(y_s))
            roc = float(roc_auc_score(y_s, p_s)) if len(np.unique(y_s)) > 1 else 0.5
            k10 = max(1, int(round(0.10 * n)))
            top10_rate = float(np.mean(y_s[np.argsort(p_s)[-k10:]]))
            lift10 = float(top10_rate / rate) if rate > 0 else 1.0
            airline_slices.append({
                "carrier": a,
                "flight_count": n,
                "observed_delay_rate": round(rate, 4),
                "roc_auc": round(roc, 4),
                "top_10_lift": round(lift10, 2)
            })
    test_slices["airline_slices"] = sorted(airline_slices, key=lambda x: x["flight_count"], reverse=True)
    
    # Slice B: Top 10 Origin Airports
    origin_counts = test_raw["ORIGIN"].value_counts()
    top10_origins = origin_counts.head(10).index.tolist()
    origin_slices = []
    for o in top10_origins:
        mask = (test_raw["ORIGIN"] == o).values
        n = int(np.sum(mask))
        if n >= slice_min_thresh:
            y_s = y_test[mask]
            p_s = p_champ_cal[mask]
            rate = float(np.mean(y_s))
            roc = float(roc_auc_score(y_s, p_s)) if len(np.unique(y_s)) > 1 else 0.5
            k10 = max(1, int(round(0.10 * n)))
            top10_rate = float(np.mean(y_s[np.argsort(p_s)[-k10:]]))
            lift10 = float(top10_rate / rate) if rate > 0 else 1.0
            origin_slices.append({
                "origin": o,
                "flight_count": n,
                "observed_delay_rate": round(rate, 4),
                "roc_auc": round(roc, 4),
                "top_10_lift": round(lift10, 2)
            })
    test_slices["origin_slices"] = origin_slices
    
    # Slice C: Part of Day
    pod_series = tree_test["part_of_day"].values
    pod_slices = []
    for p in ["night", "morning", "afternoon", "evening"]:
        mask = (pod_series == p)
        n = int(np.sum(mask))
        if n >= slice_min_thresh:
            y_s = y_test[mask]
            p_s = p_champ_cal[mask]
            rate = float(np.mean(y_s))
            roc = float(roc_auc_score(y_s, p_s)) if len(np.unique(y_s)) > 1 else 0.5
            k10 = max(1, int(round(0.10 * n)))
            top10_rate = float(np.mean(y_s[np.argsort(p_s)[-k10:]]))
            lift10 = float(top10_rate / rate) if rate > 0 else 1.0
            pod_slices.append({
                "part_of_day": p,
                "flight_count": n,
                "observed_delay_rate": round(rate, 4),
                "roc_auc": round(roc, 4),
                "top_10_lift": round(lift10, 2)
            })
    test_slices["part_of_day_slices"] = pod_slices
    
    # 10. Generate High-Resolution Diagnostic Figures in reports/figures/
    print("\n[*] Generating High-Resolution Figures in reports/figures/...")
    os.makedirs("reports/figures", exist_ok=True)
    
    # Figure 15: Test Reliability Curve
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
    plt.plot(curve10["bin_confs"], curve10["bin_accs"], marker="o", lw=2, color="#2563eb",
             label=f"Champion (ECE-10 = {ece10:.4f}, Brier = {champ_metrics['test_brier']:.4f})")
    plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Observed Delay Frequency", fontsize=11, fontweight="bold")
    plt.title("Reliability Curve on 2023 Test Set (304,458 flights)", fontsize=12, fontweight="bold")
    plt.xlim(0.0, 0.7)
    plt.ylim(0.0, 0.7)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("reports/figures/fig15_test_reliability_curve.png", dpi=300)
    plt.close()
    print("  -> Saved reports/figures/fig15_test_reliability_curve.png")
    
    # Figure 16: Decile Lift Bar Chart
    plt.figure(figsize=(9, 5), dpi=300)
    deciles_x = np.arange(1, 11)
    observed_pct = decile_df_test["observed_delay_rate"].values * 100
    predicted_pct = decile_df_test["mean_pred_prob"].values * 100
    bar_width = 0.35
    
    plt.bar(deciles_x - bar_width/2, observed_pct, width=bar_width, label="Observed Delay Rate (%)", color="#1e3a8a")
    plt.bar(deciles_x + bar_width/2, predicted_pct, width=bar_width, label="Mean Predicted Probability (%)", color="#38bdf8")
    plt.axhline(test_base_rate * 100, color="#ef4444", linestyle="--", label=f"Average Test Base Rate ({test_base_rate*100:.1f}%)")
    plt.xlabel("Predicted Risk Decile (1 = Lowest Risk, 10 = Highest Risk)", fontsize=11, fontweight="bold")
    plt.ylabel("Delay Rate (%)", fontsize=11, fontweight="bold")
    plt.title("Risk Stratification: 10-Decile Observed vs. Predicted Delay Rate on 2023 Test Data", fontsize=12, fontweight="bold")
    plt.xticks(deciles_x, [f"D{d}\n({decile_df_test['lift'].values[d-1]:.2f}x)" for d in deciles_x])
    plt.legend(loc="upper left", frameon=True)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/fig16_test_decile_lift.png", dpi=300)
    plt.close()
    print("  -> Saved reports/figures/fig16_test_decile_lift.png")
    
    # Figure 17: ROC and PR Curves comparing Baselines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    # ROC Curves
    fpr_champ, tpr_champ, _ = roc_curve(y_test, p_champ_cal)
    fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, p_lgbm_v1)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, p_lr)
    
    ax1.plot(fpr_champ, tpr_champ, color="#2563eb", lw=2, label=f"Champion CatBoost (AUC = {champ_metrics['test_roc_auc']:.4f})")
    ax1.plot(fpr_lgbm, tpr_lgbm, color="#10b981", lw=1.8, linestyle="-.", label=f"LightGBM v1 (AUC = {m_lgbm_v1['test_lgbm_v1_roc_auc']:.4f})")
    ax1.plot(fpr_lr, tpr_lr, color="#f59e0b", lw=1.8, linestyle=":", label=f"Logistic Regression (AUC = {m_lr['test_lr_roc_auc']:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Chance Baseline (AUC = 0.5000)")
    ax1.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax1.set_title("Test ROC Curves", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # PR Curves
    rec_champ, prec_champ, _ = precision_recall_curve(y_test, p_champ_cal)
    rec_lgbm, prec_lgbm, _ = precision_recall_curve(y_test, p_lgbm_v1)
    rec_lr, prec_lr, _ = precision_recall_curve(y_test, p_lr)
    
    ax2.plot(rec_champ, prec_champ, color="#2563eb", lw=2, label=f"Champion CatBoost (PR-AUC = {champ_metrics['test_pr_auc']:.4f})")
    ax2.plot(rec_lgbm, prec_lgbm, color="#10b981", lw=1.8, linestyle="-.", label=f"LightGBM v1 (PR-AUC = {m_lgbm_v1['test_lgbm_v1_pr_auc']:.4f})")
    ax2.plot(rec_lr, prec_lr, color="#f59e0b", lw=1.8, linestyle=":", label=f"Logistic Regression (PR-AUC = {m_lr['test_lr_pr_auc']:.4f})")
    ax2.axhline(test_base_rate, color="k", linestyle="--", lw=1, label=f"Chance Baseline (Prevalence = {test_base_rate:.4f})")
    ax2.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax2.set_title("Test Precision-Recall Curves", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reports/figures/fig17_test_roc_pr_curves.png", dpi=300)
    plt.close()
    print("  -> Saved reports/figures/fig17_test_roc_pr_curves.png")
    
    # Figure 18: Monthly Performance Plot (Jan–Aug 2023)
    m_x = [d["month"] for d in monthly_test]
    m_roc = [d["roc_auc"] for d in monthly_test]
    m_lift = [d["top_10_lift"] for d in monthly_test]
    m_rate = [d["base_rate"] * 100 for d in monthly_test]
    
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    color1 = "#2563eb"
    ax1.set_xlabel("Month of 2023 (Jan – Aug)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("ROC-AUC / Lift Ratio", color=color1, fontsize=11, fontweight="bold")
    line1 = ax1.plot(m_x, m_roc, marker="o", color=color1, lw=2, label="Monthly ROC-AUC")
    line2 = ax1.plot(m_x, m_lift, marker="s", color="#10b981", lw=2, linestyle="-.", label="Top 10% Lift")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(m_x)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = "#ef4444"
    ax2.set_ylabel("Observed Delay Rate (%)", color=color2, fontsize=11, fontweight="bold")
    line3 = ax2.plot(m_x, m_rate, marker="^", color=color2, lw=2, linestyle=":", label="Observed Delay Rate (%)")
    ax2.tick_params(axis="y", labelcolor=color2)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon=True)
    plt.title("Champion Performance & Delay Rate Trend Across 2023 Test Months", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/fig18_test_monthly_performance.png", dpi=300)
    plt.close()
    print("  -> Saved reports/figures/fig18_test_monthly_performance.png")
    
    # 11. Save Full Test Metrics JSON
    total_time = time.time() - t0_all
    test_metrics_payload = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_flight_count": n_test,
        "test_days_count": len(np.unique(test_dates)),
        "test_base_rate": round(test_base_rate, 4),
        "execution_time_sec": round(total_time, 2),
        "champion_test_metrics": champ_metrics,
        "champion_test_bootstrap_ci": champ_boot_ci,
        "brier_skill_scores": {
            "bss_vs_test_base_rate": round(bss_test, 4),
            "bss_vs_train_base_rate": round(bss_train, 4),
            "reference_brier_test": round(brier_ref_test, 4),
            "reference_brier_train": round(brier_ref_train, 4)
        },
        "calibration_test": {
            "ece_10": round(ece10, 4),
            "ece_20": round(ece20, 4),
            "mean_predicted_prob": round(mean_pred_p, 4),
            "calibration_bias": round(cal_bias, 4)
        },
        "decile_table_test": decile_df_test.to_dict(orient="records"),
        "baselines_comparison": baselines_comparison,
        "oracle_calibration_diagnostic": oracle_diag,
        "monthly_diagnostics": monthly_test,
        "slice_diagnostics": test_slices,
        "unseen_categories_audit": unseen_audit
    }
    
    with open("reports/test_evaluation_metrics.json", "w") as f:
        json.dump(test_metrics_payload, f, indent=2)
    print("\n[+] Successfully saved test_evaluation_metrics.json")
    print(f"[*] Total Phase 5 test evaluation completed in {total_time:.1f} seconds.")
    return test_metrics_payload

if __name__ == "__main__":
    run_test_evaluation()
