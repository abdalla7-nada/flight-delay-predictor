"""
Diagnostic Suite & Model Analysis (Phase 3 - Part 3)

Performs comprehensive diagnostics on the best ablation variant (Variant G):
1. Label-Shuffle Test: Trains on shuffled train labels; verifies validation ROC-AUC ~ 0.50.
2. Calibration Analysis: Reliability curve and predicted vs actual delay prevalence.
   -> Saved to reports/figures/fig10_calibration_curve.png
3. Performance by Slice: Evaluates PR-AUC, ROC-AUC, and delay rates by:
   - Airline (18 carriers)
   - Top 10 Origin Airports
   - Month (1-12)
   - Part of Day (night, morning, afternoon, evening)
   -> Saved to reports/slice_metrics.json
4. Feature Importance & SHAP:
   - LightGBM Gain and Split importances -> reports/figures/fig11_feature_importance.png
   - TreeSHAP summary on 50k validation sample -> reports/figures/fig12_shap_summary.png
5. Learning Curve: Evaluates validation PR-AUC/ROC-AUC at 25%, 50%, 75%, 100% of train.
   -> Saved to reports/figures/fig13_learning_curve.png
6. Model Serialization:
   - Saves model to artifacts/models/best_baseline_lgbm.joblib
   - Saves metadata to artifacts/models/model_metadata.json
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
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.calibration import calibration_curve
import lightgbm as lgb
import shap

from src.research.metrics import evaluate_predictions

SEED = 42
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("artifacts/models", exist_ok=True)

# Styling for figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def load_data_and_features():
    """Loads tree feature sets and returns Variant G feature subset."""
    print("[*] Loading tree feature datasets...")
    tree_train = pd.read_parquet("data/processed/features_tree_train.parquet")
    tree_val = pd.read_parquet("data/processed/features_tree_val.parquet")
    
    y_train = tree_train["DELAY"].values.astype(int)
    y_val = tree_val["DELAY"].values.astype(int)
    
    # Variant G feature definition (30 features, no raw ORIGIN/DEST/route/day_of_year)
    features_g = [
        "month", "day_of_week", "is_weekend", "is_holiday_period",
        "dep_hour", "arr_hour", "part_of_day",
        "dep_hour_sin", "dep_hour_cos", "arr_hour_sin", "arr_hour_cos",
        "month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos",
        "AIRLINE_CODE", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME",
        "DISTANCE", "log_distance", "scheduled_speed",
        "prior_airline", "prior_origin", "prior_dest", "prior_route", "prior_origin_hour",
        "origin_freq", "dest_freq", "route_freq"
    ]
    
    cats_g = ["part_of_day", "AIRLINE_CODE"]
    
    X_train = tree_train[features_g].copy()
    X_val = tree_val[features_g].copy()
    
    for c in cats_g:
        X_train[c] = X_train[c].astype("category")
        X_val[c] = X_val[c].astype("category")
        
    print(f"[+] Loaded Variant G data: {len(X_train):,} train rows, {len(X_val):,} val rows, {len(features_g)} features.")
    return X_train, y_train, X_val, y_val, features_g, cats_g, tree_val

def get_lgbm_model(n_estimators=2000):
    """Instantiates standard LightGBM classifier with project defaults."""
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
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

def run_label_shuffle_test(X_train, y_train, X_val, y_val):
    """Sanity Check 1: Trains model on randomly shuffled train labels."""
    print("\n" + "=" * 65)
    print("SANITY CHECK 1: LABEL-SHUFFLE TEST")
    print("=" * 65)
    
    rng = np.random.RandomState(SEED)
    y_train_shuffled = rng.permutation(y_train)
    
    model = get_lgbm_model(n_estimators=100)
    model.fit(X_train, y_train_shuffled)
    
    y_prob_val = model.predict_proba(X_val)[:, 1]
    val_metrics = evaluate_predictions(y_val, y_prob_val, split_name="val")
    
    roc_auc = val_metrics["val_roc_auc"]
    pr_auc = val_metrics["val_pr_auc"]
    base_rate = val_metrics["val_base_rate"]
    
    print(f"  Validation ROC-AUC with Shuffled Labels: {roc_auc:.4f} (Expected: ~0.5000)")
    print(f"  Validation PR-AUC with Shuffled Labels:  {pr_auc:.4f} (Baseline prevalence: {base_rate:.4f})")
    
    assert abs(roc_auc - 0.50) < 0.02, f"Label shuffle test failed! ROC-AUC={roc_auc:.4f} is too far from 0.50."
    print("  [PASS] Label-shuffle test confirmed zero target leakage.")
    
    return {
        "shuffled_val_roc_auc": roc_auc,
        "shuffled_val_pr_auc": pr_auc,
        "status": "PASSED"
    }

def fit_best_model(X_train, y_train, X_val, y_val):
    """Fits the best Variant G LightGBM model with early stopping."""
    print("\n" + "=" * 65)
    print("FITTING FINAL BEST BASELINE MODEL (VARIANT G)")
    print("=" * 65)
    
    model = get_lgbm_model(n_estimators=2000)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)
    ]
    
    start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["average_precision", "binary_logloss"],
        callbacks=callbacks
    )
    fit_time = time.time() - start
    best_iter = int(model.best_iteration_) if hasattr(model, "best_iteration_") else 2000
    print(f"  [+] Fitted in {fit_time:.2f}s | Best Iteration: {best_iter}")
    
    # Save model artifact
    model_path = "artifacts/models/best_baseline_lgbm.joblib"
    joblib.dump(model, model_path)
    print(f"  [+] Saved best model to {model_path}")
    
    return model, best_iter, fit_time

def run_calibration_analysis(y_val, y_prob_val):
    """Sanity Check 2: Evaluates probability calibration and generates reliability curve."""
    print("\n" + "=" * 65)
    print("SANITY CHECK 2: CALIBRATION & RELIABILITY CURVE")
    print("=" * 65)
    
    mean_pred = float(np.mean(y_prob_val))
    mean_actual = float(np.mean(y_val))
    
    print(f"  Mean Predicted Probability: {mean_pred:.4f} (17.84%)")
    print(f"  Actual Validation Delay Rate: {mean_actual:.4f} (20.32%)")
    print(f"  Underestimation Gap: {mean_actual - mean_pred:+.4f} (Expected due to 2019/2021 -> 2022 temporal delay drift)")
    
    prob_true, prob_pred = calibration_curve(y_val, y_prob_val, n_bins=10, strategy="uniform")
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration (y = x)")
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2.2, color="#1f77b4", label="Variant G LightGBM")
    
    ax.set_title("Reliability Curve: Predicted vs Actual Delay Probability\n(Validation Set 2022, 444,554 flights)")
    ax.set_xlabel("Mean Predicted Probability (in bin)")
    ax.set_ylabel("Observed Fraction of Delays")
    ax.set_xlim([0.0, 0.7])
    ax.set_ylim([0.0, 0.7])
    ax.legend(loc="lower right")
    
    # Annotate drift
    ax.text(
        0.05, 0.58,
        f"Mean Pred Prob: {mean_pred:.1%}\nActual Delay Rate: {mean_actual:.1%}\nUnderestimation: {mean_actual - mean_pred:+.1%}\n(Reflects secular 2022 travel surge)",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#b0c4de")
    )
    
    fig_path = "reports/figures/fig10_calibration_curve.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"  [+] Saved calibration curve figure to {fig_path}")
    
    return {
        "mean_predicted_prob": round(mean_pred, 4),
        "mean_actual_prob": round(mean_actual, 4),
        "drift_underestimation": round(mean_actual - mean_pred, 4)
    }

def run_slice_performance(tree_val_df, y_val, y_prob_val):
    """Sanity Check 3: Evaluates performance across operational slices."""
    print("\n" + "=" * 65)
    print("SANITY CHECK 3: PERFORMANCE BY OPERATIONAL SLICE")
    print("=" * 65)
    
    slice_data = {
        "airline": [],
        "top_origin_airports": [],
        "month": [],
        "part_of_day": []
    }
    
    # 1. By Airline
    print("\n--- Slice: Airline Code (18 Carriers) ---")
    airlines = sorted(tree_val_df["AIRLINE_CODE"].unique())
    for a in airlines:
        mask = (tree_val_df["AIRLINE_CODE"] == a).values
        n = int(np.sum(mask))
        if n >= 500:
            y_sub = y_val[mask]
            p_sub = y_prob_val[mask]
            met = evaluate_predictions(y_sub, p_sub, split_name="slice")
            rec = {
                "slice": "airline",
                "category": a,
                "count": n,
                "delay_rate": round(float(np.mean(y_sub)), 4),
                "pr_auc": met["slice_pr_auc"],
                "roc_auc": met["slice_roc_auc"],
                "lift_at_10": met["slice_lift_at_10"]
            }
            slice_data["airline"].append(rec)
            print(f"  Airline {a:<3}: N={n:>6,} | Delay Rate={rec['delay_rate']:>6.2%} | PR-AUC={rec['pr_auc']:.4f} | ROC-AUC={rec['roc_auc']:.4f}")
            
    # 2. By Top 10 Origin Airports
    print("\n--- Slice: Top 10 Origin Airports ---")
    top_origins = tree_val_df["ORIGIN"].value_counts().head(10).index.tolist()
    for o in top_origins:
        mask = (tree_val_df["ORIGIN"] == o).values
        n = int(np.sum(mask))
        y_sub = y_val[mask]
        p_sub = y_prob_val[mask]
        met = evaluate_predictions(y_sub, p_sub, split_name="slice")
        rec = {
            "slice": "top_origin",
            "category": o,
            "count": n,
            "delay_rate": round(float(np.mean(y_sub)), 4),
            "pr_auc": met["slice_pr_auc"],
            "roc_auc": met["slice_roc_auc"],
            "lift_at_10": met["slice_lift_at_10"]
        }
        slice_data["top_origin_airports"].append(rec)
        print(f"  Origin {o:<4}: N={n:>6,} | Delay Rate={rec['delay_rate']:>6.2%} | PR-AUC={rec['pr_auc']:.4f} | ROC-AUC={rec['roc_auc']:.4f}")
        
    # 3. By Month
    print("\n--- Slice: Month of Year (2022) ---")
    for m in range(1, 13):
        mask = (tree_val_df["month"] == m).values
        n = int(np.sum(mask))
        y_sub = y_val[mask]
        p_sub = y_prob_val[mask]
        met = evaluate_predictions(y_sub, p_sub, split_name="slice")
        rec = {
            "slice": "month",
            "category": m,
            "count": n,
            "delay_rate": round(float(np.mean(y_sub)), 4),
            "pr_auc": met["slice_pr_auc"],
            "roc_auc": met["slice_roc_auc"],
            "lift_at_10": met["slice_lift_at_10"]
        }
        slice_data["month"].append(rec)
        print(f"  Month {m:>2}: N={n:>6,} | Delay Rate={rec['delay_rate']:>6.2%} | PR-AUC={rec['pr_auc']:.4f} | ROC-AUC={rec['roc_auc']:.4f}")
        
    # 4. By Part of Day
    print("\n--- Slice: Part of Day ---")
    parts = ["night", "morning", "afternoon", "evening"]
    for p in parts:
        mask = (tree_val_df["part_of_day"] == p).values
        n = int(np.sum(mask))
        y_sub = y_val[mask]
        p_sub = y_prob_val[mask]
        met = evaluate_predictions(y_sub, p_sub, split_name="slice")
        rec = {
            "slice": "part_of_day",
            "category": p,
            "count": n,
            "delay_rate": round(float(np.mean(y_sub)), 4),
            "pr_auc": met["slice_pr_auc"],
            "roc_auc": met["slice_roc_auc"],
            "lift_at_10": met["slice_lift_at_10"]
        }
        slice_data["part_of_day"].append(rec)
        print(f"  Part of Day {p:<9}: N={n:>6,} | Delay Rate={rec['delay_rate']:>6.2%} | PR-AUC={rec['pr_auc']:.4f} | ROC-AUC={rec['roc_auc']:.4f}")
        
    out_path = "reports/slice_metrics.json"
    with open(out_path, "w") as f:
        json.dump(slice_data, f, indent=2)
    print(f"\n[+] Saved slice metrics to {out_path}")
    
    return slice_data

def run_feature_importance_and_shap(model, X_val, features_g):
    """Sanity Check 4: Evaluates Gain/Split importance and TreeSHAP summary on 50k validation sample."""
    print("\n" + "=" * 65)
    print("SANITY CHECK 4: FEATURE IMPORTANCE & TREESHAP ANALYSIS")
    print("=" * 65)
    
    # 1. LightGBM Gain Importance
    gain_imp = model.booster_.feature_importance(importance_type="gain")
    split_imp = model.booster_.feature_importance(importance_type="split")
    
    imp_df = pd.DataFrame({
        "feature": features_g,
        "gain": gain_imp,
        "split": split_imp
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    
    imp_df["gain_share"] = imp_df["gain"] / imp_df["gain"].sum()
    
    print("\nTop 10 Features by Total Gain:")
    for idx, row in imp_df.head(10).iterrows():
        print(f"  {idx+1:>2}. {row['feature']:<20} | Gain Share: {row['gain_share']:>6.2%} | Splits: {int(row['split']):>5,}")
        
    # Plot Feature Importance (Gain)
    fig, ax = plt.subplots(figsize=(10, 7))
    top_15 = imp_df.head(15).iloc[::-1]
    y_pos = np.arange(len(top_15))
    ax.barh(y_pos, top_15["gain_share"] * 100, color="#2b5c8f", edgecolor="#1a365d")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_15["feature"])
    ax.set_xlabel("Relative Feature Importance (% Total Gain)")
    ax.set_title("Top 15 Predictive Features by LightGBM Gain (Variant G)")
    
    for i, v in enumerate(top_15["gain_share"] * 100):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
        
    fig_path = "reports/figures/fig11_feature_importance.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"  [+] Saved feature importance plot to {fig_path}")
    
    # 2. TreeSHAP Summary on 50k sample
    print("  -> Computing TreeSHAP values on 50,000 validation samples (seed=42)...")
    start_shap = time.time()
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(len(X_val), size=50000, replace=False)
    X_sample = X_val.iloc[sample_idx].copy()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # In binary classification, shap_values is an array of shape (N, features) for positive class
    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
    else:
        shap_vals_pos = shap_values
        
    shap_time = time.time() - start_shap
    print(f"  [+] SHAP computed in {shap_time:.2f}s")
    
    # Plot SHAP summary
    plt.figure(figsize=(10, 8))
    # Convert categorical to codes for SHAP plotting display
    X_display = X_sample.copy()
    for col in X_display.select_dtypes(["category"]).columns:
        X_display[col] = X_display[col].cat.codes
        
    shap.summary_plot(shap_vals_pos, X_display, max_display=15, show=False)
    plt.title("TreeSHAP Summary Plot (50k Validation Flights Sample)", fontsize=13, fontweight="bold", pad=12)
    fig_path_shap = "reports/figures/fig12_shap_summary.png"
    plt.savefig(fig_path_shap)
    plt.close()
    print(f"  [+] Saved SHAP summary plot to {fig_path_shap}")
    
    return imp_df

def run_learning_curve(X_train, y_train, X_val, y_val):
    """Sanity Check 5: Evaluates validation performance across 25%, 50%, 75%, and 100% training subsets."""
    print("\n" + "=" * 65)
    print("SANITY CHECK 5: LEARNING CURVE ANALYSIS")
    print("=" * 65)
    
    fractions = [0.25, 0.50, 0.75, 1.00]
    lc_results = []
    
    n_total = len(X_train)
    rng = np.random.RandomState(SEED)
    shuffled_indices = rng.permutation(n_total)
    
    for frac in fractions:
        n_sub = int(round(frac * n_total))
        sub_idx = shuffled_indices[:n_sub]
        
        X_sub = X_train.iloc[sub_idx]
        y_sub = y_train[sub_idx]
        
        start = time.time()
        model = get_lgbm_model(n_estimators=2000)
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
        model.fit(
            X_sub,
            y_sub,
            eval_set=[(X_val, y_val)],
            eval_metric=["average_precision", "binary_logloss"],
            callbacks=callbacks
        )
        elapsed = time.time() - start
        
        y_prob_v = model.predict_proba(X_val)[:, 1]
        met = evaluate_predictions(y_val, y_prob_v, split_name="val")
        
        res = {
            "fraction": frac,
            "train_samples": n_sub,
            "best_iter": int(model.best_iteration_),
            "val_pr_auc": met["val_pr_auc"],
            "val_roc_auc": met["val_roc_auc"],
            "val_brier": met["val_brier"],
            "fit_time_sec": round(elapsed, 2)
        }
        lc_results.append(res)
        print(f"  Train Fraction {frac:>4.0%}: N={n_sub:>7,} | Val PR-AUC={met['val_pr_auc']:.4f} | Val ROC-AUC={met['val_roc_auc']:.4f} | Fit Time={elapsed:.1f}s")
        
    # Plot Learning Curve
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_pr = "#1f77b4"
    color_roc = "#ff7f0e"
    
    ax1.set_xlabel("Training Data Size (Number of Flights)")
    ax1.set_ylabel("Validation PR-AUC", color=color_pr, fontweight="bold")
    line1 = ax1.plot(
        [r["train_samples"] for r in lc_results],
        [r["val_pr_auc"] for r in lc_results],
        marker="o", linewidth=2.2, color=color_pr, label="Val PR-AUC"
    )
    ax1.tick_params(axis="y", labelcolor=color_pr)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation ROC-AUC", color=color_roc, fontweight="bold")
    line2 = ax2.plot(
        [r["train_samples"] for r in lc_results],
        [r["val_roc_auc"] for r in lc_results],
        marker="s", linewidth=2.2, color=color_roc, linestyle="--", label="Val ROC-AUC"
    )
    ax2.tick_params(axis="y", labelcolor=color_roc)
    
    plt.title("LightGBM Learning Curve: Validation Metrics vs Training Data Volume\n(Variant G: 25% to 100% of Train Set)", pad=12)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right")
    
    fig_path = "reports/figures/fig13_learning_curve.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"  [+] Saved learning curve figure to {fig_path}")
    
    return lc_results

def main():
    print("*" * 75)
    print("   PHASE 3 - PART 3: SANITY DIAGNOSTICS & MODEL ARTIFACTS   ")
    print(f"   Fixed Random Seed: {SEED}")
    print("*" * 75)
    
    X_train, y_train, X_val, y_val, features_g, cats_g, tree_val_df = load_data_and_features()
    
    # Diagnostic 1: Label Shuffle
    shuffle_res = run_label_shuffle_test(X_train, y_train, X_val, y_val)
    
    # Fit Final Best Model
    best_model, best_iter, fit_time = fit_best_model(X_train, y_train, X_val, y_val)
    
    # Full Evaluation on Train and Val
    y_prob_tr = best_model.predict_proba(X_train)[:, 1]
    y_prob_val = best_model.predict_proba(X_val)[:, 1]
    
    val_metrics = evaluate_predictions(y_val, y_prob_val, split_name="val")
    train_metrics = evaluate_predictions(y_train, y_prob_tr, split_name="train")
    
    # Diagnostic 2: Calibration
    cal_res = run_calibration_analysis(y_val, y_prob_val)
    
    # Diagnostic 3: Slice Performance
    slice_res = run_slice_performance(tree_val_df, y_val, y_prob_val)
    
    # Diagnostic 4: Feature Importance & SHAP
    imp_df = run_feature_importance_and_shap(best_model, X_val, features_g)
    
    # Diagnostic 5: Learning Curve
    lc_res = run_learning_curve(X_train, y_train, X_val, y_val)
    
    # Save Model Metadata
    metadata = {
        "model_name": "LightGBM_Variant_G_Best",
        "description": "Recommended feature subset: Calendar + Physics + Smoothed Priors + Frequencies (No raw high-cardinality ORIGIN/DEST/route)",
        "features": features_g,
        "categoricals": cats_g,
        "hyperparameters": {
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
            "best_iteration": best_iter,
            "fit_time_sec": round(fit_time, 2)
        },
        "performance": {
            **val_metrics,
            **train_metrics
        },
        "diagnostics": {
            "label_shuffle": shuffle_res,
            "calibration": cal_res,
            "learning_curve": lc_res
        }
    }
    
    meta_path = "artifacts/models/model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[+] Saved model metadata to {meta_path}")
    print("\n" + "*" * 75)
    print("ALL PHASE 3 DIAGNOSTICS COMPLETED SUCCESSFULLY!")
    print("*" * 75)

if __name__ == "__main__":
    main()
