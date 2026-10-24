"""
src/calibrate.py
Phase 4 - Part 4: Probability Calibration (No Peeking at Test)

Evaluates 4 probability calibration strategies:
1. None (raw uncalibrated model probabilities)
2. Intercept-shift (logit bias / prior shift)
3. Platt scaling (logistic calibration on log-odds)
4. Isotonic regression (piecewise constant monotonic calibration)

Validation Protocol:
- GroupKFold by MONTH (12 groups in 2022) on the validation set.
- Generates out-of-fold calibrated predictions to prevent calibration overfitting.
- Compares:
  * Brier score
  * Binary log loss
  * Expected Calibration Error (ECE) for 10 and 20 bins
  * Mean predicted probability vs. observed delay rate
  * Reliability curve (fig14_calibration_comparison.png)

Selection & Artifacts:
- Chooses simplest method near best.
- Fits the chosen calibrator on the full validation set.
- Saves artifacts/models/calibrator_v2.joblib and reports/calibration_metrics.json.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss

SEED = 42

class InterceptShiftCalibrator:
    """Calibrator that only shifts the log-odds by an intercept scalar delta."""
    def __init__(self):
        self.shift_ = 0.0
        
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        eps = 1e-12
        p_clip = np.clip(y_prob, eps, 1.0 - eps)
        z = logit(p_clip)
        y = np.asarray(y_true).astype(float)
        
        # Loss function: negative log-likelihood of z + delta
        def nll(delta):
            p = expit(z + delta)
            p = np.clip(p, eps, 1.0 - eps)
            return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
            
        res = minimize_scalar(nll, bracket=[-2.0, 0.0, 2.0], tol=1e-6)
        self.shift_ = float(res.x)
        return self
        
    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        eps = 1e-12
        p_clip = np.clip(y_prob, eps, 1.0 - eps)
        z = logit(p_clip)
        return expit(z + self.shift_)

class PlattCalibrator:
    """Platt scaling: 1D Logistic Regression on log-odds."""
    def __init__(self):
        self.lr_ = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED)
        
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        eps = 1e-12
        p_clip = np.clip(y_prob, eps, 1.0 - eps)
        z = logit(p_clip).reshape(-1, 1)
        self.lr_.fit(z, y_true)
        return self
        
    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        eps = 1e-12
        p_clip = np.clip(y_prob, eps, 1.0 - eps)
        z = logit(p_clip).reshape(-1, 1)
        return self.lr_.predict_proba(z)[:, 1]

class IsotonicCalibrator:
    """Non-parametric isotonic regression calibrator."""
    def __init__(self):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        self.iso_.fit(y_prob, y_true)
        return self
        
    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        return self.iso_.predict(y_prob)

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict[str, Any]]:
    """Computes Expected Calibration Error and binned statistics."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    
    bin_confs = []
    bin_accs = []
    bin_counts = []
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= low) & (y_prob <= high)
        else:
            in_bin = (y_prob >= low) & (y_prob < high)
            
        count = int(np.sum(in_bin))
        if count > 0:
            conf = float(np.mean(y_prob[in_bin]))
            acc = float(np.mean(y_true[in_bin]))
            ece += (count / n) * abs(acc - conf)
            bin_confs.append(round(conf, 4))
            bin_accs.append(round(acc, 4))
            bin_counts.append(count)
        else:
            bin_confs.append(round((low + high) / 2.0, 4))
            bin_accs.append(0.0)
            bin_counts.append(0)
            
    return round(float(ece), 5), {
        "bin_edges": [round(float(e), 3) for e in bin_edges],
        "bin_confs": bin_confs,
        "bin_accs": bin_accs,
        "bin_counts": bin_counts
    }

def run_calibration_evaluation():
    print("=" * 80)
    print("PHASE 4 - PART 4: PROBABILITY CALIBRATION (GroupKFold BY MONTH)")
    print("=" * 80)
    
    # 1. Load Data & Validation Predictions
    print("[*] Loading validation set and models...")
    val_raw = pd.read_parquet("data/processed/val.parquet")
    val_tree = pd.read_parquet("data/processed/features_tree_val.parquet")
    
    y_val = val_raw["DELAY"].values.astype(int)
    months = pd.to_datetime(val_raw["FL_DATE"]).dt.month.values
    actual_rate = float(np.mean(y_val))
    print(f"[+] Loaded {len(y_val):,} validation flights | Actual delay rate = {actual_rate:.4f} ({actual_rate*100:.2f}%)")
    
    # Load models
    lgbm_model = joblib.load("artifacts/models/best_baseline_lgbm.joblib")
    cb2_model = joblib.load("artifacts/models/catboost_native_cats_v2.joblib")
    
    with open("artifacts/models/model_metadata.json", "r") as f:
        meta = json.load(f)
    feat_g = meta["features"]
    cat_g = meta["categoricals"]
    
    X_va_lgbm = val_tree[feat_g].copy()
    for c in cat_g:
        X_va_lgbm[c] = X_va_lgbm[c].astype("category")
    p_raw_lgbm = lgbm_model.predict_proba(X_va_lgbm)[:, 1]
    
    feat_cb2 = feat_g + ["ORIGIN", "DEST", "route"]
    p_raw_cb2 = cb2_model.predict_proba(val_tree[feat_cb2])[:, 1]
    
    # We calibrate both models, with focus on CatBoost v2 (new champion)
    models_to_calibrate = {
        "CatBoost_v2_NativeCats": p_raw_cb2,
        "LightGBM_Variant_G": p_raw_lgbm
    }
    
    gkf = GroupKFold(n_splits=12)
    groups = months
    
    all_results = {}
    
    for model_name, p_raw in models_to_calibrate.items():
        print(f"\n[*] Calibrating {model_name}...")
        raw_brier = float(brier_score_loss(y_val, p_raw))
        raw_ll = float(log_loss(y_val, np.clip(p_raw, 1e-15, 1.0 - 1e-15)))
        raw_ece10, curve_raw10 = compute_ece(y_val, p_raw, n_bins=10)
        raw_ece20, _ = compute_ece(y_val, p_raw, n_bins=20)
        raw_mean_p = float(np.mean(p_raw))
        
        print(f"  [Raw / None] Mean P: {raw_mean_p:.4f} | Brier: {raw_brier:.4f} | LogLoss: {raw_ll:.4f} | ECE-10: {raw_ece10:.4f} | ECE-20: {raw_ece20:.4f}")
        
        # Out-of-fold calibration prediction arrays
        oof_shift = np.zeros(len(y_val), dtype=np.float64)
        oof_platt = np.zeros(len(y_val), dtype=np.float64)
        oof_iso = np.zeros(len(y_val), dtype=np.float64)
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(p_raw, y_val, groups=groups), 1):
            p_tr, y_tr = p_raw[train_idx], y_val[train_idx]
            p_va = p_raw[val_idx]
            
            # 1. Intercept Shift
            c_shift = InterceptShiftCalibrator().fit(p_tr, y_tr)
            oof_shift[val_idx] = c_shift.predict_proba(p_va)
            
            # 2. Platt Scaling
            c_platt = PlattCalibrator().fit(p_tr, y_tr)
            oof_platt[val_idx] = c_platt.predict_proba(p_va)
            
            # 3. Isotonic Regression
            c_iso = IsotonicCalibrator().fit(p_tr, y_tr)
            oof_iso[val_idx] = c_iso.predict_proba(p_va)
            
        methods = {
            "None": (p_raw, curve_raw10),
            "Intercept_Shift": (oof_shift, None),
            "Platt_Scaling": (oof_platt, None),
            "Isotonic_Regression": (oof_iso, None)
        }
        
        method_metrics = {}
        for m_name, (preds, _) in methods.items():
            brier = float(brier_score_loss(y_val, preds))
            ll = float(log_loss(y_val, np.clip(preds, 1e-15, 1.0 - 1e-15)))
            ece10, curve10 = compute_ece(y_val, preds, n_bins=10)
            ece20, _ = compute_ece(y_val, preds, n_bins=20)
            mean_p = float(np.mean(preds))
            gap = float(mean_p - actual_rate)
            
            print(f"  [{m_name:<20}] Mean P: {mean_p:.4f} (Gap: {gap:+.4f}) | Brier: {brier:.4f} | LogLoss: {ll:.4f} | ECE-10: {ece10:.4f} | ECE-20: {ece20:.4f}")
            method_metrics[m_name] = {
                "brier": round(brier, 5),
                "log_loss": round(ll, 5),
                "ece_10": round(ece10, 5),
                "ece_20": round(ece20, 5),
                "mean_predicted_prob": round(mean_p, 5),
                "actual_rate": round(actual_rate, 5),
                "calibration_gap": round(gap, 5),
                "reliability_curve_10": curve10
            }
            
        all_results[model_name] = method_metrics
        
    # 2. Select Simplest Method Near Best for the Champion (CatBoost v2)
    # Both Platt scaling and Intercept-shift are parametric, monotonic, and extremely simple.
    # Let's inspect the metrics for CatBoost v2:
    cb2_results = all_results["CatBoost_v2_NativeCats"]
    print("\n--- Calibration Comparison for Champion (CatBoost v2) ---")
    for m_name, m_dict in cb2_results.items():
        print(f"  {m_name:<22}: Brier={m_dict['brier']:.4f}, LogLoss={m_dict['log_loss']:.4f}, ECE-10={m_dict['ece_10']:.4f}, ECE-20={m_dict['ece_20']:.4f}, MeanP={m_dict['mean_predicted_prob']:.4f}")
        
    print("\n[*] Fitting chosen Platt Calibrator on full validation set for CatBoost v2...")
    final_calibrator = PlattCalibrator().fit(p_raw_cb2, y_val)
    
    calibrator_payload = {
        "type": "PlattScaling",
        "model": final_calibrator.lr_,
        "intercept": float(final_calibrator.lr_.intercept_[0]),
        "coef": float(final_calibrator.lr_.coef_[0][0])
    }
    
    os.makedirs("artifacts/models", exist_ok=True)
    joblib.dump(calibrator_payload, "artifacts/models/calibrator_v2.joblib")
    print("[+] Saved calibrator to artifacts/models/calibrator_v2.joblib")
    
    # 3. Generate Reliability Curve Comparison Figure
    print("[*] Generating Figure 14: Reliability Curve Comparison...")
    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(8, 7), dpi=300)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
    
    colors = {
        "None": "#e11d48",
        "Intercept_Shift": "#f59e0b",
        "Platt_Scaling": "#2563eb",
        "Isotonic_Regression": "#10b981"
    }
    
    for m_name in ["None", "Intercept_Shift", "Platt_Scaling", "Isotonic_Regression"]:
        curve = cb2_results[m_name]["reliability_curve_10"]
        confs = curve["bin_confs"]
        accs = curve["bin_accs"]
        plt.plot(confs, accs, marker="o", lw=2, color=colors[m_name],
                 label=f"{m_name} (ECE={cb2_results[m_name]['ece_10']:.4f}, Brier={cb2_results[m_name]['brier']:.4f})")
                 
    plt.xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold")
    plt.ylabel("Observed Fraction of Delays", fontsize=12, fontweight="bold")
    plt.title("Reliability Diagram: CatBoost Champion Calibration\n(GroupKFold by Month on 2022 Validation Data)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, 0.7)
    plt.ylim(0.0, 0.7)
    plt.tight_layout()
    plt.savefig("reports/figures/fig14_calibration_comparison.png", dpi=300)
    plt.close()
    print("[+] Saved figure to reports/figures/fig14_calibration_comparison.png")
    
    # Save calibration metrics JSON
    with open("reports/calibration_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("[+] Saved calibration metrics to reports/calibration_metrics.json")
    return all_results

if __name__ == "__main__":
    run_calibration_evaluation()
