"""
src/stats.py
Statistical Yardstick and Bootstrap Validation Module (Phase 4 - Part 0)

Provides:
- Cluster-bootstrap confidence intervals for validation metrics (resampling by DAY).
- Paired cluster-bootstrap testing for metric differences between two models.
- Statistical significance evaluation: a change is "real" only if the 95% CI
  of the paired difference in PR-AUC excludes 0.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss
)

def compute_metrics_fast(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes core validation metrics: PR-AUC, ROC-AUC, Brier score."""
    # Guard if constant predictions
    if len(np.unique(y_prob)) <= 1:
        pr_auc = float(np.mean(y_true))
        roc_auc = 0.5
    else:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier
    }

def bootstrap_day_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dates: np.ndarray,
    n_resamples: int = 250,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Computes bootstrap confidence intervals for PR-AUC, ROC-AUC, and Brier,
    resampling clusters by DAY (flights on the same day are correlated).
    
    Args:
        y_true: 1D array of ground truth labels
        y_prob: 1D array of predicted probabilities
        dates: 1D array of flight dates (FL_DATE or day strings)
        n_resamples: Number of bootstrap iterations (>= 200)
        seed: Random seed for reproducibility
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        Dict containing point estimates, 95% CIs, and standard errors.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    dates = np.asarray(dates)
    
    # Pre-index flight rows by unique date
    unique_dates = np.unique(dates)
    n_days = len(unique_dates)
    day_indices = [np.where(dates == d)[0] for d in unique_dates]
    
    # Point estimates on full dataset
    point_est = compute_metrics_fast(y_true, y_prob)
    
    boot_pr_auc = np.zeros(n_resamples, dtype=np.float64)
    boot_roc_auc = np.zeros(n_resamples, dtype=np.float64)
    boot_brier = np.zeros(n_resamples, dtype=np.float64)
    
    for b in range(n_resamples):
        sampled_day_idx = rng.choice(n_days, size=n_days, replace=True)
        # Concatenate rows from sampled days
        sampled_rows = np.concatenate([day_indices[i] for i in sampled_day_idx])
        
        y_b = y_true[sampled_rows]
        p_b = y_prob[sampled_rows]
        
        m = compute_metrics_fast(y_b, p_b)
        boot_pr_auc[b] = m["pr_auc"]
        boot_roc_auc[b] = m["roc_auc"]
        boot_brier[b] = m["brier"]
        
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    
    return {
        "pr_auc": {
            "point": round(point_est["pr_auc"], 4),
            "ci_lower": round(float(np.percentile(boot_pr_auc, low_pct)), 4),
            "ci_upper": round(float(np.percentile(boot_pr_auc, high_pct)), 4),
            "std_err": round(float(np.std(boot_pr_auc)), 4)
        },
        "roc_auc": {
            "point": round(point_est["roc_auc"], 4),
            "ci_lower": round(float(np.percentile(boot_roc_auc, low_pct)), 4),
            "ci_upper": round(float(np.percentile(boot_roc_auc, high_pct)), 4),
            "std_err": round(float(np.std(boot_roc_auc)), 4)
        },
        "brier": {
            "point": round(point_est["brier"], 4),
            "ci_lower": round(float(np.percentile(boot_brier, low_pct)), 4),
            "ci_upper": round(float(np.percentile(boot_brier, high_pct)), 4),
            "std_err": round(float(np.std(boot_brier)), 4)
        },
        "n_resamples": n_resamples,
        "n_days": n_days,
        "n_samples": len(y_true)
    }

def paired_bootstrap_difference(
    y_true: np.ndarray,
    y_prob_base: np.ndarray,
    y_prob_cand: np.ndarray,
    dates: np.ndarray,
    n_resamples: int = 250,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Computes paired cluster-bootstrap differences (Candidate - Baseline)
    resampling clusters by DAY.
    
    Evaluates rule: change is "real" only if the 95% CI of the paired
    difference in PR-AUC excludes 0 (specifically lower_ci > 0 for improvement).
    
    Args:
        y_true: 1D array of ground truth labels
        y_prob_base: 1D array of baseline model predicted probabilities
        y_prob_cand: 1D array of candidate model predicted probabilities
        dates: 1D array of flight dates (FL_DATE or day strings)
        n_resamples: Number of bootstrap iterations (>= 200)
        seed: Random seed for reproducibility
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        Dict containing delta point estimates, 95% CIs of differences,
        and statistical significance verdict.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob_base = np.asarray(y_prob_base).astype(float)
    y_prob_cand = np.asarray(y_prob_cand).astype(float)
    dates = np.asarray(dates)
    
    unique_dates = np.unique(dates)
    n_days = len(unique_dates)
    day_indices = [np.where(dates == d)[0] for d in unique_dates]
    
    # Point estimates on full dataset
    base_m = compute_metrics_fast(y_true, y_prob_base)
    cand_m = compute_metrics_fast(y_true, y_prob_cand)
    
    delta_pr_auc_point = cand_m["pr_auc"] - base_m["pr_auc"]
    delta_roc_auc_point = cand_m["roc_auc"] - base_m["roc_auc"]
    delta_brier_point = cand_m["brier"] - base_m["brier"]
    
    diff_pr_auc = np.zeros(n_resamples, dtype=np.float64)
    diff_roc_auc = np.zeros(n_resamples, dtype=np.float64)
    diff_brier = np.zeros(n_resamples, dtype=np.float64)
    
    for b in range(n_resamples):
        sampled_day_idx = rng.choice(n_days, size=n_days, replace=True)
        sampled_rows = np.concatenate([day_indices[i] for i in sampled_day_idx])
        
        y_b = y_true[sampled_rows]
        p_base_b = y_prob_base[sampled_rows]
        p_cand_b = y_prob_cand[sampled_rows]
        
        m_base_b = compute_metrics_fast(y_b, p_base_b)
        m_cand_b = compute_metrics_fast(y_b, p_cand_b)
        
        diff_pr_auc[b] = m_cand_b["pr_auc"] - m_base_b["pr_auc"]
        diff_roc_auc[b] = m_cand_b["roc_auc"] - m_base_b["roc_auc"]
        diff_brier[b] = m_cand_b["brier"] - m_base_b["brier"]
        
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    
    ci_pr_low = float(np.percentile(diff_pr_auc, low_pct))
    ci_pr_high = float(np.percentile(diff_pr_auc, high_pct))
    
    ci_roc_low = float(np.percentile(diff_roc_auc, low_pct))
    ci_roc_high = float(np.percentile(diff_roc_auc, high_pct))
    
    ci_brier_low = float(np.percentile(diff_brier, low_pct))
    ci_brier_high = float(np.percentile(diff_brier, high_pct))
    
    # Statistical significance check: does 95% CI exclude 0?
    # Improvement in PR-AUC means delta > 0, so CI lower bound > 0
    is_pr_sig_better = bool(ci_pr_low > 0.0)
    is_pr_sig_worse = bool(ci_pr_high < 0.0)
    excludes_zero = bool(ci_pr_low > 0.0 or ci_pr_high < 0.0)
    
    # Empirical p-value for H0: delta <= 0 (testing if candidate is better)
    p_val_one_tailed = float(np.mean(diff_pr_auc <= 0.0))
    p_val_two_tailed = float(2.0 * min(np.mean(diff_pr_auc <= 0.0), np.mean(diff_pr_auc >= 0.0)))
    
    return {
        "delta_pr_auc": {
            "point": round(delta_pr_auc_point, 5),
            "ci_lower": round(ci_pr_low, 5),
            "ci_upper": round(ci_pr_high, 5),
            "std_err": round(float(np.std(diff_pr_auc)), 5),
            "excludes_zero": excludes_zero,
            "statistically_significant_improvement": is_pr_sig_better,
            "p_value_two_tailed": round(p_val_two_tailed, 4)
        },
        "delta_roc_auc": {
            "point": round(delta_roc_auc_point, 5),
            "ci_lower": round(ci_roc_low, 5),
            "ci_upper": round(ci_roc_high, 5),
            "std_err": round(float(np.std(diff_roc_auc)), 5)
        },
        "delta_brier": {
            "point": round(delta_brier_point, 5),
            "ci_lower": round(ci_brier_low, 5),
            "ci_upper": round(ci_brier_high, 5),
            "std_err": round(float(np.std(diff_brier)), 5)
        },
        "verdict": "keep" if is_pr_sig_better else "drop"
    }
