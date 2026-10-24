"""
Evaluation Metrics Utility for Flight Arrival Delay Prediction (Phase 3)

Provides standardized evaluation across all baseline models and ablation variants:
- PR-AUC (Average Precision)
- ROC-AUC
- Brier Score Loss
- Log Loss (Binary Cross-Entropy)
- Best-F1 and Optimal Decision Threshold
- Top 10% Risk Metrics: Precision@10%, Recall@10%, Lift@10%
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    f1_score,
    precision_recall_curve
)

def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    split_name: str = "val"
) -> Dict[str, Any]:
    """
    Computes all project-standard evaluation metrics given true binary labels and predicted probabilities.
    
    Args:
        y_true: 1D array of binary labels {0, 1}
        y_prob: 1D array of predicted probabilities in [0, 1]
        split_name: Prefix for metric keys (e.g. 'val' or 'train')
        
    Returns:
        Dictionary of formatted metrics
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    # Clip probabilities to guard log loss against log(0)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    
    n_samples = len(y_true)
    base_rate = float(np.mean(y_true))
    
    # 1. PR-AUC and ROC-AUC
    # Guard if all predictions are identical (e.g. dummy constant predictor)
    if len(np.unique(y_prob)) <= 1:
        pr_auc = base_rate
        roc_auc = 0.5
    else:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        
    # 2. Probability Calibration Loss: Brier and Log Loss
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob_clipped))
    
    # 3. Best-F1 and Optimal Threshold
    # Evaluate across a grid of thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = 0.0
    best_thresh = 0.5
    
    # Quick scan
    for t in thresholds:
        pred_binary = (y_prob >= t).astype(int)
        tp = np.sum((pred_binary == 1) & (y_true == 1))
        fp = np.sum((pred_binary == 1) & (y_true == 0))
        fn = np.sum((pred_binary == 0) & (y_true == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
            
    # 4. Top 10% Risk Flights Metrics
    k_10pct = max(1, int(round(0.10 * n_samples)))
    top_10_indices = np.argsort(y_prob)[-k_10pct:]
    top_10_delayed = int(np.sum(y_true[top_10_indices]))
    total_delayed = int(np.sum(y_true))
    
    prec_at_10 = float(top_10_delayed / k_10pct)
    rec_at_10 = float(top_10_delayed / total_delayed) if total_delayed > 0 else 0.0
    lift_at_10 = float(prec_at_10 / base_rate) if base_rate > 0 else 1.0
    
    return {
        f"{split_name}_pr_auc": round(pr_auc, 4),
        f"{split_name}_roc_auc": round(roc_auc, 4),
        f"{split_name}_brier": round(brier, 4),
        f"{split_name}_log_loss": round(ll, 4),
        f"{split_name}_best_f1": round(best_f1, 4),
        f"{split_name}_best_threshold": round(best_thresh, 2),
        f"{split_name}_prec_at_10": round(prec_at_10, 4),
        f"{split_name}_rec_at_10": round(rec_at_10, 4),
        f"{split_name}_lift_at_10": round(lift_at_10, 2),
        f"{split_name}_base_rate": round(base_rate, 4),
        f"{split_name}_samples": n_samples
    }
