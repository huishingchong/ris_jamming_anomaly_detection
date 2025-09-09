#!/usr/bin/env python3
"""
Metrics & threshold helpers
"""
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    brier_score_loss,
)
import pandas as pd

def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    tpr = recall

    sensitivity = tpr
    specificity = tnr
    balanced_accuracy = (sensitivity + specificity) / 2.0

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    precision_normal, precision_ris = precisions
    recall_normal,  recall_ris  = recalls
    f1_normal,      f1_ris      = f1s
    support_normal, support_ris = supports

    macro_f1 = (f1_normal + f1_ris) / 2.0
    weighted_f1 = (
        (f1_normal * support_normal + f1_ris * support_ris) / (support_normal + support_ris)
        if (support_normal + support_ris) > 0 else 0.0
    )

    # AUC metrics
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc  = average_precision_score(y_true, y_scores)
    else:
        roc_auc, pr_auc = 0.0, 0.0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'tpr': float(tpr),
        'tnr': float(tnr),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'balanced_accuracy': float(balanced_accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'f1_positive': float(f1_ris),
        'f1_negative': float(f1_normal),
        'precision_positive': float(precision_ris),
        'precision_negative': float(precision_normal),
        'recall_positive': float(recall_ris),
        'recall_negative': float(recall_normal),
    }

def ece(scores: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (used by RQ1 test & val calibration)"""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(scores, bins) - 1
    error = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        if not np.any(mask):
            continue
        p_hat = scores[mask].mean()
        y_bar = y_true[mask].mean()
        w = mask.mean()
        error += w * abs(p_hat - y_bar)
    return float(error)

def threshold_for_fpr(y_true: np.ndarray, p_scores: np.ndarray, target_fpr: float = 0.10, guard: float = 0.0) -> Tuple[float, float]:
    """Return (threshold, achieved_fpr) with FPR <= target_fpr - guard where possible."""
    fpr, tpr, thr = roc_curve(y_true, p_scores)
    cap = max(0.0, target_fpr - guard)
    ok = np.where(fpr <= cap)[0]
    if len(ok) == 0:
        idx = int(np.argmin(fpr))
        return float(thr[idx]), float(fpr[idx])
    idx = int(ok[np.argmin(thr[ok])])  # lowest thr among those meeting cap
    return float(thr[idx]), float(fpr[idx])

def threshold_for_fnr(y_true: np.ndarray, p_scores: np.ndarray, target_fnr: float = 0.10, guard: float = 0.0) -> Tuple[float, float]:
    """Return (threshold, achieved_fnr) with FNR <= target_fnr - guard where possible."""
    fpr, tpr, thr = roc_curve(y_true, p_scores)
    fnr = 1.0 - tpr
    cap = max(0.0, target_fnr - guard)
    ok = np.where(fnr <= cap)[0]
    if len(ok) == 0:
        idx = int(np.argmin(fnr))
        return float(thr[idx]), float(fnr[idx])
    idx = int(ok[np.argmax(thr[ok])])  # highest thr among those meeting cap
    return float(thr[idx]), float(fnr[idx])

def analyse_band_performance(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Analyse performance by effectiveness band."""
    if 'band_name' not in df.columns:
        return {}
    band_analysis = {}
    for band, group in df.groupby('band_name'):
        indices = group.index.tolist()
        band_y_true = y_true[indices]
        band_y_pred = y_pred[indices]
        
        normal_mask = band_y_true == 0
        attack_mask = band_y_true == 1
        
        if np.sum(normal_mask) > 0:
            band_fpr = np.sum((band_y_true[normal_mask] == 0) & (band_y_pred[normal_mask] == 1)) / np.sum(normal_mask)
        else:
            band_fpr = None
            
        if np.sum(attack_mask) > 0:
            band_fnr = np.sum((band_y_true[attack_mask] == 1) & (band_y_pred[attack_mask] == 0)) / np.sum(attack_mask)
        else:
            band_fnr = None
        
        band_analysis[str(band)] = {
            'fpr': band_fpr,
            'fnr': band_fnr,
            'counts': {'normal': int(np.sum(normal_mask)), 'jam': int(np.sum(attack_mask))}
        }
    return band_analysis
