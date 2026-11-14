"""Metric definitions, identical to the original evaluation scripts
(scripts/virchow_test_bootstrap_inference.py::_metrics_on_sample, scripts/train_virchow_preprocessed_colab.py)."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def fit_temperature_binary(logits, y, grid: int = 60) -> float:
    """Grid search T in [0.2, 5.0] (log-spaced) minimising mean NLL on source-validation logits (original protocol)."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    best_T, best = 1.0, float("inf")
    for T in np.logspace(math.log10(0.2), math.log10(5.0), grid):
        p = np.clip(sigmoid(logits / T), 1e-7, 1 - 1e-7)
        nll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        if nll < best:
            best, best_T = nll, float(T)
    return best_T


def ece15(y, prob, n_bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e, n = 0.0, len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (prob >= lo) & (prob <= hi if i == n_bins - 1 else prob < hi)
        c = int(m.sum())
        if c:
            e += (c / n) * abs(float(prob[m].mean()) - float(y[m].mean()))
    return float(e)


def sens_at_spec(y, score, spec: float) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= (1 - spec) + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def spec_at_sens(y, score, sens: float) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    ok = tpr >= sens - 1e-12
    return float(1 - fpr[ok].min()) if ok.any() else 0.0


def all_metrics(y, prob, operating_points: bool = True) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    prob = np.clip(np.asarray(prob, dtype=np.float64).reshape(-1), 1e-7, 1 - 1e-7)
    pred = prob >= 0.5
    tp = float((pred & (y == 1)).sum()); tn = float((~pred & (y == 0)).sum())
    fp = float((pred & (y == 0)).sum()); fn = float((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    out = {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (rec + spec),
        "precision": prec, "recall": rec, "specificity": spec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "brier": float(np.mean((prob - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))),
        "ece15": ece15(y, prob),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    if operating_points:
        out["sens_at_spec95"] = sens_at_spec(y, prob, 0.95)
        out["sens_at_spec98"] = sens_at_spec(y, prob, 0.98)
        out["spec_at_sens90"] = spec_at_sens(y, prob, 0.90)
    return out
