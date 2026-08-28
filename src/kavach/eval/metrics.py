"""Metrics that survive a 3.5% base rate.

ROC-AUC is close to meaningless here: with 96.5% negatives it stays high while
the model drowns an analyst in false positives. Precision, recall at a
precision a review team could actually staff, and PR-AUC are the honest set.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def recall_at_precision(y_true: np.ndarray, p: np.ndarray, target: float) -> float:
    """Best recall achievable while holding precision at or above `target`."""
    prec, rec, _ = precision_recall_curve(y_true, p)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


def precision_at_k(y_true: np.ndarray, p: np.ndarray, k: int) -> float:
    """Precision in the k highest-scoring orders - the analyst's daily queue."""
    idx = np.argsort(p)[::-1][:k]
    return float(y_true[idx].mean())


def summary(y_true: np.ndarray, p: np.ndarray, name: str) -> dict:
    return {
        "model": name,
        "pr_auc": round(float(average_precision_score(y_true, p)), 4),
        "recall@prec50": round(recall_at_precision(y_true, p, 0.50), 4),
        "recall@prec60": round(recall_at_precision(y_true, p, 0.60), 4),
        "recall@prec70": round(recall_at_precision(y_true, p, 0.70), 4),
        "prec@1000": round(precision_at_k(y_true, p, 1000), 4),
    }
