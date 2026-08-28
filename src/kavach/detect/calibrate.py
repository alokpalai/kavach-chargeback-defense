"""Probability calibration.

The detector ranks well but its probabilities are too low: mean predicted
0.0251 against a 0.0353 base rate. Ranking metrics (ROC-AUC, PR-AUC) are
invariant to that squashing and never complain. Our cost policy does complain,
because it compares p against an absolute threshold derived from rupees.

Isotonic regression is fit on validation and applied to test. Note that the
validation split also drove early stopping, so it is mildly reused; the effect
is small because early stopping only selects a tree count, but it is recorded
in MODEL_CARD.md rather than hidden.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def fit_calibrator(p_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    """Monotone map from raw score to observed frequency.

    Isotonic rather than Platt: we have thousands of positives and no reason to
    assume the miscalibration is sigmoid-shaped.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val, y_val)
    return iso


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> float:
    """Weighted mean |predicted - observed| over equal-frequency bins."""
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        return float("nan")
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    err = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def calibration_report(y: np.ndarray, p: np.ndarray, name: str) -> dict:
    return {
        "split": name,
        "mean_predicted": round(float(p.mean()), 4),
        "actual_rate": round(float(y.mean()), 4),
        "bias": round(float(p.mean() - y.mean()), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "ece": round(expected_calibration_error(y, p), 4),
    }
