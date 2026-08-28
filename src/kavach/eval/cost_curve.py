"""Sweep decision thresholds and price each one.

The output of this module is the project's headline chart: not "which
threshold maximises F1" but "which threshold loses the merchant least money".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kavach import config
from kavach.policy.costs import realised_cost


def sweep(y_true, p, amount_inr, n_points: int = 400) -> pd.DataFrame:
    grid = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_points)))
    n = len(y_true)
    rows = []
    for t in grid:
        block = p >= t
        tp = int((block & (y_true == 1)).sum())
        fp = int((block & (y_true == 0)).sum())
        fn = int((~block & (y_true == 1)).sum())
        cost = realised_cost(y_true, p, amount_inr, t)
        rows.append({
            "threshold": float(t),
            "block_rate": block.mean(),
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "cost_inr": cost,
            "cost_per_10k": cost / n * 10_000,
        })
    return pd.DataFrame(rows)


def compare_policies(y_true, p, amount_inr) -> pd.DataFrame:
    """Optimal threshold against the two policies it must beat."""
    curve = sweep(y_true, p, amount_inr)
    best = curve.loc[curve["cost_inr"].idxmin()]
    n = len(y_true)

    approve_all = realised_cost(y_true, p, amount_inr, threshold=2.0)
    naive_half = realised_cost(y_true, p, amount_inr, threshold=0.5)

    rows = [
        {"policy": "approve everything", "threshold": None, "cost_inr": approve_all},
        {"policy": "naive 0.50", "threshold": 0.50, "cost_inr": naive_half},
        {"policy": "cost-optimal", "threshold": round(best.threshold, 4),
         "cost_inr": best.cost_inr},
    ]
    out = pd.DataFrame(rows)
    out["cost_per_10k"] = out["cost_inr"] / n * 10_000
    out["saved_vs_approve_all"] = approve_all - out["cost_inr"]
    return out, curve, best
