"""Three-way decision policy: approve / step_up / block.

A binary policy forces every uncertain order into approve-or-decline, and a
wrong decline costs the merchant a whole sale. A 3DS step-up is a third option
that is far cheaper than either mistake:

  * a genuine customer usually passes the challenge; only the fraction who
    abandon (STEPUP_ABANDON_RATE) costs us anything, and
  * most fraudsters fail it (STEPUP_FRAUD_BLOCK_RATE), so a challenged fraud
    only costs us the leakage that gets through.

The middle band is therefore where uncertainty belongs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kavach import config

APPROVE, STEP_UP, BLOCK = 0, 1, 2


def decide(p: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """Vector of APPROVE / STEP_UP / BLOCK decisions."""
    return np.where(p >= t_high, BLOCK, np.where(p >= t_low, STEP_UP, APPROVE))


def cost_three_way(
    y_true: np.ndarray,
    p: np.ndarray,
    amount_inr: np.ndarray,
    t_low: float,
    t_high: float,
) -> float:
    """Expected INR lost under a three-way policy.

    Blocking a fraud and approving a good order are both free. Everything else
    is priced, with the step-up band priced in expectation rather than assumed
    to work perfectly.
    """
    d = decide(p, t_low, t_high)
    fraud = y_true == 1
    good = ~fraud

    approved_fraud = (d == APPROVE) & fraud
    blocked_good = (d == BLOCK) & good
    challenged_fraud = (d == STEP_UP) & fraud
    challenged_good = (d == STEP_UP) & good

    return float(
        config.cost_false_negative(amount_inr[approved_fraud]).sum()
        + config.cost_false_positive(amount_inr[blocked_good]).sum()
        # fraud that survives the challenge
        + (1.0 - config.STEPUP_FRAUD_BLOCK_RATE)
        * config.cost_false_negative(amount_inr[challenged_fraud]).sum()
        # good customers who abandon at the challenge
        + config.STEPUP_ABANDON_RATE
        * config.cost_false_positive(amount_inr[challenged_good]).sum()
    )


def optimise(
    y_true: np.ndarray,
    p: np.ndarray,
    amount_inr: np.ndarray,
    n_points: int = 60,
) -> tuple[float, float, float]:
    """Grid-search (t_low, t_high) on the split it is given.

    Must be called on validation, never on test - the pair is a fitted
    parameter like any other.
    """
    grid = np.unique(np.quantile(p, np.linspace(0.50, 0.9995, n_points)))
    best = (None, None, np.inf)
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            c = cost_three_way(y_true, p, amount_inr, lo, hi)
            if c < best[2]:
                best = (float(lo), float(hi), c)
    return best


def summarise(y_true, p, amount_inr, t_low, t_high) -> pd.DataFrame:
    """Decision mix and outcome rates, for the eval report."""
    d = decide(p, t_low, t_high)
    rows = []
    for name, code in (("approve", APPROVE), ("step_up", STEP_UP), ("block", BLOCK)):
        m = d == code
        rows.append({
            "decision": name,
            "share": round(float(m.mean()), 4),
            "n": int(m.sum()),
            "fraud_rate": round(float(y_true[m].mean()), 4) if m.sum() else np.nan,
        })
    return pd.DataFrame(rows)
