"""Turn model scores and order values into rupees.

Every number the project reports in INR passes through this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kavach import config


def amounts_inr(df: pd.DataFrame) -> np.ndarray:
    """Order values in INR. The USD assumption lives in config, not here."""
    return df["TransactionAmt"].to_numpy(dtype=float) * config.USD_TO_INR


def realised_cost(
    y_true: np.ndarray, p: np.ndarray, amount_inr: np.ndarray, threshold: float
) -> float:
    """Total INR lost under a block-above-threshold policy.

    Correct decisions cost nothing: a blocked fraud is loss avoided, and an
    approved good order is business as usual. Only mistakes are priced.
    """
    block = p >= threshold
    fn = (~block) & (y_true == 1)
    fp = block & (y_true == 0)
    return float(
        config.cost_false_negative(amount_inr[fn]).sum()
        + config.cost_false_positive(amount_inr[fp]).sum()
    )
