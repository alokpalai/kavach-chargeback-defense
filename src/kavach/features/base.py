"""
Baseline feature preparation.

Deliberately minimal. LightGBM handles categoricals and missing values
natively, so the first model uses the columns close to as they arrive.
This exists to establish a floor: later feature work must beat this number
or it is not earning its complexity.
"""

from __future__ import annotations

import pandas as pd

# TransactionDT is dropped on purpose. It is a raw offset that only ever
# increases, so a tree would happily split on "timestamp > X" and score well
# in-sample while learning nothing that survives into next month. Cyclical
# time-of-day and day-of-week do generalise, so we derive those instead.
DROP_COLS = ["TransactionID", "isFraud", "TransactionDT", "day"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = (out["TransactionDT"] / 3600) % 24
    out["dayofweek"] = (out["TransactionDT"] / 86400) % 7
    return out


def make_xy(df: pd.DataFrame, label: str = "isFraud"):
    df = add_time_features(df)
    y = df[label].astype("int8")
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return X, y
