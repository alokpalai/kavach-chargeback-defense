"""
Temporal splitting.

Deliberately kept short and free of any feature logic, so a reviewer can
confirm in thirty seconds that no future data reaches the training set.
"""

from __future__ import annotations

import pandas as pd

from kavach import config

SECONDS_PER_DAY = 86_400


def add_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """TransactionDT is a seconds offset from an unspecified origin."""
    out = df.copy()
    out["day"] = out["TransactionDT"] / SECONDS_PER_DAY
    return out


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically into (train, val, test).

    Rows in the embargo window between val and test are returned by nobody.
    They are not "wasted" - discarding them is what makes the test set an
    honest simulation of scoring traffic whose labels have not arrived yet.
    """
    if "day" not in df.columns:
        df = add_day_index(df)

    train = df[df["day"] < config.TRAIN_END_DAY]
    val = df[(df["day"] >= config.TRAIN_END_DAY) & (df["day"] < config.VAL_END_DAY)]
    test = df[df["day"] >= config.TEST_START_DAY]
    return train, val, test


def describe_split(train, val, test, label: str = "isFraud") -> pd.DataFrame:
    """Table for the eval report - reviewers should not take the split on faith."""
    rows = []
    for name, part in (("train", train), ("val", val), ("test", test)):
        rows.append({
            "split": name,
            "rows": len(part),
            "day_min": round(part["day"].min(), 1),
            "day_max": round(part["day"].max(), 1),
            "positive_rate": round(part[label].mean(), 4) if label in part else None,
        })
    return pd.DataFrame(rows)
