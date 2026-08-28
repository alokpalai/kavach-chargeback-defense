"""
Guard tests for the temporal split.

If someone later 'optimises' the split into a random shuffle, these fail.
"""

import numpy as np
import pandas as pd
import pytest

from kavach import config
from kavach.data.split import SECONDS_PER_DAY, split


@pytest.fixture
def frame() -> pd.DataFrame:
    days = np.arange(0, 183, 0.01)
    return pd.DataFrame({
        "TransactionDT": days * SECONDS_PER_DAY,
        "isFraud": np.zeros(len(days), dtype=int),
    })


def test_train_is_strictly_before_val(frame):
    train, val, _ = split(frame)
    assert train["day"].max() < val["day"].min()


def test_val_is_strictly_before_test(frame):
    _, val, test = split(frame)
    assert val["day"].max() < test["day"].min()


def test_embargo_gap_is_enforced(frame):
    _, val, test = split(frame)
    gap = test["day"].min() - val["day"].max()
    assert gap >= config.EMBARGO_DAYS - 0.05


def test_no_row_appears_in_two_splits(frame):
    train, val, test = split(frame)
    combined = pd.concat([train, val, test]).index
    assert combined.is_unique