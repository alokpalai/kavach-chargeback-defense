"""Backward-looking velocity features.

Card testing and enumeration attacks are invisible in any single transaction
and obvious across a short window: forty small authorisations against one card
BIN in an hour. These features give the model that view.

Every window here is strictly backward-looking and half-open, (t - W, t].
A transaction sees its own entity's past and never its future, which is
exactly the information available at authorisation time. Transactions sharing
an identical timestamp are counted only if they sort before the current row:
at scoring time an order that has not reached the gateway yet is unknowable,
so the conservative reading is the correct one.

Note on the embargo: velocity is computed over the whole frame, including the
embargoed days. That is not leakage - it uses transaction facts (counts,
amounts) that exist the moment an order is placed, never fraud labels, which
are what the embargo protects. Label-based target encoding is deliberately
absent for that reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Entities the model can key on. card1/addr1 are present on every row;
# DeviceInfo covers only ~24% of traffic, so it enriches rather than anchors.
KEYS = ["card1", "addr1", "P_emaildomain", "uid", "DeviceInfo"]
WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 604_800}

_BIG = 1e9  # separates groups in the composite sort key; exceeds max TransactionDT


def _rolling_count_sum(codes, times, values, window):
    """Per-group count and sum over (t - window, t], vectorised.

    Groups are kept apart by offsetting each time by code * _BIG, which makes
    the composite key globally sorted and lets one searchsorted find every
    left boundary at once.
    """
    n = len(codes)
    order = np.lexsort((times, codes))
    c, t, v = codes[order], times[order], values[order]

    key = c * _BIG + t
    left = np.searchsorted(key, c * _BIG + (t - window), side="right")
    idx = np.arange(n)

    cs = np.concatenate([[0.0], np.cumsum(v)])
    count = (idx - left + 1).astype("float32")
    total = (cs[idx + 1] - cs[left]).astype("float32")

    out_c = np.empty(n, dtype="float32")
    out_s = np.empty(n, dtype="float32")
    out_c[order] = count
    out_s[order] = total
    return out_c, out_s


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["uid"] = (
        out["card1"].astype(str) + "_" + out["addr1"].astype(str)
    ).astype("category")

    times = out["TransactionDT"].to_numpy(dtype="float64")
    amts = out["TransactionAmt"].to_numpy(dtype="float64")

    for key in KEYS:
        codes = pd.factorize(out[key], use_na_sentinel=True)[0].astype("float64")
        missing = codes < 0

        # seconds since this entity's previous transaction
        order = np.lexsort((times, codes))
        prev = np.full(len(out), np.nan)
        tc, cc = times[order], codes[order]
        same = np.concatenate([[False], cc[1:] == cc[:-1]])
        gap = np.concatenate([[np.nan], np.diff(tc)])
        prev[order] = np.where(same, gap, np.nan)
        prev[missing] = np.nan
        out[f"{key}__secs_since_prev"] = prev.astype("float32")

        for name, w in WINDOWS.items():
            cnt, tot = _rolling_count_sum(codes, times, amts, w)
            cnt[missing] = np.nan
            tot[missing] = np.nan
            out[f"{key}__n_{name}"] = cnt
            out[f"{key}__amt_{name}"] = tot
            # is this order unusual for this entity right now?
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{key}__amt_ratio_{name}"] = (
                    amts / np.where(cnt > 0, tot / cnt, np.nan)
                ).astype("float32")

    return out
