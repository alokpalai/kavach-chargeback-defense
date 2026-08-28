"""Ring detection via causal entity co-occurrence.

A per-transaction model asks "does this order look fraudulent?". It cannot ask
"how many different cards has this device tried this week?", because that is a
property of a group, not of a row. One device against forty cards is the
signature of a testing ring and every individual attempt in it looks ordinary.

Implementation note on why this is co-occurrence rather than connected
components. The obvious approach - build a graph over shared attributes, take
components, use component size as a feature - leaks badly: a transaction on day
10 would carry a component size that includes transactions from day 150. Every
metric would improve and the model would be unusable in production, because at
authorisation time that component does not exist yet.

So each feature here answers a strictly backward-looking question: "as of this
moment, how many distinct B values have been seen with this row's A value in
the trailing window?" Connected-component analysis still has a place - see
graph.py - but for investigation and display, never as a model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (anchor, counted) pairs. Each reads as: how many distinct <counted> values
# has this row's <anchor> been seen with recently?
PAIRS = [
    ("DeviceInfo", "card1"),    # one device, many cards -> card testing
    ("addr1", "card1"),         # one address, many cards -> mule cluster
    ("card1", "addr1"),         # one card, many addresses -> stolen card in use
    ("DeviceInfo", "addr1"),    # one device shipping many places
    ("uid", "DeviceInfo"),      # one account hopping devices
    ("P_emaildomain", "card1"), # weak: domains are hubs, kept as a control
]
WINDOWS = {"24h": 86_400, "7d": 604_800}


def _distinct_in_window(a_codes, b_codes, times, window):
    """Distinct b-values co-occurring with each row's a-value in (t - W, t].

    Single pass over rows grouped by anchor and ordered by time, maintaining a
    sliding multiset. The window never extends past the current row, and never
    reaches back beyond the start of the anchor's own group.
    """
    n = len(a_codes)
    order = np.lexsort((times, a_codes))
    a, b, t = a_codes[order], b_codes[order], times[order]

    out = np.empty(n, dtype="float32")
    counts: dict[float, int] = {}
    left = 0

    for i in range(n):
        if i > 0 and a[i] != a[i - 1]:
            counts.clear()
            left = i

        counts[b[i]] = counts.get(b[i], 0) + 1

        cutoff = t[i] - window
        while t[left] <= cutoff:
            bl = b[left]
            remaining = counts.get(bl, 0) - 1
            if remaining <= 0:
                counts.pop(bl, None)
            else:
                counts[bl] = remaining
            left += 1

        out[i] = len(counts)

    res = np.empty(n, dtype="float32")
    res[order] = out
    return res


def add_ring_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "uid" not in out.columns:
        out["uid"] = (
            out["card1"].astype(str) + "_" + out["addr1"].astype(str)
        ).astype("category")

    times = out["TransactionDT"].to_numpy(dtype="float64")
    codes = {
        c: pd.factorize(out[c], use_na_sentinel=True)[0].astype("float64")
        for c in {p for pair in PAIRS for p in pair}
    }

    for anchor, counted in PAIRS:
        a, b = codes[anchor], codes[counted]
        missing = (a < 0) | (b < 0)
        for name, w in WINDOWS.items():
            v = _distinct_in_window(a, b, times, w)
            v[missing] = np.nan
            out[f"ring__{anchor}_x_{counted}_{name}"] = v

    return out
