"""Precompute everything the dashboard needs.

The app must be instant, so nothing is scored at render time. This writes a
small frame of held-out test orders with their scores, their value in rupees,
and - for the highest-risk orders - the feature contributions behind the score.

Reason codes come from LightGBM's own pred_contrib (SHAP values computed by the
booster), so there is no extra dependency and the explanation is exactly the
decomposition of the score the model produced.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from kavach import config
from kavach.data.load import load_raw
from kavach.data.split import split
from kavach.features.base import make_xy
from kavach.policy.costs import amounts_inr

OUT = config.DATA_PROCESSED / "dashboard.parquet"
TOP_N_EXPLAINED = 1500

tr, va, te = split(load_raw())
model = lgb.Booster(model_file="models/baseline.txt")
X, y = make_xy(te)
X = X[model.feature_name()]
p = model.predict(X)

frame = pd.DataFrame({
    # float64 for both score and amount: float32 rounding moves a handful of
    # orders across the thresholds and shifts the total by ~0.02%. The
    # dashboard must reconcile exactly with reports/metrics.json.
    "score": p,
    "is_fraud": y.to_numpy().astype("int8"),
    # float64 on purpose: float32 rounding shifts the total by ~0.02%,
    # and the dashboard must reconcile exactly with reports/metrics.json.
    "amount_inr": amounts_inr(te),
    "day": te["day"].to_numpy().astype("float32"),
    "card": te["card1"].to_numpy(),
    "device": te["DeviceInfo"].astype(str).to_numpy(),
    "email": te["P_emaildomain"].astype(str).to_numpy(),
})

# Feature contributions only for the riskiest orders - the ones an analyst
# would ever open. pred_contrib on the full test set would be 78k x 436 floats.
top = np.argsort(p)[::-1][:TOP_N_EXPLAINED]
contrib = model.predict(X.iloc[top], pred_contrib=True)[:, :-1]  # drop bias term
names = np.array(model.feature_name())

reasons = np.full(len(frame), "", dtype=object)
for row, idx in enumerate(top):
    c = contrib[row]
    order = np.argsort(c)[::-1][:3]          # three strongest push-toward-fraud
    reasons[idx] = ", ".join(
        f"{names[j]} ({c[j]:+.2f})" for j in order if c[j] > 0
    )
frame["top_reasons"] = reasons

# Persist only the columns the dashboard actually reads. card / device / email
# are raw IEEE-CIS values with no use in the app, and this file is committed so
# the hosted dashboard can boot - it should carry as little source data as it can.
frame = frame[["score", "is_fraud", "amount_inr", "top_reasons"]]

config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
frame.to_parquet(OUT, index=False)
print(f"wrote {OUT}  rows={len(frame)}  explained={TOP_N_EXPLAINED}")
print(frame.nlargest(3, "score")[["score", "amount_inr", "is_fraud", "top_reasons"]].to_string())
