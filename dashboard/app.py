"""Kavach dashboard - the cost argument, made draggable.

Everything shown is the held-out test split, days 155-183, scored by the model
in models/baseline.txt. Nothing is scored at render time; scripts/prepare_dashboard.py
precomputes it. The sliders re-price decisions, they do not re-fit anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kavach import config
from kavach.policy.decide import APPROVE, BLOCK, STEP_UP, decide

DATA = config.DATA_PROCESSED / "dashboard.parquet"

# The policy this repo ships, fitted on validation. See reports/metrics.json.
SHIPPED_LOW, SHIPPED_HIGH = 0.0232, 0.4360
BINARY_OPTIMAL = 0.0904

st.set_page_config(page_title="Kavach", page_icon="🛡️", layout="wide")


@st.cache_data
def load() -> pd.DataFrame:
    return pd.read_parquet(DATA)


def price(
    df: pd.DataFrame,
    t_low: float,
    t_high: float,
    margin: float,
    dispute_fee: float,
    churn: float,
    abandon: float,
    fraud_blocked: float,
) -> dict:
    """Total INR lost under a three-way policy, at the given cost assumptions.

    Kept pure so the same arithmetic can be checked outside Streamlit.
    """
    amt = df["amount_inr"].to_numpy(dtype=float)
    y = df["is_fraud"].to_numpy()
    d = decide(df["score"].to_numpy(dtype=float), t_low, t_high)

    c_fn = amt + dispute_fee + config.OPS_COST_PER_DISPUTE_INR
    c_fp = margin * amt + churn

    fraud, good = y == 1, y == 0
    cost = (
        c_fn[(d == APPROVE) & fraud].sum()
        + c_fp[(d == BLOCK) & good].sum()
        + (1.0 - fraud_blocked) * c_fn[(d == STEP_UP) & fraud].sum()
        + abandon * c_fp[(d == STEP_UP) & good].sum()
    )
    baseline = c_fn[fraud].sum()
    n = len(df)

    blocked = d == BLOCK
    return {
        "cost_per_10k": cost / n * 1e4,
        "saved_per_10k": (baseline - cost) / n * 1e4,
        "approve_share": float((d == APPROVE).mean()),
        "stepup_share": float((d == STEP_UP).mean()),
        "block_share": float(blocked.mean()),
        "block_precision": float(y[blocked].mean()) if blocked.any() else float("nan"),
        "recall": float(((d != APPROVE) & fraud).sum() / max(fraud.sum(), 1)),
        "decisions": d,
    }


df = load()

st.title("🛡️ Kavach — chargeback defense, priced in rupees")
st.caption(
    f"Held-out test split · {len(df):,} orders · days 155–183 · "
    f"{df.is_fraud.mean():.2%} fraud · model never saw this data"
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Policy")
    preset = st.radio(
        "Preset",
        ["Shipped (step-up ≤ 10%)", "Cost-optimal binary", "Naive 0.50", "Custom"],
        help="Presets were fitted on validation, never on the data shown here.",
    )
    if preset == "Shipped (step-up ≤ 10%)":
        d_low, d_high = SHIPPED_LOW, SHIPPED_HIGH
    elif preset == "Cost-optimal binary":
        d_low, d_high = BINARY_OPTIMAL, BINARY_OPTIMAL
    elif preset == "Naive 0.50":
        d_low, d_high = 0.50, 0.50
    else:
        d_low, d_high = SHIPPED_LOW, SHIPPED_HIGH

    disabled = preset != "Custom"
    t_low = st.slider("Step-up above", 0.0, 1.0, d_low, 0.0001,
                      format="%.4f", disabled=disabled)
    t_high = st.slider("Block above", 0.0, 1.0, d_high, 0.0001,
                       format="%.4f", disabled=disabled)
    if t_low > t_high:
        t_low = t_high

    st.header("Cost assumptions")
    st.caption("Every rupee figure below is downstream of these. Change them.")
    margin = st.slider("Gross margin", 0.05, 0.60, config.GROSS_MARGIN, 0.01)
    dispute_fee = st.slider("Dispute fee ₹", 0, 4000, int(config.DISPUTE_FEE_INR), 100)
    churn = st.slider("Churn cost of a wrong block ₹", 0, 3000,
                      int(config.CHURN_COST_INR), 100)
    abandon = st.slider("Good customers abandoning at step-up",
                        0.0, 0.40, config.STEPUP_ABANDON_RATE, 0.01)
    fraud_blocked = st.slider("Fraudsters failing step-up",
                              0.30, 1.0, config.STEPUP_FRAUD_BLOCK_RATE, 0.05)

kw = dict(margin=margin, dispute_fee=dispute_fee, churn=churn,
          abandon=abandon, fraud_blocked=fraud_blocked)
now = price(df, t_low, t_high, **kw)
binary = price(df, BINARY_OPTIMAL, BINARY_OPTIMAL, **kw)
naive = price(df, 0.50, 0.50, **kw)

# ---------------------------------------------------------------- headline
a, b, c, d = st.columns(4)
a.metric("Saved per 10,000 orders", f"₹{now['saved_per_10k']:,.0f}",
         delta=f"₹{now['saved_per_10k'] - binary['saved_per_10k']:,.0f} vs binary")
b.metric("Cost per 10,000 orders", f"₹{now['cost_per_10k']:,.0f}")
c.metric("Good customers blocked", f"{now['block_share'] * (1 - now['block_precision']):.2%}",
         help="Share of all traffic that is a legitimate order we declined.")
d.metric("Fraud caught or challenged", f"{now['recall']:.1%}")

if now["saved_per_10k"] < binary["saved_per_10k"]:
    st.warning(
        "This policy is worse than a plain binary threshold under these assumptions — "
        "the step-up band is not paying for itself here."
    )

# ---------------------------------------------------------------- mix
left, right = st.columns([3, 2])

with left:
    st.subheader("Where the traffic goes")
    mix = pd.DataFrame({
        "decision": ["approve", "step-up", "block"],
        "share": [now["approve_share"], now["stepup_share"], now["block_share"]],
    })
    st.bar_chart(mix.set_index("decision"), horizontal=True, height=200)
    st.caption(
        f"Block band is **{now['block_precision']:.1%}** fraud. "
        f"The rest of that band is real customers being turned away — "
        f"the cost the model is trading against."
    )

    st.subheader("Cost across every threshold")
    grid = np.unique(np.quantile(df["score"].to_numpy(), np.linspace(0, 1, 220)))
    curve = pd.DataFrame({
        "threshold": grid,
        "cost per 10k": [price(df, g, g, **kw)["cost_per_10k"] for g in grid],
    })
    # Near threshold 0 the policy blocks everything and cost runs to ~7x the
    # approve-all baseline, flattening the region anyone actually cares about.
    # Clip to just above baseline so the shape of the optimum is visible.
    ceiling = curve["cost per 10k"].iloc[-1] * 1.15
    curve = curve[curve["cost per 10k"] <= ceiling].set_index("threshold")
    st.line_chart(curve, height=260)
    st.caption(
        "Binary policy only, clipped just above the approve-everything baseline "
        "(blocking everything costs several times more and hides the shape). "
        "The flat right-hand side is 'approve everything'."
    )

with right:
    st.subheader("Policy comparison")
    st.dataframe(
        pd.DataFrame([
            {"policy": "Naive 0.50", "saved / 10k": f"₹{naive['saved_per_10k']:,.0f}",
             "blocked": f"{naive['block_share']:.2%}"},
            {"policy": "Cost-optimal binary", "saved / 10k": f"₹{binary['saved_per_10k']:,.0f}",
             "blocked": f"{binary['block_share']:.2%}"},
            {"policy": "Current", "saved / 10k": f"₹{now['saved_per_10k']:,.0f}",
             "blocked": f"{now['block_share']:.2%}"},
        ]),
        hide_index=True, use_container_width=True,
    )

    st.subheader("Review queue")
    st.caption("Highest-risk orders, with the model's own contribution breakdown.")
    dec = now["decisions"]
    queue = (df.assign(decision=np.where(dec == BLOCK, "block",
                                         np.where(dec == STEP_UP, "step-up", "approve")))
               .query("top_reasons != ''")
               .nlargest(15, "score")[["score", "amount_inr", "decision",
                                       "is_fraud", "top_reasons"]])
    st.dataframe(
        queue.rename(columns={"amount_inr": "₹", "is_fraud": "was fraud",
                              "top_reasons": "why"}),
        hide_index=True, use_container_width=True,
        column_config={"score": st.column_config.NumberColumn(format="%.4f"),
                       "₹": st.column_config.NumberColumn(format="₹%.0f")},
    )

st.divider()
st.caption(
    "Precision here is measured only on orders the historical system approved — "
    "blocked orders never produce a chargeback label. Treat it as an upper bound. "
    "See MODEL_CARD.md."
)
