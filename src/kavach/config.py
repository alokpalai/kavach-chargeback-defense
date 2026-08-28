"""
Central configuration for Kavach.

Every downstream module imports its numbers from here. If a reviewer wants to
know what a false positive costs us, there is exactly one file to read.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

SEED = 42

# ---------------------------------------------------------------------------
# Temporal split (days since the first transaction in the dataset)
#
# We never split randomly. Fraud is non-stationary and entities (cards, devices)
# recur, so a random split leaks the future into training and inflates every
# metric. Days are derived from TransactionDT, which is a seconds offset.
# ---------------------------------------------------------------------------
TRAIN_END_DAY = 120
VAL_END_DAY = 140

# Chargebacks surface 30-90 days after the transaction. At serving time we
# would not yet have labels for the most recent weeks, so we discard a window
# between validation and test rather than pretending we could train on it.
EMBARGO_DAYS = 15
TEST_START_DAY = VAL_END_DAY + EMBARGO_DAYS

# ---------------------------------------------------------------------------
# Cost model  (all values in INR)
#
# IEEE-CIS does not state its currency. We assume USD, which is the community
# consensus for this dataset. This is an assumption, not a fact, and it is
# reported as one.
# ---------------------------------------------------------------------------
USD_TO_INR = 88.0

GROSS_MARGIN = 0.25              # merchant keeps 25p of every rupee of GMV
DISPUTE_FEE_INR = 1500.0         # network/issuer fee — charged even if you win
OPS_COST_PER_DISPUTE_INR = 300.0 # analyst time to assemble evidence
CHURN_COST_INR = 500.0           # expected LTV lost when a good customer is blocked

# 3DS step-up: cheap friction instead of an expensive decline
STEPUP_ABANDON_RATE = 0.08       # good customers who drop off at the challenge
STEPUP_FRAUD_BLOCK_RATE = 0.85   # fraudsters who fail the challenge

# Visa VDMP / Mastercard EFM monitoring threshold. Crossing it triggers fines
# and, sustained, account termination — a cost no per-transaction model sees.
CHARGEBACK_RATIO_THRESHOLD = 0.0065


def cost_false_negative(amount_inr: float) -> float:
    """We approved a fraudulent order: goods gone, plus the dispute costs."""
    return amount_inr + DISPUTE_FEE_INR + OPS_COST_PER_DISPUTE_INR


def cost_false_positive(amount_inr: float) -> float:
    """We blocked a good order: lost margin, plus the customer we annoyed."""
    return GROSS_MARGIN * amount_inr + CHURN_COST_INR


def breakeven_threshold(amount_inr: float) -> float:
    """
    Probability above which blocking is cheaper than approving.

    Approving costs  p * C_fn ; blocking costs (1 - p) * C_fp.
    Setting them equal gives  p* = C_fp / (C_fp + C_fn).
    """
    c_fp = cost_false_positive(amount_inr)
    c_fn = cost_false_negative(amount_inr)
    return c_fp / (c_fp + c_fn)