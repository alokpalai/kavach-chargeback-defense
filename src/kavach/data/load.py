"""
Loading and joining the raw IEEE-CIS tables.

Read naively, the 394-column transaction table costs several GB in memory.
We downcast on load and cache to Parquet so that price is paid once.
"""

from __future__ import annotations

import pandas as pd

from kavach import config

TRANSACTION_CSV = config.DATA_RAW / "train_transaction.csv"
IDENTITY_CSV = config.DATA_RAW / "train_identity.csv"
MERGED_PARQUET = config.DATA_INTERIM / "merged.parquet"


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, int64 -> smallest safe int, text -> category.

    pandas 3.x gives text columns a dedicated ``string`` dtype rather than
    ``object``, so both are matched here.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("category")
    return df


def load_raw(force: bool = False) -> pd.DataFrame:
    """Merged transaction + identity frame, cached to Parquet after first call."""
    if MERGED_PARQUET.exists() and not force:
        return pd.read_parquet(MERGED_PARQUET)

    txn = downcast(pd.read_csv(TRANSACTION_CSV))
    idn = downcast(pd.read_csv(IDENTITY_CSV))

    # Left join, never inner. Identity data exists for only a minority of
    # transactions, and that missingness is itself signal: guest checkouts and
    # unrecognised devices carry different risk. Dropping unmatched rows would
    # discard most of the data and bias the survivors.
    merged = txn.merge(idn, on="TransactionID", how="left")
    merged["has_identity"] = merged["id_01"].notna().astype("int8")

    config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MERGED_PARQUET, index=False)
    return merged
