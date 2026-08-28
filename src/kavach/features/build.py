"""Feature pipeline: raw merge -> velocity -> cached Parquet."""

from __future__ import annotations

import pandas as pd

from kavach import config
from kavach.data.load import load_raw
from kavach.features.velocity import add_velocity_features

FEATURES_PARQUET = config.DATA_PROCESSED / "features.parquet"


def build_features(force: bool = False) -> pd.DataFrame:
    if FEATURES_PARQUET.exists() and not force:
        return pd.read_parquet(FEATURES_PARQUET)
    df = add_velocity_features(load_raw())
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PARQUET, index=False)
    return df
