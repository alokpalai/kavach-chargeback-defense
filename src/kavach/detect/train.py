"""Baseline gradient-boosted detector."""

from __future__ import annotations

import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

from kavach import config
from kavach.features.base import make_xy

PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "verbosity": -1,
    "seed": config.SEED,
    # No scale_pos_weight on purpose. Reweighting improves ranking metrics but
    # destroys the probability scale, and our cost policy needs probabilities
    # that mean what they say. Imbalance is handled at the threshold instead.
}


def train(train_df, val_df, num_boost_round: int = 3000):
    X_tr, y_tr = make_xy(train_df)
    X_va, y_va = make_xy(val_df)

    model = lgb.train(
        PARAMS,
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=num_boost_round,
        valid_sets=[lgb.Dataset(X_va, y_va)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )
    return model


def evaluate(model, df, name: str = "test") -> dict:
    X, y = make_xy(df)
    p = model.predict(X, num_iteration=model.best_iteration)
    return {
        "split": name,
        "n": len(y),
        "positives": int(y.sum()),
        "pr_auc": round(average_precision_score(y, p), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "base_rate": round(float(y.mean()), 4),
    }
