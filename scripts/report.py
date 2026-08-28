"""Regenerate every number the README quotes.

The README must never contain a hand-typed metric. This script writes
reports/metrics.json and the figures; the README cites that file.
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from kavach import config
from kavach.data.load import load_raw
from kavach.data.split import describe_split, split
from kavach.eval.cost_curve import sweep
from kavach.eval.metrics import summary
from kavach.features.base import make_xy
from kavach.policy.costs import amounts_inr
from kavach.policy.decide import cost_three_way, optimise, summarise

t0 = time.time()
out: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "seed": config.SEED}

# ---------------------------------------------------------------- data & split
raw = load_raw()
tr, va, te = split(raw)
out["dataset"] = {
    "source": "IEEE-CIS Fraud Detection (Kaggle)",
    "rows": len(raw),
    "columns": int(raw.shape[1]),
    "span_days": round(float(raw.TransactionDT.max() - raw.TransactionDT.min()) / 86400, 1),
    "fraud_rate": round(float(raw.isFraud.mean()), 4),
}
out["split"] = json.loads(describe_split(tr, va, te).to_json(orient="records"))
out["split_policy"] = {
    "type": "temporal",
    "train_end_day": config.TRAIN_END_DAY,
    "val_end_day": config.VAL_END_DAY,
    "embargo_days": config.EMBARGO_DAYS,
    "test_start_day": config.TEST_START_DAY,
    "embargo_rows_discarded": int(len(raw) - len(tr) - len(va) - len(te)),
}
out["cost_model"] = {
    "usd_to_inr": config.USD_TO_INR,
    "gross_margin": config.GROSS_MARGIN,
    "dispute_fee_inr": config.DISPUTE_FEE_INR,
    "ops_cost_per_dispute_inr": config.OPS_COST_PER_DISPUTE_INR,
    "churn_cost_inr": config.CHURN_COST_INR,
    "stepup_abandon_rate": config.STEPUP_ABANDON_RATE,
    "stepup_fraud_block_rate": config.STEPUP_FRAUD_BLOCK_RATE,
}

# ---------------------------------------------------------------- detector
base = lgb.Booster(model_file="models/baseline.txt")
Xva, yva = make_xy(va); yva = yva.to_numpy()
Xte, yte = make_xy(te); yte = yte.to_numpy()
bf = base.feature_name()
pva, pte = base.predict(Xva[bf]), base.predict(Xte[bf])
ava, ate = amounts_inr(va), amounts_inr(te)
n = len(yte)

out["detector"] = {
    "model": "LightGBM", "trees": int(base.num_trees()),
    "val": summary(yva, pva, "val"), "test": summary(yte, pte, "test"),
}

# ---------------------------------------------------------------- policies
approve_all = float(config.cost_false_negative(ate[yte == 1]).sum())
cv = sweep(yva, pva, ava)
t_bin = float(cv.loc[cv.cost_inr.idxmin()].threshold)
lo_u, hi_u, _ = optimise(yva, pva, ava)
lo_c, hi_c, _ = optimise(yva, pva, ava, max_stepup_rate=0.10)

# Round BEFORE pricing. Publishing a threshold of 0.0232 while costing it at
# 0.02317... means anyone reproducing from the published parameters gets a
# different number. The published params must be the ones that were used.
t_bin, lo_u, hi_u, lo_c, hi_c = (round(v, 4) for v in (t_bin, lo_u, hi_u, lo_c, hi_c))

policies = {
    "approve_everything": (approve_all, None),
    "binary_naive_0.50": (cost_three_way(yte, pte, ate, 0.50, 0.50), {"t": 0.50}),
    "binary_cost_optimal": (cost_three_way(yte, pte, ate, t_bin, t_bin), {"t": t_bin}),
    "three_way": (cost_three_way(yte, pte, ate, lo_u, hi_u),
                  {"t_low": lo_u, "t_high": hi_u}),
    "three_way_stepup_capped_10pct": (cost_three_way(yte, pte, ate, lo_c, hi_c),
                                      {"t_low": lo_c, "t_high": hi_c}),
}
out["policies"] = {
    k: {"cost_per_10k_inr": round(c / n * 1e4),
        "saved_vs_approve_all_per_10k_inr": round((approve_all - c) / n * 1e4),
        "params": prm}
    for k, (c, prm) in policies.items()
}
out["headline_saving_per_10k_inr"] = out["policies"]["three_way_stepup_capped_10pct"][
    "saved_vs_approve_all_per_10k_inr"]
out["decision_mix_capped"] = json.loads(
    summarise(yte, pte, ate, lo_c, hi_c).to_json(orient="records"))

# ---------------------------------------------------------------- honesty check
ct = sweep(yte, pte, ate)
oracle = float(ct.cost_inr.min())
held = cost_three_way(yte, pte, ate, t_bin, t_bin)
out["optimism_penalty"] = {
    "note": "threshold chosen on val vs the best achievable had we tuned on test",
    "val_selected_cost_per_10k_inr": round(held / n * 1e4),
    "test_tuned_cost_per_10k_inr": round(oracle / n * 1e4),
    "penalty_pct": round(100 * (held - oracle) / oracle, 2),
}

# ---------------------------------------------------------------- ablations
abl = [dict(summary(yte, pte, "detector only"), verdict="shipped")]
for name, path, pq in (("+ velocity", "models/velocity.txt", "features.parquet"),
                       ("+ rings", "models/rings.txt", "features_rings.parquet")):
    try:
        mm = lgb.Booster(model_file=path)
        dfp = pd.read_parquet(config.DATA_PROCESSED / pq)
        _, _, tep = split(dfp)
        Xp, yp = make_xy(tep)
        pp = mm.predict(Xp[mm.feature_name()])
        abl.append(dict(summary(yp.to_numpy(), pp, name), verdict="rejected - no measurable gain"))
    except (OSError, ValueError, KeyError) as e:  # a missing artefact must not kill the report
        abl.append({"model": name, "error": str(e)[:120]})
out["ablations"] = abl

# ---------------------------------------------------------------- figures
config.FIGURES.mkdir(parents=True, exist_ok=True)
c10 = ct.cost_inr / n * 1e4
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(ct.threshold, c10, lw=1.8)
ax.axvline(t_bin, ls="--", c="crimson", label=f"chosen on val ({t_bin:.3f})")
ax.axhline(approve_all / n * 1e4, ls=":", c="grey", label="approve everything")
ax.set_xscale("log"); ax.set_xlabel("block threshold"); ax.set_ylabel("cost per 10,000 orders (INR)")
ax.set_title("Cost of every threshold, test split"); ax.legend(); fig.tight_layout()
fig.savefig(config.FIGURES / "cost_curve.png", dpi=140); plt.close(fig)


pr, rc, _ = precision_recall_curve(yte, pte)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(rc, pr, lw=1.8)
ax.axhline(yte.mean(), ls=":", c="grey", label=f"base rate {yte.mean():.3f}")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title(f"Precision-recall, test split (PR-AUC {out['detector']['test']['pr_auc']})")
ax.legend(); fig.tight_layout()
fig.savefig(config.FIGURES / "pr_curve.png", dpi=140); plt.close(fig)

mix = pd.DataFrame(out["decision_mix_capped"])
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh(mix.decision, mix.share, color=["#2e7d32", "#f9a825", "#c62828"])
for i, r in mix.iterrows():
    ax.text(r["share"] + .01, i, f"{r['share']*100:.1f}%  (fraud {r['fraud_rate']*100:.1f}%)", va="center")
ax.set_xlim(0, 1.05); ax.set_xlabel("share of test traffic")
ax.set_title("Decision mix, step-up capped at 10%"); fig.tight_layout()
fig.savefig(config.FIGURES / "decision_mix.png", dpi=140); plt.close(fig)

config.REPORTS.mkdir(parents=True, exist_ok=True)
(config.REPORTS / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps({k: out[k] for k in
                  ("headline_saving_per_10k_inr", "policies", "optimism_penalty")}, indent=2))
print("\nwrote reports/metrics.json + 3 figures in %.0fs" % (time.time() - t0))
