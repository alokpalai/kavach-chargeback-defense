"""Step 12 ablation: does the ring layer add recall the detector cannot reach?"""
import time

import lightgbm as lgb
import pandas as pd

from kavach import config
from kavach.data.load import load_raw
from kavach.data.split import split
from kavach.detect.train import train
from kavach.eval.metrics import summary
from kavach.features.base import make_xy
from kavach.policy.costs import amounts_inr
from kavach.policy.decide import cost_three_way, optimise
from kavach.sentinel.rings import add_ring_features

t0 = time.time()
df = add_ring_features(load_raw())
print(f"ring features built in {time.time() - t0:.0f}s -> {df.shape[1]} cols")
df.to_parquet(config.DATA_PROCESSED / "features_rings.parquet", index=False)

tr, va, te = split(df)
m = train(tr, va)
m.save_model("models/rings.txt")
print(f"trained {m.best_iteration} trees in {time.time() - t0:.0f}s")

Xte, yte = make_xy(te); yte = yte.to_numpy()
Xva, yva = make_xy(va); yva = yva.to_numpy()
p_ring_te, p_ring_va = m.predict(Xte), m.predict(Xva)

base = lgb.Booster(model_file="models/baseline.txt")
bf = base.feature_name()
p_base_te, p_base_va = base.predict(Xte[bf]), base.predict(Xva[bf])

print("\n--- ABLATION (test split) ---")
res = pd.DataFrame([summary(yte, p_base_te, "detector only"),
                    summary(yte, p_ring_te, "detector + rings")])
print(res.to_string(index=False))

ate, ava = amounts_inr(te), amounts_inr(va)
n = len(yte)
approve_all = float(config.cost_false_negative(ate[yte == 1]).sum())
print("\n--- MONEY (three-way, step-up capped at 10%, fitted on val) ---")
for nm, pv, pt in (("detector only", p_base_va, p_base_te),
                   ("detector + rings", p_ring_va, p_ring_te)):
    lo, hi, _ = optimise(yva, pv, ava, max_stepup_rate=0.10)
    c = cost_three_way(yte, pt, ate, lo, hi)
    print(f"  {nm:<18} cost/10k Rs {c/n*1e4:9.0f} | saves Rs {(approve_all-c)/n*1e4:9.0f}")

imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
ring_cols = [c for c in imp.index if c.startswith("ring__")]
print("\nring features in top 30:", sum(c.startswith("ring__") for c in imp.head(30).index))
print(imp[ring_cols].head(6).round(0).to_string())
