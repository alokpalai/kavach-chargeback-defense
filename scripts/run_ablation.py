"""Step 12 ablation: does the ring layer add recall the detector cannot reach?"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from kavach.data.load import load_raw
from kavach.data.split import split
from kavach.sentinel.rings import add_ring_features
from kavach.features.base import make_xy
from kavach.detect.train import train
from kavach.eval.metrics import summary
from kavach.policy.costs import amounts_inr
from kavach.policy.decide import optimise, cost_three_way, summarise
from kavach import config

t0 = time.time()
df = add_ring_features(load_raw())
print("ring features built in %.0fs -> %d cols" % (time.time() - t0, df.shape[1]))
df.to_parquet(config.DATA_PROCESSED / "features_rings.parquet", index=False)

tr, va, te = split(df)
m = train(tr, va)
m.save_model("models/rings.txt")
print("trained %d trees in %.0fs" % (m.best_iteration, time.time() - t0))

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
    print("  %-18s cost/10k Rs %9.0f | saves Rs %9.0f" % (nm, c/n*1e4, (approve_all-c)/n*1e4))

imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
ring_cols = [c for c in imp.index if c.startswith("ring__")]
print("\nring features in top 30:", sum(c.startswith("ring__") for c in imp.head(30).index))
print(imp[ring_cols].head(6).round(0).to_string())
