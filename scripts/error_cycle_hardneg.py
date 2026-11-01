"""Focus second-stage training on competitive mistakes of a first-stage classifier."""

import argparse, json, time, pickle
import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from sweettv.experiment import Fold, SEED
from sweettv.improve import cached, WINDOWS, DEST as OLD
from sweettv.error_cycles import DEST
from sweettv.horizon import rescale
from sweettv.metrics import evaluate
from sweettv.improve_final import export

p = argparse.ArgumentParser()
p.add_argument("--fold", default="dev")
p.add_argument("--top", type=int, default=30)
a = p.parse_args()
with threadpool_limits(limits=1):
    keys = (
        ["early1", "early2", "early3"]
        + (["dev"] if a.fold in ["audit", "final"] else [])
        + (["audit"] if a.fold == "final" else [])
    )
    base = pickle.load(
        open(
            OLD
            / (
                {
                    "dev": "dev_cat6.pkl",
                    "audit": "audit_cat6.pkl",
                    "final": "v2_cat200.pkl",
                }[a.fold]
            ),
            "rb",
        )
    )
    xs = []
    ys = []
    t = time.perf_counter()
    for k in keys:
        assert WINDOWS[k][1] <= WINDOWS[a.fold][0]
        d = cached(k)
        x = d["X"]
        score = base.predict_proba(x, ntree_end=200)[:, 1]
        keep = d["y"] > 0
        for u in np.unique(d["u"]):
            idx = np.flatnonzero(d["u"] == u)
            keep[idx[np.argsort(-score[idx], kind="stable")[: a.top]]] = True
        keep &= d["valid"]
        xs.append(x[keep])
        ys.append(d["y"][keep])
    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        random_seed=SEED,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    model.fit(x, y)
    model.save_model(str(DEST / f"{a.fold}_hard{a.top}.cbm"))
    res = []
    for k in [a.fold] + (["long"] if a.fold == "dev" else []):
        d = cached(k)
        f = Fold(*WINDOWS[k])
        xx = (
            rescale(d["X"], d["names"].tolist(), (f.end - f.start) / 604800)
            if k in ["long", "final"]
            else d["X"]
        )
        v = model.predict_proba(xx)[:, 1]
        z = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
        z[d["u"], d["i"]] = v
        first = base.predict_proba(xx, ntree_end=200)[:, 1]
        gate = np.zeros(len(v), dtype=bool)
        for user in np.unique(d["u"]):
            idx = np.flatnonzero(d["u"] == user)
            gate[idx[np.argsort(-first[idx], kind="stable")[: a.top]]] = True
        z[d["u"][~gate], d["i"][~gate]] = -1e9
        np.save(DEST / f"{k}_hard{a.top}.npy", z)
        r = {
            "eval": k,
            "top": a.top,
            "rows": len(y),
            "seconds": time.perf_counter() - t,
            **{
                k: v
                for k, v in evaluate(f.truth, f.rank(z)).items()
                if k != "per_user_ap"
            },
        }
        if k == "final":
            r.update(export(f, z, f"v4_hard{a.top}"))
        print("RESULT", json.dumps(r), flush=True)
        res.append(r)
    (DEST / f"{a.fold}_hard{a.top}.json").write_text(json.dumps(res, indent=2))
