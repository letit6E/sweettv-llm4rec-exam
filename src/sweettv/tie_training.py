"""Soft training labels for unresolved Top-5 frequency ties; validation unchanged."""

import argparse, json, time
import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from .experiment import Fold, SEED
from .improve import cached, WINDOWS
from .discovery_context import DEST
from .horizon import rescale
from .metrics import evaluate


def soft_targets(f, u, i):
    grouped = f.truth_events.groupby(["user", "tv_show_id"]).size()
    target = np.zeros(len(u), dtype="float32")
    rowmap = {int(v): r for r, v in enumerate(f.eval_users)}
    lookup = {}
    for user, g in grouped.groupby(level=0):
        if int(user) not in rowmap:
            continue
        values = g.to_numpy()
        if len(values) < 5:
            continue
        threshold = np.sort(values)[-5]
        tied = (values == threshold).sum()
        credit = (5 - int((values > threshold).sum())) / tied
        for (_, item), n in g.items():
            lookup[(rowmap[int(user)], f.im[int(item)])] = (
                1.0 if n > threshold else credit if n == threshold else 0.0
            )
    for j, (r, it) in enumerate(zip(u, i)):
        target[j] = lookup.get((int(r), int(it)), 0)
    return target


def run(key):
    keys = (
        ["early1", "early2", "early3"]
        + (["dev"] if key in ["audit", "final"] else [])
        + (["audit"] if key == "final" else [])
    )
    xs = []
    ys = []
    for k in keys:
        assert WINDOWS[k][1] <= WINDOWS[key][0]
        d = cached(k)
        f = Fold(*WINDOWS[k])
        mask = d["valid"]
        xs.append(d["X"][mask])
        ys.append(soft_targets(f, d["u"], d["i"])[mask])
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    t = time.perf_counter()
    model = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        loss_function="CrossEntropy",
        random_seed=SEED,
        thread_count=5,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)
    model.save_model(str(DEST / f"{key}_soft_ties.cbm"))
    records = []
    for k in [key] + (["long"] if key == "dev" else []):
        d = cached(k)
        f = Fold(*WINDOWS[k])
        x = rescale(d["X"], d["names"].tolist(), (f.end - f.start) / 604800)
        for trees in [100, 200, 400, 600]:
            v = model.predict_proba(x, ntree_end=trees)[:, 1]
            s = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
            s[d["u"], d["i"]] = v
            np.save(DEST / f"{k}_soft_ties_{trees}.npy", s)
            r = dict(
                fold=k,
                trees=trees,
                training_rows=len(y),
                fractional_labels=int(((y > 0) & (y < 1)).sum()),
                seconds=time.perf_counter() - t,
                **{
                    n: v
                    for n, v in evaluate(f.truth, f.rank(s)).items()
                    if n != "per_user_ap"
                },
            )
            records.append(r)
            print("RESULT", json.dumps(r), flush=True)
    (DEST / f"{key}_soft_ties.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="dev")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.fold)
