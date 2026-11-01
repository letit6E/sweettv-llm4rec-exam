"""Add replay-based candidate discovery and features to the existing ranker."""

import argparse, json, time
import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from .simple_search import DEST
from .session_replay import cached as replay
from .improve import WINDOWS, DEST as OLD, extra_features
from .rerank import build_features
from .experiment import Fold, SEED
from .horizon import rescale
from .metrics import evaluate
from .improve_final import export


def data(key):
    path = DEST / f"{key}_replay_features.npz"
    if path.exists():
        return dict(np.load(path))
    f = Fold(*WINDOWS[key])
    a = replay(key, 112, False)
    old = np.load(OLD / f"{key}-scores.npz")
    scores = {k: old[k] for k in old.files if k != "hgb"}
    ranks = {}
    for k, v in list(scores.items()) + [("replay", a[3])]:
        z = v.copy()
        z[:, ~f.eligible] = -np.inf
        order = np.argsort(-z, axis=1, kind="stable")
        r = np.empty_like(order)
        np.put_along_axis(r, order, np.arange(len(f.items))[None, :], axis=1)
        ranks[k] = r
    x, u, i, n, c = build_features(f, scores, ranks)
    e, en = extra_features(f, u, i)
    x = np.column_stack([x, e])
    n += en
    # Match incumbent feature scale for five-week folds; normalize long/final only.
    if key in ["long", "final"]:
        x = rescale(x, n, (f.end - f.start) / 604800)
    for mode in range(4):
        z = a[mode]
        order = np.argsort(-z, axis=1, kind="stable")
        r = np.empty_like(order)
        np.put_along_axis(r, order, np.arange(len(f.items))[None, :], axis=1)
        x = np.column_stack(
            [
                x,
                np.log1p(z[u, i] * 5 / ((f.end - f.start) / 604800)),
                z[u, i] / np.maximum(z.max(axis=1)[u], 1e-8),
                1 / (r[u, i] + 1),
            ]
        )
        n += [
            f"replay_{mode}_count",
            f"replay_{mode}_normalized",
            f"replay_{mode}_rank",
        ]
    truth = [set(t) for t in f.truth]
    y = np.array([int(f.items[j] in truth[row]) for row, j in zip(u, i)], dtype="int8")
    valid = np.array([len(truth[row]) == 5 and not f.cold[row] for row in u])
    d = dict(X=x.astype("float32"), u=u, i=i, y=y, valid=valid, names=np.array(n))
    np.savez_compressed(path, **d)
    five = np.array([len(t) == 5 for t in f.truth])
    oracle = (
        float(
            np.mean(
                [
                    sum(c[row, f.im[it]] for it in truth[row]) / 5
                    for row in np.flatnonzero(five)
                ]
            )
        )
        if five.any()
        else None
    )
    (DEST / f"{key}_replay_features.json").write_text(
        json.dumps({"features": n, "rows": len(y), "oracle_map5": oracle}, indent=2)
    )
    print("FEATURES", key, len(y), "oracle", oracle, flush=True)
    return d


def run(key):
    keys = (
        ["early1", "early2", "early3"]
        + (["dev"] if key in ["audit", "final"] else [])
        + (["audit"] if key == "final" else [])
    )
    xs = []
    ys = []
    t = time.perf_counter()
    for k in keys:
        assert WINDOWS[k][1] <= WINDOWS[key][0]
        d = data(k)
        xs.append(d["X"][d["valid"]])
        ys.append(d["y"][d["valid"]])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        random_seed=SEED,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y)
    model.save_model(str(DEST / f"{key}_replay_ranker.cbm"))
    out = []
    for ev in [key] + (["long"] if key == "dev" else []):
        d = data(ev)
        f = Fold(*WINDOWS[ev])
        for it in [200, 400]:
            s = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
            s[d["u"], d["i"]] = model.predict_proba(d["X"], ntree_end=it)[:, 1]
            np.save(DEST / f"{ev}_replay_ranker{it}.npy", s)
            r = {
                "eval": ev,
                "trees": it,
                "training_rows": len(y),
                "seconds": time.perf_counter() - t,
                **{
                    k: v
                    for k, v in evaluate(f.truth, f.rank(s)).items()
                    if k != "per_user_ap"
                },
            }
            if ev == "final":
                r.update(export(f, s, f"v5_replay_ranker{it}"))
            out.append(r)
            print("RESULT", json.dumps(r), flush=True)
    (DEST / f"{key}_replay_ranker.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="dev")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.fold)
