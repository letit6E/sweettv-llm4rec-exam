"""Test whether a separate unseen-item classifier repairs repeat-dominated ranking."""

import argparse, json, time
import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from .experiment import Fold, SEED, OUT
from .improve import cached, WINDOWS, DEST as OLD
from .horizon import rescale
from .metrics import evaluate
from .simple_search import baseline
from .blend import rrf

DEST = OUT / "discovery-v6"
DEST.mkdir(exist_ok=True)


def evaluate_grid(key, model):
    d = cached(key)
    f = Fold(*WINDOWS[key])
    names = d["names"].tolist()
    x = (
        rescale(d["X"], names, (f.end - f.start) / 604800)
        if key in ["long", "final"]
        else d["X"]
    )
    unseen = x[:, names.index("days_since_last")] >= 998
    current = baseline(key)
    channel = np.load(OUT / "approaches-v5" / f"{key}_channel.npy")
    records = []
    candidates = [("calibration", 0, current[d["u"][unseen], d["i"][unseen]])]
    for trees in [200, 500]:
        candidates.append(
            ("specialist", trees, model.predict_proba(x[unseen], ntree_end=trees)[:, 1])
        )
    for method, trees, v in candidates:
        for scale in [0.5, 1, 1.5, 2, 3, 5]:
            score = current.copy()
            score[d["u"][unseen], d["i"][unseen]] = np.clip(v * scale, 0, 1)
            score = 0.9 * rrf(f, score) + 0.1 * rrf(f, channel)
            m = evaluate(f.truth, f.rank(score))["map5_five_targets"]
            records.append(
                dict(method=method, trees=trees, scale=scale, fold=key, map5=m)
            )
    return records


def main(fold, allusers):
    keys = (
        ["early1", "early2", "early3"]
        + (["dev"] if fold in ["audit", "final"] else [])
        + (["audit"] if fold == "final" else [])
    )
    xs = []
    ys = []
    t = time.perf_counter()
    for key in keys:
        assert WINDOWS[key][1] <= WINDOWS[fold][0]
        d = cached(key)
        names = d["names"].tolist()
        mask = (d["X"][:, names.index("days_since_last")] >= 998) & d["valid"]
        xs.append(d["X"][mask])
        ys.append(d["y"][mask])
        if allusers:
            a = np.load(OLD / f"{key}-allusers.npz")
            mask = a["X"][:, names.index("days_since_last")] >= 998
            xs.append(a["X"][mask])
            ys.append(a["y"][mask])
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        thread_count=5,
        verbose=False,
        allow_writing_files=False,
        random_seed=SEED,
    )
    model.fit(X, y)
    tag = f"{fold}_novel_{allusers}"
    model.save_model(str(DEST / f"{tag}.cbm"))
    records = []
    for key in [fold] + (["long"] if fold == "dev" else []):
        records += evaluate_grid(key, model)
        print(
            "BEST",
            key,
            max([r for r in records if r["fold"] == key], key=lambda x: x["map5"]),
            flush=True,
        )
    payload = {
        "train_rows": len(y),
        "positive_rows": int(y.sum()),
        "seconds": time.perf_counter() - t,
        "allusers": allusers,
        "train_keys": keys,
        "grid": records,
    }
    (DEST / f"{tag}.json").write_text(json.dumps(payload, indent=2))
    print("DONE", tag, len(y), y.sum(), time.perf_counter() - t, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="dev")
    p.add_argument("--allusers", action="store_true")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        main(a.fold, a.allusers)
