"""Expand supervised examples to other supplied users; evaluation stays fixed.

Each pseudo-window's labels are used only after its end. Training batches are
feature-memory batches, not random temporal validation splits.
"""

import json, pickle, time
import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from .experiment import Fold, SEED
from .improve import (
    DEST,
    WINDOWS,
    generate,
    build_features,
    neighbor_scores,
    extra_features,
    cached,
)
from .metrics import evaluate
from .improve_final import export


def prepare(key):
    path = DEST / (key + "-allusers.npz")
    if path.exists():
        return dict(np.load(path))
    f = Fold(*WINDOWS[key])
    template = set(f.eval_users.tolist())
    counts = (
        f.truth_events.groupby(["user", "tv_show_id"])
        .size()
        .reset_index(name="count")
        .sort_values(["user", "count", "tv_show_id"], ascending=[True, False, True])
    )
    truth = (
        counts.groupby("user").head(5).groupby("user").tv_show_id.apply(set).to_dict()
    )
    # Existing template rows are already cached; generate only additional eligible users.
    users = [
        u
        for u, t in truth.items()
        if len(t) == 5 and u not in template and f.base[int(u)].nnz > 0
    ]
    xs = []
    ys = []
    sizes = []
    t0 = time.perf_counter()
    for off in range(0, len(users), 800):
        f.eval_users = np.array(users[off : off + 800])
        s, r = generate(f)
        for n, v in neighbor_scores(f).items():
            s[n] = v
            a = v.copy()
            a[:, ~f.eligible] = -np.inf
            order = np.argsort(-a, axis=1, kind="stable")
            rank = np.empty_like(order, dtype="int32")
            np.put_along_axis(rank, order, np.arange(len(f.items))[None, :], axis=1)
            r[n] = rank
        x, u, i, names, c = build_features(f, s, r)
        ex, en = extra_features(f, u, i)
        x = np.column_stack([x, ex])
        names += en
        y = np.array(
            [int(f.items[j] in truth[int(f.eval_users[a])]) for a, j in zip(u, i)],
            dtype="int8",
        )
        xs.append(x)
        ys.append(y)
        sizes.extend(np.unique(u, return_counts=True)[1].tolist())
        print(
            "ALLUSERFEATURES",
            key,
            off,
            len(x),
            round(time.perf_counter() - t0, 1),
            flush=True,
        )
    d = {
        "X": np.concatenate(xs),
        "y": np.concatenate(ys),
        "groups": np.array(sizes),
        "names": np.array(names),
    }
    np.savez_compressed(path, **d)
    (DEST / (key + "-allusers-info.json")).write_text(
        json.dumps(
            {
                "additional_users": len(users),
                "pairs": len(d["y"]),
                "seconds": time.perf_counter() - t0,
                "scope": "non-template supplied users only; all labels completed before downstream prediction cutoff",
            },
            indent=2,
        )
    )
    return d


def data(keys):
    xs = []
    ys = []
    sizes = []
    for k in keys:
        d = cached(k)
        m = d["valid"]
        xs.append(d["X"][m])
        ys.append(d["y"][m])
        sizes.extend(np.unique(d["u"][m], return_counts=True)[1].tolist())
        d = prepare(k)
        xs.append(d["X"])
        ys.append(d["y"])
        sizes.extend(d["groups"].tolist())
    return np.concatenate(xs), np.concatenate(ys), np.array(sizes)


def run():
    records = []
    for key in ["dev", "audit", "final"]:
        keys = (
            ["early1", "early2", "early3"]
            + (["dev"] if key in ["audit", "final"] else [])
            + (["audit"] if key == "final" else [])
        )
        assert all(WINDOWS[k][1] <= WINDOWS[key][0] for k in keys)
        X, y, g = data(keys)
        d = cached(key)
        f = Fold(*WINDOWS[key])
        t = time.perf_counter()
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=10,
            loss_function="Logloss",
            random_seed=SEED,
            thread_count=5,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X, y)
        for checkpoint in [200, 500]:
            v = model.predict_proba(d["X"], ntree_end=checkpoint)[:, 1]
            score = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
            score[d["u"], d["i"]] = v
            name = f"{key}_allusers_cat{checkpoint}"
            np.save(DEST / (name + ".npy"), score)
            row = {
                "name": name,
                "train_folds": keys,
                "training_pairs": len(y),
                "training_queries": len(g),
                "seconds": time.perf_counter() - t,
            }
            if key == "final":
                row.update(export(f, score, f"v3_allusers_cat{checkpoint}"))
            else:
                row.update(
                    {
                        k: v
                        for k, v in evaluate(f.truth, f.rank(score)).items()
                        if k != "per_user_ap"
                    }
                )
            records.append(row)
            print("ALLUSERRESULT", json.dumps(row), flush=True)
            (DEST / "allusers-results.json").write_text(json.dumps(records, indent=2))
        with (DEST / f"{key}_allusers_cat.pkl").open("wb") as h:
            pickle.dump(model, h)
        del X, y, g, model


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
