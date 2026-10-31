"""Train score fusion on earlier temporal windows, never development labels."""

import json, time, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from .experiment import Fold, OUT, SEED, normalized
from .metrics import evaluate

COMPONENTS = {
    "repeat": {"method": "repeat", "threshold": 0.5, "half": None},
    "content": {"method": "content", "half": 28, "title": True},
    "als": {
        "method": "als",
        "factors": 32,
        "alpha": 10,
        "reg": 1,
        "iterations": 25,
        "half": None,
    },
    "ease": {"method": "ease", "reg": 10},
    "channel": {"method": "channel", "half": 56, "hourly": True},
    "ppr": {"method": "ppr", "alpha": 0.5, "neighbors": 100},
}


def generate(fold):
    scores = {}
    ranks = {}
    t = time.perf_counter()
    for n, c in COMPONENTS.items():
        scores[n] = fold.model(c).astype(np.float32)
        eligible_scores = scores[n].copy()
        eligible_scores[:, ~fold.eligible] = -np.inf
        order = np.argsort(-eligible_scores, axis=1, kind="stable")
        rank = np.empty_like(order, dtype=np.int32)
        np.put_along_axis(rank, order, np.arange(len(fold.items))[None, :], axis=1)
        ranks[n] = rank
        print("component", n, "seconds", round(time.perf_counter() - t, 2), flush=True)
    return scores, ranks


def build_features(fold, scores, ranks):
    # Candidate generation is independent of validation labels.
    # Union top 100 for each component plus 100 training-popularity items.
    candidates = np.zeros_like(scores["repeat"], dtype=bool)
    for r in ranks.values():
        candidates |= r < 100
    poporder = np.lexsort((fold.items, -fold.pop))
    poporder = poporder[fold.eligible[poporder]][:100]
    candidates[:, poporder] = True
    candidates &= fold.eligible[None, :]
    uu, ii = np.nonzero(candidates)
    features = []
    names = []

    def add(name, value):
        names.append(name)
        features.append(np.asarray(value, dtype=np.float32))

    for n, s in scores.items():
        add(n + "_score", s[uu, ii])
        add(n + "_normalized", normalized(s)[uu, ii])
        add(n + "_reciprocal_rank", 1 / (ranks[n][uu, ii] + 1))
    for threshold in [0.2, 0.5, 0.8]:
        for half in [None, 14, 28, 56]:
            x = fold.matrix(threshold, half)[fold.eval_users].toarray()
            add(f"count_{threshold}_{half}", np.log1p(x[uu, ii]))
            add(
                f"share_{threshold}_{half}",
                x[uu, ii] / np.maximum(x.sum(axis=1)[uu], 1e-9),
            )
    full = fold.matrix(0.8, None)
    hist = np.asarray(full.sum(axis=1)).ravel()[fold.eval_users]
    add("user_total_full", np.log1p(hist[uu]))
    add("user_distinct", np.diff(full.indptr)[fold.eval_users][uu])
    add("item_popularity", np.log1p(fold.pop[ii]))
    last = fold.train.groupby(["user", "item_idx"]).end.max().reset_index()
    ages = np.full((len(fold.eval_users), len(fold.items)), 999, dtype=np.float32)
    mapping = {int(u): j for j, u in enumerate(fold.eval_users)}
    last = last[last.user.isin(mapping)]
    ages[last.user.map(mapping).to_numpy(), last.item_idx.to_numpy()] = (
        fold.start - last.end.to_numpy()
    ) / 86400
    add("days_since_last", ages[uu, ii])
    freq = (
        fold.future.groupby("tv_show_id")
        .size()
        .reindex(fold.items, fill_value=0)
        .to_numpy()
    )
    oldfreq = (
        fold.past.groupby("tv_show_id")
        .size()
        .reindex(fold.items, fill_value=0)
        .to_numpy()
    )
    add("future_airings", np.log1p(freq[ii]))
    add("past_airings", np.log1p(oldfreq[ii]))
    add("airing_ratio", np.log1p(freq[ii]) / np.maximum(np.log1p(oldfreq[ii]), 1))
    X = np.column_stack(features)
    assert np.isfinite(X).all()
    return X, uu, ii, names, candidates


def candidate_recall(fold, candidates):
    hit = total = 0
    for row, t in zip(candidates, fold.truth):
        hit += sum(row[fold.im[i]] for i in t)
        total += len(t)
    return float(hit / total) if total else None


def development():
    allx = []
    ally = []
    allw = []
    training = []
    names = None
    # Each label window ends before 18 May, the outer development cutoff.
    for start, end in [
        ("2020-03-23", "2020-04-27"),
        ("2020-04-06", "2020-05-11"),
        ("2020-04-13", "2020-05-18"),
    ]:
        f = Fold(start, end)
        s, r = generate(f)
        x, u, i, names, c = build_features(f, s, r)
        truth = [set(t) for t in f.truth]
        y = np.array([int(f.items[j] in truth[a]) for a, j in zip(u, i)])
        valid = np.array([len(truth[a]) == 5 and not f.cold[a] for a in u])
        # Equal total sample weight per user, keeping all generated hard negatives.
        per = np.bincount(u, minlength=len(f.eval_users))
        w = 1 / np.maximum(per[u], 1)
        allx.append(x[valid])
        ally.append(y[valid])
        allw.append(w[valid])
        training.append(
            {
                **f.audit,
                "candidate_recall": candidate_recall(f, c),
                "training_pairs": int(valid.sum()),
                "positive_pairs": int(y[valid].sum()),
            }
        )
        print("pseudo fold", start, end, training[-1], flush=True)
    X = np.concatenate(allx)
    y = np.concatenate(ally)
    w = np.concatenate(allw)
    w *= len(w) / w.sum()
    f = Fold("2020-05-18", "2020-06-22")
    s, r = generate(f)
    x, u, i, devnames, c = build_features(f, s, r)
    assert names == devnames
    folder = OUT / "reranker"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / "development-features.npz", X=x, user=u, item=i)
    # Retain fit data to reproduce model without rebuilding all pseudo-folds.
    np.savez_compressed(folder / "training-features.npz", X=X, y=y, weight=w)
    results = []
    best = None
    for leaves in [7, 15, 31]:
        for iterations in [100, 250]:
            cfg = {
                "max_leaf_nodes": leaves,
                "max_iter": iterations,
                "learning_rate": 0.05,
                "l2_regularization": 10,
                "min_samples_leaf": 50,
                "random_state": SEED,
                "early_stopping": False,
            }
            t = time.perf_counter()
            model = HistGradientBoostingClassifier(**cfg)
            model.fit(X, y, sample_weight=w)
            score = np.zeros((len(f.eval_users), len(f.items)), dtype=np.float32)
            score[u, i] = model.predict_proba(x)[:, 1]
            metrics = evaluate(f.truth, f.rank(score))
            name = f"hgb_{leaves}_{iterations}"
            with (folder / (name + ".pkl")).open("wb") as file:
                pickle.dump(model, file)
            np.save(folder / (name + ".npy"), score)
            row = {
                "name": name,
                "config": cfg,
                "seconds": time.perf_counter() - t,
                **metrics,
            }
            results.append(row)
            print(
                "reranker",
                name,
                row["map5_five_targets"],
                "seconds",
                round(row["seconds"], 2),
                flush=True,
            )
            if best is None or row["map5_five_targets"] > best["map5_five_targets"]:
                best = row
            (folder / "results.json").write_text(json.dumps(results, indent=2))
    info = {
        "training": training,
        "feature_names": names,
        "training_pairs": len(y),
        "training_positive_pairs": int(y.sum()),
        "validation_candidate_recall": candidate_recall(f, c),
        "selected": best,
        "leakage_check": "all supervised labels end by 2020-05-18; development labels used only for configuration selection",
    }
    (folder / "selection.json").write_text(json.dumps(info, indent=2))
    print("BEST_RERANKER", best["name"], best["map5_five_targets"], flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        development()
