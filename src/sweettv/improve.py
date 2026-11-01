"""Post-first-submission experiments. Earlier audit is now a validation set.

All new features are fitted separately at each temporal cutoff. No hidden test
labels are available. Original frozen-selection and audit files stay immutable.
"""

import argparse, json, pickle, time
import numpy as np
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from .experiment import Fold, OUT, SEED, csr
from .rerank import generate, build_features
from .metrics import evaluate

DEST = OUT / "improvement-v2"
DEST.mkdir(exist_ok=True)
WINDOWS = {
    "early1": ("2020-03-23", "2020-04-27"),
    "early2": ("2020-04-06", "2020-05-11"),
    "early3": ("2020-04-13", "2020-05-18"),
    "dev": ("2020-05-18", "2020-06-22"),
    "audit": ("2020-06-22", "2020-07-27"),
    "long": ("2020-05-18", "2020-07-27"),
    "final": ("2020-07-27", "2020-10-19"),
}


def neighbor_scores(f):
    """BM25-weighted user neighbors and cosine item neighbors, train-only."""
    from implicit.nearest_neighbours import bm25_weight

    x = f.matrix(0.5, 56)
    b = bm25_weight(x, K1=1.2, B=0.75).tocsr()
    # BM25's log(N/(1+df)) is slightly negative for universal items.
    b.data = np.maximum(b.data, 0)
    b = normalize(b)
    sim = (b[f.eval_users] @ b.T).toarray()
    sim[np.arange(len(f.eval_users)), f.eval_users] = 0
    # Deterministic 100 nearest users; no validation interaction enters similarity.
    top = np.argsort(-sim, axis=1, kind="stable")[:, :100]
    weights = np.take_along_axis(sim, top, axis=1)
    knn = csr(
        np.repeat(np.arange(len(top)), 100),
        top.ravel(),
        weights.ravel(),
        (len(top), len(f.users)),
    )
    history = f.matrix(0.8, None)
    history.data = np.log1p(history.data)
    user = (normalize(knn, norm="l1") @ history).toarray()
    binary = f.matrix(0.5, None)
    binary.data[:] = 1
    item = normalize(binary.T)
    cos = (item @ item.T).tocsr()
    cos.setdiag(0)
    cos.eliminate_zeros()
    # Shrink similarities supported by only a few co-viewers.
    co = (binary.T @ binary).tocsr()
    co.data = co.data / (co.data + 10)
    cos = cos.multiply(co).tocsr()
    for row in range(cos.shape[0]):
        a, b = cos.indptr[row : row + 2]
        if b - a > 100:
            keep = np.argpartition(cos.data[a:b], -100)[-100:]
            mask = np.ones(b - a, dtype=bool)
            mask[keep] = False
            cos.data[a:b][mask] = 0
    cos.eliminate_zeros()
    iknn = (history[f.eval_users] @ cos).toarray()
    return {"userknn": user.astype("float32"), "itemknn": iknn.astype("float32")}


def extra_features(f, u, i):
    values = []
    names = []

    def add(n, a):
        a = np.asarray(a, dtype="float32")
        assert a.shape == u.shape and np.isfinite(a).all(), n
        names.append(n)
        values.append(a)

    tr = f.train.copy()
    rowmap = {int(v): k for k, v in enumerate(f.eval_users)}
    shape = (len(f.eval_users), len(f.items))
    # Aggregate counts/rates at recent cutoffs, not future labels.
    period_days = (f.end - f.start) / 86400
    future = (
        f.future.groupby("tv_show_id").size().reindex(f.items, fill_value=0).to_numpy()
    )
    add("future_per_week", np.log1p(future[i] / period_days * 7))
    add("horizon_weeks", np.full(len(u), period_days / 7))
    for days in [14, 28, 56, 112]:
        t = tr[tr.end >= f.start - days * 86400]
        p = f.past[f.past.end >= f.start - days * 86400]
        air = p.groupby("tv_show_id").size().reindex(f.items, fill_value=0).to_numpy()
        add(f"airings_per_week_{days}", np.log1p(air[i] / days * 7))
        for th in [0.0, 0.5, 0.8]:
            t1 = t[t.fraction >= th]
            a = csr(
                t1.user.to_numpy(), t1.item_idx.to_numpy(), np.ones(len(t1)), f.shape
            )[f.eval_users].toarray()
            add(f"recent_count_{days}_{th}", np.log1p(a[u, i]))
            rate = a[u, i] / np.maximum(air[i], 1)
            add(f"watch_rate_{days}_{th}", rate)
            add(f"expected_count_{days}_{th}", np.log1p(rate * future[i]))
            pop = np.asarray(
                (
                    csr(
                        t1.user.to_numpy(),
                        t1.item_idx.to_numpy(),
                        np.ones(len(t1)),
                        f.shape,
                    )
                    > 0
                ).sum(axis=0)
            ).ravel()
            add(f"recent_pop_{days}_{th}", np.log1p(pop[i]))
    # Continuous completeness, single-log completeness, distinct viewing days.
    t = tr[tr.user.isin(rowmap)].copy()
    t["r"] = t.user.map(rowmap)
    for field in ["fraction", "single_fraction"]:
        for agg in ["mean", "max"]:
            z = t.groupby(["r", "item_idx"])[field].agg(agg)
            mat = np.zeros(shape, dtype="float32")
            mat[z.index.get_level_values(0), z.index.get_level_values(1)] = z.to_numpy()
            add(field + "_" + agg, mat[u, i])
    t["day"] = t.start // 86400
    z = t.groupby(["r", "item_idx"]).day.nunique()
    mat = np.zeros(shape, dtype="float32")
    mat[z.index.get_level_values(0), z.index.get_level_values(1)] = z.to_numpy()
    add("unique_watch_days", np.log1p(mat[u, i]))
    age = t.groupby("r").end.max()
    ua = np.full(len(f.eval_users), 999, dtype="float32")
    ua[age.index] = (f.start - age.to_numpy()) / 86400
    add("user_last_days", ua[u])
    # Channel affinity from all matched events, including incomplete views.
    channels = sorted(set(t.channel_id) | set(f.future.channel_id))
    cm = {v: k for k, v in enumerate(channels)}
    for half in [14, 56, None]:
        decay = (
            np.ones(len(t))
            if half is None
            else np.exp2(-(f.start - t.end.to_numpy()) / (half * 86400))
        )
        uc = csr(
            t.r.to_numpy(),
            t.channel_id.map(cm).to_numpy(),
            t.seconds.to_numpy() * decay,
            (len(f.eval_users), len(channels)),
        )
        uc = normalize(uc, norm="l1")
        futuremap = csr(
            f.future.channel_id.map(cm).to_numpy(),
            f.future.tv_show_id.map(f.im).to_numpy(),
            np.ones(len(f.future)),
            (len(channels), len(f.items)),
        )
        expect = (uc @ futuremap).toarray()
        add(f"channel_exposure_{half}", np.log1p(expect[u, i] / period_days * 7))
        add(f"channel_affinity_{half}", expect[u, i] / np.maximum(future[i], 1))
    return np.column_stack(values), names


def cached(name):
    path = DEST / (name + "-features.npz")
    if path.exists():
        return dict(np.load(path, allow_pickle=False))
    f = Fold(*WINDOWS[name])
    s, r = generate(f)
    oldx, oldu, oldi, oldnames, oldc = build_features(f, s, r)
    with (OUT / "reranker/hgb_15_250.pkl").open("rb") as h:
        old = pickle.load(h)
    oldscore = np.zeros_like(s["repeat"])
    oldscore[oldu, oldi] = old.predict_proba(oldx)[:, 1]
    extra = neighbor_scores(f)
    # Add two independently fitted retrieval sources to the common candidate union.
    for n, v in extra.items():
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
    truth = [set(t) for t in f.truth]
    y = np.array([int(f.items[b] in truth[a]) for a, b in zip(u, i)], dtype="int8")
    valid = np.array([len(truth[a]) == 5 and not f.cold[a] for a in u])
    # Save independent-model scores for comparison and ensemble, not as labels.
    np.savez_compressed(DEST / (name + "-scores.npz"), hgb=oldscore, **s)
    payload = dict(
        X=x, u=u, i=i, y=y, valid=valid, names=np.array(names), items=f.items
    )
    np.savez_compressed(path, **payload)
    five = np.array([len(t) == 5 for t in truth])
    pred = f.rank(oldscore)
    ap = np.array(evaluate(f.truth, pred)["per_user_ap"])
    hist = np.asarray(f.base[f.eval_users].sum(axis=1)).ravel()
    cohorts = {}
    for label, mask in [
        ("light", hist < 30),
        ("medium", (hist >= 30) & (hist < 100)),
        ("heavy", hist >= 100),
        ("cold", f.cold),
    ]:
        mask = mask & five
        cohorts[label] = {
            "users": int(mask.sum()),
            "map5": float(ap[mask].mean()) if mask.any() else None,
        }
    groups = {
        k: {"targets": 0, "retrieved_old": 0, "retrieved_new": 0, "hit_top5": 0}
        for k in ["seen", "new_to_user", "new_to_catalog"]
    }
    oracle = []
    oldoracle = []
    for row in np.flatnonzero(five):
        hits = oh = 0
        for item in truth[row]:
            j = f.im[item]
            group = (
                "new_to_catalog"
                if f.pop[j] == 0
                else ("seen" if f.base[f.eval_users[row], j] > 0 else "new_to_user")
            )
            g = groups[group]
            g["targets"] += 1
            g["retrieved_old"] += int(oldc[row, j])
            g["retrieved_new"] += int(c[row, j])
            g["hit_top5"] += int(item in pred[row])
            hits += int(c[row, j])
            oh += int(oldc[row, j])
        oracle.append(hits / 5)
        oldoracle.append(oh / 5)
    result = {
        "original_hgb_evaluation_scope": (
            "in-sample diagnostic only" if name.startswith("early")
            else "no test labels" if name == "final"
            else "already-open temporal validation"
        ),
        "fold": f.audit,
        "original_hgb_map5": float(ap[five].mean()) if five.any() else None,
        "candidate_oracle_map5_old": float(np.mean(oldoracle)) if oldoracle else None,
        "candidate_oracle_map5_expanded": float(np.mean(oracle)) if oracle else None,
        "target_groups": groups,
        "history_cohorts": cohorts,
        "features": names,
        "rows": len(y),
        "individual_metrics": {
            n: {
                k: v
                for k, v in evaluate(f.truth, f.rank(z)).items()
                if k != "per_user_ap"
            }
            for n, z in extra.items()
        },
    }
    (DEST / (name + "-analysis.json")).write_text(json.dumps(result, indent=2))
    print(
        "ANALYSIS",
        name,
        json.dumps({k: v for k, v in result.items() if k not in ["features", "fold"]}),
        flush=True,
    )
    return payload


def fit_data(keys):
    data = [cached(k) for k in keys]
    xs = []
    ys = []
    groups = []
    for d in data:
        m = d["valid"]
        xs.append(d["X"][m])
        ys.append(d["y"][m])
        groups.extend(np.unique(d["u"][m], return_counts=True)[1].tolist())
    return np.concatenate(xs), np.concatenate(ys), np.array(groups)


def run_models(train_keys, eval_key, prefix):
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    X, y, groups = fit_data(train_keys)
    d = cached(eval_key)
    f = Fold(*WINDOWS[eval_key])
    assert all(WINDOWS[k][1] <= WINDOWS[eval_key][0] for k in train_keys)
    configs = [
        ("lgb_bin_15", False, 15, 400),
        ("lgb_bin_31", False, 31, 400),
        ("lgb_rank_15", True, 15, 400),
        ("lgb_rank_31", True, 31, 400),
    ]
    records = []
    for name, ranking, leaves, it in configs:
        t = time.perf_counter()
        params = dict(
            n_estimators=it,
            num_leaves=leaves,
            learning_rate=0.03,
            min_child_samples=80,
            reg_lambda=10,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=6,
            verbosity=-1,
        )
        if ranking:
            model = lgb.LGBMRanker(
                objective="lambdarank", lambdarank_truncation_level=8, **params
            )
            model.fit(X, y, group=groups)
        else:
            model = lgb.LGBMClassifier(**params)
            model.fit(X, y)
        for checkpoint in [100, 200, 400]:
            vals = (
                model.predict(d["X"], num_iteration=checkpoint)
                if ranking
                else model.predict_proba(d["X"], num_iteration=checkpoint)[:, 1]
            )
            score = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
            score[d["u"], d["i"]] = vals
            m = evaluate(f.truth, f.rank(score))
            full = prefix + "_" + name + "_" + str(checkpoint)
            np.save(DEST / (full + ".npy"), score)
            row = {
                "name": full,
                "train_folds": train_keys,
                "eval_fold": eval_key,
                "checkpoint": checkpoint,
                "seconds": time.perf_counter() - t,
                **{k: v for k, v in m.items() if k != "per_user_ap"},
            }
            records.append(row)
            print("RESULT", json.dumps(row), flush=True)
        with (DEST / (prefix + "_" + name + ".pkl")).open("wb") as h:
            pickle.dump(model, h)
        (DEST / (prefix + "-results.json")).write_text(json.dumps(records, indent=2))
    t = time.perf_counter()
    model = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        loss_function="Logloss",
        random_seed=SEED,
        thread_count=6,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)
    for checkpoint in [200, 400, 600]:
        vals = model.predict_proba(d["X"], ntree_end=checkpoint)[:, 1]
        score = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
        score[d["u"], d["i"]] = vals
        m = evaluate(f.truth, f.rank(score))
        full = prefix + "_cat6_" + str(checkpoint)
        np.save(DEST / (full + ".npy"), score)
        row = {
            "name": full,
            "train_folds": train_keys,
            "eval_fold": eval_key,
            "checkpoint": checkpoint,
            "seconds": time.perf_counter() - t,
            **{k: v for k, v in m.items() if k != "per_user_ap"},
        }
        records.append(row)
        print("RESULT", json.dumps(row), flush=True)
    with (DEST / (prefix + "_cat6.pkl")).open("wb") as h:
        pickle.dump(model, h)
    (DEST / (prefix + "-results.json")).write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "train"])
    p.add_argument("--fold", default="dev")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.action == "prepare":
            cached(a.fold)
        else:
            run_models(
                ["early1", "early2", "early3"] + (["dev"] if a.fold == "audit" else []),
                a.fold,
                a.fold,
            )
