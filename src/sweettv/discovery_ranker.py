"""Expanded discovery candidates and past-only semantic/context features."""

import argparse, json, time
import numpy as np
from catboost import CatBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from .experiment import Fold, SEED, csr
from .improve import WINDOWS, DEST as OLD, extra_features
from .rerank import build_features
from .discovery_context import DEST, context_scores, title_key
from .horizon import rescale
from .metrics import evaluate


def signals(f):
    out = context_scores(f)
    meta = (
        f.past.sort_values("end")
        .drop_duplicates("tv_show_id", keep="last")
        .set_index("tv_show_id")
        .reindex(f.items)
        .fillna("")
    )
    history = f.matrix(0.2, 56, weight="fraction")
    history.data = np.log1p(history.data)
    for mode in ["genre", "actors", "title_family"]:
        docs = []
        for row in meta.itertuples():
            if mode == "genre":
                vals = [
                    str(getattr(row, f"tv_show_genre_{j}")).strip().lower()
                    for j in [1, 2, 3]
                ]
            elif mode == "actors":
                vals = [s.strip().lower() for s in str(row.actors).split(",")]
            else:
                vals = [title_key(row.tv_show_title, row.tv_show_category)]
            docs.append({k: 1 for k in vals if k})
        enc = DictVectorizer(dtype=np.float32).fit_transform(docs)
        df = np.asarray((enc > 0).sum(axis=0)).ravel()
        enc = normalize(enc.multiply(np.log((1 + len(f.items)) / (1 + df)) + 1)).tocsr()
        profile = normalize(history[f.eval_users] @ enc)
        out[mode] = (profile @ enc.T).toarray().astype("float32")
    # Long-window channel evidence gives new programs a retrieval route even
    # when there is no recent-channel activity and no program interaction.
    from .simple_search import score

    for days in [28, 112]:
        out[f"raw_channel_{days}"] = score(
            f, dict(family="channel", days=days, duration=True, mode="future_freq")
        )
    return out


def prepare(key):
    path = DEST / f"{key}_discovery_features.npz"
    if path.exists():
        return dict(np.load(path))
    t = time.perf_counter()
    f = Fold(*WINDOWS[key])
    old = dict(np.load(OLD / f"{key}-scores.npz"))
    old.pop("hgb")
    new = signals(f)
    np.savez_compressed(DEST / f"{key}_discovery_signals.npz", **new)
    ranks = {}
    for name, score in {**old, **new}.items():
        s = score.copy()
        s[:, ~f.eligible] = -np.inf
        order = np.argsort(-s, axis=1, kind="stable")
        rank = np.empty_like(order, dtype=np.int32)
        np.put_along_axis(rank, order, np.arange(len(f.items))[None, :], axis=1)
        ranks[name] = rank
    x, u, i, names, c = build_features(f, old, ranks)
    extra, en = extra_features(f, u, i)
    x = np.column_stack([x, extra])
    names += en
    assert names == np.load(OLD / f"{key}-features.npz")["names"].tolist()
    x = rescale(x, names, (f.end - f.start) / 604800)
    vals = []
    nn = []

    def add(n, v):
        nn.append(n)
        vals.append(np.asarray(v, dtype="float32"))

    for name, s in new.items():
        v = s[u, i]
        if name.startswith("raw_channel"):
            v = v * 5 / ((f.end - f.start) / 604800)
        add(name + "_value", np.log1p(v))
        add(name + "_maxnorm", s[u, i] / np.maximum(s.max(axis=1)[u], 1e-8))
        add(name + "_rank", 1 / (ranks[name][u, i] + 1))
    meta = (
        f.past.sort_values("end")
        .drop_duplicates("tv_show_id", keep="last")
        .set_index("tv_show_id")
        .reindex(f.items)
        .fillna("")
    )
    categories = [
        "Инфо",
        "Познавательное",
        "Развлечения",
        "Сериалы",
        "Детям",
        "Другое",
        "Спорт",
        "Музыка",
        "Фильмы",
        "",
    ]
    for category in categories:
        add(
            "item_category_" + category,
            (meta.tv_show_category.to_numpy()[i] == category).astype(float),
        )
    # Global audience growth and new availability; all audience counts are past.
    for days in [14, 28, 56]:
        t1 = f.train[
            (f.train.end >= f.start - days * 86400) & (f.train.fraction >= 0.8)
        ]
        t2 = f.train[
            (f.train.end >= f.start - 2 * days * 86400)
            & (f.train.end < f.start - days * 86400)
            & (f.train.fraction >= 0.8)
        ]
        a = np.asarray(
            (csr(t1.user, t1.item_idx, np.ones(len(t1)), f.shape) > 0).sum(axis=0)
        ).ravel()
        b = np.asarray(
            (csr(t2.user, t2.item_idx, np.ones(len(t2)), f.shape) > 0).sum(axis=0)
        ).ravel()
        add(f"audience_growth_{days}", np.log((a[i] + 3) / (b[i] + 3)))
    x = np.column_stack([x, *vals]).astype("float32")
    names += nn
    truth = [set(t) for t in f.truth]
    y = np.array([int(f.items[j] in truth[r]) for r, j in zip(u, i)], dtype="int8")
    valid = np.array([len(truth[r]) == 5 and not f.cold[r] for r in u])
    payload = dict(
        X=x, u=u, i=i, y=y, valid=valid, names=np.array(names), items=f.items
    )
    np.savez_compressed(path, **payload)
    five = [r for r, t in enumerate(truth) if len(t) == 5]
    oracle = (
        float(np.mean([sum(c[r, f.im[v]] for v in truth[r]) / 5 for r in five]))
        if five
        else None
    )
    info = {
        "rows": len(y),
        "features": names,
        "oracle_map5": oracle,
        "seconds": time.perf_counter() - t,
    }
    (DEST / f"{key}_discovery_features.json").write_text(json.dumps(info, indent=2))
    print("FEATURES", key, len(y), len(names), oracle, info["seconds"], flush=True)
    return payload


def run(fold, mode):
    keys = (
        ["early1", "early2", "early3"]
        + (["dev"] if fold in ["audit", "final"] else [])
        + (["audit"] if fold == "final" else [])
    )
    xs = []
    ys = []
    for key in keys:
        assert WINDOWS[key][1] <= WINDOWS[fold][0]
        d = prepare(key)
        mask = d["valid"]
        if mode == "novel":
            mask = mask & (
                d["X"][:, d["names"].tolist().index("days_since_last")] >= 998
            )
        xs.append(d["X"][mask])
        ys.append(d["y"][mask])
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    t = time.perf_counter()
    model = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        random_seed=SEED,
        thread_count=5,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)
    model.save_model(str(DEST / f"{fold}_discovery_{mode}.cbm"))
    records = []
    for key in [fold] + (["long"] if fold == "dev" else []):
        d = prepare(key)
        f = Fold(*WINDOWS[key])
        x = d["X"]
        for trees in [200, 400, 600]:
            v = model.predict_proba(x, ntree_end=trees)[:, 1]
            s = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
            s[d["u"], d["i"]] = v
            np.save(DEST / f"{key}_discovery_{mode}_{trees}.npy", s)
            m = (
                {"evaluation": "Unavailable: official test labels are hidden"}
                if key == "final"
                else {
                    k: v
                    for k, v in evaluate(f.truth, f.rank(s)).items()
                    if k != "per_user_ap"
                }
            )
            r = dict(
                fold=key,
                mode=mode,
                trees=trees,
                train_rows=len(y),
                positive_rows=int(y.sum()),
                seconds=time.perf_counter() - t,
                **m,
            )
            records.append(r)
            print("RESULT", json.dumps(r), flush=True)
    (DEST / f"{fold}_discovery_{mode}.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="dev")
    p.add_argument("--mode", default="joint", choices=["joint", "novel"])
    p.add_argument("--prepare")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.prepare:
            prepare(a.prepare)
        else:
            run(a.fold, a.mode)
