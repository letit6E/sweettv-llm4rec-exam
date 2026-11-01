"""Error-driven, causal feature ablations with fixed CatBoost hyperparameters."""

import argparse, json, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from .experiment import Fold, ROOT, SEED, csr
from .improve import cached, WINDOWS
from .horizon import rescale
from .metrics import evaluate

DEST = ROOT / "artifacts/error-cycles-v4"
DEST.mkdir(exist_ok=True)


def calibration(f, u, i):
    out = []
    names = []

    def add(n, v):
        v = np.asarray(v, dtype="float32")
        assert np.isfinite(v).all(), n
        out.append(v)
        names.append(n)

    def item(g, field):
        return g[field].reindex(f.items, fill_value=0).to_numpy()

    tr = f.train.copy()
    tr["full"] = (tr.fraction >= 0.8).astype(float)
    tr["singlefull"] = (tr.single_fraction >= 0.8).astype(float)
    tr["minutes"] = tr.seconds / 60
    for days in [28, 112]:
        t = tr[tr.end >= f.start - days * 86400]
        g = t.groupby("tv_show_id").agg(
            count=("user", "size"),
            users=("user", "nunique"),
            full=("full", "sum"),
            singlefull=("singlefull", "sum"),
            fraction=("fraction", "mean"),
            seconds=("seconds", "mean"),
            duration=("duration", "mean"),
            minutes=("minutes", "sum"),
        )
        counts = item(g, "count")
        full = item(g, "full")
        users = item(g, "users")
        for field in ["count", "full", "users", "duration", "seconds", "minutes"]:
            add(f"cal_{days}_{field}", np.log1p(item(g, field)[i]))
        add(f"cal_{days}_completion", (full / (counts + 10))[i])
        add(f"cal_{days}_single_completion", (item(g, "singlefull") / (counts + 10))[i])
        add(f"cal_{days}_views_per_user", (full / (users + 10))[i])
        add(f"cal_{days}_fraction", item(g, "fraction")[i])
        # Expected item views conditional on user's channel exposure, using ONLY past events.
        c = t.groupby(["user", "channel_id"]).full.sum().unstack(fill_value=0)
        rates = t.groupby(["channel_id", "tv_show_id"]).full.sum().unstack(fill_value=0)
        rates = rates.div(rates.sum(axis=1).clip(lower=1), axis=0)
        c = c.reindex(index=f.eval_users, columns=rates.index, fill_value=0)
        z = np.einsum(
            "uc,ci->ui",
            np.asarray(c, dtype="float32"),
            np.asarray(rates, dtype="float32"),
            optimize=False,
        )
        z = (
            pd.DataFrame(z, columns=rates.columns)
            .reindex(columns=f.items, fill_value=0)
            .to_numpy()
        )
        add(f"cal_{days}_channel_conditional", np.log1p(z[u, i]))
    # Only historical metadata, not future labels or future titles.
    p = f.past.groupby("tv_show_id").duration.agg(["median", "min", "max"])
    for field in p:
        add(
            "duration_" + field,
            np.log1p(p[field].reindex(f.items, fill_value=0).to_numpy()[i]),
        )
    rowmap = {int(v): k for k, v in enumerate(f.eval_users)}
    t = tr[tr.user.isin(rowmap)].copy()
    t["r"] = t.user.map(rowmap)
    for days in [28, 112]:
        q = t[t.end >= f.start - days * 86400]
        for field in ["seconds", "full"]:
            a = csr(
                q.r, q.item_idx, q[field], (len(f.eval_users), len(f.items))
            ).toarray()
            add(f"user_share_{field}_{days}", a[u, i] / np.maximum(a.sum(axis=1)[u], 1))
        # Fraction of own full views attributed to program, relative to catalog popularity.
    return np.column_stack(out), names


def slot_features(f, u, i):
    """Occupancy by channel and clock bin, projected over future airing duration.

    30-minute bins approximate stationarity within a bin. Split intervals at UTC
    half hours, map bins to Kyiv civil time (DST safe); truncate history at cutoff.
    Uses all raw sessions, including unlabeled airtime; no future user behavior.
    """
    raw = np.load(ROOT / "data/processed/views.npz")
    rowmap = np.full(len(f.users), -1, dtype=int)
    rowmap[f.eval_users] = np.arange(len(f.eval_users))
    mask = (
        (raw["start"] < f.start)
        & (raw["end"] > f.start - 112 * 86400)
        & (rowmap[raw["user"]] >= 0)
    )
    ru = rowmap[raw["user"][mask]]
    rs = np.maximum(raw["start"][mask], f.start - 112 * 86400)
    re = np.minimum(raw["end"][mask], f.start)
    chans = np.union1d(raw["channel"][mask], f.future.channel_id.to_numpy())
    cm = {int(c): j for j, c in enumerate(chans)}
    rc = np.array([cm[int(c)] for c in raw["channel"][mask]])
    fu = f.future.tv_show_id.map(f.im).to_numpy()
    fc = f.future.channel_id.map(cm).to_numpy()
    fs = f.future.start.to_numpy()
    fe = f.future.end.to_numpy()

    def segments(a, b, ids, ch):
        arrays = [[], [], [], [], []]
        cur = a.copy()
        while (cur < b).any():
            m = cur < b
            nxt = np.minimum((cur // 1800 + 1) * 1800, b)
            dt = pd.to_datetime(cur[m], unit="s", utc=True).tz_convert("Europe/Kyiv")
            arrays[0].append(ids[m])
            arrays[1].append(ch[m])
            arrays[2].append((dt.hour * 2 + dt.minute // 30).to_numpy())
            arrays[3].append(dt.dayofweek.to_numpy())
            arrays[4].append((nxt - cur)[m])
            cur = nxt
        return [np.concatenate(x) for x in arrays]

    hu, hc, hh, hd, hs = segments(rs, re, ru, rc)
    fi, cc, h, d, sec = segments(fs, fe, fu, fc)
    out = []
    names = []
    nusers = len(f.eval_users)
    nitems = len(f.items)
    for mode, bins in [("hour", 48), ("weekday", 336), ("weekend", 96)]:
        hb = hh + (
            hd * 48 if mode == "weekday" else (hd >= 5) * 48 if mode == "weekend" else 0
        )
        fb = h + (
            d * 48 if mode == "weekday" else (d >= 5) * 48 if mode == "weekend" else 0
        )
        hist = csr(hu, hc * bins + hb, hs, (nusers, len(chans) * bins))
        # User-conditioned distribution over channel / clock bins.
        fut = csr(cc * bins + fb, fi, sec / 1800, (len(chans) * bins, nitems))
        z = (normalize(hist, norm="l1") @ fut).toarray()
        weeks = (f.end - f.start) / 604800
        out.append(np.log1p(z[u, i] / weeks))
        names.append("slot_" + mode + "_duration")
        # Unit-mass per airing, so long broadcasts do not win merely by duration.
        durations = f.future.duration.to_numpy()
        # Segment output fi is item index; per-segment duration requires second pass via airing ids.
        af, ac, ah, ad, asec = segments(fs, fe, np.arange(len(fs)), fc)
        ab = ah + (
            ad * 48 if mode == "weekday" else (ad >= 5) * 48 if mode == "weekend" else 0
        )
        ff = csr(
            ac * bins + ab,
            fu[af],
            asec / np.maximum(durations[af], 1),
            (len(chans) * bins, nitems),
        )
        z = (normalize(hist, norm="l1") @ ff).toarray()
        out.append(np.log1p(z[u, i] / weeks))
        names.append("slot_" + mode + "_airings")
        # Project total occupied seconds, preserving viewer's intensity (not only shares).
        z = (hist @ ff).toarray() / 3600 / 16 / weeks
        out.append(np.log1p(z[u, i]))
        names.append("slot_" + mode + "_hours")
    return np.column_stack(out).astype("float32"), names


def features(key, stage):
    d = cached(key)
    p = DEST / f"{key}-extra-{stage}.npz"
    if p.exists():
        return d, dict(np.load(p))
    f = Fold(*WINDOWS[key])
    t = time.perf_counter()
    from .propensity import propensity_features, repeat_channel_features
    from .neighbor_cycles import neighbor_features
    from .category_cycles import category_features

    x, n = (
        {
            1: calibration,
            2: slot_features,
            3: propensity_features,
            4: repeat_channel_features,
            5: neighbor_features,
            6: category_features,
        }[stage]
    )(f, d["u"], d["i"])
    assert x.shape[0] == len(d["y"]) and np.isfinite(x).all()
    np.savez_compressed(p, X=x, names=np.array(n))
    print(
        "FEATURES", key, stage, x.shape, round(time.perf_counter() - t, 1), flush=True
    )
    return d, {"X": x, "names": np.array(n)}


def matrix(key, stage):
    d = cached(key)
    fstart, fend = WINDOWS[key]
    from .experiment import epoch

    weeks = (epoch(fend) - epoch(fstart)) / 604800
    # Keep five-week equivalent schedule exposure in every model, including training.
    x = (
        d["X"]
        if stage >= 10 and weeks <= 5.01
        else rescale(d["X"], d["names"].tolist(), weeks)
    )
    parts = [x]
    names = d["names"].tolist()
    parts_stage = (
        [stage - 10]
        if stage >= 11
        else []
        if stage == 10
        else [stage]
        if stage >= 3
        else range(1, stage + 1)
    )
    for st in parts_stage:
        _, e = features(key, st)
        parts.append(e["X"])
        names += e["names"].tolist()
    return d, np.column_stack(parts), names


def train(stage, key, trees=200):
    keys = (
        ["early1", "early2", "early3"]
        + (["dev"] if key in ["audit", "final"] else [])
        + (["audit"] if key == "final" else [])
    )
    X = []
    y = []
    t = time.perf_counter()
    for k in keys:
        assert WINDOWS[k][1] <= WINDOWS[key][0]
        d, x, n = matrix(k, stage)
        X.append(x[d["valid"]])
        y.append(d["y"][d["valid"]])
    X = np.concatenate(X)
    y = np.concatenate(y)
    model = CatBoostClassifier(
        iterations=trees,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        loss_function="Logloss",
        random_seed=SEED,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)
    name = f"{key}_stage{stage}_{trees}"
    model.save_model(str(DEST / (name + ".cbm")))
    keys_eval = [key] + (["long"] if key == "dev" else [])
    records = []
    for ev in keys_eval:
        d, x, n = matrix(ev, stage)
        f = Fold(*WINDOWS[ev])
        score = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
        score[d["u"], d["i"]] = model.predict_proba(x)[:, 1]
        np.save(DEST / f"{ev}_stage{stage}_{trees}.npy", score)
        m = evaluate(f.truth, f.rank(score))
        rec = {
            "train": key,
            "eval": ev,
            "stage": stage,
            "trees": trees,
            "train_rows": len(y),
            "features": n,
            "seconds": time.perf_counter() - t,
            **{k: v for k, v in m.items() if k != "per_user_ap"},
        }
        if ev == "final":
            from .improve_final import export

            rec.update(export(f, score, f"v4_stage{stage}_{trees}"))
        records.append(rec)
        print(
            "RESULT",
            json.dumps({k: v for k, v in rec.items() if k != "features"}),
            flush=True,
        )
    (DEST / f"{name}.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--fold", default="dev")
    p.add_argument("--trees", type=int, default=200)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        train(a.stage, a.fold, a.trees)
