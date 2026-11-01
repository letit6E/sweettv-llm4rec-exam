"""Broad explicit simple baselines; canonical validation labels remain unchanged."""

import argparse, json, time
import numpy as np
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from .experiment import Fold, ROOT, csr
from .improve import WINDOWS, DEST as OLD
from .metrics import evaluate
from .blend import rrf

DEST = ROOT / "artifacts/approaches-v5"
DEST.mkdir(exist_ok=True)


def configurations():
    out = []
    for unit in ["airings", "seconds", "fraction", "days", "weeks", "single"]:
        for threshold in [0.8] if unit == "single" else [0, 0.5, 0.8]:
            for half in [None, 28]:
                out.append(
                    dict(family="repeat", unit=unit, threshold=threshold, half=half)
                )
    for days in [28, 112]:
        for duration in [False, True]:
            for mode in [
                "future_freq",
                "future_share",
                "past_full",
                "past_rate",
                "hybrid",
            ]:
                out.append(
                    dict(family="channel", days=days, duration=duration, mode=mode)
                )
    for days in [28, 112]:
        for k in [10, 50, 200]:
            for axis in ["items", "channels"]:
                out.append(dict(family="neighbors", days=days, k=k, axis=axis))
    return out


def channel_data(f, days):
    raw = np.load(ROOT / "data/processed/views.npz")
    m = (raw["start"] < f.start) & (raw["end"] > f.start - days * 86400)
    channels = np.union1d(raw["channel"][m], f.future.channel_id)
    cm = {int(c): i for i, c in enumerate(channels)}
    u = raw["user"][m]
    c = np.array([cm[int(v)] for v in raw["channel"][m]])
    sec = np.minimum(raw["end"][m], f.start) - np.maximum(
        raw["start"][m], f.start - days * 86400
    )
    return channels, cm, u, c, sec


def score(f, c):
    family = c["family"]
    if family == "repeat":
        t = f.train[
            (f.train.single_fraction if c["unit"] == "single" else f.train.fraction)
            >= c["threshold"]
        ].copy()
        if c["unit"] in ["days", "weeks"]:
            t["period"] = t.start // (86400 if c["unit"] == "days" else 604800)
            t = t.sort_values("end").drop_duplicates(
                ["user", "item_idx", "period"], keep="last"
            )
        v = (
            t.seconds.to_numpy() / 60
            if c["unit"] == "seconds"
            else t.fraction.to_numpy()
            if c["unit"] == "fraction"
            else np.ones(len(t))
        )
        if c["half"]:
            v = v * np.exp2(-(f.start - t.end.to_numpy()) / (86400 * c["half"]))
        return csr(t.user, t.item_idx, v, f.shape)[f.eval_users].toarray()
    days = c["days"]
    channels, cm, u, ch, sec = channel_data(f, days)
    h = csr(
        u,
        ch,
        sec if c.get("duration", True) else np.ones(len(u)),
        (len(f.users), len(channels)),
    )
    if family == "channel":
        h = normalize(h, norm="l1")[f.eval_users]
        future = csr(
            f.future.channel_id.map(cm),
            f.future.tv_show_id.map(f.im),
            np.ones(len(f.future)),
            (len(cm), len(f.items)),
        )
        if c["mode"] == "future_freq":
            return (h @ future).toarray()
        if c["mode"] == "future_share":
            return (h @ normalize(future, norm="l1")).toarray()
        t = f.train[(f.train.end >= f.start - days * 86400) & (f.train.fraction >= 0.8)]
        full = csr(t.channel_id.map(cm), t.item_idx, np.ones(len(t)), future.shape)
        full = normalize(full, norm="l1")
        if c["mode"] == "past_full":
            return (h @ full).toarray()
        past = f.past[f.past.end >= f.start - days * 86400]
        air = csr(
            past.channel_id.map(cm),
            past.tv_show_id.map(f.im),
            np.ones(len(past)),
            future.shape,
        )
        rate = full.toarray() / (air.toarray() + 5)
        rate = rate * future.toarray()
        if c["mode"] == "hybrid":
            rate = (
                0.5 * normalize(rate, norm="l1")
                + 0.5 * normalize(future, norm="l1").toarray()
            )
        return np.asarray(h @ rate)
    t = f.train[(f.train.end >= f.start - days * 86400) & (f.train.fraction >= 0.8)]
    x = csr(t.user, t.item_idx, np.ones(len(t)), f.shape)
    profile = x.copy() if c["axis"] == "items" else h.copy()
    profile.data = np.log1p(profile.data)
    df = np.asarray((profile > 0).sum(axis=0)).ravel()
    idf = np.log((len(f.users) + 1) / (df + 1))
    profile = normalize(profile.multiply(idf)).tocsr()
    sim = (profile[f.eval_users] @ profile.T).toarray()
    sim[np.arange(len(f.eval_users)), f.eval_users] = 0
    k = c["k"]
    idx = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    w = np.take_along_axis(sim, idx, axis=1) ** 2
    knn = csr(
        np.repeat(np.arange(len(idx)), k),
        idx.ravel(),
        w.ravel(),
        (len(idx), len(f.users)),
    )
    x.data = np.log1p(x.data)
    return (normalize(knn, norm="l1") @ normalize(x, norm="l1")).toarray()


def baseline(key):
    return np.load(
        OLD
        / (
            f"{key}_cat6_200.npy"
            if key in ["dev", "audit"]
            else f"{key}_horizon_True.npy"
        )
    )


def run(stage):
    if stage == "select":
        f = Fold(*WINDOWS["dev"])
        base = rrf(f, baseline("dev"))
        rows = []
        best = {}
        t = time.perf_counter()
        for idx, c in enumerate(configurations()):
            s = score(f, c)
            r = rrf(f, s)
            one = {
                "config": c,
                "standalone": evaluate(f.truth, f.rank(s))["map5_five_targets"],
                "blend": [],
            }
            for w in [0.1, 0.25, 0.5]:
                m = evaluate(f.truth, f.rank((1 - w) * base + w * r))[
                    "map5_five_targets"
                ]
                one["blend"].append({"weight": w, "dev": m})
                family = c["family"]
                if family not in best or m > best[family]["dev"]:
                    best[family] = {
                        "config": c,
                        "weight": w,
                        "dev": m,
                        "standalone": one["standalone"],
                    }
                    np.save(DEST / f"dev_{family}.npy", s)
            rows.append(one)
            if idx % 10 == 0:
                print("GRID", idx, round(time.perf_counter() - t, 1), best, flush=True)
        (DEST / "simple-selection.json").write_text(
            json.dumps(
                {
                    "grid": rows,
                    "selected": best,
                    "baseline": 0.43835698924731187,
                    "label_policy": "No target definition changed. Input thresholds and units vary; truth remains unioned 80% airing count Top5.",
                    "scope": "Previously inspected temporal validation; adaptive model selection.",
                },
                indent=2,
            )
        )
        print("SELECTED", best, flush=True)
    else:
        selected = json.loads((DEST / "simple-selection.json").read_text())["selected"]
        out = {}
        for key in ["audit", "long"]:
            f = Fold(*WINDOWS[key])
            base = rrf(f, baseline(key))
            out[key] = {}
            for name, r in selected.items():
                s = score(f, r["config"])
                np.save(DEST / f"{key}_{name}.npy", s)
                z = (1 - r["weight"]) * base + r["weight"] * rrf(f, s)
                out[key][name] = {
                    "standalone": evaluate(f.truth, f.rank(s))["map5_five_targets"],
                    "blend": evaluate(f.truth, f.rank(z))["map5_five_targets"],
                }
            print("CONFIRM", key, out[key], flush=True)
        (DEST / "simple-confirmation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "confirm"])
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.stage)
