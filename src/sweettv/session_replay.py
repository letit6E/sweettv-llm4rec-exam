"""Replay past clock-time sessions against future schedule, never future viewers."""

import argparse, json, time
import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits
from .experiment import Fold, ROOT
from .improve import WINDOWS
from .simple_search import DEST, baseline
from .blend import rrf
from .metrics import evaluate


@njit(cache=True)
def accumulate(users, starts, ends, fs, fe, fi, freq, period, result):
    prefix = fe.copy()
    for j in range(1, len(prefix)):
        prefix[j] = max(prefix[j - 1], prefix[j])
    for j in range(len(users)):
        duration = min(ends[j] - starts[j], period)
        for shift in [-period, 0, period]:
            s = starts[j] + shift
            e = s + duration
            lo = np.searchsorted(prefix, s, side="right")
            hi = np.searchsorted(fs, e, side="left")
            for k in range(lo, hi):
                overlap = min(e, fe[k]) - max(s, fs[k])
                if overlap <= 0:
                    continue
                fraction = overlap / (fe[k] - fs[k])
                u = users[j]
                it = fi[k]
                w = freq[k]
                if fraction >= 0.5:
                    result[0, u, it] += w
                if fraction >= 0.8:
                    result[1, u, it] += w
                if fraction >= 0.95:
                    result[2, u, it] += w
                result[3, u, it] += w * fraction


def wall(ts):
    return (
        pd.to_datetime(ts, unit="s", utc=True)
        .tz_convert("Europe/Kyiv")
        .tz_localize(None)
        .asi8
        // 10**9
    )


def compute(f, days, weekly):
    v = np.load(ROOT / "data/processed/views.npz")
    rmap = np.full(len(f.users), -1, dtype=int)
    rmap[f.eval_users] = np.arange(len(f.eval_users))
    mask = (
        (v["start"] < f.start)
        & (v["end"] > f.start - days * 86400)
        & (rmap[v["user"]] >= 0)
    )
    user = rmap[v["user"][mask]]
    channel = v["channel"][mask]
    st = wall(np.maximum(v["start"][mask], f.start - days * 86400))
    en = wall(np.minimum(v["end"][mask], f.start))
    assert np.all(en >= st)
    period = 604800 if weekly else 86400
    offset = 3 * 86400 if weekly else 0
    duration = en - st
    st = (st + offset) % period
    en = st + duration
    future = f.future.copy()
    future["slot"] = (wall(future.start.to_numpy()) + offset) % period
    future["finish"] = future.slot + future.duration
    future["item"] = future.tv_show_id.map(f.im)
    future = (
        future.groupby(["channel_id", "slot", "finish", "item"])
        .size()
        .reset_index(name="freq")
    )
    result = np.zeros((4, len(f.eval_users), len(f.items)), dtype="float32")
    for ch, g in future.groupby("channel_id"):
        m = channel == ch
        if not m.any():
            continue
        g = g.sort_values("slot")
        accumulate(
            user[m],
            st[m],
            en[m],
            g.slot.to_numpy(),
            g.finish.to_numpy(),
            g["item"].to_numpy(),
            g.freq.to_numpy(),
            period,
            result,
        )
    assert np.isfinite(result).all()
    return result


def cached(key, days, weekly):
    p = DEST / f"{key}_replay_{days}_{weekly}.npz"
    if p.exists():
        return np.load(p)["scores"]
    f = Fold(*WINDOWS[key])
    t = time.perf_counter()
    x = compute(f, days, weekly)
    np.savez_compressed(p, scores=x)
    print("REPLAY", key, days, weekly, round(time.perf_counter() - t, 1), flush=True)
    return x


def run(stage):
    if stage == "select":
        f = Fold(*WINDOWS["dev"])
        base = rrf(f, baseline("dev"))
        rows = []
        for days in [28, 112]:
            for weekly in [False, True]:
                a = cached("dev", days, weekly)
                for mode, s in enumerate(a):
                    standalone = evaluate(f.truth, f.rank(s))["map5_five_targets"]
                    rs = rrf(f, s)
                    for w in [0.1, 0.25, 0.5, 1.0]:
                        z = (1 - w) * base + w * rs
                        m = evaluate(f.truth, f.rank(z))["map5_five_targets"]
                        rows.append(
                            {
                                "days": days,
                                "weekly": weekly,
                                "mode": mode,
                                "weight": w,
                                "dev": m,
                                "standalone": standalone,
                            }
                        )
        best = max(rows, key=lambda r: r["dev"])
        (DEST / "replay-selection.json").write_text(
            json.dumps({"grid": rows, "selected": best}, indent=2)
        )
        print("SELECTED", best, flush=True)
    else:
        cfg = json.loads((DEST / "replay-selection.json").read_text())["selected"]
        out = {}
        for key in ["audit", "long"]:
            f = Fold(*WINDOWS[key])
            s = cached(key, cfg["days"], cfg["weekly"])[cfg["mode"]]
            z = (1 - cfg["weight"]) * rrf(f, baseline(key)) + cfg["weight"] * rrf(f, s)
            out[key] = {
                "standalone": evaluate(f.truth, f.rank(s))["map5_five_targets"],
                "blend": evaluate(f.truth, f.rank(z))["map5_five_targets"],
            }
            print("CONFIRM", key, out[key], flush=True)
        (DEST / "replay-confirmation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "confirm"])
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.stage)
