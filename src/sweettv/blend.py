"""Predeclared small rank-fusion grid across two already-open temporal windows."""

import json
import numpy as np
from .experiment import Fold
from .improve import DEST, WINDOWS
from .improve_final import export
from .metrics import evaluate

COMP = {
    "cat": "cat6_200",
    "rank": "lgb_rank_15_400",
    "bin": "lgb_bin_31_200",
    "xgb": "xgbmap_d3_500",
}


def rrf(f, s, k=5):
    v = s.copy()
    v[:, ~f.eligible] = -np.inf
    order = np.lexsort(
        (np.broadcast_to(f.items, v.shape), np.broadcast_to(-f.pop, v.shape), -v),
        axis=1,
    )
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(len(f.items))[None, :], axis=1)
    out = 1 / (k + ranks.astype("float32"))
    out[:, ~f.eligible] = 0
    return out


def run():
    configs = [
        {"cat": 1},
        {"rank": 1},
        {"bin": 1},
        {"xgb": 1},
        {"cat": 0.5, "rank": 0.5},
        {"cat": 0.5, "bin": 0.5},
        {"cat": 1 / 3, "rank": 1 / 3, "bin": 1 / 3},
        {"cat": 0.5, "hgb": 0.5},
        {"cat": 0.75, "hgb": 0.25},
        {"cat": 0.5, "rank": 0.25, "hgb": 0.25},
        {"cat": 0.5, "xgb": 0.5},
        {"cat": 0.4, "rank": 0.2, "bin": 0.2, "xgb": 0.2},
    ]
    results = []
    for key in ["dev", "audit"]:
        f = Fold(*WINDOWS[key])
        scores = {
            n: np.load(DEST / f"{key}_{s}.npy")
            for n, s in COMP.items()
            if (DEST / f"{key}_{s}.npy").exists()
        }
        scores["hgb"] = np.load(DEST / f"{key}-scores.npz")["hgb"]
        # Broader graph / sequence blends are diagnostic and never use test truth.
        for n in ["userknn", "itemknn"]:
            scores[n] = np.load(DEST / f"{key}-scores.npz")[n]
        seq = DEST / f"{key}_esasrec_20_mean10.npy"
        if seq.exists():
            scores["sequence"] = np.load(seq)
        ranks = {n: rrf(f, s) for n, s in scores.items()}
        current = configs + [
            {"cat": 1 - w, n: w}
            for n in ["userknn", "itemknn", "sequence"]
            if n in scores
            for w in [0.05, 0.1, 0.2, 0.3]
        ]
        for cfg in current:
            if not all(n in ranks for n in cfg):
                continue
            s = sum(w * ranks[n] for n, w in cfg.items())
            m = evaluate(f.truth, f.rank(s))
            row = {"fold": key, "weights": cfg, "map5": m["map5_five_targets"]}
            results.append(row)
        print(
            key,
            sorted([r for r in results if r["fold"] == key], key=lambda r: -r["map5"])[
                :6
            ],
            flush=True,
        )
    (DEST / "blend-validation.json").write_text(json.dumps(results, indent=2))
    # One equal-weight ensemble, plus one conservative blend with original model.
    f = Fold(*WINDOWS["final"])
    ranks = {
        n: rrf(f, np.load(DEST / file))
        for n, file in [
            ("cat", "v2_cat200.npy"),
            ("rank", "v2_lgb_rank400.npy"),
            ("bin", "v2_lgb_bin200.npy"),
        ]
    }
    ranks["hgb"] = rrf(f, np.load(DEST / "final-scores.npz")["hgb"])
    out = []
    for name, cfg in [
        ("v2_ensemble", {"cat": 1 / 3, "rank": 1 / 3, "bin": 1 / 3}),
        ("v2_conservative", {"cat": 0.5, "rank": 0.25, "hgb": 0.25}),
    ]:
        s = sum(w * ranks[n] for n, w in cfg.items())
        out.append({**export(f, s, name), "weights": cfg, "rrf_offset": 5})
    (DEST / "blend-exports.json").write_text(json.dumps(out, indent=2))
    print(out, flush=True)


if __name__ == "__main__":
    run()
