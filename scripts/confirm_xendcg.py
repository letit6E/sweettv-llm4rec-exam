"""Freeze XE_NDCG blend on development; evaluate later and longer windows."""

import argparse
import datetime
import json
import numpy as np
import lightgbm as lgb

from sweettv.neural_rank import DEST
from sweettv.improve import DEST as OLD, WINDOWS, cached
from sweettv.experiment import Fold, SEED
from sweettv.metrics import evaluate
from sweettv.blend import rrf
from sweettv.horizon import rescale


def compare(f, baseline, candidate):
    a = evaluate(f.truth, f.rank(baseline))
    b = evaluate(f.truth, f.rank(candidate))
    mask = np.array([len(t) == 5 for t in f.truth])
    delta = (np.array(b["per_user_ap"]) - np.array(a["per_user_ap"]))[mask]
    rng = np.random.default_rng(SEED)
    ci = np.quantile(
        [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)],
        [0.025, 0.975],
    )
    return dict(
        baseline=a["map5_five_targets"],
        candidate=b["map5_five_targets"],
        delta=float(delta.mean()),
        bootstrap95=ci.tolist(),
    )


def select():
    f = Fold(*WINDOWS["dev"])
    baseline = np.load(OLD / "dev_cat6_200.npy")
    candidate = np.load(DEST / f"dev_xendcg_s{SEED}" / "fixed-scores.npy")
    a, b = rrf(f, baseline), rrf(f, candidate)
    grid = []
    for w in [0, 0.25, 0.5, 0.75, 1]:
        m = evaluate(f.truth, f.rank((1 - w) * a + w * b))
        grid.append(dict(weight=w, map5=m["map5_five_targets"]))
    weight = max(grid, key=lambda r: r["map5"])["weight"]
    decision = dict(
        time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model="XE_NDCG_MART",
        seed=SEED,
        trees=1200,
        grid=grid,
        frozen_xendcg_weight=weight,
        selected_on="Earlier development window only",
        promotion_rule="Positive MAP difference on both five-week windows; long-horizon difference at least -0.001; seed replication shown separately. No significance claim if confidence interval crosses zero.",
    )
    (DEST / "xendcg-selection.json").write_text(json.dumps(decision, indent=2))
    print("FROZEN", json.dumps(decision), flush=True)


def evaluate_frozen():
    decision = json.loads((DEST / "xendcg-selection.json").read_text())
    weight = decision["frozen_xendcg_weight"]
    comparisons = {}
    seeds = []
    for s in range(SEED, SEED + 3):
        f = Fold(*WINDOWS["dev"])
        p = np.load(DEST / f"dev_xendcg_s{s}" / "fixed-scores.npy")
        seeds.append(
            dict(seed=s, map5=evaluate(f.truth, f.rank(p))["map5_five_targets"])
        )
    for key in ["dev", "audit"]:
        f = Fold(*WINDOWS[key])
        base = np.load(OLD / f"{key}_cat6_200.npy")
        candidate = np.load(DEST / f"{key}_xendcg_s{SEED}" / "fixed-scores.npy")
        blend = (1 - weight) * rrf(f, base) + weight * rrf(f, candidate)
        comparisons[key] = compare(f, base, blend)
        comparisons[key]["standalone_xendcg_map5"] = evaluate(
            f.truth, f.rank(candidate)
        )["map5_five_targets"]
        np.save(DEST / f"{key}_xendcg_frozen_blend.npy", blend)
    f = Fold(*WINDOWS["long"])
    d = cached("long")
    model = lgb.Booster(model_file=str(DEST / f"dev_xendcg_s{SEED}" / "model.txt"))
    import pickle

    with (OLD / "dev_cat6.pkl").open("rb") as handle:
        cat = pickle.load(handle)
    normalized_x = rescale(d["X"], d["names"].tolist(), 10)
    rawbase = cat.predict_proba(normalized_x, ntree_end=200)[:, 1]
    base = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
    base[d["u"], d["i"]] = rawbase
    for label, x in [("raw", d["X"]), ("normalized", normalized_x)]:
        p = model.predict(x, num_iteration=1200)
        candidate = np.full_like(base, -1e9)
        candidate[d["u"], d["i"]] = p
        blend = (1 - weight) * rrf(f, base) + weight * rrf(f, candidate)
        comparisons["long_" + label] = compare(f, base, blend)
    horizon = max(
        ["raw", "normalized"], key=lambda k: comparisons["long_" + k]["candidate"]
    )
    promote = (
        weight > 0
        and comparisons["dev"]["delta"] > 0
        and comparisons["audit"]["delta"] > 0
        and comparisons["long_" + horizon]["delta"] >= -0.001
    )
    result = dict(
        **decision,
        seed_replications=seeds,
        comparisons=comparisons,
        final_horizon_transform=horizon,
        promote=promote,
        caveat="All windows previously inspected; long diagnostic overlaps both five-week windows. Bootstrap omits model-selection uncertainty.",
    )
    (DEST / "xendcg-confirmation.json").write_text(json.dumps(result, indent=2))
    print("CONFIRMED", json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "evaluate"])
    a = p.parse_args()
    select() if a.stage == "select" else evaluate_frozen()
