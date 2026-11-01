"""Three-seed replication; freeze epoch/blend before later-window evaluation."""

import json
import numpy as np

from sweettv.neural_rank import DEST, CONFIGS, run
from sweettv.improve import WINDOWS, DEST as OLD
from sweettv.experiment import Fold, SEED
from sweettv.metrics import evaluate
from sweettv.blend import rrf


def ensemble(fold, config, suffix):
    f = Fold(*WINDOWS[fold])
    scores = [
        np.load(DEST / f"{fold}_{config}_s{s}" / suffix) for s in range(SEED, SEED + 3)
    ]
    return f, sum(rrf(f, x) for x in scores) / len(scores)


def main():
    results = [
        json.loads((DEST / f"dev_{c}_s{SEED}" / "result.json").read_text())
        for c in CONFIGS
    ]
    chosen = max(results, key=lambda d: d["best_validation_map5"])
    config = chosen["name"].split("_")[1]
    for seed in [SEED + 1, SEED + 2]:
        p = DEST / f"dev_{config}_s{seed}" / "result.json"
        if not p.exists():
            run("dev", config, seed=seed)
    reps = [
        json.loads((DEST / f"dev_{config}_s{s}" / "result.json").read_text())
        for s in range(SEED, SEED + 3)
    ]
    epochs = int(np.median([r["best_epoch"] for r in reps]))
    f, neural = ensemble("dev", config, "best-scores.npy")
    base = np.load(OLD / "dev_cat6_200.npy")
    base_rrf = rrf(f, base)
    grid = []
    for weight in [0, 0.1, 0.25, 0.5, 1]:
        metric = evaluate(f.truth, f.rank((1 - weight) * base_rrf + weight * neural))
        grid.append(dict(neural_weight=weight, map5=metric["map5_five_targets"]))
    weight = max(grid, key=lambda r: r["map5"])["neural_weight"]
    decision = dict(
        config=config,
        fixed_audit_epochs=epochs,
        seeds=list(range(SEED, SEED + 3)),
        dev_seed_map5=[r["best_validation_map5"] for r in reps],
        dev_seed_epochs=[r["best_epoch"] for r in reps],
        dev_blend_grid=grid,
        frozen_neural_weight=weight,
        note="Selected on dev only before this run's later-window neural training; both windows were inspected historically.",
    )
    (DEST / "neural-confirmation-selection.json").write_text(
        json.dumps(decision, indent=2)
    )
    print("FROZEN_NEURAL_SELECTION", json.dumps(decision), flush=True)
    for seed in range(SEED, SEED + 3):
        p = DEST / f"audit_{config}_s{seed}" / "result.json"
        if not p.exists():
            run("audit", config, seed=seed, epochs=epochs, fixed_epochs=True)
    f, neural = ensemble("audit", config, "fixed-scores.npy")
    base = np.load(OLD / "audit_cat6_200.npy")
    baseline = evaluate(f.truth, f.rank(base))
    single = evaluate(f.truth, f.rank(neural))
    blend = evaluate(f.truth, f.rank((1 - weight) * rrf(f, base) + weight * neural))
    mask = np.array([len(t) == 5 for t in f.truth])
    delta = (np.array(blend["per_user_ap"]) - np.array(baseline["per_user_ap"]))[mask]
    rng = np.random.default_rng(SEED)
    means = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)]
    output = dict(
        **decision,
        later_baseline_map5=baseline["map5_five_targets"],
        later_neural_ensemble_map5=single["map5_five_targets"],
        later_frozen_blend_map5=blend["map5_five_targets"],
        later_delta=float(delta.mean()),
        paired_bootstrap95=np.quantile(means, [0.025, 0.975]).tolist(),
        bootstrap_caveat="User sampling uncertainty only; does not cover prior validation reuse or hyperparameter selection.",
    )
    np.save(DEST / "audit_neural_ensemble.npy", neural)
    np.save(
        DEST / "audit_neural_frozen_blend.npy",
        (1 - weight) * rrf(f, base) + weight * neural,
    )
    (DEST / "neural-confirmation.json").write_text(json.dumps(output, indent=2))
    print("NEURAL_CONFIRMATION", json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
