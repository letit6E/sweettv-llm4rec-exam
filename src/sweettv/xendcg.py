"""Cross-entropy LambdaMART alternative with the same temporal training data."""

import argparse
import json
import time
import numpy as np
import lightgbm as lgb
from .allusers import data
from .neural_rank import DEST, training_keys
from .improve import cached, WINDOWS
from .experiment import Fold, SEED
from .metrics import evaluate


def run(fold="dev", seed=SEED, iterations=1200):
    folder = DEST / f"{fold}_xendcg_s{seed}"
    folder.mkdir(exist_ok=True)
    x, y, groups = data(training_keys(fold))
    d = cached(fold)
    f = Fold(*WINDOWS[fold])
    config = dict(
        n_estimators=iterations,
        objective="rank_xendcg",
        num_leaves=31,
        learning_rate=0.03,
        min_child_samples=100,
        reg_lambda=10,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=3,
        verbosity=-1,
    )
    metadata = dict(
        name=folder.name,
        config=config,
        windows=training_keys(fold),
        queries=len(groups),
        pairs=len(y),
        selection="Full five-target MAP at stored checkpoints",
    )
    (folder / "config.json").write_text(json.dumps(metadata, indent=2))
    t = time.perf_counter()
    model = lgb.LGBMRanker(**config)
    print("START", json.dumps(metadata), flush=True)
    model.fit(x, y, group=groups)
    model.booster_.save_model(str(folder / "model.txt"))
    best, best_iteration, rows = -1, 0, []
    for it in sorted(set([100, 200, 400, 800, iterations])):
        if it > iterations:
            continue
        raw = model.predict(d["X"], num_iteration=it)
        scores = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
        scores[d["u"], d["i"]] = raw
        if it == iterations:
            np.save(folder / "fixed-scores.npy", scores)
        if fold != "final":
            m = evaluate(f.truth, f.rank(scores))
            row = dict(
                iteration=it, **{k: v for k, v in m.items() if k != "per_user_ap"}
            )
            if m["map5_five_targets"] > best:
                best, best_iteration = m["map5_five_targets"], it
                np.save(folder / "best-scores.npy", scores)
        else:
            row = dict(iteration=it)
            np.save(folder / f"scores-{it}.npy", scores)
        rows.append(row)
        print("CHECKPOINT", folder.name, json.dumps(row), flush=True)
    (folder / "history.json").write_text(json.dumps(rows, indent=2))
    result = dict(
        **metadata,
        best_iteration=best_iteration,
        best_validation_map5=best,
        elapsed_seconds=time.perf_counter() - t,
    )
    if fold != "final":
        result["fixed_validation_map5"] = rows[-1]["map5_five_targets"]
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    print("FINISHED", json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="dev", choices=["dev", "audit", "final"])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--iterations", type=int, default=1200)
    a = p.parse_args()
    run(a.fold, a.seed, a.iterations)
