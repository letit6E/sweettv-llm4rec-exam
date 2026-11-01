"""Train MAP-oriented CatBoost ranker; select using full-ground-truth MAP@5.

CatBoost's internal MAP uses retrieved positives as its denominator, so it is
only a monitoring diagnostic. Checkpoint selection below uses the same five
actual target programs and complete evaluation protocol as every other model.
"""

import argparse
import json
import time

import numpy as np
from catboost import CatBoostRanker, Pool

from .allusers import data
from .experiment import Fold, SEED
from .improve import WINDOWS, cached
from .metrics import evaluate
from .neural_rank import DEST, training_keys


def run(fold="dev", seed=SEED, depth=6, iterations=1000):
    name = f"{fold}_yeti{depth}_s{seed}"
    folder = DEST / name
    folder.mkdir(exist_ok=True)
    x, y, groups = data(training_keys(fold))
    train = Pool(x, y, group_id=np.repeat(np.arange(len(groups)), groups))
    del x, y
    d = cached(fold)
    valid = d["valid"]
    validation = (
        Pool(d["X"][valid], d["y"][valid], group_id=d["u"][valid])
        if fold != "final"
        else None
    )
    f = Fold(*WINDOWS[fold])
    config = dict(
        iterations=iterations,
        depth=depth,
        learning_rate=0.03,
        l2_leaf_reg=10,
        loss_function="YetiRank:mode=MAP;top=5",
        eval_metric="MAP:top=5",
        random_seed=seed,
        thread_count=4,
        verbose=100,
        allow_writing_files=False,
    )
    metadata = dict(
        name=name,
        config=config,
        windows=training_keys(fold),
        queries=len(groups),
        pairs=int(groups.sum()),
        selection="External MAP@5 with full true Top-5 at checkpoints; internal MAP diagnostic only",
    )
    (folder / "config.json").write_text(json.dumps(metadata, indent=2))
    print("START", json.dumps(metadata), flush=True)
    t = time.perf_counter()
    model = CatBoostRanker(**config)
    # Fixed maximum budget; retain all trees so true-metric checkpoints can be compared.
    model.fit(train, eval_set=validation, use_best_model=False)
    model.save_model(str(folder / "model.cbm"))
    (folder / "internal-history.json").write_text(
        json.dumps(model.get_evals_result(), indent=2)
    )
    history, best, best_epoch = [], -1, 0
    for it in sorted(set([50] + list(range(100, iterations + 1, 100)) + [iterations])):
        if it > iterations:
            continue
        raw = model.predict(d["X"], ntree_end=it)
        scores = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
        scores[d["u"], d["i"]] = raw
        row = dict(iteration=it)
        if fold != "final":
            m = evaluate(f.truth, f.rank(scores))
            row.update({k: v for k, v in m.items() if k != "per_user_ap"})
            if m["map5_five_targets"] > best:
                best, best_epoch = m["map5_five_targets"], it
                np.save(folder / "best-scores.npy", scores)
        else:
            np.save(folder / f"scores-{it}.npy", scores)
        history.append(row)
        print("CHECKPOINT", name, json.dumps(row), flush=True)
    (folder / "history.json").write_text(json.dumps(history, indent=2))
    result = dict(
        **metadata,
        best_iteration=best_epoch,
        best_validation_map5=best,
        elapsed_seconds=time.perf_counter() - t,
    )
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    print("FINISHED", json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="dev", choices=["dev", "audit", "final"])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--iterations", type=int, default=1000)
    args = p.parse_args()
    run(args.fold, args.seed, args.depth, args.iterations)
