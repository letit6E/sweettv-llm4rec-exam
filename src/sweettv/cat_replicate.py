"""Three-seed replication of fixed incumbent hyperparameters, no new grid."""

import json
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier

from .experiment import Fold, SEED
from .improve import DEST as OLD, WINDOWS, cached, fit_data
from .neural_rank import DEST, training_keys
from .metrics import evaluate
from .horizon import rescale


def run():
    results = []
    for fold in ["dev", "audit"]:
        f = Fold(*WINDOWS[fold])
        d = cached(fold)
        x, y, groups = fit_data(training_keys(fold))
        predictions = [np.load(OLD / f"{fold}_cat6_200.npy")]
        for seed in [SEED + 1, SEED + 2]:
            folder = DEST / f"{fold}_cat200_s{seed}"
            folder.mkdir(exist_ok=True)
            t = time.perf_counter()
            model = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=10,
                loss_function="Logloss",
                random_seed=seed,
                thread_count=3,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(x, y)
            model.save_model(str(folder / "model.cbm"))
            raw = model.predict_proba(d["X"])[:, 1]
            scores = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
            scores[d["u"], d["i"]] = raw
            predictions.append(scores)
            np.save(folder / "best-scores.npy", scores)
            m = evaluate(f.truth, f.rank(scores))
            row = dict(
                name=folder.name,
                training_windows=training_keys(fold),
                pairs=len(y),
                queries=len(groups),
                parameters=model.get_params(),
                seconds=time.perf_counter() - t,
                best_validation_map5=m["map5_five_targets"],
            )
            (folder / "result.json").write_text(json.dumps(row, indent=2))
            print("CAT_REPLICATION", json.dumps(row), flush=True)
        combined = sum(predictions) / 3
        np.save(DEST / f"{fold}_cat_three_seed.npy", combined)
        before = evaluate(f.truth, f.rank(predictions[0]))
        after = evaluate(f.truth, f.rank(combined))
        mask = np.array([len(t) == 5 for t in f.truth])
        delta = (np.array(after["per_user_ap"]) - np.array(before["per_user_ap"]))[mask]
        rng = np.random.default_rng(SEED)
        ci = np.quantile(
            [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)],
            [0.025, 0.975],
        )
        row = dict(
            fold=fold,
            baseline=before["map5_five_targets"],
            ensemble=after["map5_five_targets"],
            delta=float(delta.mean()),
            bootstrap95=ci.tolist(),
        )
        results.append(row)
        print("CAT_ENSEMBLE", json.dumps(row), flush=True)
        (DEST / "cat-replication.json").write_text(json.dumps(results, indent=2))
    # Additional long-horizon diagnostic using the same already trained dev models.
    f = Fold(*WINDOWS["long"])
    d = cached("long")
    x = rescale(d["X"], d["names"].tolist(), 10)
    with (OLD / "dev_cat6.pkl").open("rb") as handle:
        old = pickle.load(handle)
    predictions = [old.predict_proba(x, ntree_end=200)[:, 1]]
    for seed in [SEED + 1, SEED + 2]:
        m = CatBoostClassifier()
        m.load_model(str(DEST / f"dev_cat200_s{seed}" / "model.cbm"))
        predictions.append(m.predict_proba(x)[:, 1])
    rows = []
    for label, raw in [
        ("baseline", predictions[0]),
        ("ensemble", np.mean(predictions, axis=0)),
    ]:
        scores = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
        scores[d["u"], d["i"]] = raw
        rows.append(
            dict(
                model=label, map5=evaluate(f.truth, f.rank(scores))["map5_five_targets"]
            )
        )
    (DEST / "cat-replication-long.json").write_text(json.dumps(rows, indent=2))
    print("CAT_LONG", json.dumps(rows), flush=True)


if __name__ == "__main__":
    run()
