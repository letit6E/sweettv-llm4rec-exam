"""XGBoost MAP-targeted LambdaMART and controlled feature ablations."""

import argparse, json, pickle, time
import numpy as np
from threadpoolctl import threadpool_limits
from .improve import DEST, WINDOWS, cached, fit_data
from .experiment import Fold, SEED
from .metrics import evaluate


def run(key):
    from xgboost import XGBRanker

    keys = ["early1", "early2", "early3"] + (["dev"] if key == "audit" else [])
    X, y, g = fit_data(keys)
    d = cached(key)
    f = Fold(*WINDOWS[key])
    results = []
    for depth in [3, 5]:
        t = time.perf_counter()
        model = XGBRanker(
            objective="rank:map",
            n_estimators=500,
            max_depth=depth,
            learning_rate=0.03,
            tree_method="hist",
            n_jobs=4,
            reg_lambda=10,
            min_child_weight=20,
            colsample_bytree=0.9,
            random_state=SEED,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=8,
        )
        model.fit(X, y, group=g, verbose=False)
        for n in [100, 250, 500]:
            score = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
            score[d["u"], d["i"]] = model.predict(d["X"], iteration_range=(0, n))
            name = f"{key}_xgbmap_d{depth}_{n}"
            np.save(DEST / (name + ".npy"), score)
            m = evaluate(f.truth, f.rank(score))
            row = {
                "name": name,
                "seconds": time.perf_counter() - t,
                "train_folds": keys,
                **{k: v for k, v in m.items() if k != "per_user_ap"},
            }
            results.append(row)
            print("XGBRESULT", json.dumps(row), flush=True)
        with (DEST / f"{key}_xgbmap_d{depth}.pkl").open("wb") as h:
            pickle.dump(model, h)
        (DEST / f"{key}-xgb-results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="dev")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.fold)
