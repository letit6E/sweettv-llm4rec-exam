"""Change historical target horizon; compare on identical outer evaluations."""

import argparse, json, time
import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from sweettv.error_cycles import DEST, matrix
from sweettv.improve import WINDOWS
from sweettv.experiment import Fold, SEED
from sweettv.metrics import evaluate
from sweettv.improve_final import export

WINDOWS.update(
    wide1=("2020-03-23", "2020-05-18"),
    wide2=("2020-04-06", "2020-05-18"),
    wide3=("2020-04-13", "2020-06-22"),
)
p = argparse.ArgumentParser()
p.add_argument("--fold", default="dev")
p.add_argument("--stage", type=int, default=0)
p.add_argument("--trees", type=int, default=200)
a = p.parse_args()
with threadpool_limits(limits=1):
    keys = ["wide1", "wide2", "early3"]
    if a.fold in ["audit", "final"]:
        keys += ["wide3", "dev"]
    if a.fold == "final":
        keys += ["long", "audit"]
    xs = []
    ys = []
    t = time.perf_counter()
    for k in keys:
        assert WINDOWS[k][1] <= WINDOWS[a.fold][0]
        d, x, n = matrix(k, a.stage)
        xs.append(x[d["valid"]])
        ys.append(d["y"][d["valid"]])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    model = CatBoostClassifier(
        iterations=a.trees,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        random_seed=SEED,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y)
    name = f"{a.fold}_wide_stage{a.stage}_{a.trees}"
    model.save_model(str(DEST / (name + ".cbm")))
    results = []
    for k in [a.fold] + (["long"] if a.fold == "dev" else []):
        d, x, n = matrix(k, a.stage)
        f = Fold(*WINDOWS[k])
        z = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
        z[d["u"], d["i"]] = model.predict_proba(x)[:, 1]
        np.save(DEST / f"{k}_wide_stage{a.stage}_{a.trees}.npy", z)
        r = {
            "name": name,
            "eval": k,
            "train_windows": keys,
            "train_rows": len(y),
            "seconds": time.perf_counter() - t,
            **{
                k: v
                for k, v in evaluate(f.truth, f.rank(z)).items()
                if k != "per_user_ap"
            },
        }
        if k == "final":
            r.update(export(f, z, f"v4_wide_stage{a.stage}_{a.trees}"))
        results.append(r)
        print("RESULT", json.dumps(r), flush=True)
    (DEST / f"{name}.json").write_text(json.dumps(results, indent=2))
