"""Learn recurring program calibration from strictly prior label windows."""

import argparse, json, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from sweettv.error_cycles import DEST, matrix
from sweettv.improve import WINDOWS
from sweettv.experiment import Fold, SEED
from sweettv.metrics import evaluate
from sweettv.improve_final import export

p = argparse.ArgumentParser()
p.add_argument("--fold", default="dev")
p.add_argument("--identity", default="item", choices=["item", "both"])
p.add_argument("--trees", type=int, default=400)
a = p.parse_args()


def data(k):
    d, x, n = matrix(k, 0)
    f = Fold(*WINDOWS[k])
    df = pd.DataFrame(x, columns=n)
    df["program_id"] = f.items[d["i"]].astype(str)
    # User numeric source indices treated only as categories, not continuous quantities.
    if a.identity == "both":
        df["viewer_id"] = f.eval_users[d["u"]].astype(str)
    return d, df, f


with threadpool_limits(limits=1):
    keys = (
        ["early1", "early2", "early3"]
        + (["dev"] if a.fold in ["audit", "final"] else [])
        + (["audit"] if a.fold == "final" else [])
    )
    xs = []
    ys = []
    t = time.perf_counter()
    for k in keys:
        assert WINDOWS[k][1] <= WINDOWS[a.fold][0]
        d, x, _ = data(k)
        xs.append(x[d["valid"]])
        ys.append(d["y"][d["valid"]])
    x = pd.concat(xs, ignore_index=True)
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
        cat_features=["program_id"] + (["viewer_id"] if a.identity == "both" else []),
    )
    model.fit(x, y)
    name = f"{a.fold}_identity_{a.identity}_{a.trees}"
    model.save_model(str(DEST / (name + ".cbm")))
    res = []
    for k in [a.fold] + (["long"] if a.fold == "dev" else []):
        d, x, f = data(k)
        for it in [200, a.trees] if a.trees != 200 and a.fold == "dev" else [a.trees]:
            z = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
            z[d["u"], d["i"]] = model.predict_proba(x, ntree_end=it)[:, 1]
            np.save(DEST / f"{k}_identity_{a.identity}_{it}.npy", z)
            r = {
                "name": name,
                "trees": it,
                "eval": k,
                "train_windows": keys,
                "training_pairs": len(y),
                "seconds": time.perf_counter() - t,
                **{
                    k: v
                    for k, v in evaluate(f.truth, f.rank(z)).items()
                    if k != "per_user_ap"
                },
            }
            if k == "final":
                r.update(export(f, z, f"v4_identity_{a.identity}_{it}"))
            print("RESULT", json.dumps(r), flush=True)
            res.append(r)
    (DEST / (name + ".json")).write_text(json.dumps(res, indent=2))
