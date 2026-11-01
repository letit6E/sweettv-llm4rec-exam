import json, pickle, time
import numpy as np
from xgboost import XGBRanker
from threadpoolctl import threadpool_limits
from sweettv.improve import DEST, WINDOWS, fit_data, cached
from sweettv.improve_final import export
from sweettv.experiment import Fold, SEED
from sweettv.blend import rrf

with threadpool_limits(limits=1):
    t = time.perf_counter()
    keys = ["early1", "early2", "early3", "dev", "audit"]
    X, y, g = fit_data(keys)
    d = cached("final")
    f = Fold(*WINDOWS["final"])
    model = XGBRanker(
        objective="rank:map",
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        tree_method="hist",
        n_jobs=3,
        reg_lambda=10,
        min_child_weight=20,
        colsample_bytree=0.9,
        random_state=SEED,
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=8,
    )
    model.fit(X, y, group=g, verbose=False)
    s = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
    s[d["u"], d["i"]] = model.predict(d["X"])
    out = [export(f, s, "v2_xgbmap500")]
    c = np.load(DEST / "v2_cat200.npy")
    out.append(export(f, 0.5 * rrf(f, s) + 0.5 * rrf(f, c), "v2_cat_xgb"))
    with (DEST / "v2_xgbmap500.pkl").open("wb") as h:
        pickle.dump(model, h)
    (DEST / "xgb-final-exports.json").write_text(
        json.dumps(
            {
                "exports": out,
                "seconds": time.perf_counter() - t,
                "training_pairs": len(y),
            },
            indent=2,
        )
    )
    print(out, flush=True)
