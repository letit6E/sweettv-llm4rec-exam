"""Separate new feature benefit from refreshing supervised training windows."""

import json, time
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from sweettv.improve import cached, fit_data, DEST, WINDOWS
from sweettv.experiment import Fold, SEED
from sweettv.metrics import evaluate

with threadpool_limits(limits=1):
    d = cached("audit")
    f = Fold(*WINDOWS["audit"])
    names = d["names"].tolist()
    # Original features are shifted by the six new retrieval features after column 18.
    old = list(range(18)) + list(range(24, 55))
    records = []
    for recent in [False, True]:
        keys = ["early1", "early2", "early3"] + (["dev"] if recent else [])
        X, y, g = fit_data(keys)
        for expanded in [False, True]:
            cols = list(range(X.shape[1])) if expanded else old
            model = lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.03,
                min_child_samples=80,
                reg_lambda=10,
                colsample_bytree=0.9,
                random_state=SEED,
                n_jobs=3,
                verbosity=-1,
            )
            t = time.perf_counter()
            model.fit(X[:, cols], y)
            z = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
            z[d["u"], d["i"]] = model.predict_proba(d["X"][:, cols])[:, 1]
            m = evaluate(f.truth, f.rank(z))
            row = {
                "recent_training": recent,
                "expanded_features": expanded,
                "feature_count": len(cols),
                "training_pairs": len(y),
                "seconds": time.perf_counter() - t,
                **{k: v for k, v in m.items() if k != "per_user_ap"},
            }
            records.append(row)
            print(json.dumps(row), flush=True)
            (DEST / "ablation.json").write_text(json.dumps(records, indent=2))
