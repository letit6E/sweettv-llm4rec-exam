"""Select small representation ensemble on earlier validation, log longer check."""

import json
import numpy as np
from sweettv.experiment import Fold
from sweettv.improve import WINDOWS, DEST as OLD
from sweettv.error_cycles import DEST
from sweettv.metrics import evaluate

names = ["stage1_200", "stage2_200", "stage13_200", "stage14_200", "identity_item_400"]
res = []
folds = {k: Fold(*WINDOWS[k]) for k in ["dev", "long"]}
base = {
    k: np.load(OLD / (f"{k}_cat6_200.npy" if k == "dev" else "long_horizon_True.npy"))
    for k in folds
}
for name in names + ["all_features"]:
    for weight in [0.25, 0.5, 0.75]:
        rec = {"component": name, "weight": weight}
        for k, f in folds.items():
            p = (
                np.mean([np.load(DEST / f"{k}_{n}.npy") for n in names], axis=0)
                if name == "all_features"
                else np.load(DEST / f"{k}_{name}.npy")
            )
            z = (1 - weight) * base[k] + weight * p
            rec[k] = evaluate(f.truth, f.rank(z))["map5_five_targets"]
        res.append(rec)
best = max(res, key=lambda r: r["dev"])
(DEST / "ensemble-selection.json").write_text(
    json.dumps(
        {
            "grid": res,
            "selected": best,
            "selection": "Max dev, then inspect long; later audit not yet evaluated",
        },
        indent=2,
    )
)
print(json.dumps(best), flush=True)
