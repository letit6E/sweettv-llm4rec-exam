"""Validation-only calibration motivated by novel-item under-retrieval into Top5."""

import json
import numpy as np
from sweettv.experiment import Fold
from sweettv.improve import DEST as OLD, WINDOWS
from sweettv.error_cycles import DEST
from sweettv.metrics import evaluate


def adjust(f, s, novel, pop):
    # Logistic intercept for non-fully-watched programs and popularity shrinkage.
    seen = f.matrix(0.8, None)[f.eval_users].toarray() > 0
    z = np.log(np.clip(s, 1e-8, 1 - 1e-8) / (1 - np.clip(s, 1e-8, 1 - 1e-8)))
    z += novel * (~seen)
    z -= pop * np.log1p(f.pop)[None, :]
    z[s < -1e8] = -1e9
    return z.astype("float32")


res = []
f = Fold(*WINDOWS["dev"])
s = np.load(OLD / "dev_cat6_200.npy")
for novel in [-0.5, 0, 0.5, 1, 1.5]:
    for pop in [0, 0.1, 0.2]:
        z = adjust(f, s, novel, pop)
        m = evaluate(f.truth, f.rank(z))["map5_five_targets"]
        res.append({"novel": novel, "pop": pop, "dev": m})
best = max(res, key=lambda r: r["dev"])
(DEST / "calibration-selection.json").write_text(
    json.dumps(
        {
            "grid": res,
            "selected": best,
            "selection": "dev only; audit and overlapping long inspected after selection",
        },
        indent=2,
    )
)
for key in ["audit", "long"]:
    f = Fold(*WINDOWS[key])
    s = np.load(
        OLD / (f"{key}_cat6_200.npy" if key == "audit" else "long_horizon_True.npy")
    )
    z = adjust(f, s, best["novel"], best["pop"])
    best[key] = evaluate(f.truth, f.rank(z))["map5_five_targets"]
(DEST / "calibration-confirmation.json").write_text(json.dumps(best, indent=2))
print(best, flush=True)
