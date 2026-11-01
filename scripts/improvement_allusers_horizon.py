import json, pickle
import numpy as np
from threadpoolctl import threadpool_limits
from sweettv.improve import DEST, WINDOWS, cached
from sweettv.experiment import Fold
from sweettv.horizon import rescale
from sweettv.improve_final import export
from sweettv.metrics import evaluate

with threadpool_limits(limits=1):
    out = []
    for key in ["long", "final"]:
        f = Fold(*WINDOWS[key])
        d = cached(key)
        names = d["names"].tolist()
        with (
            DEST
            / ("dev_allusers_cat.pkl" if key == "long" else "final_allusers_cat.pkl")
        ).open("rb") as h:
            model = pickle.load(h)
        for n in [200, 500]:
            for scale in [False, True]:
                x = (
                    rescale(d["X"], names, (f.end - f.start) / 86400 / 7)
                    if scale
                    else d["X"]
                )
                s = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
                s[d["u"], d["i"]] = model.predict_proba(x, ntree_end=n)[:, 1]
                row = {"fold": key, "trees": n, "horizon_normalized": scale}
                if key == "long":
                    row.update(
                        {
                            k: v
                            for k, v in evaluate(f.truth, f.rank(s)).items()
                            if k != "per_user_ap"
                        }
                    )
                elif scale:
                    row.update(export(f, s, f"v3_allusers_cat{n}_horizon"))
                out.append(row)
                print(json.dumps(row), flush=True)
    (DEST / "allusers-horizon-results.json").write_text(json.dumps(out, indent=2))
