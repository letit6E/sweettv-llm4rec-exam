"""Count supervision: predict frequency rather than arbitrary binary Top5 ties."""

import json, pickle, time
import numpy as np
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
from .improve import cached, DEST, WINDOWS
from .experiment import Fold, SEED, csr
from .metrics import evaluate
from .improve_final import export


def fit_data(keys):
    xs = []
    ys = []
    for k in keys:
        f = Fold(*WINDOWS[k])
        d = cached(k)
        v = f.truth_events
        count = csr(
            v.user.to_numpy(),
            v.tv_show_id.map(f.im).to_numpy(),
            np.ones(len(v)),
            f.shape,
        )[f.eval_users]
        y = np.asarray(count[d["u"], d["i"]]).ravel()
        m = d["valid"]
        xs.append(d["X"][m])
        ys.append(y[m])
    return np.concatenate(xs), np.concatenate(ys)


def run():
    rows = []
    for key in ["dev", "audit", "final"]:
        keys = (
            ["early1", "early2", "early3"]
            + (["dev"] if key in ["audit", "final"] else [])
            + (["audit"] if key == "final" else [])
        )
        assert all(WINDOWS[k][1] <= WINDOWS[key][0] for k in keys)
        X, y = fit_data(keys)
        d = cached(key)
        f = Fold(*WINDOWS[key])
        for loss in ["Poisson", "RMSE"]:
            target = y if loss == "Poisson" else np.log1p(y)
            t = time.perf_counter()
            model = CatBoostRegressor(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=10,
                loss_function=loss,
                random_seed=SEED,
                thread_count=3,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(X, target)
            for n in [200, 400]:
                pred = model.predict(d["X"], ntree_end=n)
                s = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
                s[d["u"], d["i"]] = pred
                name = f"{key}_count_{loss.lower()}_{n}"
                np.save(DEST / (name + ".npy"), s)
                row = {
                    "name": name,
                    "loss": loss,
                    "train_windows": keys,
                    "seconds": time.perf_counter() - t,
                    "training_pairs": len(y),
                }
                if key == "final":
                    row.update(export(f, s, f"v2_count_{loss.lower()}_{n}"))
                else:
                    row.update(
                        {
                            k: v
                            for k, v in evaluate(f.truth, f.rank(s)).items()
                            if k != "per_user_ap"
                        }
                    )
                rows.append(row)
                print("COUNTRESULT", json.dumps(row), flush=True)
            with (DEST / f"{key}_count_{loss.lower()}.pkl").open("wb") as h:
                pickle.dump(model, h)
            (DEST / "count-results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
