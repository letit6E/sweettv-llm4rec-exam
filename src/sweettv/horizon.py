"""Test horizon covariate shift without fitting on later validation labels."""

import json, pickle
import numpy as np
from threadpoolctl import threadpool_limits
from .improve import cached, DEST, WINDOWS
from .experiment import Fold
from .metrics import evaluate
from .improve_final import export


def rescale(X, names, weeks, target=5):
    """Convert exposure-dependent features to the five-week training scale."""
    x = X.copy()
    ratio = target / weeks
    for j, n in enumerate(names):
        if n == "channel_score":
            x[:, j] *= ratio
        elif n == "future_airings" or n.startswith("expected_count_"):
            x[:, j] = np.log1p(np.expm1(x[:, j]) * ratio)
        elif n == "horizon_weeks":
            x[:, j] = target
    x[:, names.index("airing_ratio")] = x[
        :, names.index("future_airings")
    ] / np.maximum(x[:, names.index("past_airings")], 1)
    assert np.isfinite(x).all()
    return x


def run():
    results = []
    for key in ["long", "final"]:
        d = cached(key)
        f = Fold(*WINDOWS[key])
        names = d["names"].tolist()
        weeks = (f.end - f.start) / 86400 / 7
        path = DEST / ("dev_cat6.pkl" if key == "long" else "v2_cat200.pkl")
        with path.open("rb") as h:
            model = pickle.load(h)
        for scale in [False, True]:
            x = rescale(d["X"], names, weeks) if scale else d["X"]
            vals = model.predict_proba(x, ntree_end=200)[:, 1]
            s = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
            s[d["u"], d["i"]] = vals
            row = {
                "fold": key,
                "five_week_normalization": scale,
                "weeks": weeks,
                "model_training_labels_end": "2020-05-18"
                if key == "long"
                else "2020-07-27",
            }
            if key == "long":
                row.update(
                    {
                        k: v
                        for k, v in evaluate(f.truth, f.rank(s)).items()
                        if k != "per_user_ap"
                    }
                )
            elif scale:
                row.update(export(f, s, "v2_cat200_horizon"))
            np.save(DEST / f"{key}_horizon_{scale}.npy", s)
            results.append(row)
            print("HORIZON", json.dumps(row), flush=True)
        (DEST / "horizon-results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
