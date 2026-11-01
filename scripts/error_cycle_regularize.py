"""Focused feature removal and training-recency ablations after error analysis."""

import argparse, json, time
import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from sweettv.error_cycles import DEST
from sweettv.improve import cached, WINDOWS
from sweettv.horizon import rescale
from sweettv.experiment import Fold, SEED
from sweettv.metrics import evaluate
from sweettv.improve_final import export

p = argparse.ArgumentParser()
p.add_argument("--fold", default="dev")
p.add_argument("--config", default="all")
a = p.parse_args()
configs = (
    ["no_pop", "no_collaborative", "recent_half", "recent_only", "shallow"]
    if a.config == "all"
    else [a.config]
)
with threadpool_limits(limits=1):
    keys = (
        ["early1", "early2", "early3"]
        + (["dev"] if a.fold in ["audit", "final"] else [])
        + (["audit"] if a.fold == "final" else [])
    )
    data = [cached(k) for k in keys]
    names = data[0]["names"].tolist()
    for cfg in configs:
        cols = [
            j
            for j, n in enumerate(names)
            if not (cfg == "no_pop" and ("pop" in n))
            and not (
                cfg == "no_collaborative"
                and n.startswith(
                    ("als_", "ppr_", "ease_", "content_", "itemknn_", "userknn_")
                )
            )
        ]
        xs = []
        ys = []
        ws = []
        t = time.perf_counter()
        for idx, d in enumerate(data):
            if cfg == "recent_only" and idx < len(data) - 1:
                continue
            m = d["valid"]
            xs.append(d["X"][m][:, cols])
            ys.append(d["y"][m])
            ws.append(
                np.full(
                    m.sum(),
                    0.5 ** (len(data) - 1 - idx) if cfg == "recent_half" else 1,
                    dtype=float,
                )
            )
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        w = np.concatenate(ws)
        w *= len(w) / w.sum()
        model = CatBoostClassifier(
            iterations=200,
            depth=4 if cfg == "shallow" else 6,
            learning_rate=0.05,
            l2_leaf_reg=10,
            random_seed=SEED,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x, y, sample_weight=w)
        model.save_model(str(DEST / f"{a.fold}_{cfg}.cbm"))
        res = []
        for key in [a.fold] + (["long"] if a.fold == "dev" else []):
            d = cached(key)
            f = Fold(*WINDOWS[key])
            xx = (
                rescale(d["X"], names, (f.end - f.start) / 604800)
                if key in ["long", "final"]
                else d["X"]
            )
            s = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
            s[d["u"], d["i"]] = model.predict_proba(xx[:, cols])[:, 1]
            np.save(DEST / f"{key}_{cfg}.npy", s)
            r = {
                "config": cfg,
                "eval": key,
                "train_windows": keys[-1:] if cfg == "recent_only" else keys,
                "features": [names[j] for j in cols],
                "rows": len(y),
                "seconds": time.perf_counter() - t,
                **{
                    k: v
                    for k, v in evaluate(f.truth, f.rank(s)).items()
                    if k != "per_user_ap"
                },
            }
            if key == "final":
                r.update(export(f, s, "v4_" + cfg))
            res.append(r)
            print(
                "RESULT",
                json.dumps({k: v for k, v in r.items() if k != "features"}),
                flush=True,
            )
        (DEST / f"{a.fold}_{cfg}.json").write_text(json.dumps(res, indent=2))
