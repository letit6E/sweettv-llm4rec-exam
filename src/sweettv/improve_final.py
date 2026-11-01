"""Refit v2 learners on all allowed rolling windows and export independent files."""

import json, pickle, time, hashlib
import numpy as np
from catboost import CatBoostClassifier
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from .improve import cached, fit_data, DEST, WINDOWS
from .experiment import Fold, ROOT, SEED


def export(f, score, name):
    pred = f.rank(score)
    frame = f.template.copy()
    frame["tv_show_id"] = [" ".join(map(str, p)) for p in pred]
    allowed = set(f.future.tv_show_id)
    assert len(frame) == 1260 and frame.user_id.is_unique
    assert all(
        len(set(p)) == len(p) == 5 and set(p) <= allowed and 0 not in p for p in pred
    )
    path = ROOT / "submissions" / (name + ".csv")
    frame.to_csv(path, index=False)
    np.save(DEST / (name + ".npy"), score)
    return {
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
        "candidate_count": len(allowed),
    }


def run():
    keys = ["early1", "early2", "early3", "dev", "audit"]
    X, y, g = fit_data(keys)
    d = cached("final")
    f = Fold(*WINDOWS["final"])
    records = []
    configs = [
        ("v2_cat200", "cat", 200),
        ("v2_lgb_rank400", "rank", 400),
        ("v2_lgb_bin200", "bin", 200),
    ]
    for name, kind, it in configs:
        t = time.perf_counter()
        if kind == "cat":
            model = CatBoostClassifier(
                iterations=it,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=10,
                loss_function="Logloss",
                random_seed=SEED,
                thread_count=4,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(X, y)
            v = model.predict_proba(d["X"])[:, 1]
        else:
            params = dict(
                n_estimators=it,
                num_leaves=15 if kind == "rank" else 31,
                learning_rate=0.03,
                min_child_samples=80,
                reg_lambda=10,
                colsample_bytree=0.9,
                random_state=SEED,
                n_jobs=4,
                verbosity=-1,
            )
            if kind == "rank":
                model = lgb.LGBMRanker(
                    objective="lambdarank", lambdarank_truncation_level=8, **params
                )
                model.fit(X, y, group=g)
                v = model.predict(d["X"])
            else:
                model = lgb.LGBMClassifier(**params)
                model.fit(X, y)
                v = model.predict_proba(d["X"])[:, 1]
        s = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
        s[d["u"], d["i"]] = v
        row = {
            **export(f, s, name),
            "name": name,
            "train_windows": keys,
            "seconds": time.perf_counter() - t,
            "training_pairs": len(y),
        }
        records.append(row)
        with (DEST / (name + ".pkl")).open("wb") as h:
            pickle.dump(model, h)
        print("EXPORT", json.dumps(row), flush=True)
        (DEST / "final-exports.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
