"""Measured ranking errors by history size and target novelty."""

import json
from collections import Counter
import numpy as np

from sweettv.neural_rank import DEST
from sweettv.improve import WINDOWS, DEST as OLD
from sweettv.experiment import Fold
from sweettv.metrics import evaluate


def run(fold="dev"):
    f = Fold(*WINDOWS[fold])
    paths = [("previous_cat200", OLD / f"{fold}_cat6_200.npy")]
    for folder in sorted(DEST.glob(f"{fold}_*")):
        score = folder / "fixed-scores.npy"
        if not score.exists():
            score = folder / "best-scores.npy"
        if score.exists():
            paths.append((folder.name, score))
    five = np.array([len(t) == 5 for t in f.truth])
    volume = np.asarray(f.base[f.eval_users].sum(axis=1)).ravel()
    cohorts = {
        "light_under30": five & (volume < 30),
        "medium30_99": five & (volume >= 30) & (volume < 100),
        "heavy100plus": five & (volume >= 100),
    }
    known_catalog = set(f.past.tv_show_id)
    known_views = set(f.items[f.pop > 0])
    rows = []
    for name, path in paths:
        rec = f.rank(np.load(path))
        metric = evaluate(f.truth, rec)
        ap = np.array(metric["per_user_ap"])
        classes = {
            k: dict(targets=0, hits=0)
            for k in [
                "seen_by_user",
                "new_to_user_with_qualifying_views",
                "known_catalog_without_qualifying_views",
                "new_to_schedule_catalog",
            ]
        }
        fp, fn = Counter(), Counter()
        for row, (user, truth, pred) in enumerate(zip(f.eval_users, f.truth, rec)):
            if not five[row]:
                continue
            seen = set(f.items[f.base[user].indices])
            for item in truth:
                if item in seen:
                    category = "seen_by_user"
                elif item in known_views:
                    category = "new_to_user_with_qualifying_views"
                elif item in known_catalog:
                    category = "known_catalog_without_qualifying_views"
                else:
                    category = "new_to_schedule_catalog"
                classes[category]["targets"] += 1
                classes[category]["hits"] += int(item in pred)
            fp.update(set(pred) - set(truth))
            fn.update(set(truth) - set(pred))
        rows.append(
            dict(
                model=name,
                map5=metric["map5_five_targets"],
                cohorts={
                    k: dict(users=int(m.sum()), map5=float(ap[m].mean()))
                    for k, m in cohorts.items()
                },
                target_classes=classes,
                false_positive_programs=fp.most_common(15),
                missed_programs=fn.most_common(15),
            )
        )
    (DEST / f"{fold}-error-analysis.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(r["model"], r["map5"], r["target_classes"])


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="dev", choices=["dev", "audit"])
    run(p.parse_args().fold)
