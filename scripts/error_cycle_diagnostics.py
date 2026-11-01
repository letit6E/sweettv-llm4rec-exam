"""Compare existing ranking errors and counterfactual oracle ceilings."""

import json
import numpy as np
from collections import Counter
from sweettv.experiment import Fold
from sweettv.improve import cached, WINDOWS, DEST as OLD
from sweettv.error_cycles import DEST
from sweettv.metrics import evaluate

res = {}
for key in ["dev", "audit", "long"]:
    f = Fold(*WINDOWS[key])
    d = cached(key)
    s = np.load(
        OLD / (f"{key}_cat6_200.npy" if key != "long" else "long_horizon_True.npy")
    )
    pred = f.rank(s)
    five = np.array([len(t) == 5 for t in f.truth])
    s = s.copy()
    s[f.cold] = f.pop
    eligible = np.flatnonzero(f.eligible)
    inverse = np.full(s.shape, len(f.items), dtype=np.int32)
    for row, score in enumerate(s):
        order = eligible[
            np.lexsort((f.items[eligible], -f.pop[eligible], -score[eligible]))
        ]
        inverse[row, order] = np.arange(len(order))
        assert f.items[order[:5]].tolist() == pred[row]
    seen = f.matrix(0.8, None)[f.eval_users].toarray() > 0
    partial = f.matrix(0, None)[f.eval_users].toarray() > 0
    c = {}
    for name in ["full_seen", "partial_only", "no_overlap"]:
        ranks = []
        for r in np.flatnonzero(five):
            for item in f.truth[r]:
                j = f.im[item]
                label = (
                    "full_seen"
                    if seen[r, j]
                    else "partial_only"
                    if partial[r, j]
                    else "no_overlap"
                )
                if label == name:
                    ranks.append(int(inverse[r, j]) + 1)
        c[name] = {
            "targets": len(ranks),
            "top5": sum(r <= 5 for r in ranks),
            "top10": sum(r <= 10 for r in ranks),
            "top50": sum(r <= 50 for r in ranks),
        }
    miss = Counter()
    false = Counter()
    for t, p, m in zip(f.truth, pred, five):
        if m:
            miss.update(set(t) - set(p))
            false.update(set(p) - set(t))
    meta = (
        f.past.sort_values("end")
        .drop_duplicates("tv_show_id", keep="last")
        .set_index("tv_show_id")
    )

    def details(counter):
        return [
            {
                "id": int(k),
                "title": str(meta.loc[k, "tv_show_title"])
                if k in meta.index
                else "unknown historical title",
                "users": int(n),
            }
            for k, n in counter.most_common(10)
        ]

    res[key] = {
        "baseline_map5": evaluate(f.truth, pred)["map5_five_targets"],
        "target_ranks": c,
        "misses": details(miss),
        "false_positives": details(false),
    }
(DEST / "baseline-errors.json").write_text(
    json.dumps(res, indent=2, ensure_ascii=False)
)
print(
    json.dumps(
        {
            k: {"map": v["baseline_map5"], "classes": v["target_ranks"]}
            for k, v in res.items()
        },
        indent=2,
    )
)
