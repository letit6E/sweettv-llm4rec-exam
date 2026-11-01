"""Separate missing-candidate, unseen-program and ranking errors without changing labels."""

import json
import numpy as np
import pandas as pd
from sweettv.experiment import Fold, OUT
from sweettv.improve import WINDOWS, cached
from sweettv.simple_search import baseline
from sweettv.blend import rrf
from sweettv.metrics import evaluate

DEST = OUT / "discovery-v6"
DEST.mkdir(exist_ok=True)
summary = {}
for key in ["dev", "audit", "long"]:
    f = Fold(*WINDOWS[key])
    d = cached(key)
    raw = np.load(OUT / "approaches-v5" / f"{key}_channel.npy")
    score = 0.9 * rrf(f, baseline(key)) + 0.1 * rrf(f, raw)
    score[f.cold] = f.pop
    rank = np.full(score.shape, len(f.items), dtype=np.int32)
    ids = np.flatnonzero(f.eligible)
    for u, s in enumerate(score):
        ix = ids[np.lexsort((f.items[ids], -f.pop[ids], -s[ids]))]
        rank[u, ix] = np.arange(len(ix)) + 1
    candidate = np.zeros(score.shape, bool)
    candidate[d["u"], d["i"]] = True
    anyview = f.matrix(0)[f.eval_users].toarray()
    full = f.base[f.eval_users].toarray()
    counts = f.truth_events.groupby(["user", "tv_show_id"]).size().to_dict()
    pastair = f.past.groupby("tv_show_id").size().to_dict()
    futureair = f.future.groupby("tv_show_id").size().to_dict()
    meta = (
        f.past.sort_values("end")
        .drop_duplicates("tv_show_id", keep="last")
        .set_index("tv_show_id")
    )
    globalview = np.asarray((f.matrix(0) > 0).sum(axis=0)).ravel()
    rows = []
    for r, user in enumerate(f.eval_users):
        if len(f.truth[r]) != 5:
            continue
        for item in f.truth[r]:
            j = f.im[item]
            rows.append(
                dict(
                    user_row=r,
                    item=item,
                    kind="full_seen"
                    if full[r, j] > 0
                    else "partial_only"
                    if anyview[r, j] > 0
                    else "unseen",
                    rank=int(rank[r, j]),
                    candidate=bool(candidate[r, j]),
                    global_full_users=int(f.pop[j]),
                    global_any_users=int(globalview[j]),
                    past_airings=pastair.get(item, 0),
                    future_airings=futureair.get(item, 0),
                    future_count=counts[(user, item)],
                    title=meta.loc[item, "tv_show_title"] if item in meta.index else "",
                    category=meta.loc[item, "tv_show_category"]
                    if item in meta.index
                    else "",
                    channel_signal=float(raw[r, j]),
                    cold_user=bool(f.cold[r]),
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(DEST / f"{key}-target-diagnostics.csv", index=False)
    res = {}
    for name, t in table.groupby("kind"):
        res[name] = {
            "targets": len(t),
            "retrieved": int(t.candidate.sum()),
            "hit5": int((t["rank"] <= 5).sum()),
            "hit20": int((t["rank"] <= 20).sum()),
            "hit100": int((t["rank"] <= 100).sum()),
            "median_rank": float(t["rank"].median()),
            "single_future_view": int((t.future_count == 1).sum()),
            "catalog_no_full_history": int((t.global_full_users == 0).sum()),
            "catalog_no_any_history": int((t.global_any_users == 0).sum()),
            "absent_from_past_schedule": int((t.past_airings == 0).sum()),
            "channel_signal_nonzero": int((t.channel_signal > 0).sum()),
            "median_future_count": float(t.future_count.median()),
        }
    # Perfect new-item ordering alone is a ceiling, not an evaluated model.
    ceiling = score.copy()
    for r in range(len(f.truth)):
        for item in f.truth[r]:
            if anyview[r, f.im[item]] == 0:
                ceiling[r, f.im[item]] = 1e6
    res["overall"] = {
        "map5": evaluate(f.truth, f.rank(score))["map5_five_targets"],
        "oracle_unseen_only": evaluate(f.truth, f.rank(ceiling))["map5_five_targets"],
    }
    summary[key] = res
    print(key, json.dumps(res), flush=True)
(DEST / "baseline-diagnosis.json").write_text(json.dumps(summary, indent=2))
