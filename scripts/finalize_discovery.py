"""Freeze one validated discovery candidate; reload models before export."""

import json, datetime, pickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sweettv.experiment import ROOT, Fold
from sweettv.improve import WINDOWS, DEST as OLD
from sweettv.discovery_context import DEST
from sweettv.discovery_ranker import prepare
from sweettv.horizon import rescale
from sweettv.simple_search import baseline, score
from sweettv.blend import rrf
from sweettv.improve_final import export


def main():
    cfg = json.loads((DEST / "context-selection.json").read_text())["selected"][
        "config"
    ]
    assert cfg == dict(family="joint", trees=600, mode="all", scale=1, weight=0.5)
    evidence = {
        k: json.loads((DEST / p).read_text())
        for k, p in [
            ("dev", "context-selection.json"),
            ("audit", "context-audit-confirmation.json"),
            ("long", "context-long-confirmation.json"),
        ]
    }
    assert all(v["selected"]["map5"] > v["baseline"] for v in evidence.values())
    selection = {
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": cfg,
        "validation": {
            k: {"old": v["baseline"], "new": v["selected"]["map5"]}
            for k, v in evidence.items()
        },
        "rule": "Early-selected configuration unchanged across two later checks; small improvements, bootstrap includes zero. Exactly one v6 candidate planned.",
    }
    p = DEST / "submission-selection.json"
    if not p.exists():
        p.write_text(json.dumps(selection, indent=2))
    f = Fold(*WINDOWS["final"])
    d = prepare("final")
    model = CatBoostClassifier()
    model.load_model(str(DEST / "final_discovery_joint.cbm"))
    new = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
    new[d["u"], d["i"]] = model.predict_proba(d["X"], ntree_end=600)[:, 1]
    np.testing.assert_array_equal(new, np.load(DEST / "final_discovery_joint_600.npy"))
    olddata = np.load(OLD / "final-features.npz")
    with (OLD / "v2_cat200.pkl").open("rb") as handle:
        oldmodel = pickle.load(handle)
    x = rescale(olddata["X"], olddata["names"].tolist(), (f.end - f.start) / 604800)
    old = np.full(new.shape, -1e9, dtype="float32")
    old[olddata["u"], olddata["i"]] = oldmodel.predict_proba(x, ntree_end=200)[:, 1]
    np.testing.assert_array_equal(old, baseline("final"))
    channel = score(
        f, dict(family="channel", days=28, duration=True, mode="future_freq")
    )
    # Keep precisely the arithmetic order used in validation; float32 rank ties matter.
    scores = 0.9 * (0.5 * rrf(f, old) + 0.5 * rrf(f, new)) + 0.1 * rrf(f, channel)
    np.save(DEST / "final_discovery_blend.npy", scores)
    result = export(f, scores, "v6_discovery_blend")
    csv = pd.read_csv(ROOT / result["file"], dtype=str)
    allowed = set(
        pd.read_csv(
            ROOT / "data/raw/export_arh_31-42-final.csv", usecols=["tv_show_id"]
        ).tv_show_id
    ) - {0}
    assert csv.user_id.tolist() == f.template.user_id.tolist()
    assert all(
        len(x.split()) == len(set(x.split())) == 5
        and set(map(int, x.split())) <= allowed
        for x in csv.tv_show_id
    )
    result.update(
        models_reloaded=True,
        new_scores_exactly_reproduced=True,
        old_scores_exactly_reproduced=True,
        channel_recomputed=True,
        official_schedule_verified=True,
    )
    (DEST / "submission-verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
