"""Freeze the development-selected strategy before the audit is opened."""

import json, datetime, numpy as np
from .experiment import Fold, OUT


def main():
    if (OUT / "audit-results.json").exists():
        raise RuntimeError(
            "Audit is already open. Do not change the selection after seeing its outcomes."
        )
    best = json.loads((OUT / "development-best.json").read_text())
    fusion = json.loads((OUT / "development-hybrids.json").read_text())
    simple = max(fusion, key=lambda x: x["map5_five_targets"])
    selected = json.loads((OUT / "reranker/selection.json").read_text())["selected"]
    f = Fold("2020-05-18", "2020-06-22")
    hist = np.asarray(f.base[f.eval_users].sum(axis=1)).ravel()
    order = np.flatnonzero(hist > 0)
    order = order[np.argsort(hist[order], kind="stable")]
    rows = [int(order[int(len(order) * q)]) for q in [0.25, 0.9]]
    config = {
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selected_method": "hgb",
        "selected_reranker": selected["name"],
        "selected_config": selected["config"],
        "development_map5": selected["map5_five_targets"],
        "best_individuals": {m: r["config"] for m, r in best.items()},
        "simple_hybrid": simple["config"],
        "simple_hybrid_development_map5": simple["map5_five_targets"],
        "development": ["2020-05-18", "2020-06-22"],
        "audit": ["2020-06-22", "2020-07-27"],
        "example_rows": rows,
        "example_users": [str(f.template.user_id.iloc[r]) for r in rows],
        "example_selection": "25th and 90th percentile positive training watch counts, selected before audit outcomes",
        "example_training_counts": [int(hist[r]) for r in rows],
        "reranker_training_labels_end_no_later_than": "2020-05-18",
        "audit_policy": "One evaluation of frozen choices; no audit-driven tuning",
    }
    (OUT / "frozen-selection.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
