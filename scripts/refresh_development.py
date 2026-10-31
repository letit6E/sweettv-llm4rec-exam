import json, time, numpy as np
from sweettv.experiment import Fold, OUT, CACHE, normalized
from sweettv.metrics import evaluate
from threadpoolctl import threadpool_limits

if (OUT / "audit-results.json").exists():
    raise RuntimeError("Historical pre-audit refresh only; cannot run after the audit.")
with threadpool_limits(1):
    f = Fold("2020-05-18", "2020-06-22")
    p = OUT / "development-results.json"
    results = json.loads(p.read_text())
    (OUT / "development-results-before-dst-hour-fix.json").write_text(p.read_text())
    for row in results:
        c = row["config"]
        if c["method"] == "channel" and c["hourly"]:
            t = time.perf_counter()
            s = f.model(c)
            np.save(CACHE / (row["key"] + ".npy"), s)
            row.update(evaluate(f.truth, f.rank(s)))
            row["runtime_seconds"] = time.perf_counter() - t
            row["note"] = "Local hour resolved per interval segment across DST"
            print(c, row["map5_five_targets"], flush=True)
    best = {}
    for r in results:
        m = r["config"]["method"]
        if m not in best or r["map5_five_targets"] > best[m]["map5_five_targets"]:
            best[m] = r
    p.write_text(json.dumps(results, indent=2))
    (OUT / "development-best.json").write_text(json.dumps(best, indent=2))
    fusion = []
    anchor = best["repeat"]
    a = normalized(np.load(CACHE / (anchor["key"] + ".npy")))
    for m in ["content", "als", "association", "ppr", "ease", "channel"]:
        b = normalized(np.load(CACHE / (best[m]["key"] + ".npy")))
        for w in [0.1, 0.25, 0.5, 0.75]:
            fusion.append(
                {
                    "config": {
                        "method": "hybrid",
                        "components": [anchor["config"], best[m]["config"]],
                        "weights": [1 - w, w],
                    },
                    **evaluate(f.truth, f.rank((1 - w) * a + w * b)),
                }
            )
    (OUT / "development-hybrids.json").write_text(json.dumps(fusion, indent=2))
    winner = max(fusion, key=lambda x: x["map5_five_targets"])
    selected = json.loads((OUT / "reranker/selection.json").read_text())["selected"]
    # Freeze before accessing audit outcomes. User examples selected from training only.
    hist = np.asarray(f.base[f.eval_users].sum(axis=1)).ravel()
    order = np.flatnonzero(hist > 0)
    order = order[np.argsort(hist[order], kind="stable")]
    rows = [int(order[int(len(order) * q)]) for q in [0.25, 0.9]]
    cfg = {
        "frozen_at_utc": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "selected_method": "hgb",
        "selected_reranker": selected["name"],
        "selected_config": selected["config"],
        "development_map5": selected["map5_five_targets"],
        "best_individuals": {m: r["config"] for m, r in best.items()},
        "simple_hybrid": winner["config"],
        "simple_hybrid_development_map5": winner["map5_five_targets"],
        "development": ["2020-05-18", "2020-06-22"],
        "audit": ["2020-06-22", "2020-07-27"],
        "example_rows": rows,
        "example_users": [str(f.template.user_id.iloc[r]) for r in rows],
        "example_selection": "25th and 90th percentile positive training watch counts, selected before audit outcomes",
        "example_training_counts": [int(hist[r]) for r in rows],
        "reranker_training_labels_end_no_later_than": "2020-05-18",
        "audit_policy": "One evaluation of frozen choices; no audit-driven tuning",
    }
    (OUT / "frozen-selection.json").write_text(json.dumps(cfg, indent=2))
    print("FROZEN", cfg, flush=True)
