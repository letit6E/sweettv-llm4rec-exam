"""Evaluate the frozen choice once, then refit base models on all allowed history."""

import argparse, json, time, pickle, hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .experiment import Fold, ROOT, OUT, normalized, SEED
from .rerank import COMPONENTS, build_features, candidate_recall
from .metrics import evaluate


def compact(metrics):
    return {k: v for k, v in metrics.items() if k != "per_user_ap"}


def score_models(f, config):
    scores = {}
    timings = {}
    checks = {}
    for name, c in config["best_individuals"].items():
        t = time.perf_counter()
        scores[name] = f.model(c).astype(np.float32)
        assert np.isfinite(scores[name]).all()
        timings[name] = time.perf_counter() - t
        if name == "ppr":
            checks["ppr"] = f.last_ppr
        print("scored", name, "seconds", round(timings[name], 2), flush=True)
    hc = config["simple_hybrid"]
    scores["simple_hybrid"] = sum(
        w * normalized(scores[c["method"]])
        for c, w in zip(hc["components"], hc["weights"])
    )
    component_scores = {n: scores[n] for n in COMPONENTS}
    ranks = {}
    for n, s in component_scores.items():
        v = s.copy()
        v[:, ~f.eligible] = -np.inf
        order = np.argsort(-v, axis=1, kind="stable")
        rank = np.empty_like(order, dtype=np.int32)
        np.put_along_axis(rank, order, np.arange(len(f.items))[None, :], axis=1)
        ranks[n] = rank
    t = time.perf_counter()
    x, u, i, names, candidates = build_features(f, component_scores, ranks)
    selection = json.loads((OUT / "reranker/selection.json").read_text())
    assert names == selection["feature_names"]
    with (OUT / "reranker" / (config["selected_reranker"] + ".pkl")).open("rb") as h:
        model = pickle.load(h)
    s = np.zeros_like(scores["repeat"])
    s[u, i] = model.predict_proba(x)[:, 1]
    scores["hgb"] = s
    timings["hgb_features_and_inference"] = time.perf_counter() - t
    checks["candidate_recall_all_nonempty"] = candidate_recall(f, candidates)
    return scores, timings, checks


def metadata(f):
    # Display lookup only. Test attributes never enter any model feature.
    old = f.past.sort_values("start").drop_duplicates("tv_show_id", keep="last")
    table = (
        old[["tv_show_id", "tv_show_title", "tv_show_category"]]
        .set_index("tv_show_id")
        .to_dict("index")
    )
    if f.start >= __import__("sweettv.experiment", fromlist=["epoch"]).epoch(
        "2020-07-27"
    ):
        for c in pd.read_csv(
            ROOT / "data/raw/export_arh_31-42-final.csv",
            usecols=["tv_show_id", "tv_show_title", "tv_show_category"],
            keep_default_na=False,
            chunksize=200000,
        ):
            for row in (
                c[c.tv_show_id > 0]
                .drop_duplicates("tv_show_id")
                .itertuples(index=False)
            ):
                if int(row.tv_show_id) not in table:
                    table[int(row.tv_show_id)] = {
                        "tv_show_title": row.tv_show_title,
                        "tv_show_category": row.tv_show_category,
                    }
    return table


def examples(f, predictions, scores, config):
    table = metadata(f)
    result = []

    def item(i):
        return {
            "id": int(i),
            "title": table.get(int(i), {}).get(
                "tv_show_title", "Unknown historical title"
            ),
            "category": table.get(int(i), {}).get("tv_show_category", "Unknown"),
        }

    for row in config["example_rows"]:
        u = int(f.eval_users[row])
        history = (
            f.train[(f.train.user == u) & (f.train.fraction >= 0.8)]
            .groupby("tv_show_id")
            .size()
            .sort_values(ascending=False)
        )
        result.append(
            {
                "user_id": str(f.template.user_id.iloc[row]),
                "template_row": row,
                "qualifying_airings": int(history.sum()),
                "distinct_programs": len(history),
                "training_top10": [
                    {**item(i), "count": int(n)} for i, n in history.head(10).items()
                ],
                "truth": [item(i) for i in f.truth[row]],
                "recommendations": {
                    m: [
                        {
                            **item(i),
                            "score": float(scores[m][row, f.im[i]]),
                            "train_watch_count": int(history.get(i, 0)),
                        }
                        for i in p[row]
                    ]
                    for m, p in predictions.items()
                },
            }
        )
    return result


def cohort_details(f, predictions, metrics):
    five = np.array([len(t) == 5 for t in f.truth])
    rng = np.random.default_rng(SEED)
    paired = {}
    a = np.array(metrics["hgb"]["per_user_ap"])[five]
    for m in ["popularity", "repeat", "channel", "ppr", "simple_hybrid"]:
        diff = a - np.array(metrics[m]["per_user_ap"])[five]
        boot = diff[rng.integers(0, len(diff), (2000, len(diff)))].mean(axis=1)
        paired[m] = {
            "mean_difference": float(diff.mean()),
            "bootstrap_95_percentile_interval": np.quantile(
                boot, [0.025, 0.975]
            ).tolist(),
        }
    counts = f.truth_events.groupby(["user", "tv_show_id"]).size()
    ties = 0
    for u in f.eval_users[five]:
        values = sorted(counts.loc[u].tolist(), reverse=True)
        ties += len(values) > 5 and values[4] == values[5]
    groups = {}
    for label, mask in [("cold", five & f.cold), ("warm", five & ~f.cold)]:
        groups[label] = {
            "users": int(mask.sum()),
            "map5": {
                m: float(np.array(v["per_user_ap"])[mask].mean())
                if mask.any()
                else None
                for m, v in metrics.items()
            },
        }
    return {
        "paired_bootstrap": paired,
        "groups": groups,
        "fifth_place_target_frequency_ties": int(ties),
        "tie_rule": "ascending numeric program ID; organizer tie rule unspecified",
    }


def run(mode):
    config = json.loads((OUT / "frozen-selection.json").read_text())
    window = config["audit"] if mode == "audit" else ["2020-07-27", "2020-10-19"]
    if mode == "audit" and (OUT / "audit-results.json").exists():
        raise RuntimeError(
            "Audit already evaluated. Preserve it; do not silently tune or overwrite."
        )
    f = Fold(*window)
    print("FOLD", f.audit, flush=True)
    scores, timing, checks = score_models(f, config)
    pred = {m: f.rank(s) for m, s in scores.items()}
    for lists in pred.values():
        assert len(lists) == 1260
        for p in lists:
            assert (
                len(p) == len(set(p)) == 5
                and 0 not in p
                and all(f.eligible[f.im[i]] for i in p)
            )
    result = {
        "protocol": f.audit,
        "timing_seconds": timing,
        "checks": checks,
        "selection_frozen_at": config["frozen_at_utc"],
    }
    if mode == "audit":
        metrics = {m: evaluate(f.truth, p) for m, p in pred.items()}
        result["metrics"] = {m: compact(v) for m, v in metrics.items()}
        result.update(cohort_details(f, pred, metrics))
        for m, v in result["metrics"].items():
            print("AUDIT", m, v["map5_five_targets"], flush=True)
        (OUT / "audit-results.json").write_text(json.dumps(result, indent=2))
        (OUT / "audit-examples.json").write_text(
            json.dumps(examples(f, pred, scores, config), ensure_ascii=False, indent=2)
        )
    else:
        folder = ROOT / "submissions"
        folder.mkdir(exist_ok=True)
        path = folder / "final_hgb.csv"
        frame = f.template.copy()
        assert len(frame.columns) == 2
        frame.iloc[:, 1] = [" ".join(map(str, p)) for p in pred["hgb"]]
        frame.to_csv(path, index=False)
        read = pd.read_csv(path, dtype=str)
        assert read.user_id.tolist() == f.template.user_id.tolist()
        assert read.iloc[:, 1].tolist() == frame.iloc[:, 1].tolist()
        result["submission"] = {
            "file": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(read),
            "prediction_column": str(read.columns[1]),
            "selected_model": config["selected_reranker"],
        }
        result["novelty"] = {
            m: float(
                np.mean(
                    [
                        sum(f.base[int(u), f.im[i]] == 0 for i in p) / 5
                        for u, p in zip(f.eval_users, ps)
                    ]
                )
            )
            for m, ps in pred.items()
        }
        (OUT / "final-results.json").write_text(json.dumps(result, indent=2))
        (OUT / "final-examples.json").write_text(
            json.dumps(examples(f, pred, scores, config), ensure_ascii=False, indent=2)
        )
        print("FINAL", result["submission"], flush=True)
    out = OUT / "predictions"
    out.mkdir(exist_ok=True)
    (out / (mode + ".json")).write_text(json.dumps(pred))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["audit", "final"])
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.mode)
