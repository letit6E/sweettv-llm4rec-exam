"""Freeze locally selected ensembles, export and independently validate CSVs."""

import argparse, json, datetime, pickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sweettv.error_cycles import DEST, matrix
from sweettv.improve import DEST as OLD, WINDOWS
from sweettv.experiment import Fold, ROOT
from sweettv.metrics import evaluate
from sweettv.horizon import rescale
from sweettv.improve_final import export

p = argparse.ArgumentParser()
p.add_argument("action", choices=["select", "export"])
a = p.parse_args()
if a.action == "select":
    rows = []
    for w in [0.75, 0.25]:
        row = {
            "weight": w,
            "selection": "max earlier validation"
            if w == 0.75
            else "conservative alternative with gains on all three windows",
        }
        for key in ["dev", "audit", "long"]:
            f = Fold(*WINDOWS[key])
            s = np.load(DEST / f"{key}_identity_item_400.npy")
            b = np.load(
                OLD
                / (f"{key}_cat6_200.npy" if key != "long" else "long_horizon_True.npy")
            )
            m = evaluate(f.truth, f.rank((1 - w) * b + w * s))
            bm = evaluate(f.truth, f.rank(b))
            mask = np.array([len(t) == 5 for t in f.truth])
            delta = (np.array(m["per_user_ap"]) - bm["per_user_ap"])[mask]
            rng = np.random.default_rng(20261101)
            ci = np.quantile(
                [
                    rng.choice(delta, len(delta), replace=True).mean()
                    for _ in range(2000)
                ],
                [0.025, 0.975],
            )
            row[key] = {
                "baseline": bm["map5_five_targets"],
                "candidate": m["map5_five_targets"],
                "delta": float(delta.mean()),
                "paired_bootstrap95": ci.tolist(),
            }
        rows.append(row)
    (DEST / "submission-selection.json").write_text(
        json.dumps(
            {
                "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "candidates": rows,
                "caveat": "All local windows have been reused for selection; long overlaps two shorter windows. Bootstrap does not include selection uncertainty. Both weights frozen before new Kaggle scores.",
            },
            indent=2,
        )
    )
    print(json.dumps(rows), flush=True)
else:
    selection = json.loads((DEST / "submission-selection.json").read_text())
    f = Fold(*WINDOWS["final"])
    d, x, n = matrix("final", 0)
    frame = pd.DataFrame(x, columns=n)
    frame["program_id"] = f.items[d["i"]].astype(str)
    model = CatBoostClassifier()
    model.load_model(str(DEST / "final_identity_item_400.cbm"))
    v = model.predict_proba(frame)[:, 1]
    z = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
    z[d["u"], d["i"]] = v
    np.testing.assert_allclose(
        z, np.load(DEST / "final_identity_item_400.npy"), rtol=1e-7, atol=1e-7
    )
    base = pickle.load(open(OLD / "v2_cat200.pkl", "rb"))
    b = np.full_like(z, -1e9)
    b[d["u"], d["i"]] = base.predict_proba(
        rescale(d["X"], d["names"].tolist(), 12), ntree_end=200
    )[:, 1]
    np.testing.assert_allclose(
        b, np.load(OLD / "final_horizon_True.npy"), rtol=1e-7, atol=1e-7
    )
    allowed = set(
        pd.read_csv(
            ROOT / "data/raw/export_arh_31-42-final.csv", usecols=["tv_show_id"]
        ).tv_show_id
    ) - {0}
    rows = []
    for r in selection["candidates"]:
        w = r["weight"]
        name = f"v4_identity_blend{round(w * 100)}"
        score = (1 - w) * b + w * z
        record = export(f, score, name)
        csv = pd.read_csv(ROOT / record["file"], dtype=str)
        assert csv.user_id.tolist() == f.template.user_id.tolist()
        for s in csv.tv_show_id:
            ids = list(map(int, s.split()))
            assert len(set(ids)) == len(ids) == 5 and set(ids) <= allowed
        record.update(
            weight=w,
            model_reload_verified=True,
            raw_schedule_eligibility_verified=True,
            local_results=r,
        )
        rows.append(record)
    (DEST / "exports.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows), flush=True)
