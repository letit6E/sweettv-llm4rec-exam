"""Independent process reloads both trained models and reproduces the CSV."""

import hashlib
import io
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

from sweettv.neural_rank import DEST
from sweettv.improve import DEST as OLD, WINDOWS, cached
from sweettv.experiment import Fold, ROOT, SEED
from sweettv.horizon import rescale
from sweettv.blend import rrf


def main():
    manifest = json.loads((DEST / "final-candidate.json").read_text())
    path = ROOT / manifest["file"]
    frame = pd.read_csv(path, dtype=str)
    template = pd.read_csv(ROOT / "data/raw/submission.csv", dtype=str)
    assert frame.columns.tolist() == template.columns.tolist()
    assert frame.user_id.tolist() == template.user_id.tolist()
    raw_schedule = next((ROOT / "data/raw").glob("export_arh_31-42*.csv"))
    allowed = set(pd.read_csv(raw_schedule, usecols=["tv_show_id"]).tv_show_id) - {0}
    predictions = [list(map(int, s.split())) for s in frame.tv_show_id]
    assert len(predictions) == 1260
    assert all(len(p) == len(set(p)) == 5 and set(p) <= allowed for p in predictions)
    f = Fold(*WINDOWS["final"])
    d = cached("final")
    x = d["X"]
    if manifest["horizon_transform"] == "normalized":
        x = rescale(x, d["names"].tolist(), 12)
    model = lgb.Booster(model_file=str(DEST / f"final_xendcg_s{SEED}" / "model.txt"))
    new = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
    new[d["u"], d["i"]] = model.predict(x, num_iteration=1200)
    with (OLD / "v2_cat200.pkl").open("rb") as handle:
        incumbent = pickle.load(handle)
    old = np.full_like(new, -1e9)
    old[d["u"], d["i"]] = incumbent.predict_proba(
        rescale(d["X"], d["names"].tolist(), 12)
    )[:, 1]
    w = manifest["xendcg_weight"]
    reproduced = f.rank(w * rrf(f, new) + (1 - w) * rrf(f, old))
    assert reproduced == predictions
    result = template.copy()
    result["tv_show_id"] = [" ".join(map(str, p)) for p in reproduced]
    buf = io.StringIO()
    result.to_csv(buf, index=False)
    digest = hashlib.sha256(buf.getvalue().encode()).hexdigest()
    assert digest == manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    checks = dict(
        file=str(path),
        sha256=digest,
        rows=1260,
        allowed_programs=len(allowed),
        exact_template_user_order=True,
        five_unique_allowed_ids=True,
        reloaded_both_models=True,
        reproduced_all_top5=True,
        byte_identical_csv=True,
        note="Feature cache reused; model scores recomputed independently. Original v2 independently rebuilt these unchanged features from processed inputs.",
    )
    (DEST / "submission-verification.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
