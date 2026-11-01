"""Refit only after frozen local checks pass; export one Kaggle candidate."""

import hashlib
import json
import numpy as np
import lightgbm as lgb

from sweettv.xendcg import run
from sweettv.neural_rank import DEST
from sweettv.improve import DEST as OLD, WINDOWS, cached
from sweettv.experiment import Fold, ROOT, SEED
from sweettv.horizon import rescale
from sweettv.blend import rrf


def main():
    decision = json.loads((DEST / "xendcg-confirmation.json").read_text())
    if not decision["promote"]:
        print("NOT_PROMOTED: frozen local checks did not pass", flush=True)
        return
    folder = DEST / f"final_xendcg_s{SEED}"
    if not (folder / "result.json").exists():
        run("final", SEED, 1200)
    f = Fold(*WINDOWS["final"])
    d = cached("final")
    x = d["X"]
    if decision["final_horizon_transform"] == "normalized":
        x = rescale(x, d["names"].tolist(), (f.end - f.start) / (7 * 86400))
    model = lgb.Booster(model_file=str(folder / "model.txt"))
    raw = model.predict(x, num_iteration=1200)
    score = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
    score[d["u"], d["i"]] = raw
    base = np.load(OLD / "v2_cat200_horizon.npy")
    weight = decision["frozen_xendcg_weight"]
    combined = weight * rrf(f, score) + (1 - weight) * rrf(f, base)
    pred = f.rank(combined)
    frame = f.template.copy()
    frame["tv_show_id"] = [" ".join(map(str, p)) for p in pred]
    allowed = set(f.future.tv_show_id)
    assert len(frame) == 1260 and frame.user_id.is_unique
    assert frame.user_id.tolist() == f.template.user_id.tolist()
    assert all(
        len(p) == len(set(p)) == 5 and set(p) <= allowed and 0 not in p for p in pred
    )
    path = ROOT / "submissions" / "trained_xendcg1200.csv"
    frame.to_csv(path, index=False)
    np.save(DEST / "final_xendcg_frozen_blend.npy", combined)
    old = f.rank(base)
    manifest = dict(
        file=str(path.relative_to(ROOT)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model="XE_NDCG_MART1200 + incumbent CatBoost200",
        xendcg_weight=weight,
        horizon_transform=decision["final_horizon_transform"],
        rows=len(frame),
        changed_top5_users=sum(a != b for a, b in zip(pred, old)),
        mean_top5_overlap=float(
            np.mean([len(set(a) & set(b)) for a, b in zip(pred, old)])
        ),
        local_confirmation="artifacts/deep-training-v3/xendcg-confirmation.json",
        kaggle_status="not submitted",
    )
    (DEST / "final-candidate.json").write_text(json.dumps(manifest, indent=2))
    print("FINAL_CANDIDATE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
