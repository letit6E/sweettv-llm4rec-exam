"""Collect actual experiment records without treating training fits as holdouts."""

import csv, json, hashlib, platform
from pathlib import Path
from importlib.metadata import version

root = Path(__file__).resolve().parents[1]
dest = root / "artifacts/improvement-v2"
records = []
for name in [
    "dev-results.json",
    "audit-results.json",
    "dev-xgb-results.json",
    "audit-xgb-results.json",
    "dev-sequence-results.json",
    "audit-sequence-results.json",
    "allusers-results.json",
    "count-results.json",
    "horizon-results.json",
    "allusers-horizon-results.json",
]:
    p = dest / name
    if not p.exists():
        continue
    for r in json.loads(p.read_text()):
        if r.get("map5_five_targets") is not None:
            records.append({"source": name, **r})
sub = json.loads((dest / "kaggle-submissions.json").read_text())
best = max(sub, key=lambda x: x["private"])
summary = {
    "best_observed_submission": best,
    "improvement_absolute": best["private"] - sub[0]["private"],
    "improvement_relative": best["private"] / sub[0]["private"] - 1,
    "kaggle_submissions_total": len(sub),
    "local_model_checkpoint_evaluations": len(records),
    "additional_ablation_evaluations": len(
        json.loads((dest / "ablation.json").read_text())
    ),
    "additional_fusion_evaluations": len(
        json.loads((dest / "blend-validation.json").read_text())
    ),
    "selection_caveat": "Both former validation/audit windows and returned Kaggle feedback were used for adaptive development. No unseen holdout claim. The overlapping 10-week diagnostic is not a new independent dataset.",
    "runtime": {
        "python": platform.python_version(),
        **{
            n: version(n)
            for n in [
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "lightgbm",
                "catboost",
                "xgboost",
                "torch",
            ]
        },
    },
    "models": records,
    "code_sha256": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "src/sweettv").glob("*.py"))
    },
}
(dest / "experiment-summary.json").write_text(json.dumps(summary, indent=2))
with (dest / "experiment-table.csv").open("w") as h:
    w = csv.DictWriter(
        h,
        fieldnames=[
            "source",
            "name",
            "fold",
            "map5_five_targets",
            "map5_all",
            "seconds",
        ],
    )
    w.writeheader()
    for r in records:
        w.writerow({k: r.get(k, "") for k in w.fieldnames})
print(
    {k: v for k, v in summary.items() if k not in ["models", "code_sha256", "runtime"]}
)
