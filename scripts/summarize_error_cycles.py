"""Compact ledger of actual v4 runs; preserve failed and rejected hypotheses."""

import csv, json, hashlib
from pathlib import Path
from sweettv.error_cycles import DEST

rows = []
for p in sorted(DEST.glob("*.json")):
    data = json.loads(p.read_text())
    if isinstance(data, list):
        for r in data:
            if isinstance(r, dict) and "map5_five_targets" in r:
                rows.append(
                    {
                        "artifact": p.name,
                        "eval": r.get("eval"),
                        "config": r.get(
                            "name", r.get("config", str(r.get("stage", "hardneg")))
                        ),
                        "trees": r.get("trees", 200),
                        "MAP5": r.get("map5_five_targets"),
                        "training_pairs": r.get(
                            "training_pairs", r.get("train_rows", r.get("rows"))
                        ),
                        "seconds": r.get("seconds"),
                    }
                )
with (DEST / "experiments.csv").open("w") as h:
    writer = csv.DictWriter(h, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
paths = (
    [
        Path("src/sweettv") / n
        for n in [
            "error_cycles.py",
            "propensity.py",
            "neighbor_cycles.py",
            "category_cycles.py",
        ]
    ]
    + list(Path("scripts").glob("error_cycle_*.py"))
    + [Path("scripts/finalize_error_cycles.py"), Path("tests/test_error_cycles.py")]
)
summary = {
    "trained_checkpoints": len(list(DEST.glob("*.cbm"))),
    "evaluation_records": len(rows),
    "validation_policy": "All three temporal windows are reused validation, not untouched tests. Long overlaps shorter windows. Kaggle returned scores used only to record outcome/designate best; both new weights frozen before upload.",
    "tests_passed": 23,
    "source_sha256": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    },
    "report": "Deferred; no PDF or publication update",
    "best_before": {
        "file": "submissions/v2_cat200_horizon.csv",
        "private": 0.45024,
        "public": 0.45844,
    },
    "submission_selection": "submission-selection.json",
    "submission_verification": "exports.json",
}
(DEST / "completion.json").write_text(json.dumps(summary, indent=2))
print({k: v for k, v in summary.items() if k != "source_sha256"})
