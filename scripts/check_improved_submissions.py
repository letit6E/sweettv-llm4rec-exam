"""Verify every improvement CSV against the original, unprocessed input files."""

import hashlib, json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
template = pd.read_csv(root / "data/raw/submission.csv", dtype=str)
allowed = set()
for chunk in pd.read_csv(
    root / "data/raw/export_arh_31-42-final.csv",
    usecols=["tv_show_id"],
    chunksize=200000,
):
    allowed.update(int(v) for v in chunk.tv_show_id if v != 0)
results = []
for path in sorted((root / "submissions").glob("v*.csv")):
    frame = pd.read_csv(path, dtype=str)
    assert list(frame.columns) == list(template.columns), path
    assert (
        frame.user_id.tolist() == template.user_id.tolist() and frame.user_id.is_unique
    ), path
    for text in frame.tv_show_id:
        ids = list(map(int, text.split()))
        assert (
            len(ids) == len(set(ids)) == 5 and set(ids) <= allowed and 0 not in ids
        ), path
    results.append(
        {
            "file": path.name,
            "rows": len(frame),
            "all_checks_passed": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
(root / "artifacts/improvement-v2/submission-checks.json").write_text(
    json.dumps(results, indent=2)
)
print(json.dumps(results, indent=2))
