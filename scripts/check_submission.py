"""Independent final-file contract check against the original CSVs."""

import hashlib, json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
f = pd.read_csv(root / "submissions/final_hgb.csv", dtype=str)
template = pd.read_csv(root / "data/raw/submission.csv", dtype=str)
allowed = set()
for c in pd.read_csv(
    root / "data/raw/export_arh_31-42-final.csv",
    usecols=["tv_show_id"],
    chunksize=200000,
):
    allowed.update(int(v) for v in c.tv_show_id if v != 0)
assert list(f.columns) == list(template.columns)
assert (
    f.user_id.tolist() == template.user_id.tolist()
    and f.user_id.is_unique
    and len(f) == 1260
)
for text in f.tv_show_id:
    ids = list(map(int, text.split()))
    assert len(ids) == len(set(ids)) == 5 and set(ids) <= allowed and 0 not in ids
report = {
    "rows": len(f),
    "unique_users": f.user_id.nunique(),
    "official_test_candidate_ids": len(allowed),
    "all_five_unique_and_available": True,
    "user_strings_and_order_preserved": True,
    "sha256": hashlib.sha256(
        (root / "submissions/final_hgb.csv").read_bytes()
    ).hexdigest(),
}
(root / "artifacts/submission-check.json").write_text(json.dumps(report, indent=2))
print(report)
