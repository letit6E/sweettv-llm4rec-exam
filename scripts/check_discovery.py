"""Recompute new historical signals and compare shared features with the old cache."""

import json, hashlib
import numpy as np
import pandas as pd
from sweettv.experiment import Fold, ROOT
from sweettv.improve import WINDOWS, DEST as OLD
from sweettv.discovery_ranker import signals
from sweettv.discovery_context import DEST
from sweettv.horizon import rescale

f = Fold(*WINDOWS["final"])
expected = np.load(DEST / "final_discovery_signals.npz")
actual = signals(f)
assert set(actual) == set(expected.files)
for name, value in actual.items():
    np.testing.assert_array_equal(value, expected[name])
new = np.load(DEST / "final_discovery_features.npz")
old = np.load(OLD / "final-features.npz")
assert np.array_equal(new["items"], old["items"])
key = new["u"].astype("int64") * len(f.items) + new["i"]
lookup = np.searchsorted(key, old["u"].astype("int64") * len(f.items) + old["i"])
x = rescale(old["X"], old["names"].tolist(), (f.end - f.start) / 604800)
np.testing.assert_allclose(new["X"][lookup, :121], x, rtol=1e-6, atol=1e-6)
info = json.loads((DEST / "submission-verification.json").read_text())
file = ROOT / info["file"]
assert hashlib.sha256(file.read_bytes()).hexdigest() == info["sha256"]
frame = pd.read_csv(file, dtype=str)
assert frame.user_id.tolist() == f.template.user_id.tolist()
assert frame.tv_show_id.tolist() == [
    " ".join(map(str, r)) for r in f.rank(np.load(DEST / "final_discovery_blend.npy"))
]
(DEST / "independent-check.json").write_text(
    json.dumps(
        {
            "new_signals_recomputed": list(actual),
            "shared_feature_pairs_checked": len(x),
            "prediction_rows_checked": len(frame),
            "sha256": info["sha256"],
            "passed": True,
        },
        indent=2,
    )
)
print(
    "PASS",
    len(x),
    "shared feature pairs;",
    len(frame),
    "prediction rows;",
    len(actual),
    "recomputed signals",
    flush=True,
)
