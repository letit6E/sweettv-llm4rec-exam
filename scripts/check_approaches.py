import json, numpy as np, hashlib
from sweettv.simple_search import DEST, score, baseline
from sweettv.session_replay import compute
from sweettv.experiment import Fold, ROOT
from sweettv.improve import WINDOWS
from sweettv.blend import rrf
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1):
    f = Fold(*WINDOWS["final"])
    b = rrf(f, baseline("final"))
    s = json.loads((DEST / "submission-selection.json").read_text())
    checks = []
    for name in ["channel", "replay"]:
        r = s[name]
        a = (
            score(f, r["config"])
            if name == "channel"
            else compute(f, r["days"], r["weekly"])[r["mode"]]
        )
        z = (1 - r["weight"]) * b + r["weight"] * rrf(f, a)
        stored = np.load(DEST / f"final_{name}_blend.npy")
        assert np.array_equal(z, stored)
        frame = f.template.copy()
        frame["tv_show_id"] = [" ".join(map(str, p)) for p in f.rank(z)]
        path = ROOT / f"submissions/v5_{name}_blend.csv"
        assert frame.to_csv(index=False).encode() == path.read_bytes()
        checks.append(
            {
                "name": name,
                "recomputed_from_raw_history": True,
                "byte_identical_csv": True,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (DEST / "independent-check.json").write_text(json.dumps(checks, indent=2))
    print(checks)
