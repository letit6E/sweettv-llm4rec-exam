import json
import numpy as np
from sweettv.improve import DEST, WINDOWS
from sweettv.experiment import Fold
from sweettv.blend import rrf
from sweettv.improve_final import export

f = Fold(*WINDOWS["final"])
cat = np.load(DEST / "v2_cat200.npy")
seq = np.load(DEST / "final_esasrec_20_mean10.npy")
r = export(f, 0.9 * rrf(f, cat) + 0.1 * rrf(f, seq), "v2_cat_esasrec")
r.update(
    {
        "weights": {"catboost": 0.9, "esasrec_mean10": 0.1},
        "rrf_offset": 5,
        "choice": "Tiny positive MAP differences on both already-open local windows; test as one predefined hybrid, not a claim of significance.",
    }
)
(DEST / "neural-blend-export.json").write_text(json.dumps(r, indent=2))
print(r)
