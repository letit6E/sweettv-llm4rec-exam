import json
from collections import Counter
import numpy as np
from sweettv.experiment import Fold
from sweettv.improve import DEST, WINDOWS

f = Fold(*WINDOWS["audit"])
z = np.load(DEST / "audit-scores.npz")["hgb"]
p = f.rank(z)
meta = (
    f.past.sort_values("end")
    .drop_duplicates("tv_show_id", keep="last")
    .set_index("tv_show_id")
)
miss = Counter()
false = Counter()
for row, (truth, pred) in enumerate(zip(f.truth, p)):
    if len(truth) != 5:
        continue
    miss.update(set(truth) - set(pred))
    false.update(set(pred) - set(truth))


def details(c):
    return [
        {
            "id": int(i),
            "title": str(meta.loc[i, "tv_show_title"])
            if i in meta.index
            else "unknown historical metadata",
            "category": str(meta.loc[i, "tv_show_category"]) if i in meta.index else "",
            "users": n,
            "train_popularity": int(f.pop[f.im[i]]),
        }
        for i, n in c.most_common(15)
    ]


r = {
    "scope": "Previously opened audit, now diagnostic validation. Missing means outside reconstructed frequency-based Top5, not necessarily disliked.",
    "most_missed": details(miss),
    "most_false_positive": details(false),
}
(DEST / "error-programs.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
print(json.dumps(r, ensure_ascii=False))
