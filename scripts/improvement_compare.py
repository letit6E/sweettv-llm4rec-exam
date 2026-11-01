import json, pickle
import numpy as np
from sweettv.experiment import Fold, SEED
from sweettv.improve import DEST, WINDOWS
from sweettv.metrics import evaluate

f = Fold(*WINDOWS["audit"])
old = np.load(DEST / "audit-scores.npz")["hgb"]
new = np.load(DEST / "audit_cat6_200.npy")
five = np.array([len(t) == 5 for t in f.truth])
rng = np.random.default_rng(SEED)
p = {n: f.rank(s) for n, s in [("original", old), ("cat200", new)]}
aps = {n: np.array(evaluate(f.truth, v)["per_user_ap"]) for n, v in p.items()}
delta = (aps["cat200"] - aps["original"])[five]
boot = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(axis=1)
hist = np.asarray(f.base[f.eval_users].sum(axis=1)).ravel()
groups = {}
for label, mask in [
    ("light", hist < 30),
    ("medium", (hist >= 30) & (hist < 100)),
    ("heavy", hist >= 100),
]:
    mask = mask & five
    groups[label] = {
        "users": int(mask.sum()),
        **{n: float(a[mask].mean()) for n, a in aps.items()},
    }
for label in ["seen", "new_to_user", "new_to_catalog"]:
    total = 0
    hits = {n: 0 for n in p}
    for row in np.flatnonzero(five):
        for item in f.truth[row]:
            j = f.im[item]
            g = (
                "new_to_catalog"
                if f.pop[j] == 0
                else ("seen" if f.base[f.eval_users[row], j] > 0 else "new_to_user")
            )
            if g == label:
                total += 1
                for n, v in p.items():
                    hits[n] += int(item in v[row])
    groups[label] = {"targets": total, "hits": hits}
with (DEST / "audit_cat6.pkl").open("rb") as h:
    model = pickle.load(h)
# These are fitted-model importances (600 trees), not causal feature effects.
names = json.load(open(DEST / "audit-analysis.json"))["features"]
importance = sorted(
    zip(names, model.feature_importances_.tolist()), key=lambda x: -x[1]
)
r = {
    "note": "Adaptive development diagnostic on already-seen validation; bootstrap reflects users, not model-selection uncertainty. Feature importance from 600-tree parent, not selected 200-tree checkpoint.",
    "map5_difference": float(delta.mean()),
    "bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
    "groups": groups,
    "parent_model_importance": importance,
}
(DEST / "error-improvement.json").write_text(json.dumps(r, indent=2))
print(json.dumps({k: v for k, v in r.items() if k != "parent_model_importance"}))
print(importance[:12])
