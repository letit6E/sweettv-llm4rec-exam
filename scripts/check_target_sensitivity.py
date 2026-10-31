import json, numpy as np
from sweettv.experiment import Fold, OUT
from sweettv.metrics import evaluate

f = Fold("2020-06-22", "2020-07-27")
p = json.loads((OUT / "predictions/audit.json").read_text())
out = {}
for version in ["reverse_id", "random_11", "random_22", "random_33", "single_session"]:
    e = (
        f.truth_events
        if version != "single_session"
        else f.truth_events[f.truth_events.single_fraction >= 0.8]
    )
    c = e.groupby(["user", "tv_show_id"]).size().reset_index(name="count")
    if version.startswith("random"):
        c["tie"] = np.random.default_rng(int(version.split("_")[1])).random(len(c))
    else:
        c["tie"] = -c.tv_show_id if version == "reverse_id" else c.tv_show_id
    c = c.sort_values(["user", "count", "tie"], ascending=[True, False, True])
    lookup = c.groupby("user").head(5).groupby("user").tv_show_id.apply(list).to_dict()
    truth = [lookup.get(int(u), []) for u in f.eval_users]
    out[version] = {
        m: {k: v for k, v in evaluate(truth, ps).items() if k != "per_user_ap"}
        for m, ps in p.items()
    }
(OUT / "target-sensitivity.json").write_text(json.dumps(out, indent=2))
print(
    {
        v: {m: round(r["map5_five_targets"], 4) for m, r in ms.items()}
        for v, ms in out.items()
    }
)
