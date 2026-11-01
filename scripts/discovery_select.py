"""Choose a discovery correction on early validation, then transfer it unchanged."""

import json
import numpy as np
from sweettv.experiment import Fold, OUT
from sweettv.improve import WINDOWS
from sweettv.simple_search import baseline
from sweettv.blend import rrf
from sweettv.metrics import evaluate
from sweettv.discovery_context import DEST


def apply(f, base, model, mode, scale):
    if mode == "all":
        return model
    unseen = f.matrix(0)[f.eval_users].toarray() == 0
    score = base.copy()
    score[unseen] = np.maximum(model[unseen], 0) * scale
    return score


def run(key, chosen=None):
    f = Fold(*WINDOWS[key])
    base = baseline(key)
    channel = np.load(OUT / "approaches-v5" / f"{key}_channel.npy")
    channel = rrf(f, channel)
    br = rrf(f, base)
    incumbent = 0.9 * br + 0.1 * channel
    results = []
    best = None
    for family in ["joint", "novel"]:
        if chosen is not None and family != chosen["family"]:
            continue
        for trees in [200, 400, 600]:
            if chosen is not None and trees != chosen["trees"]:
                continue
            model = np.load(DEST / f"{key}_discovery_{family}_{trees}.npy")
            for mode in ["all", "unseen"] if family == "joint" else ["unseen"]:
                for scale in [1] if mode == "all" else [0.5, 1, 2, 4]:
                    q = apply(f, base, model, mode, scale)
                    rr = rrf(f, q)
                    for weight in [0.25, 0.5, 1]:
                        cfg = dict(
                            family=family,
                            trees=trees,
                            mode=mode,
                            scale=scale,
                            weight=weight,
                        )
                        if chosen is not None and cfg != chosen:
                            continue
                        z = 0.9 * ((1 - weight) * br + weight * rr) + 0.1 * channel
                        m = evaluate(f.truth, f.rank(z))["map5_five_targets"]
                        row = dict(config=cfg, fold=key, map5=m)
                        results.append(row)
                        if best is None or m > best["map5"]:
                            best = row
    print(key, best, flush=True)
    return dict(
        baseline=evaluate(f.truth, f.rank(incumbent))["map5_five_targets"],
        selected=best,
        grid=results,
    )


if __name__ == "__main__":
    dev = run("dev")
    (DEST / "context-selection.json").write_text(json.dumps(dev, indent=2))
    other = run("long", dev["selected"]["config"])
    (DEST / "context-long-confirmation.json").write_text(json.dumps(other, indent=2))
