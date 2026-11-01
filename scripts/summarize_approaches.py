"""Summarize the frozen v5 study and paired local error changes."""

import csv
import json
import hashlib
from pathlib import Path

import numpy as np

from sweettv.blend import rrf
from sweettv.experiment import Fold
from sweettv.improve import WINDOWS
from sweettv.metrics import evaluate
from sweettv.simple_search import DEST, baseline


def read(name):
    return json.loads((DEST / name).read_text())


rows = []
simple = read("simple-selection.json")
for entry in simple["grid"]:
    config = entry["config"]
    rows.append(dict(family=config["family"], config=json.dumps(config, sort_keys=True),
                     weight=1, fold="dev", map5=entry["standalone"]))
    for b in entry["blend"]:
        rows.append(dict(family=config["family"], config=json.dumps(config, sort_keys=True),
                         weight=b["weight"], fold="dev", map5=b["dev"]))
for name in ["replay", "public"]:
    for entry in read(f"{name}-selection.json")["grid"]:
        config = {k: v for k, v in entry.items() if k not in ["weight", "dev", "standalone"]}
        rows.append(dict(family=name, config=json.dumps(config, sort_keys=True),
                         weight=entry["weight"], fold="dev", map5=entry["dev"]))
for entry in read("dev_replay_ranker.json"):
    rows.append(dict(family="replay_ranker", config=json.dumps({"trees": entry["trees"]}),
                     weight=1, fold=entry["eval"], map5=entry["map5_five_targets"]))
with (DEST / "experiments.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=["family", "config", "weight", "fold", "map5"])
    writer.writeheader()
    writer.writerows(rows)

selected = read("submission-selection.json")
diagnostics = {}
for key in ["dev", "audit", "long"]:
    f = Fold(*WINDOWS[key])
    base = baseline(key)
    pred = f.rank(base)
    mask = np.array([len(t) == 5 for t in f.truth])
    base_ap = np.array(evaluate(f.truth, pred)["per_user_ap"])[mask]
    seen = f.matrix(0.8, None)[f.eval_users].toarray() > 0
    partial = f.matrix(0, None)[f.eval_users].toarray() > 0
    b = rrf(f, base)
    diagnostics[key] = {"baseline_map5": float(base_ap.mean()), "users": int(mask.sum())}
    for name in ["channel", "replay"]:
        if name == "channel":
            signal = np.load(DEST / f"{key}_channel.npy")
        else:
            r = selected[name]
            cache = np.load(DEST / f"{key}_replay_{r['days']}_{r['weekly']}.npz")
            signal = cache["scores"][r["mode"]]
        weight = selected[name]["weight"]
        rec = f.rank((1 - weight) * b + weight * rrf(f, signal))
        ap = np.array(evaluate(f.truth, rec)["per_user_ap"])[mask]
        delta = ap - base_ap
        rng = np.random.default_rng(20261101)
        bootstrap = np.mean(rng.choice(delta, size=(3000, len(delta)), replace=True), axis=1)
        classes = {k: {"targets": 0, "old_hits": 0, "new_hits": 0}
                   for k in ["full_seen", "partial_only", "no_overlap"]}
        for u in np.flatnonzero(mask):
            for item in f.truth[u]:
                j = f.im[item]
                cls = "full_seen" if seen[u, j] else "partial_only" if partial[u, j] else "no_overlap"
                classes[cls]["targets"] += 1
                classes[cls]["old_hits"] += int(item in pred[u])
                classes[cls]["new_hits"] += int(item in rec[u])
        diagnostics[key][name] = {
            "map5": float(ap.mean()), "delta": float(delta.mean()),
            "paired_bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "users_improved": int((delta > 1e-12).sum()),
            "users_worse": int((delta < -1e-12).sum()),
            "users_unchanged": int((abs(delta) <= 1e-12).sum()),
            "target_classes": classes,
        }
    print(key, diagnostics[key], flush=True)
(DEST / "selected-error-changes.json").write_text(json.dumps(diagnostics, indent=2))

paths = [*Path("src/sweettv").glob("*replay*.py"), Path("src/sweettv/simple_search.py"),
         Path("src/sweettv/public_formula.py"), Path("scripts/finalize_approaches.py"),
         Path("scripts/check_approaches.py"), Path(__file__), Path("tests/test_session_replay.py")]
(DEST / "source-hashes.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in paths}, indent=2))
