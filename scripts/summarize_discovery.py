"""Keep the v6 error analysis, including rejected hypotheses, reproducible."""

import json, csv, hashlib
from pathlib import Path
import numpy as np
from sweettv.experiment import Fold, OUT
from sweettv.improve import WINDOWS
from sweettv.simple_search import baseline
from sweettv.blend import rrf
from sweettv.metrics import evaluate
from sweettv.discovery_context import DEST

result = {}
for key in ["dev", "audit", "long"]:
    f = Fold(*WINDOWS[key])
    base = rrf(f, baseline(key))
    channel = rrf(f, np.load(OUT / "approaches-v5" / f"{key}_channel.npy"))
    novel = rrf(f, np.load(DEST / f"{key}_discovery_joint_600.npy"))
    old = 0.9 * base + 0.1 * channel
    # Exactly the selection arithmetic, including float32 tie resolution.
    new = 0.9 * (0.5 * base + 0.5 * novel) + 0.1 * channel
    before = f.rank(old)
    after = f.rank(new)
    seen = f.matrix(0)[f.eval_users].toarray() > 0
    full = f.base[f.eval_users].toarray() > 0
    five = np.array([len(t) == 5 for t in f.truth])
    a = np.array(evaluate(f.truth, before)["per_user_ap"])[five]
    b = np.array(evaluate(f.truth, after)["per_user_ap"])[five]
    diff = b - a
    rng = np.random.default_rng(20261101)
    ci = np.quantile(
        rng.choice(diff, (3000, len(diff))).mean(axis=1), [0.025, 0.975]
    ).tolist()
    groups = {
        x: {"targets": 0, "old": 0, "new": 0}
        for x in ["full_seen", "partial_only", "unseen"]
    }
    examples = []
    sig = np.load(DEST / f"{key}_discovery_signals.npz")
    context_targets = 0
    for u in np.flatnonzero(five):
        for item in f.truth[u]:
            j = f.im[item]
            g = (
                "full_seen"
                if full[u, j]
                else "partial_only"
                if seen[u, j]
                else "unseen"
            )
            groups[g]["targets"] += 1
            groups[g]["old"] += int(item in before[u])
            groups[g]["new"] += int(item in after[u])
            if g == "unseen" and sig["zero_112_full"][u, j] > 0:
                context_targets += 1
            if g == "unseen" and (item in before[u]) != (item in after[u]):
                examples.append(
                    {
                        "user_row": int(u),
                        "item": int(item),
                        "old": before[u],
                        "new": after[u],
                        "truth": f.truth[u],
                    }
                )
    result[key] = dict(
        old=float(a.mean()),
        new=float(b.mean()),
        ci95=ci,
        users_improved=int((diff > 1e-12).sum()),
        users_worse=int((diff < -1e-12).sum()),
        groups=groups,
        unseen_targets_with_zero_context=context_targets,
        changed_unseen_examples=examples,
    )
    print(
        key,
        {k: v for k, v in result[key].items() if k != "changed_unseen_examples"},
        flush=True,
    )
(DEST / "selected-error-changes.json").write_text(json.dumps(result, indent=2))

rows = []
for name in [
    "dev_discovery_joint",
    "dev_discovery_novel",
    "audit_discovery_joint",
    "dev_soft_ties",
]:
    for row in json.loads((DEST / f"{name}.json").read_text()):
        rows.append(
            dict(
                experiment=name,
                fold=row["fold"],
                configuration=json.dumps({"trees": row["trees"]}),
                map5=row["map5_five_targets"],
            )
        )
for row in json.loads((DEST / "dev_novel_True.json").read_text())["grid"]:
    rows.append(
        dict(
            experiment="original_features_novel_specialist",
            fold=row["fold"],
            configuration=json.dumps({k: row[k] for k in ["method", "trees", "scale"]}),
            map5=row["map5"],
        )
    )
for name in [
    "context-selection",
    "context-long-confirmation",
    "context-audit-confirmation",
]:
    for row in json.loads((DEST / f"{name}.json").read_text())["grid"]:
        rows.append(
            dict(
                experiment=name,
                fold=row["fold"],
                configuration=json.dumps(row["config"]),
                map5=row["map5"],
            )
        )
for row in json.loads((DEST / "temporal-discovery-selection.json").read_text())["grid"]:
    rows.append(
        dict(
            experiment="temporal_discovery",
            fold="dev",
            configuration=json.dumps({k: v for k, v in row.items() if k != "map5"}),
            map5=row["map5"],
        )
    )
with (DEST / "experiments.csv").open("w") as h:
    writer = csv.DictWriter(
        h, fieldnames=["experiment", "fold", "configuration", "map5"]
    )
    writer.writeheader()
    writer.writerows(rows)

sources = [
    *Path("src/sweettv").glob("*discovery*.py"),
    Path("src/sweettv/novel_specialist.py"),
    Path("src/sweettv/tie_training.py"),
    *Path("scripts").glob("*discovery*.py"),
    Path("tests/test_discovery_context.py"),
]
(DEST / "source-hashes.json").write_text(
    json.dumps(
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2
    )
)
print("Ledger:", len(rows), "records", flush=True)
