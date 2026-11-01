"""Read measured training runs; compare to fixed old baseline and plot curves."""

import json
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sweettv.neural_rank import DEST
from sweettv.improve import DEST as OLD, WINDOWS
from sweettv.experiment import Fold
from sweettv.metrics import evaluate
from sweettv.blend import rrf


def run():
    runs = [json.loads(p.read_text()) for p in sorted(DEST.glob("*/result.json"))]
    for result in runs:
        if "fixed_validation" in result:
            result["fixed_validation"] = {
                k: v
                for k, v in result["fixed_validation"].items()
                if k != "per_user_ap"
            }
    with (DEST / "training-runs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run",
                "neural_parameter_count",
                "epochs_completed",
                "selected_epoch_or_trees",
                "window_best_MAP5_diagnostic",
                "fixed_budget_MAP5",
                "seconds",
            ]
        )
        for r in runs:
            writer.writerow(
                [
                    r["name"],
                    r.get("parameters", "")
                    if isinstance(r.get("parameters"), int)
                    else "",
                    r.get("completed_epochs", ""),
                    r.get("best_epoch", r.get("best_iteration", "")),
                    r.get("best_validation_map5", ""),
                    r.get(
                        "fixed_validation_map5",
                        r.get("fixed_validation", {}).get("map5_five_targets", ""),
                    ),
                    r.get("elapsed_seconds", r.get("seconds", "")),
                ]
            )
    rows = []
    for key in ["dev", "audit"]:
        paths = sorted(DEST.glob(f"{key}_*/best-scores.npy"))
        if not paths:
            continue
        f = Fold(*WINDOWS[key])
        base = np.load(OLD / f"{key}_cat6_200.npy")
        baseline = evaluate(f.truth, f.rank(base))
        rb = rrf(f, base)
        rows.append(
            dict(
                fold=key,
                model="previous_cat200",
                weight=0,
                map5=baseline["map5_five_targets"],
            )
        )
        for path in paths:
            fixed = path.parent / "fixed-scores.npy"
            if fixed.exists():
                path = fixed
            scores = np.load(path)
            metric = evaluate(f.truth, f.rank(scores))
            row = dict(
                fold=key,
                model=path.parent.name,
                weight=1,
                map5=metric["map5_five_targets"],
            )
            rows.append(row)
            rs = rrf(f, scores)
            for weight in [0.1, 0.25, 0.5]:
                m = evaluate(f.truth, f.rank((1 - weight) * rb + weight * rs))
                rows.append(
                    dict(
                        fold=key,
                        model=path.parent.name,
                        weight=weight,
                        map5=m["map5_five_targets"],
                    )
                )
    (DEST / "summary.json").write_text(
        json.dumps(
            dict(
                runs=runs,
                comparisons=rows,
                caveat="Repeatedly inspected validation; blend grid is adaptive and not a test estimate.",
            ),
            indent=2,
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for family, col in [("rank", 0), ("sequence", 1)]:
        for path in sorted(DEST.glob("dev_*/history.json")):
            name = path.parent.name
            if "yeti" in name or ("seq" in name) != (family == "sequence"):
                continue
            h = json.loads(path.read_text())
            if not h or "epoch" not in h[0]:
                continue
            epochs = [r["epoch"] for r in h]
            axes[0, col].plot(
                epochs,
                [r["train_loss"] / h[0]["train_loss"] for r in h],
                label=name.replace("dev_", ""),
            )
            val = [r for r in h if "map5_five_targets" in r]
            axes[1, col].plot(
                [r["epoch"] for r in val], [r["map5_five_targets"] for r in val]
            )
        axes[0, col].set(
            title=family.title() + " training", ylabel="Loss / first-epoch loss"
        )
        axes[0, col].legend(fontsize=7)
        axes[1, col].axhline(
            0.4383569892473118, color="black", ls="--", label="Previous CatBoost"
        )
        axes[1, col].set(xlabel="Epoch", ylabel="Validation MAP@5")
        axes[1, col].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.savefig(DEST / "learning-curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for col, model in enumerate(["xendcg", "yeti"]):
        for path in sorted(DEST.glob(f"dev_*{model}*/history.json")):
            h = json.loads(path.read_text())
            axes[col].plot(
                [r["iteration"] for r in h],
                [r["map5_five_targets"] for r in h],
                marker="o",
                label=path.parent.name.replace("dev_", ""),
            )
        axes[col].axhline(
            0.43835698924731187,
            color="black",
            linestyle="--",
            label="Incumbent CatBoost200",
        )
        axes[col].set(title=model, xlabel="Trees", ylabel="Full-ground-truth MAP@5")
        axes[col].legend(fontsize=7)
        axes[col].grid(alpha=0.2)
    fig.savefig(DEST / "boosting-curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps(sorted(rows, key=lambda r: (r["fold"], -r["map5"])), indent=2))


if __name__ == "__main__":
    run()
