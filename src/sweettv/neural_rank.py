"""Temporal, query-wise neural ranking with complete candidate sets.

Each training query is a user at a historical cutoff. Targets are that user's
five most frequent full-view programs in the following five weeks, not the
next event. All training windows must END before the prediction cutoff.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .allusers import data
from .experiment import Fold, OUT, SEED
from .improve import WINDOWS, cached
from .metrics import evaluate

DEST = OUT / "deep-training-v3"
DEST.mkdir(exist_ok=True)
CONFIGS = {
    "list128": dict(width=128, depth=2, dropout=0.1, objective="listwise"),
    "list256": dict(width=256, depth=3, dropout=0.1, objective="listwise"),
    "drop128": dict(width=128, depth=2, dropout=0.3, objective="listwise"),
    "bce128": dict(width=128, depth=2, dropout=0.1, objective="balanced_bce"),
    "map128": dict(width=128, depth=2, dropout=0.1, objective="lambda_map"),
}


def training_keys(fold):
    keys = ["early1", "early2", "early3"]
    if fold in ["audit", "final"]:
        keys.append("dev")
    if fold == "final":
        keys.append("audit")
    assert all(WINDOWS[k][1] <= WINDOWS[fold][0] for k in keys)
    return keys


def prepare(fold):
    prefix = DEST / (fold + "-train")
    if not Path(str(prefix) + "-meta.json").exists():
        x, y, groups = data(training_keys(fold))
        rng = np.random.default_rng(SEED)
        sample = x[rng.choice(len(x), min(200000, len(x)), replace=False)]
        mean = sample.mean(axis=0, dtype=np.float64).astype("float32")
        std = np.maximum(sample.std(axis=0, dtype=np.float64), 1e-5).astype("float32")
        # Transform uses training-only moments; clip heavy-tailed features.
        for off in range(0, len(x), 100000):
            x[off : off + 100000] = np.clip(
                (x[off : off + 100000] - mean) / std, -10, 10
            )
        assert np.isfinite(x).all() and groups.sum() == len(x)
        np.save(str(prefix) + "-x.npy", x)
        np.savez(str(prefix) + "-other.npz", y=y, groups=groups, mean=mean, std=std)
        Path(str(prefix) + "-meta.json").write_text(
            json.dumps(
                {
                    "windows": training_keys(fold),
                    "pairs": len(x),
                    "queries": len(groups),
                    "features": x.shape[1],
                    "max_candidates": int(groups.max()),
                    "normalization_sample": len(sample),
                    "labels_end_before": WINDOWS[fold][0],
                },
                indent=2,
            )
        )
        print("PREPARED", fold, len(x), len(groups), flush=True)
    a = dict(np.load(str(prefix) + "-other.npz"))
    a["x"] = np.load(str(prefix) + "-x.npy", mmap_mode="r")
    a["offsets"] = np.r_[0, np.cumsum(a["groups"])]
    return a


class RankMLP(nn.Module):
    def __init__(self, features, width, depth, dropout, **kwargs):
        super().__init__()
        self.input = nn.Linear(features, width)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, width * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(width * 2, width),
                )
                for _ in range(depth)
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.linear = nn.Linear(features, 1)

    def forward(self, x):
        h = F.gelu(self.input(x))
        for block in self.blocks:
            h = h + block(h)
        return (self.output(h) + self.linear(x)).squeeze(-1)


def map_swap_weights(logits, labels, mask):
    """Exact absolute change in AP@5 from swapping positive/negative ranks.

    Detached weights for LambdaRank-style pairwise gradients. Ground-truth
    denominator remains FIVE even if retrieval missed some true programs.
    Select the top-20 negatives as hard negatives for computational focus;
    other negative pairs are omitted. All retrieved positives are used.
    """
    with torch.no_grad():
        scores = logits.masked_fill(~mask, -1e9)
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        rank = torch.empty_like(order)
        positions = torch.arange(1, scores.shape[1] + 1, device=scores.device)[
            None
        ].expand_as(order)
        rank.scatter_(1, order, positions)
        y = (labels * mask).gather(1, order)
        cumulative = y.cumsum(dim=1)
        reciprocal = (y / positions * (positions <= 5)).cumsum(dim=1)
        pos = torch.topk(labels * mask, min(5, scores.shape[1]), dim=1).indices
        neg = torch.topk(
            scores.masked_fill((labels > 0) | ~mask, -1e9),
            min(20, scores.shape[1]),
            dim=1,
        ).indices
        p, q = rank.gather(1, pos)[:, :, None], rank.gather(1, neg)[:, None, :]
        cp = cumulative.gather(1, p.squeeze(-1) - 1)[:, :, None]
        cq = cumulative.gather(1, q.squeeze(1) - 1)[:, None, :]
        # Prefix sum of relevant reciprocal ranks strictly between the pair.
        lo, hi = torch.minimum(p, q), torch.maximum(p, q)
        prefix = F.pad(reciprocal, (1, 0))
        upper = prefix.gather(1, (hi - 1).flatten(1)).reshape_as(hi)
        lower = prefix.gather(1, lo.flatten(1)).reshape_as(lo)
        between = upper - lower
        moved = (cq + (q < p)) / q * (q <= 5)
        original = cp / p * (p <= 5)
        delta = (moved - original + torch.where(q < p, between, -between)).abs() / 5
        valid = (labels.gather(1, pos) > 0)[:, :, None] & mask.gather(1, pos)[
            :, :, None
        ]
        valid = (
            valid
            & (labels.gather(1, neg) == 0)[:, None, :]
            & mask.gather(1, neg)[:, None, :]
        )
        return pos, neg, delta * valid


def query_loss(logits, labels, mask, objective):
    """Equal query weight; padding contributes neither logits nor labels."""
    labels = labels * mask
    positives = labels.sum(dim=-1).clamp_min(1)
    if objective == "listwise":
        logp = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
        return (-(labels * logp).sum(dim=-1) / positives).mean()
    if objective == "lambda_map":
        pos, neg, weights = map_swap_weights(logits, labels, mask)
        margin = logits.gather(1, pos)[:, :, None] - logits.gather(1, neg)[:, None, :]
        losses = (F.softplus(-margin) * weights).sum(dim=(1, 2))
        return (losses / weights.sum(dim=(1, 2)).clamp_min(1e-6)).mean()
    positive = (F.softplus(-logits) * labels).sum(dim=-1) / positives
    negatives = mask.float() - labels
    negative = (F.softplus(logits) * negatives).sum(dim=-1) / negatives.sum(
        dim=-1
    ).clamp_min(1)
    return ((positive + negative) * 0.5).mean()


def batch(a, queries, device):
    sizes = a["groups"][queries]
    maxsize = int(sizes.max())
    positions = np.arange(maxsize)[None, :]
    mask = positions < sizes[:, None]
    rows = np.minimum(a["offsets"][queries, None] + positions, len(a["y"]) - 1)
    x = torch.from_numpy(np.asarray(a["x"][rows])).to(device)
    y = torch.from_numpy(a["y"][rows].astype("float32")).to(device)
    return x, y, torch.from_numpy(mask).to(device)


def score(model, d, mean, std, f, device):
    model.eval()
    result = np.empty(len(d["X"]), dtype="float32")
    with torch.inference_mode():
        for off in range(0, len(result), 16384):
            z = np.clip((d["X"][off : off + 16384] - mean) / std, -10, 10)
            result[off : off + len(z)] = (
                model(torch.from_numpy(z).to(device)).cpu().numpy()
            )
    assert np.isfinite(result).all()
    dense = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
    dense[d["u"], d["i"]] = result
    return dense


def run(fold, config, seed=SEED, epochs=80, fixed_epochs=False, resume=False):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    settings = CONFIGS[config]
    name = f"{fold}_{config}_s{seed}"
    folder = DEST / name
    folder.mkdir(exist_ok=True)
    a = prepare(fold)
    d = cached(fold)
    f = Fold(*WINDOWS[fold])
    model = RankMLP(a["x"].shape[1], **settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=80, eta_min=1e-5
    )
    history, start, best, best_epoch = [], 0, -1.0, 0
    if resume and (folder / "latest.pt").exists():
        state = torch.load(
            folder / "latest.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "mps" and "mps_rng" in state:
            torch.mps.set_rng_state(state["mps_rng"].cpu())
        rng.bit_generator.state = state["numpy_rng"]
        history, start, best, best_epoch = (
            state["history"],
            state["epoch"],
            state["best"],
            state["best_epoch"],
        )
    meta = {
        "name": name,
        "config": settings,
        "seed": seed,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "max_epochs": epochs,
        "batch_queries": 64,
        "lr": 0.001,
        "weight_decay": 0.001,
        "schedule": "cosine over 80 epochs, minimum 1e-5",
        "early_stop_epochs": 16,
        "fixed_epochs": fixed_epochs,
        "training_windows": training_keys(fold),
        "queries": len(a["groups"]),
        "pairs": len(a["y"]),
        "validation_scope": "Previously inspected temporal window; no untouched-test claim.",
    }
    (folder / "config.json").write_text(json.dumps(meta, indent=2))
    print("START", json.dumps(meta), flush=True)
    t = time.perf_counter()
    for epoch in range(start + 1, epochs + 1):
        model.train()
        order = rng.permutation(len(a["groups"]))
        total = torch.zeros((), device=device)
        for off in range(0, len(order), 64):
            q = order[off : off + 64]
            x, y, mask = batch(a, q, device)
            optimizer.zero_grad(set_to_none=True)
            loss = query_loss(model(x), y, mask, settings["objective"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.detach() * len(q)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(total.cpu()) / len(order),
            "lr": lr,
            "seconds": time.perf_counter() - t,
        }
        should_eval = fold != "final" and (
            epoch % 2 == 0 or epoch == 1 or epoch == epochs
        )
        if should_eval:
            pred = score(model, d, a["mean"], a["std"], f, device)
            metric = evaluate(f.truth, f.rank(pred))
            row.update({k: v for k, v in metric.items() if k != "per_user_ap"})
            current = metric["map5_five_targets"]
            if current > best:
                best, best_epoch = current, epoch
                np.save(folder / "best-scores.npy", pred)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": settings,
                        "mean": a["mean"],
                        "std": a["std"],
                        "names": d["names"],
                        "epoch": epoch,
                        "metric": metric,
                        "seed": seed,
                    },
                    folder / "best.pt",
                )
        history.append(row)
        (folder / "history.json").write_text(json.dumps(history, indent=2))
        print("EPOCH", name, json.dumps(row), flush=True)
        if epoch % 2 == 0 or epoch == epochs:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "history": history,
                "best": best,
                "best_epoch": best_epoch,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
            }
            if device.type == "mps":
                state["mps_rng"] = torch.mps.get_rng_state()
            torch.save(state, folder / "latest.pt")
        if not fixed_epochs and epoch >= 20 and epoch - best_epoch >= 16:
            break
    if fixed_epochs:
        pred = score(model, d, a["mean"], a["std"], f, device)
        np.save(folder / "fixed-scores.npy", pred)
        torch.save(
            {
                "model": model.state_dict(),
                "config": settings,
                "mean": a["mean"],
                "std": a["std"],
                "names": d["names"],
                "epoch": epoch,
            },
            folder / "fixed.pt",
        )
    result = {
        **meta,
        "completed_epochs": epoch,
        "best_epoch": best_epoch,
        "best_validation_map5": best if best >= 0 else None,
        "elapsed_seconds": time.perf_counter() - t,
    }
    if fixed_epochs and fold != "final":
        result["fixed_validation"] = evaluate(f.truth, f.rank(pred))
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    print("FINISHED", json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", choices=["dev", "audit", "final"], default="dev")
    p.add_argument("--config", choices=list(CONFIGS), default="list128")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--fixed-epochs", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run(args.fold, args.config, args.seed, args.epochs, args.fixed_epochs, args.resume)
