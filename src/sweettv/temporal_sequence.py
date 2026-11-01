"""Multi-week full-softmax sequence recommender, not next-event prediction.

Uses the attributed LiGR block from sequence.py. Training examples contain
only events completed at their cutoff; all target windows finish before the
outer evaluation cutoff. Repeated programs remain eligible. No sampled negatives.
"""

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from .experiment import Fold, SEED, epoch, csr
from .improve import WINDOWS
from .metrics import evaluate
from .neural_rank import DEST
from .sequence import GatedBlock

CONFIGS = {
    "seq64": dict(d=64, blocks=2, prior=False),
    "seq128": dict(d=128, blocks=2, prior=False),
    "seq128prior": dict(d=128, blocks=2, prior=True),
    "seq256prior": dict(d=256, blocks=2, prior=True),
    "seq64prior": dict(d=64, blocks=2, prior=True),
    "seq128reg": dict(d=128, blocks=2, prior=True, dropout=0.5),
}


def snapshot(events, cutoff, users, im, n, length=100):
    h = events[events.end <= cutoff]
    mapping = {int(u): j for j, u in enumerate(users)}
    h = h[h.user.isin(mapping)].copy()
    h["row"] = h.user.map(mapping)
    h["col"] = h.tv_show_id.map(im)
    assert h.col.notna().all()
    rows, cols = h.row.to_numpy(), h.col.to_numpy(dtype="int64")
    count = csr(rows, cols, np.ones(len(h)), (len(users), n))
    recent = csr(
        rows,
        cols,
        np.exp2(-(cutoff - h.end.to_numpy()) / (28 * 86400)),
        (len(users), n),
    )
    count.data = np.log1p(count.data)
    recent.data = np.log1p(recent.data)
    seq = np.zeros((len(users), length), dtype="int64")
    for u, group in h.groupby("row", sort=False):
        items = group.col.to_numpy(dtype="int64")[-length:] + 1
        seq[int(u), : len(items)] = items
    return seq, count, recent


def prepare(fold):
    path = DEST / (fold + "-temporal-sequences.pkl")
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    f = Fold(*WINDOWS[fold])
    events = f.train[f.train.fraction >= 0.8].sort_values(["user", "end", "airing_id"])
    ids = np.array(sorted(events.tv_show_id.unique()))
    im = {int(v): j for j, v in enumerate(ids)}
    n = len(ids)
    xs, counts, recent, targets, masks, metadata = [], [], [], [], [], []
    # Weekly overlapping queries improve sample coverage, not independent sample size.
    last = pd.Timestamp(WINDOWS[fold][0]) - pd.Timedelta(days=35)
    for start in pd.date_range("2020-03-23", last, freq="7D"):
        end = start + pd.Timedelta(days=35)
        cut, stop = epoch(str(start.date())), epoch(str(end.date()))
        assert stop <= f.start
        target = events[(events.start >= cut) & (events.end <= stop)]
        z = target.groupby(["user", "tv_show_id"]).size().reset_index(name="count")
        z = z.sort_values(
            ["user", "count", "tv_show_id"], ascending=[True, False, True]
        )
        truth = (
            z.groupby("user").head(5).groupby("user").tv_show_id.apply(list).to_dict()
        )
        seen = set(events.loc[events.end <= cut, "user"])
        users = np.array([u for u, t in truth.items() if len(t) == 5 and u in seen])
        if not len(users):
            continue
        x, c, r = snapshot(events, cut, users, im, n)
        y = np.array([[im[int(item)] for item in truth[int(u)]] for u in users])
        eligible = np.isin(
            ids, f.past.loc[(f.past.start >= cut) & (f.past.start < stop), "tv_show_id"]
        )
        assert eligible[y].all(), "Target must be available during its window"
        xs.append(x)
        counts.append(c)
        recent.append(r)
        targets.append(y)
        masks.append(eligible)
        metadata.append(
            {"start": str(start.date()), "end": str(end.date()), "queries": len(users)}
        )
    xeval, ceval, reval = snapshot(events, f.start, f.eval_users, im, n)
    result = dict(
        ids=ids,
        seq=np.concatenate(xs),
        count=sparse.vstack(counts, format="csr"),
        recent=sparse.vstack(recent, format="csr"),
        targets=np.concatenate(targets),
        masks=np.stack(masks),
        window=np.repeat(np.arange(len(xs)), [len(x) for x in xs]),
        eval_seq=xeval,
        eval_count=ceval,
        eval_recent=reval,
        windows=metadata,
    )
    with path.open("wb") as handle:
        pickle.dump(result, handle, protocol=4)
    print("SEQUENCE_DATA", fold, len(result["seq"]), n, metadata, flush=True)
    return result


class TemporalSequence(nn.Module):
    def __init__(self, n, d, blocks, prior, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n + 1, d, padding_idx=0)
        self.pos = nn.Embedding(100, d)
        self.blocks = nn.ModuleList([GatedBlock(d) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        for module in self.blocks.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
            elif isinstance(module, nn.MultiheadAttention):
                module.dropout = dropout
        self.bias = nn.Parameter(torch.zeros(n))
        self.prior = prior
        self.repeat_weight = nn.Parameter(torch.zeros(2))
        self.d = d
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[0].zero_()

    def forward(self, seq, count, recent):
        valid = seq != 0
        # All-empty histories get a dummy key, then their pooled representation is zero.
        safe = seq.clone()
        safe[~valid.any(dim=1), 0] = 1
        pad = safe == 0
        z = self.drop(
            self.emb(safe) * self.d**0.5
            + self.pos(torch.arange(100, device=seq.device))[None]
        )
        causal = torch.ones((100, 100), dtype=torch.bool, device=seq.device).triu(1)
        for block in self.blocks:
            z = block(z, causal, pad)
        # Multi-week interests summarized from the most recent 20 contextual states.
        take = valid & (
            torch.arange(100, device=seq.device)[None] >= valid.sum(dim=1)[:, None] - 20
        )
        q = (self.norm(z) * take[:, :, None]).sum(dim=1) / take.sum(dim=1)[
            :, None
        ].clamp_min(1)
        scores = q @ self.emb.weight[1:].T + self.bias
        if self.prior:
            w = F.softplus(self.repeat_weight)
            scores = scores + w[0] * count + w[1] * recent
        return scores


def predict(model, a, f, device):
    model.eval()
    scores = np.full((len(f.eval_users), len(f.items)), -1e4, dtype="float32")
    columns = np.array([f.im[int(v)] for v in a["ids"]])
    with torch.inference_mode():
        for off in range(0, len(scores), 128):
            sl = slice(off, off + 128)
            seq = torch.from_numpy(a["eval_seq"][sl]).to(device)
            count = torch.from_numpy(a["eval_count"][sl].toarray()).to(device)
            recent = torch.from_numpy(a["eval_recent"][sl].toarray()).to(device)
            result = model(seq, count, recent).cpu().numpy()
            scores[np.arange(off, off + len(result))[:, None], columns[None]] = result
    assert np.isfinite(scores).all()
    return scores


def run(fold, config, seed=SEED, epochs=100, fixed_epochs=False):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    a = prepare(fold)
    f = Fold(*WINDOWS[fold])
    settings = CONFIGS[config]
    name = f"{fold}_{config}_s{seed}"
    folder = DEST / name
    folder.mkdir(exist_ok=True)
    model = TemporalSequence(len(a["ids"]), **settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-5
    )
    metadata = dict(
        name=name,
        config=settings,
        parameters=sum(p.numel() for p in model.parameters()),
        seed=seed,
        windows=a["windows"],
        queries=len(a["seq"]),
        vocabulary=len(a["ids"]),
        max_epochs=epochs,
        patience_epochs=20,
        batch_size=128,
        objective="Full-softmax cross-entropy with equal mass on future Top-5",
        optimizer="AdamW lr=0.001 weight_decay=0.01, cosine100 minlr=1e-5",
        evaluation="Previously inspected temporal validation, not untouched test",
    )
    (folder / "config.json").write_text(json.dumps(metadata, indent=2))
    print("START", json.dumps(metadata), flush=True)
    history, best, best_epoch = [], -1.0, 0
    t = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        total = torch.zeros((), device=device)
        order = rng.permutation(len(a["seq"]))
        for off in range(0, len(order), 128):
            ix = order[off : off + 128]
            x = torch.from_numpy(a["seq"][ix]).to(device)
            c = torch.from_numpy(a["count"][ix].toarray()).to(device)
            r = torch.from_numpy(a["recent"][ix].toarray()).to(device)
            y = torch.from_numpy(a["targets"][ix]).to(device)
            available = torch.from_numpy(a["masks"][a["window"][ix]]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, c, r).masked_fill(~available, -1e9)
            loss = -F.log_softmax(logits, dim=1).gather(1, y).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.detach() * len(ix)
        row = dict(
            epoch=ep,
            train_loss=float(total.cpu()) / len(order),
            lr=optimizer.param_groups[0]["lr"],
            seconds=time.perf_counter() - t,
        )
        scheduler.step()
        if fold != "final" and (ep % 2 == 0 or ep == 1 or ep == epochs):
            scores = predict(model, a, f, device)
            metrics = evaluate(f.truth, f.rank(scores))
            row.update({k: v for k, v in metrics.items() if k != "per_user_ap"})
            if metrics["map5_five_targets"] > best:
                best, best_epoch = metrics["map5_five_targets"], ep
                np.save(folder / "best-scores.npy", scores)
                torch.save(
                    dict(
                        model=model.state_dict(),
                        config=settings,
                        ids=a["ids"],
                        epoch=ep,
                        metric=metrics,
                    ),
                    folder / "best.pt",
                )
        history.append(row)
        (folder / "history.json").write_text(json.dumps(history, indent=2))
        print("EPOCH", name, json.dumps(row), flush=True)
        if ep % 2 == 0 or ep == epochs:
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    history=history,
                    epoch=ep,
                    numpy_rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    mps_rng=torch.mps.get_rng_state() if device.type == "mps" else None,
                ),
                folder / "latest.pt",
            )
        if not fixed_epochs and ep >= 30 and ep - best_epoch >= 20:
            break
    if fixed_epochs:
        np.save(folder / "fixed-scores.npy", predict(model, a, f, device))
    result = dict(
        **metadata,
        completed_epochs=ep,
        best_epoch=best_epoch,
        best_validation_map5=best if best >= 0 else None,
        elapsed_seconds=time.perf_counter() - t,
    )
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    print("FINISHED", json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", choices=["dev", "audit", "final"], default="dev")
    p.add_argument("--config", choices=list(CONFIGS), default="seq128prior")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--fixed-epochs", action="store_true")
    args = p.parse_args()
    run(args.fold, args.config, args.seed, args.epochs, args.fixed_epochs)
