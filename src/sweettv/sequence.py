"""Small eSASRec-style experiment on supplied TV histories only.

LiGR gated pre-LN attention and SwiGLU adapted from transformer_benchmark,
commit 2b039927ceb4c7654131d7ac7c43ea124b49d240 (Apache-2.0).
Original authors: Daria Tikhonovich et al., eSASRec (RecSys 2025).
Changes: standalone PyTorch blocks; 64 factors, 50-token contexts; sampled
softmax and rolling training chunks; next-event versus multi-week evaluation.
This is an adaptation, NOT a reproduction of the authors' benchmark numbers.
"""

import argparse, json, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from .experiment import Fold, SEED
from .improve import DEST, WINDOWS
from .metrics import evaluate


class GatedBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 2, dropout=0.1, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.g1 = nn.Linear(d, d)
        self.g2 = nn.Linear(d, d)
        self.up = nn.Linear(d, d * 4, bias=False)
        self.gate = nn.Linear(d, d * 4, bias=False)
        self.down = nn.Linear(d * 4, d, bias=False)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, causal, pad):
        z = self.ln1(x)
        z = self.attn(
            z, z, z, attn_mask=causal, key_padding_mask=pad, need_weights=False
        )[0]
        x = x + torch.sigmoid(self.g1(x)) * self.drop(z)
        z = self.ln2(x)
        z = self.down(self.drop(F.silu(self.up(z)) * self.gate(z)))
        return (x + torch.sigmoid(self.g2(x)) * self.drop(z)).masked_fill(
            pad[:, :, None], 0
        )


class SequenceModel(nn.Module):
    def __init__(self, n, d=64, length=50):
        super().__init__()
        self.emb = nn.Embedding(n + 1, d, padding_idx=0)
        self.pos = nn.Embedding(length, d)
        self.blocks = nn.ModuleList([GatedBlock(d) for _ in range(2)])
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(0.1)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[0].zero_()

    def forward(self, seq):
        pad = seq == 0
        # Right padding means valid queries always have a nonpadding key.
        z = self.drop(
            self.emb(seq) * 8
            + self.pos(torch.arange(seq.shape[1], device=seq.device))[None]
        )
        mask = torch.triu(
            torch.ones(seq.shape[1], seq.shape[1], dtype=torch.bool, device=seq.device),
            diagonal=1,
        )
        for block in self.blocks:
            z = block(z, mask, pad)
        return self.norm(z).masked_fill(pad[:, :, None], 0)


def train(foldname, epochs=20):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    f = Fold(*WINDOWS[foldname])
    t = f.train[f.train.fraction >= 0.8].sort_values(["user", "end", "airing_id"])
    # Only training items receive embeddings; unseen catalogue items use other models.
    ids = np.array(sorted(t.tv_show_id.unique()))
    imap = {int(v): j + 1 for j, v in enumerate(ids)}
    histories = {
        int(u): np.array([imap[int(i)] for i in g.tv_show_id], dtype="int64")
        for u, g in t.groupby("user")
    }
    xs = []
    ys = []
    for seq in histories.values():
        for end in range(2, len(seq) + 25, 25):
            end = min(end, len(seq))
            a = seq[max(0, end - 51) : end]
            if len(a) < 2:
                continue
            x = np.zeros(50, dtype="int64")
            y = x.copy()
            x[: len(a) - 1] = a[:-1]
            y[: len(a) - 1] = a[1:]
            xs.append(x)
            ys.append(y)
            if end == len(seq):
                break
    loader = DataLoader(
        TensorDataset(torch.tensor(np.array(xs)), torch.tensor(np.array(ys))),
        batch_size=128,
        shuffle=True,
    )
    model = SequenceModel(len(ids)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    test = np.zeros((len(f.eval_users), 50), dtype="int64")
    lengths = np.ones(len(test), dtype="int64")
    for row, u in enumerate(f.eval_users):
        a = histories.get(int(u), np.array([], dtype="int64"))[-50:]
        test[row, : len(a)] = a
        lengths[row] = max(len(a), 1)
        if not len(a):
            test[row, 0] = 1  # discarded by common cold fallback
    test = torch.tensor(test, device=dev)
    results = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)
            h = model(x)
            valid = y > 0
            h = h[valid]
            target = y[valid]
            neg = torch.randint(1, len(ids) + 1, (256,), device=dev)
            pos = (h * model.emb(target)).sum(-1, keepdim=True)
            negative = h @ model.emb(neg).T
            negative = negative.masked_fill(target[:, None] == neg[None], -1e4)
            logits = torch.cat([pos, negative], dim=1)
            loss = F.cross_entropy(
                logits, torch.zeros(len(target), dtype=torch.long, device=dev)
            )
            assert torch.isfinite(loss), "Nonfinite loss"
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            losses.append(loss.item())
        print(
            "SEQ",
            foldname,
            epoch,
            float(np.mean(losses)),
            round(time.perf_counter() - start, 1),
            flush=True,
        )
        if epoch in [5, 10, 20, 30, 40]:
            model.eval()
            out = []
            avg = []
            with torch.no_grad():
                for a in range(0, len(test), 128):
                    h = model(test[a : a + 128])
                    lens = torch.tensor(lengths[a : a + 128], device=dev)
                    last = h[torch.arange(len(h), device=dev), lens - 1]
                    pos = torch.arange(50, device=dev)[None, :]
                    mask = (pos < lens[:, None]) & (
                        pos >= torch.clamp(lens[:, None] - 10, min=0)
                    )
                    mean = (h * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
                    out.append((last @ model.emb.weight[1:].T).cpu().numpy())
                    avg.append((mean @ model.emb.weight[1:].T).cpu().numpy())
            for name, v in [("last", out), ("mean10", avg)]:
                score = np.full((len(test), len(f.items)), -1e4, dtype="float32")
                score[:, [f.im[int(i)] for i in ids]] = np.concatenate(v)
                assert np.isfinite(score).all()
                name = f"{foldname}_esasrec_{epoch}_{name}"
                np.save(DEST / (name + ".npy"), score)
                metrics = (
                    evaluate(f.truth, f.rank(score))
                    if foldname != "final"
                    else {"test_labels_available": False}
                )
                row = {
                    "name": name,
                    "epoch": epoch,
                    "seconds": time.perf_counter() - start,
                    "device": str(dev),
                    "training_chunks": len(xs),
                    **{k: v for k, v in metrics.items() if k != "per_user_ap"},
                }
                results.append(row)
                print("SEQRESULT", json.dumps(row), flush=True)
            torch.save(
                {"state_dict": model.cpu().state_dict(), "items": ids, "epoch": epoch},
                DEST / f"{foldname}_esasrec_{epoch}.pt",
            )
            model.to(dev)
            (DEST / f"{foldname}-sequence-results.json").write_text(
                json.dumps(results, indent=2)
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="dev")
    p.add_argument("--epochs", type=int, default=20)
    a = p.parse_args()
    train(a.fold, a.epochs)
