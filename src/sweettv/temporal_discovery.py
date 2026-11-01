"""Past-to-future program associations trained only on completed historical windows."""

import json, time
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from .experiment import Fold, csr
from .improve import WINDOWS
from .discovery_context import DEST
from .simple_search import baseline
from .metrics import evaluate
from .blend import rrf


def matrix(f, im, days, threshold):
    t = f.train[
        (f.train.end >= f.start - days * 86400) & (f.train.fraction >= threshold)
    ]
    return csr(t.user, t.tv_show_id.map(im), np.ones(len(t)), (len(f.users), len(im)))


def train(outer, keys, days, kind):
    shape = outer.shape
    co = sparse.csr_matrix((shape[1], shape[1]), dtype="float32")
    source = np.zeros(shape[1])
    target = np.zeros(shape[1])
    for key in keys:
        assert WINDOWS[key][1] <= WINDOWS["dev"][0]
        f = Fold(*WINDOWS[key])
        x = matrix(f, outer.im, days, 0.8)
        x.data[:] = 1
        x = normalize(x, norm="l1")
        truth = f.truth_events
        if kind == "top5":
            counts = (
                truth.groupby(["user", "tv_show_id"]).size().reset_index(name="count")
            )
            counts = (
                counts.sort_values(
                    ["user", "count", "tv_show_id"], ascending=[True, False, True]
                )
                .groupby("user")
                .head(5)
            )
            y = csr(
                counts.user,
                counts.tv_show_id.map(outer.im),
                np.ones(len(counts)),
                shape,
            )
        else:
            y = csr(
                truth.user, truth.tv_show_id.map(outer.im), np.ones(len(truth)), shape
            )
            y.data = np.sqrt(y.data)
        seen = matrix(f, outer.im, 1000, 0)
        seen.data[:] = 1
        y = y - y.multiply(seen)
        y.eliminate_zeros()
        y = normalize(y, norm="l1")
        co += x.T @ y
        source += np.asarray(x.sum(axis=0)).ravel()
        target += np.asarray(y.sum(axis=0)).ravel()
    return co, source, target


def run():
    outer = Fold(*WINDOWS["dev"])
    t = time.perf_counter()
    records = []
    best = None
    base = 0.9 * rrf(outer, baseline("dev")) + 0.1 * rrf(
        outer, np.load(DEST.parent / "approaches-v5/dev_channel.npy")
    )
    for days in [28, 112]:
        for kind in ["top5", "all"]:
            co, a, b = train(outer, ["early1", "early2", "early3"], days, kind)
            x = matrix(outer, outer.im, days, 0.8)[outer.eval_users]
            x.data = np.log1p(x.data)
            for power in [0, 0.5, 1]:
                graph = (
                    co.multiply(1 / np.maximum(a[:, None], 1))
                    .multiply(1 / np.maximum(b[None, :], 1) ** power)
                    .tocsr()
                )
                score = (normalize(x, norm="l1") @ graph).toarray()
                score[outer.matrix(0)[outer.eval_users].toarray() > 0] = 0
                rank = rrf(outer, score)
                for weight in [0.05, 0.1, 0.25, 0.5]:
                    m = evaluate(
                        outer.truth, outer.rank((1 - weight) * base + weight * rank)
                    )["map5_five_targets"]
                    r = dict(days=days, kind=kind, power=power, weight=weight, map5=m)
                    records.append(r)
                    if best is None or m > best["map5"]:
                        best = r
                        np.save(DEST / "dev_temporal_discovery.npy", score)
                        sparse.save_npz(DEST / "temporal_discovery_graph.npz", graph)
                print("TRANSITION", days, kind, power, "best", best, flush=True)
    (DEST / "temporal-discovery-selection.json").write_text(
        json.dumps(
            {
                "selected": best,
                "grid": records,
                "seconds": time.perf_counter() - t,
                "policy": "Historical directed transitions to new-to-user programs; no validation targets used in graph training.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
