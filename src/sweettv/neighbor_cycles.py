"""Alternative neighborhood signal focused on discovering unseen programs."""

import numpy as np
from sklearn.preprocessing import normalize
from .experiment import csr


def neighbor_features(f, u, i):
    vals = []
    names = []
    for days in [28, 112]:
        t = f.train[(f.train.end >= f.start - days * 86400) & (f.train.fraction >= 0.8)]
        x = csr(t.user, t.item_idx, np.ones(len(t)), f.shape)
        # TF-IDF cosine: repeated and rare full views, no partial zapping signal.
        df = np.asarray((x > 0).sum(axis=0)).ravel()
        idf = np.log((1 + len(f.users)) / (1 + df))
        xx = x.copy()
        xx.data = np.log1p(xx.data)
        xx = normalize(xx.multiply(idf)).tocsr()
        sim = (xx[f.eval_users] @ xx.T).toarray()
        sim[np.arange(len(f.eval_users)), f.eval_users] = 0
        order = np.argsort(-sim, axis=1, kind="stable")
        # Neighbor histories are normalized: a heavy viewer cannot dominate solely by volume.
        y = x.copy()
        y.data = np.log1p(y.data)
        y = normalize(y, norm="l1")
        for k in [10, 30, 100]:
            idx = order[:, :k]
            weights = np.take_along_axis(sim, idx, axis=1) ** 2
            nn = csr(
                np.repeat(np.arange(len(idx)), idx.shape[1]),
                idx.ravel(),
                weights.ravel(),
                (len(idx), len(f.users)),
            )
            z = (normalize(nn, norm="l1") @ y).toarray()
            vals.append(z[u, i])
            names.append(f"neighbor_{days}_{k}")
            rank = np.argsort(
                np.argsort(-z, axis=1, kind="stable"), axis=1, kind="stable"
            )
            vals.append(1 / (rank[u, i] + 1))
            names.append(f"neighbor_rank_{days}_{k}")
    return np.column_stack(vals).astype("float32"), names
