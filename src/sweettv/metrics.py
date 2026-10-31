"""Ranking metrics. Relevance is binary; prediction position affects AP."""

import numpy as np


def apk(actual, predicted, k=5):
    relevant = set(actual)
    if not relevant:
        return 0.0
    hits, score, used = 0, 0.0, set()
    for rank, item in enumerate(predicted[:k], 1):
        if item in relevant and item not in used:
            hits += 1
            score += hits / rank
        used.add(item)
    return score / min(k, len(relevant))


def evaluate(truth, recommendations):
    if len(truth) != len(recommendations):
        raise ValueError("Truth and prediction users must align")
    ap = np.array([apk(t, p) for t, p in zip(truth, recommendations)])
    nonempty = np.array([len(t) > 0 for t in truth])
    five = np.array([len(t) == 5 for t in truth])
    return {
        "map5_all": float(ap.mean()) if len(ap) else 0.0,
        "map5_nonempty": float(ap[nonempty].mean()) if nonempty.any() else None,
        "map5_five_targets": float(ap[five].mean()) if five.any() else None,
        "users": len(ap),
        "nonempty_users": int(nonempty.sum()),
        "five_target_users": int(five.sum()),
        "per_user_ap": ap.tolist(),
    }


def topk(scores, item_ids, eligible, fallback, k=5):
    """One candidate policy and deterministic fallback for every method.

    Zero-scored items use training popularity, then numeric program ID.
    Negative finite scores (e.g. MF) remain legitimate model outputs.
    """
    scores = np.asarray(scores, dtype=float)
    ids = np.asarray(item_ids)
    eligible = np.asarray(eligible, dtype=bool)
    fallback = np.asarray(fallback, dtype=float)
    if not (scores.shape == ids.shape == eligible.shape == fallback.shape):
        raise ValueError("Mismatched item arrays")
    if not np.isfinite(scores).all() or not np.isfinite(fallback).all():
        raise ValueError("Nonfinite ranking score")
    idx = np.flatnonzero(eligible & (ids != 0))
    if len(idx) < k:
        raise ValueError("Fewer than five valid candidates; cannot fabricate IDs")
    order = np.lexsort((ids[idx], -fallback[idx], -scores[idx]))
    return ids[idx[order[:k]]].tolist()
