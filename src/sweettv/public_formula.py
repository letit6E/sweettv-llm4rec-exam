"""Independent adaptation of Gornyaki's public frequency formula.

Reference: https://www.kaggle.com/code/kinahem/gornyaki-final-solution
Myhaylo Turkin and Alex Tsepa, notebook version 10, Apache-2.0.
Private preprocessing inputs unavailable: uses our documented overlap data,
not an exact reproduction. Notebook future_count is joined but UNUSED in score.
"""

import argparse, json
import numpy as np
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from .experiment import Fold, csr
from .improve import WINDOWS
from .simple_search import DEST, baseline
from .metrics import evaluate
from .blend import rrf


def score(f, single, bonus, filtered):
    t = f.train.copy()
    frac = t.single_fraction if single else t.fraction
    keep = frac >= 0.8
    if filtered:
        meta = (
            f.past.sort_values("end")
            .drop_duplicates("tv_show_id", keep="last")
            .set_index("tv_show_id")
            .tv_show_category
        )
        keep &= (t.duration < 10000) & t.tv_show_id.map(meta).isin(
            ["Развлечения", "Инфо", "Сериалы", "Детям", "Другое", "Познавательное"]
        )
    t = t[keep]
    frac = frac[keep].to_numpy()
    v = np.where(frac == 1, 5, 2 + 2 * frac**2) * np.power(
        0.8, (f.start - t.start.to_numpy()) / 86400 / 60
    )
    x = csr(t.user, t.item_idx, v, f.shape)
    z = x[f.eval_users].toarray()
    air = f.past.groupby("tv_show_id").size().reindex(f.items, fill_value=0).to_numpy()
    z *= np.maximum(np.log10(np.maximum(air, 1)), 0) ** bonus
    # Neighbor fallback for warm users with fewer than five observed positive items.
    counts = csr(t.user, t.item_idx, np.ones(len(t)), f.shape)
    mx = np.asarray(counts.max(axis=0).toarray()).ravel()
    scaled = counts.multiply(1 / np.maximum(mx, 1)).tocsr()
    profiles = normalize(scaled)
    sim = (profiles[f.eval_users] @ profiles.T).toarray()
    sim[np.arange(len(f.eval_users)), f.eval_users] = 0
    ix = np.argsort(-sim, axis=1, kind="stable")[:, :10]
    a = scaled.toarray()
    order = np.argsort(-a, axis=1, kind="stable")[:, :10]
    mask = np.zeros_like(a, dtype=bool)
    np.put_along_axis(mask, order, True, axis=1)
    a[~mask] = 0
    near = a[ix].sum(axis=1)
    near /= np.maximum(near.max(axis=1, keepdims=True), 1e-9)
    z = np.where(z > 0, z, near * 1e-6)
    return z.astype("float32")


def run(stage):
    if stage == "select":
        f = Fold(*WINDOWS["dev"])
        b = rrf(f, baseline("dev"))
        rows = []
        for single in [False, True]:
            for filtered in [False, True]:
                for bonus in [0, 1, 2]:
                    s = score(f, single, bonus, filtered)
                    r = rrf(f, s)
                    plain = evaluate(f.truth, f.rank(s))["map5_five_targets"]
                    for w in [0.1, 0.25, 0.5, 1.0]:
                        rows.append(
                            {
                                "single": single,
                                "filtered": filtered,
                                "bonus": bonus,
                                "weight": w,
                                "dev": evaluate(f.truth, f.rank((1 - w) * b + w * r))[
                                    "map5_five_targets"
                                ],
                                "standalone": plain,
                            }
                        )
        best = max(rows, key=lambda r: r["dev"])
        (DEST / "public-selection.json").write_text(
            json.dumps({"grid": rows, "selected": best, "source": __doc__}, indent=2)
        )
        print(best, flush=True)
    else:
        r = json.loads((DEST / "public-selection.json").read_text())["selected"]
        out = {}
        for key in ["audit", "long"]:
            f = Fold(*WINDOWS[key])
            s = score(f, r["single"], r["bonus"], r["filtered"])
            np.save(DEST / f"{key}_public.npy", s)
            z = (1 - r["weight"]) * rrf(f, baseline(key)) + r["weight"] * rrf(f, s)
            out[key] = {
                "standalone": evaluate(f.truth, f.rank(s))["map5_five_targets"],
                "blend": evaluate(f.truth, f.rank(z))["map5_five_targets"],
            }
            print(key, out[key], flush=True)
        (DEST / "public-confirmation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "confirm"])
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.stage)
