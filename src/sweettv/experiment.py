"""Shared temporal folds, train-only models, measured selection and frozen audit."""

from pathlib import Path
import json, time, hashlib
import numpy as np
import pandas as pd
from scipy import sparse, linalg
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from .metrics import topk, evaluate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts"
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
SEED = 20261031


def epoch(s):
    return int(pd.Timestamp(s, tz="Europe/Kyiv").timestamp())


def csr(rows, cols, values, shape):
    x = sparse.coo_matrix((values, (rows, cols)), shape=shape, dtype=np.float32).tocsr()
    x.sum_duplicates()
    x.eliminate_zeros()
    return x


class Fold:
    def __init__(self, start, end):
        self.start = epoch(start)
        self.end = epoch(end)
        self.users = json.loads((ROOT / "data/processed/users.json").read_text())
        um = {u: i for i, u in enumerate(self.users)}
        self.template = pd.read_csv(ROOT / "data/raw/submission.csv", dtype=str)
        self.eval_users = np.array([um[u] for u in self.template.user_id])
        schedule = pd.read_pickle(ROOT / "data/processed/schedule.pkl")
        self.past = schedule[
            (schedule.end <= self.start) & (~schedule.source.str.contains("31-42"))
        ].copy()
        if self.start == epoch("2020-07-27"):
            # Official test file includes overnight spill past 19 October midnight.
            # Do not mix the previous file's last broadcast-day spill into test candidates.
            self.future = schedule[schedule.source.str.contains("31-42")].copy()
        else:
            self.future = schedule[
                (schedule.start >= self.start) & (schedule.start < self.end)
            ].copy()
        # Strict: future titles/genres are not model features.
        self.future = self.future[
            ["airing_id", "channel_id", "start", "end", "duration", "tv_show_id"]
        ]
        self.items = np.array(
            sorted(set(self.past.tv_show_id) | set(self.future.tv_show_id)),
            dtype=np.int64,
        )
        self.im = {int(i): j for j, i in enumerate(self.items)}
        self.shape = (len(self.users), len(self.items))
        self.eligible = np.isin(self.items, self.future.tv_show_id.unique())
        all_events = pd.read_pickle(ROOT / "data/processed/events.pkl")
        self.train = all_events[all_events.end <= self.start].copy()
        val = all_events[
            (all_events.start >= self.start)
            & (all_events.end <= self.end)
            & (all_events.fraction >= 0.8)
        ].copy()
        self.truth_events = val
        counts = (
            val.groupby(["user", "tv_show_id"])
            .size()
            .reset_index(name="count")
            .sort_values(["user", "count", "tv_show_id"], ascending=[True, False, True])
        )
        lookup = (
            counts.groupby("user")
            .head(5)
            .groupby("user")
            .tv_show_id.apply(list)
            .to_dict()
        )
        self.truth = [lookup.get(int(u), []) for u in self.eval_users]
        self.base = self.matrix(0.8, None)
        self.pop = np.asarray((self.base > 0).sum(axis=0)).ravel().astype(np.float64)
        self.cold = np.asarray(self.base[self.eval_users].sum(axis=1)).ravel() == 0
        self.audit = {
            "train_cutoff_local": start,
            "validation_end_local": end,
            "timezone": "Europe/Kyiv",
            "users": len(self.users),
            "template_users": len(self.eval_users),
            "items": len(self.items),
            "candidates": int(self.eligible.sum()),
            "train_overlap_events": len(self.train),
            "train_qualifying_events": int((self.train.fraction >= 0.8).sum()),
            "validation_qualifying_events": len(val),
            "empty_truth": sum(not t for t in self.truth),
            "five_target_users": sum(len(t) == 5 for t in self.truth),
            "cold_template_users": int(self.cold.sum()),
            "boundary_airings_excluded": int(
                ((all_events.start < self.start) & (all_events.end > self.start)).sum()
            ),
            "repeated_truth_fraction": self.repeat_truth_fraction(),
        }
        self.train["item_idx"] = self.train.tv_show_id.map(self.im).astype(int)
        self._matrices = {}

    def matrix(self, threshold=0.8, half=None, weight="count", single=False):
        tr = self.train[
            (self.train.single_fraction if single else self.train.fraction) >= threshold
        ]
        values = np.ones(len(tr), dtype=np.float32)
        if weight == "fraction":
            values = tr.fraction.to_numpy(dtype=np.float32)
        elif weight == "seconds":
            values = tr.seconds.to_numpy(dtype=np.float32)
        if half is not None:
            values = values * np.exp2(
                -(self.start - tr.end.to_numpy()) / (half * 86400)
            )
        return csr(
            tr.user.to_numpy(),
            tr.tv_show_id.map(self.im).to_numpy(),
            values,
            self.shape,
        )

    def repeat_truth_fraction(self):
        hit = total = 0
        for u, t in zip(self.eval_users, self.truth):
            seen = set(self.items[self.base[int(u)].indices])
            hit += len(set(t) & seen)
            total += len(t)
        return hit / total if total else None

    def content(self, x, title=False):
        metadata = (
            self.past.sort_values("start")
            .drop_duplicates("tv_show_id", keep="last")
            .set_index("tv_show_id")
        )
        docs = []
        for item in self.items:
            d = {}
            if item in metadata.index:
                row = metadata.loc[item]
                for field in [
                    "tv_show_category",
                    "tv_show_genre_1",
                    "tv_show_genre_2",
                    "tv_show_genre_3",
                    "director",
                    "actors",
                    "year_of_production",
                ]:
                    value = str(row[field]).strip().lower()
                    if not value:
                        continue
                    if field.startswith("tv_show_genre"):
                        field = "genre"
                    for token in (
                        value.split(",") if field in ["director", "actors"] else [value]
                    ):
                        token = token.strip()
                        if token:
                            d[field + ":" + token] = (
                                3.0
                                if field == "tv_show_category"
                                else (2.0 if field == "genre" else 1.0)
                            )
                if title:
                    import re

                    for word in re.findall(r"\w{3,}", str(row.tv_show_title).lower()):
                        d["title:" + word] = 1.0
            docs.append(d)
        features = DictVectorizer(dtype=np.float32).fit_transform(docs)
        # IDF fitted on past-metadata items only, not on unseen target rows.
        known = np.array([i in metadata.index for i in self.items])
        df = np.asarray((features[known] > 0).sum(axis=0)).ravel()
        idf = np.log((1 + known.sum()) / (1 + df)) + 1
        features = normalize(features.multiply(idf), norm="l2").tocsr()
        profile = normalize(x[self.eval_users] @ features, norm="l2")
        return (profile @ features.T).toarray()

    def model(self, c):
        name = c["method"]
        x = self.matrix(
            c.get("threshold", 0.8),
            c.get("half"),
            c.get("weight", "count"),
            c.get("single", False),
        )
        if name == "popularity":
            return np.broadcast_to(
                self.pop, (len(self.eval_users), len(self.items))
            ).copy()
        if name == "repeat":
            return x[self.eval_users].toarray()
        if name == "content":
            return self.content(x, c.get("title", False))
        if c.get("transform", "log") == "log":
            x.data = np.log1p(x.data)
        elif c.get("transform") == "binary":
            x.data[:] = 1
        if name == "als":
            from implicit.cpu.als import AlternatingLeastSquares

            # Explicit observed confidence c_ui=1+alpha*log(1+count); unobserved c=1,p=0.
            x.data = 1 + c.get("alpha", 40) * x.data
            model = AlternatingLeastSquares(
                factors=c.get("factors", 64),
                regularization=c.get("reg", 1),
                iterations=c.get("iterations", 25),
                alpha=1,
                random_state=SEED,
                num_threads=4,
            )
            model.fit(x, show_progress=False)
            a = model.user_factors[self.eval_users].astype(np.float64)
            b = model.item_factors.astype(np.float64)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError("Nonfinite ALS factors")
            # Independent dot accumulation avoids vendor matmul floating-point warnings.
            return np.einsum("uf,if->ui", a, b, optimize=False).astype(np.float32)
        if name in ["association", "ppr"]:
            b = (x > 0).astype(np.float32)
            co = (b.T @ b).toarray()
            np.fill_diagonal(co, 0)
            count = np.asarray(b.sum(axis=0)).ravel()
            if name == "association":
                co /= np.maximum(count[:, None] + c.get("shrink", 10), 1)
            else:
                co /= np.maximum(np.sqrt(count[:, None] * count[None, :]), 1)
            keep = c.get("neighbors", 100)
            if keep < len(self.items):
                selected = np.argpartition(co, -keep, axis=1)[:, -keep:]
                rows = np.repeat(np.arange(len(self.items)), keep)
                graph = csr(
                    rows,
                    selected.ravel(),
                    co[rows, selected.ravel()],
                    (len(self.items), len(self.items)),
                )
            else:
                graph = sparse.csr_matrix(co)
            del co
            if name == "association":
                return (x[self.eval_users] @ graph).toarray()
            transition = normalize(graph, norm="l1", axis=1)
            dangling = np.asarray(graph.sum(axis=1)).ravel() == 0
            seed = normalize(x[self.eval_users], norm="l1").toarray()
            p = seed.copy()
            alpha = c.get("alpha", 0.5)
            for step in range(200):
                new = (
                    alpha
                    * (
                        (transition.T @ p.T).T
                        + p[:, dangling].sum(axis=1)[:, None] * seed
                    )
                    + (1 - alpha) * seed
                )
                residual = np.abs(new - p).sum(axis=1).max()
                p = new
                if residual < 1e-6:
                    break
            self.last_ppr = {
                "iterations": step + 1,
                "residual": float(residual),
                "mass_error_nonempty": float(
                    np.max(np.abs(p.sum(axis=1)[seed.sum(axis=1) > 0] - 1))
                ),
                "converged": bool(residual < 1e-6),
            }
            return p
        if name == "ease":
            # Train only items with positive training mass; zero-pad cold items.
            active = np.flatnonzero(np.asarray(x.sum(axis=0)).ravel() > 0)
            xa = x[:, active]
            g = (xa.T @ xa).toarray().astype(np.float64)
            g.flat[:: len(g) + 1] += c.get("reg", 100)
            p = linalg.inv(g, check_finite=False)
            weights = -p / np.diag(p)[None, :]
            np.fill_diagonal(weights, 0)
            scores = np.zeros((len(self.eval_users), len(self.items)), dtype=np.float32)
            scores[:, active] = xa[self.eval_users] @ weights
            return scores
        if name == "channel":
            return self.channel_scores(c.get("half", 28), c.get("hourly", False))
        raise ValueError(name)

    def channel_scores(self, half, hourly):
        v = np.load(ROOT / "data/processed/views.npz")
        valid = v["start"] < self.start
        u = v["user"][valid]
        ch = v["channel"][valid]
        s = v["start"][valid]
        e = np.minimum(v["end"][valid], self.start)
        # Old viewing times cannot become future features; end clipped at cutoff.
        channels = sorted(set(ch) | set(self.future.channel_id))
        cm = {c: j for j, c in enumerate(channels)}
        cidx = np.array([cm[c] for c in ch])
        n = len(channels)
        decay = (
            np.ones(len(s))
            if half is None
            else np.exp2(-(self.start - e) / (half * 86400))
        )
        if not hourly:
            h = csr(u, cidx, (e - s) * decay, (len(self.users), n))
            freq = csr(
                self.future.channel_id.map(cm),
                self.future.tv_show_id.map(self.im),
                np.ones(len(self.future)),
                (n, len(self.items)),
            )
        else:
            # Exact time spent in local clock-hour bins, no future viewing.
            # Resolve local hour per segment, including intervals crossing DST.
            h = sparse.csr_matrix((len(self.users), n * 24), dtype=np.float32)
            cursor = s.copy()
            for _ in range(100):
                active = cursor < e
                if not active.any():
                    break
                nxt = np.minimum((cursor // 3600 + 1) * 3600, e)
                hours = (
                    pd.to_datetime(cursor[active], unit="s", utc=True)
                    .tz_convert("Europe/Kyiv")
                    .hour.to_numpy()
                )
                bins = cidx[active] * 24 + hours
                h += csr(
                    u[active], bins, (nxt - cursor)[active] * decay[active], h.shape
                )
                cursor = nxt
            if np.any(cursor < e):
                raise ValueError("Unexpected long viewing interval")
            future = self.future
            hour = (
                pd.to_datetime(future.start, unit="s", utc=True)
                .dt.tz_convert("Europe/Kyiv")
                .dt.hour.to_numpy()
            )
            freq = csr(
                future.channel_id.map(cm).to_numpy() * 24 + hour,
                future.tv_show_id.map(self.im),
                np.ones(len(future)),
                (n * 24, len(self.items)),
            )
        h = normalize(h, norm="l1")
        return (h[self.eval_users] @ freq).toarray()

    def rank(self, scores):
        scores = np.asarray(scores).copy()
        scores[self.cold] = self.pop
        return [topk(row, self.items, self.eligible, self.pop) for row in scores]


def key(c):
    return (
        c["method"]
        + "_"
        + hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:10]
    )


def normalized(s):
    s = np.maximum(s, 0)
    return s / np.maximum(s.max(axis=1, keepdims=True), 1e-12)


def configurations():
    out = [{"method": "popularity"}]
    for half in [None, 14, 28, 56]:
        for threshold in [0.2, 0.5, 0.8]:
            out.append({"method": "repeat", "half": half, "threshold": threshold})
    out += [
        {"method": "repeat", "half": h, "threshold": 0.8, "single": True}
        for h in [None, 28]
    ]
    for half in [None, 28]:
        for title in [False, True]:
            out.append({"method": "content", "half": half, "title": title})
    for factors in [32, 64, 128]:
        for alpha in [10, 40]:
            for reg in [0.1, 1]:
                out.append(
                    {
                        "method": "als",
                        "factors": factors,
                        "alpha": alpha,
                        "reg": reg,
                        "iterations": 25,
                        "half": None,
                    }
                )
    for alpha in [0.3, 0.5, 0.85]:
        out.append({"method": "ppr", "alpha": alpha, "neighbors": 100})
    for shrink in [1, 10, 50]:
        out.append({"method": "association", "shrink": shrink, "neighbors": 100})
    for reg in [10, 100, 1000]:
        out.append({"method": "ease", "reg": reg})
    for half in [None, 14, 28, 56]:
        for hourly in [False, True]:
            out.append({"method": "channel", "half": half, "hourly": hourly})
    return out


def run_development():
    fold = Fold("2020-05-18", "2020-06-22")
    (OUT / "development-protocol.json").write_text(json.dumps(fold.audit, indent=2))
    print(json.dumps(fold.audit), flush=True)
    results = []
    best = {}
    source = b"".join(
        p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))
    )
    provenance = hashlib.sha256(
        source
        + (OUT / "data-manifest.json").read_bytes()
        + json.dumps(fold.audit, sort_keys=True).encode()
    ).hexdigest()
    cache_dir = CACHE / provenance
    cache_dir.mkdir(exist_ok=True)
    for c in configurations():
        k = key(c)
        path = cache_dir / (k + ".npy")
        timing = cache_dir / (k + ".json")
        t = time.perf_counter()
        if path.exists():
            scores = np.load(path)
            cached = True
        else:
            scores = fold.model(c)
            np.save(path, scores)
            cached = False
        duration = time.perf_counter() - t
        if cached:
            duration = json.loads(timing.read_text())["fit_score_seconds"]
        else:
            timing.write_text(
                json.dumps({"fit_score_seconds": duration, "provenance": provenance})
            )
        predictions = fold.rank(scores)
        metrics = evaluate(fold.truth, predictions)
        row = {
            "key": k,
            "config": c,
            "runtime_seconds": duration,
            "cached": cached,
            **metrics,
        }
        results.append(row)
        (OUT / "development-results.json").write_text(json.dumps(results, indent=2))
        score = row["map5_five_targets"]
        print(
            k,
            c,
            score,
            "nonempty",
            row["map5_nonempty"],
            "seconds",
            round(duration, 2),
            flush=True,
        )
        if c["method"] not in best or score > best[c["method"]]["map5_five_targets"]:
            best[c["method"]] = row
    (OUT / "development-best.json").write_text(json.dumps(best, indent=2))
    # A small documented grid of hybrid weights, tuned only on development.
    anchor = best["repeat"]
    anchor_s = normalized(np.load(cache_dir / (anchor["key"] + ".npy")))
    fusion = []
    for method in ["content", "als", "association", "ppr", "ease", "channel"]:
        other = best[method]
        other_s = normalized(np.load(cache_dir / (other["key"] + ".npy")))
        for weight in [0.1, 0.25, 0.5, 0.75]:
            scores = (1 - weight) * anchor_s + weight * other_s
            metrics = evaluate(fold.truth, fold.rank(scores))
            row = {
                "config": {
                    "method": "hybrid",
                    "components": [anchor["config"], other["config"]],
                    "weights": [1 - weight, weight],
                },
                **metrics,
            }
            fusion.append(row)
            print("hybrid", method, weight, row["map5_five_targets"], flush=True)
    winner = max(list(best.values()) + fusion, key=lambda x: x["map5_five_targets"])
    (OUT / "development-hybrids.json").write_text(json.dumps(fusion, indent=2))
    frozen = {
        "selected": winner["config"],
        "development_map5": winner["map5_five_targets"],
        "best_individuals": {k: v["config"] for k, v in best.items()},
        "selection": "five-target template users; development only",
        "development": ["2020-05-18", "2020-06-22"],
        "audit": ["2020-06-22", "2020-07-27"],
    }
    (OUT / "phase1-selection.json").write_text(json.dumps(frozen, indent=2))
    print("FROZEN", json.dumps(frozen), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run_development()
