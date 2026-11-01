"""Rebuild final features without the cache and reproduce a scored CSV."""

import json, pickle, hashlib, time
import numpy as np
from threadpoolctl import threadpool_limits
from sweettv.improve import (
    cached,
    DEST,
    WINDOWS,
    generate,
    build_features,
    neighbor_scores,
    extra_features,
)
from sweettv.experiment import Fold, ROOT
from sweettv.horizon import rescale

with threadpool_limits(limits=1):
    t = time.perf_counter()
    f = Fold(*WINDOWS["final"])
    s, r = generate(f)
    for n, v in neighbor_scores(f).items():
        s[n] = v
        a = v.copy()
        a[:, ~f.eligible] = -np.inf
        o = np.argsort(-a, axis=1, kind="stable")
        rank = np.empty_like(o, dtype="int32")
        np.put_along_axis(rank, o, np.arange(len(f.items))[None, :], axis=1)
        r[n] = rank
    x, u, i, names, c = build_features(f, s, r)
    e, en = extra_features(f, u, i)
    x = np.column_stack([x, e])
    names += en
    old = cached("final")
    assert (
        names == old["names"].tolist()
        and np.array_equal(u, old["u"])
        and np.array_equal(i, old["i"])
    )
    diff = float(np.max(np.abs(x - old["X"])))
    np.testing.assert_allclose(x, old["X"], rtol=1e-5, atol=1e-6)
    with (DEST / "v2_cat200.pkl").open("rb") as h:
        model = pickle.load(h)
    score = np.full((len(f.eval_users), len(f.items)), -1e9, dtype="float32")
    score[u, i] = model.predict_proba(rescale(x, names, 12))[:, 1]
    frame = f.template.copy()
    frame["tv_show_id"] = [" ".join(map(str, p)) for p in f.rank(score)]
    payload = frame.to_csv(index=False).encode()
    actual = (ROOT / "submissions/v2_cat200_horizon.csv").read_bytes()
    assert payload == actual
    result = {
        "independent_feature_rebuild": True,
        "feature_max_absolute_difference": diff,
        "byte_identical_csv": "v2_cat200_horizon.csv",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "seconds": time.perf_counter() - t,
    }
    (DEST / "reproduction-check.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)
