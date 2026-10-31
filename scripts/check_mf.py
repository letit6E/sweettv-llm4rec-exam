import json, numpy as np, warnings
from sweettv.experiment import Fold, SEED, OUT
from implicit.cpu.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits

with threadpool_limits(1):
    f = Fold("2020-05-18", "2020-06-22")
    x = f.matrix()
    x.data = 1 + 10 * np.log1p(x.data)
    m = AlternatingLeastSquares(
        factors=32,
        regularization=1,
        iterations=25,
        alpha=1,
        random_state=SEED,
        num_threads=4,
    )
    m.fit(x, show_progress=False)
    a = m.user_factors[f.eval_users]
    b = m.item_factors
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        dot = a @ b.T
    exact = np.einsum(
        "uf,if->ui", a.astype("float64"), b.astype("float64"), optimize=False
    )
    assert np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(dot).all()
    result = {
        "factor_max_abs": [float(abs(a).max()), float(abs(b).max())],
        "warnings": [str(w.message) for w in ws],
        "max_abs_difference_float64_einsum": float(abs(dot - exact).max()),
        "top5_identical": f.rank(dot) == f.rank(exact),
        "score_range": [float(dot.min()), float(dot.max())],
    }
    (OUT / "mf-numerical-audit.json").write_text(json.dumps(result, indent=2))
    print(result)
