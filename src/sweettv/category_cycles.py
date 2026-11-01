"""Expose separate category affinity instead of one mixed content cosine."""

import numpy as np
from .experiment import csr


def category_features(f, u, i):
    meta = (
        f.past.sort_values("end")
        .drop_duplicates("tv_show_id", keep="last")
        .set_index("tv_show_id")
    )
    # Fixed names define schema consistently across windows; missing is its own category.
    categories = [
        "Инфо",
        "Познавательное",
        "Развлечения",
        "Сериалы",
        "Детям",
        "Другое",
        "Спорт",
        "Музыка",
        "Фильмы",
    ]
    labels = meta.tv_show_category.reindex(f.items).fillna("").astype(str).to_numpy()
    cm = {k: j for j, k in enumerate(categories)}
    idx = np.array([cm.get(k, 9) for k in labels])
    nc = 10
    vals = [(idx[i] == j).astype(float) for j in range(nc)]
    names = ["category_" + str(j) for j in range(nc)]
    rows_map = {int(k): j for j, k in enumerate(f.eval_users)}
    for days in [28, 112]:
        t = f.train[
            (f.train.end >= f.start - days * 86400) & f.train.user.isin(rows_map)
        ]
        rows = t.user.map(rows_map).to_numpy()
        cols = idx[t.item_idx.to_numpy()]
        for field in ["full", "partial", "seconds"]:
            v = (
                (t.fraction.to_numpy() >= 0.8).astype(float)
                if field == "full"
                else np.ones(len(t))
                if field == "partial"
                else t.seconds.to_numpy() / 60
            )
            a = csr(rows, cols, v, (len(rows_map), nc)).toarray()
            share = a / np.maximum(a.sum(axis=1, keepdims=True), 1)
            vals.append(share[u, idx[i]])
            names.append(f"category_share_{days}_{field}")
            vals.append(np.log1p(a[u, idx[i]]))
            names.append(f"category_count_{days}_{field}")
    return np.column_stack(vals).astype("float32"), names
