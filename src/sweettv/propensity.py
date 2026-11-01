"""Historical qualifying-view probability per scheduled channel/time slot."""

import numpy as np
import pandas as pd
from .experiment import csr


def calendar_bins(df, cm, mode):
    c = df.channel_id.map(cm).to_numpy()
    t = pd.to_datetime(df.start, unit="s", utc=True).dt.tz_convert("Europe/Kyiv")
    if mode == "channel":
        return c, len(cm)
    if mode == "hour":
        return c * 12 + t.dt.hour.to_numpy() // 2, len(cm) * 12
    if mode == "weekday":
        return c * 56 + t.dt.dayofweek.to_numpy() * 8 + t.dt.hour.to_numpy() // 3, len(
            cm
        ) * 56
    raise ValueError(mode)


def propensity_features(f, u, i):
    """All denominators are historical schedule, all numerators past views.

    Pool sparse time bins toward the user's channel rate. Forecast known future
    schedule opportunities; express counts per five weeks for horizon transfer.
    """
    chans = np.union1d(f.past.channel_id, f.future.channel_id)
    cm = {int(c): j for j, c in enumerate(chans)}
    rowmap = {int(v): j for j, v in enumerate(f.eval_users)}
    vals = []
    names = []
    shape = (len(f.eval_users), len(f.items))
    weeks = (f.end - f.start) / 604800
    for days in [28, 112]:
        past = f.past[f.past.end >= f.start - days * 86400]
        t = f.train[
            (f.train.end >= f.start - days * 86400) & f.train.user.isin(rowmap)
        ].copy()
        rows = t.user.map(rowmap).to_numpy()
        channel_air = np.bincount(past.channel_id.map(cm), minlength=len(cm))
        for threshold in [0.2, 0.8]:
            m = t.fraction.to_numpy() >= threshold
            channel_counts = csr(
                rows[m],
                t.channel_id.map(cm).to_numpy()[m],
                np.ones(m.sum()),
                (shape[0], len(cm)),
            ).toarray()
            channel_rate = channel_counts / np.maximum(channel_air, 1)
            for mode in ["channel", "hour", "weekday"]:
                bins, n = calendar_bins(past, cm, mode)
                total = np.bincount(bins, minlength=n)
                bins, _ = calendar_bins(t, cm, mode)
                num = csr(rows[m], bins[m], np.ones(m.sum()), (shape[0], n)).toarray()
                bins, nb = calendar_bins(f.future, cm, mode)
                future = csr(
                    bins,
                    f.future.tv_show_id.map(f.im),
                    np.ones(len(bins)),
                    (n, shape[1]),
                )
                prior = np.repeat(channel_rate, n // len(cm), axis=1)
                for shrink in [5, 30]:
                    rate = (num + shrink * prior) / (total + shrink)
                    expected = np.asarray(rate @ future) * 5 / weeks
                    vals.append(np.log1p(expected[u, i]))
                    names.append(f"prop_{days}_{threshold}_{mode}_{shrink}")
                    # Relative preference among all programs competes naturally within each user.
    x = np.column_stack(vals).astype("float32")
    assert np.isfinite(x).all()
    return x, names


def repeat_channel_features(f, u, i):
    """Adjust repeats for future availability on the SAME channel, with shrinkage."""
    vals = []
    names = []
    pairs = (
        pd.concat(
            [
                f.past[["channel_id", "tv_show_id"]],
                f.future[["channel_id", "tv_show_id"]],
            ]
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    pairs["slot"] = np.arange(len(pairs))
    pmap = {
        (int(c), int(it)): int(s)
        for c, it, s in pairs.itertuples(index=False, name=None)
    }

    def idx(t):
        return np.array(
            [pmap[(int(c), int(it))] for c, it in zip(t.channel_id, t.tv_show_id)]
        )

    future = np.bincount(idx(f.future), minlength=len(pairs))
    aggregate = csr(
        np.arange(len(pairs)),
        pairs.tv_show_id.map(f.im),
        future,
        (len(pairs), len(f.items)),
    )
    cm = {int(c): j for j, c in enumerate(pairs.channel_id.unique())}
    pc = pairs.channel_id.map(cm).to_numpy()
    rmap = {int(v): j for j, v in enumerate(f.eval_users)}
    for days in [28, 56, 112]:
        past = f.past[f.past.end >= f.start - days * 86400]
        den = np.bincount(idx(past), minlength=len(pairs))
        chden = np.bincount(past.channel_id.map(cm), minlength=len(cm))
        t = f.train[(f.train.end >= f.start - days * 86400) & f.train.user.isin(rmap)]
        rows = t.user.map(rmap).to_numpy()
        bins = idx(t)
        ch = t.channel_id.map(cm).to_numpy()
        for th in [0.2, 0.8]:
            m = t.fraction.to_numpy() >= th
            num = csr(
                rows[m], bins[m], np.ones(m.sum()), (len(rmap), len(pairs))
            ).toarray()
            channel = csr(
                rows[m], ch[m], np.ones(m.sum()), (len(rmap), len(cm))
            ).toarray() / np.maximum(chden, 1)
            for shrink in [0, 5, 30]:
                rate = (num + shrink * channel[:, pc]) / np.maximum(den + shrink, 1)
                expected = (
                    np.asarray(rate @ aggregate) * 5 / ((f.end - f.start) / 604800)
                )
                vals.append(np.log1p(expected[u, i]))
                names.append(f"repeat_channel_{days}_{th}_{shrink}")
    x = np.column_stack(vals).astype("float32")
    assert np.isfinite(x).all()
    return x, names
