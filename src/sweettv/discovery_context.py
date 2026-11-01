"""Past-only title-family context, including otherwise discarded ID-zero airings.

Family evidence is a feature, never a replacement label or an inferred target ID.
Future titles and future views are not read.
"""

import json
import re
import time
import numpy as np
import pandas as pd
from .experiment import ROOT, OUT, csr
from .prepare import seconds
from .intervals import AiringIndex

DEST = OUT / "discovery-v6"
DEST.mkdir(exist_ok=True)


def title_key(title, category):
    title = str(title).strip().lower().replace("ё", "е")
    quoted = re.search(r"['\"](.+?)['\"]", title)
    if quoted:
        title = quoted.group(1)
    else:
        title = re.split(r",?\s*\d+\s*(?:сезон|с\.|эп\.)", title)[0]
    title = re.sub(r"[^\w]+", " ", title).strip()
    return str(category).strip() + ":" + title if len(title) >= 4 else ""


def prepare():
    path = DEST / "zero_context.pkl"
    if path.exists():
        return pd.read_pickle(path)
    t0 = time.perf_counter()
    # Vocabulary lookup only; a fold subsequently matches families to its past IDs.
    all_sched = pd.read_pickle(ROOT / "data/processed/schedule.pkl")
    known = all_sched[~all_sched.source.str.contains("31-42")]
    keys = {
        title_key(a, b) for a, b in zip(known.tv_show_title, known.tv_show_category)
    } - {""}
    pieces = []
    for name in ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv"]:
        s = pd.read_csv(
            ROOT / "data/raw" / name,
            usecols=[
                "channel_id",
                "start_time",
                "duration",
                "tv_show_id",
                "tv_show_title",
                "tv_show_category",
            ],
            keep_default_na=False,
        )
        s = s[(s.tv_show_id == 0) & (s.duration > 0)].copy()
        pairs = s[["tv_show_title", "tv_show_category"]].drop_duplicates()
        keymap = {
            (a, b): title_key(a, b) for a, b in pairs.itertuples(index=False, name=None)
        }
        s["family"] = [
            keymap[(a, b)] for a, b in zip(s.tv_show_title, s.tv_show_category)
        ]
        s = s[s.family.isin(keys)].copy()
        s["start"] = seconds(s.start_time, "%d.%m.%Y %H:%M:%S")
        s["end"] = s.start + s.duration
        pieces.append(s[["channel_id", "start", "end", "duration", "family"]])
    schedule = pd.concat(pieces).drop_duplicates().reset_index(drop=True)
    schedule["air"] = np.arange(len(schedule))
    raw = np.load(ROOT / "data/processed/views.npz")
    records = []
    for channel, g in schedule.groupby("channel_id"):
        ix = AiringIndex(g.start.to_numpy(), g.end.to_numpy(), g.air.to_numpy())
        mask = raw["channel"] == channel
        for u, s, e in zip(raw["user"][mask], raw["start"][mask], raw["end"][mask]):
            records.extend((int(u), a, l, r) for a, l, r in ix.overlaps(int(s), int(e)))
    # Union overlapping sessions within a contextual airing.
    a = np.array(records, dtype=np.int64)
    a = a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]
    rows = []
    pu = pa = -1
    left = right = total = 0
    for u, air, l, r in a:
        if u != pu or air != pa:
            if pa != -1:
                rows.append((pu, pa, total + right - left))
            pu, pa, left, right, total = u, air, l, r, 0
        elif l <= right:
            right = max(right, r)
        else:
            total += right - left
            left, right = l, r
    if pa != -1:
        rows.append((pu, pa, total + right - left))
    events = pd.DataFrame(rows, columns=["user", "air", "seconds"]).merge(
        schedule, on="air", validate="many_to_one"
    )
    events["fraction"] = events.seconds / events.duration
    assert events.fraction.between(0, 1).all()
    events.to_pickle(path)
    (DEST / "zero-context-audit.json").write_text(
        json.dumps(
            {
                "candidate_airings": len(schedule),
                "overlap_events": len(events),
                "full_context_events": int((events.fraction >= 0.8).sum()),
                "seconds": time.perf_counter() - t0,
                "policy": "Context only. Each fold uses events and family-to-program mappings ending before cutoff. No ID-zero row is used as a labelled item.",
            },
            indent=2,
        )
    )
    print("ZERO_CONTEXT", len(events), round(time.perf_counter() - t0, 1), flush=True)
    return events


def context_scores(f):
    meta = (
        f.past[["tv_show_id", "tv_show_title", "tv_show_category"]]
        .drop_duplicates()
        .copy()
    )
    meta["family"] = [
        title_key(a, b) for a, b in zip(meta.tv_show_title, meta.tv_show_category)
    ]
    meta = meta[meta.family != ""][["tv_show_id", "family"]].drop_duplicates()
    family = {v: j for j, v in enumerate(sorted(meta.family.unique()))}
    mapping = csr(
        meta.family.map(family),
        meta.tv_show_id.map(f.im),
        np.ones(len(meta)),
        (len(family), len(f.items)),
    )
    # Divide by number of aliases so numerous title variants cannot inflate an item.
    mapping = mapping.multiply(
        1 / np.maximum(np.asarray(mapping.sum(axis=0)).ravel(), 1)
    ).tocsr()
    t = prepare()
    t = t[(t.end <= f.start) & t.family.isin(family)].copy()
    rmap = {int(v): j for j, v in enumerate(f.eval_users)}
    t = t[t.user.isin(rmap)]
    out = {}
    for days in [28, 112]:
        p = t[t.end >= f.start - days * 86400]
        for mode in ["full", "fraction", "seconds"]:
            vals = (
                (p.fraction.to_numpy() >= 0.8).astype(float)
                if mode == "full"
                else p.fraction.to_numpy()
                if mode == "fraction"
                else p.seconds.to_numpy() / 60
            )
            x = csr(
                p.user.map(rmap), p.family.map(family), vals, (len(rmap), len(family))
            )
            out[f"zero_{days}_{mode}"] = (x @ mapping).toarray()
    return out
