"""Inspect source data and construct auditable viewing/airing intersections."""

from pathlib import Path
import json, time, platform, sys
import numpy as np
import pandas as pd
from .intervals import AiringIndex

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
ART = ROOT / "artifacts"
META = [
    "tv_show_title",
    "tv_show_category",
    "tv_show_genre_1",
    "tv_show_genre_2",
    "tv_show_genre_3",
    "year_of_production",
    "director",
    "actors",
]


def seconds(series, fmt):
    return (
        pd.to_datetime(series, format=fmt, errors="raise")
        .dt.tz_localize("Europe/Kyiv", nonexistent="raise", ambiguous="raise")
        .astype("int64")
        .to_numpy()
        // 10**9
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    parts = []
    audit = {
        "schedules": {},
        "timezone": "Europe/Kyiv; inferred, all 5821987 duration differences agree; date summaries below in UTC",
    }
    for name in [
        "export_arh_11-20-final.csv",
        "export_arh_21-30-final.csv",
        "export_arh_31-42-final.csv",
    ]:
        is_test = "31-42" in name
        cols = ["channel_id", "start_time", "duration", "tv_show_id"] + (
            [] if is_test else META
        )
        counts = 0
        labelled = 0
        lo = 2**63 - 1
        hi = 0
        chunkparts = []
        for chunk in pd.read_csv(
            RAW / name, usecols=cols, chunksize=200000, keep_default_na=False
        ):
            st = seconds(chunk.start_time, "%d.%m.%Y %H:%M:%S")
            en = st + chunk.duration.to_numpy(dtype=np.int64)
            counts += len(chunk)
            lo = min(lo, int(st.min()))
            hi = max(hi, int(en.max()))
            chunk["start"] = st
            chunk["end"] = en
            chunk = chunk[(chunk.tv_show_id > 0) & (chunk.duration > 0)].copy()
            labelled += len(chunk)
            chunk["source"] = name
            chunkparts.append(chunk.drop(columns="start_time"))
        part = pd.concat(chunkparts, ignore_index=True)
        audit["schedules"][name] = {
            "rows": counts,
            "labelled_rows": labelled,
            "labelled_fraction": labelled / counts,
            "programs": int(part.tv_show_id.nunique()),
            "start": str(pd.Timestamp(lo, unit="s")),
            "end": str(pd.Timestamp(hi, unit="s")),
        }
        print(name, audit["schedules"][name], flush=True)
        parts.append(part)
    sched = pd.concat(parts, ignore_index=True).fillna("")
    keys = ["channel_id", "start", "end", "tv_show_id"]
    audit["exact_duplicate_airings"] = int(sched.duplicated(keys).sum())
    sched = (
        sched.drop_duplicates(keys)
        .sort_values(["start", "channel_id", "tv_show_id"])
        .reset_index(drop=True)
    )
    sched["airing_id"] = np.arange(len(sched))
    sched.to_pickle(OUT / "schedule.pkl")
    test_start = int(pd.Timestamp("2020-07-27", tz="Europe/Kyiv").timestamp())
    audit["test_start"] = (
        "2020-07-27 00:00:00 Europe/Kyiv; inferred from all duration checks"
    )
    print("read views", flush=True)
    views = pd.read_csv(
        RAW / "dataset11-30.csv",
        dtype={"user_id": str, "vsetv_id": "int32", "duraton": "int32"},
    )
    users, labels = pd.factorize(views.user_id, sort=True)
    user_ids = labels.tolist()
    st = seconds(views.start_time, "%Y-%m-%d %H:%M:%S")
    en = seconds(views.stop_time, "%Y-%m-%d %H:%M:%S")
    ch = views.vsetv_id.to_numpy()
    audit["views"] = {
        "rows": len(views),
        "users": len(labels),
        "channels": int(views.vsetv_id.nunique()),
        "start": str(pd.Timestamp(st.min(), unit="s")),
        "end": str(pd.Timestamp(en.max(), unit="s")),
        "duration_quantiles": {
            str(k): float(v)
            for k, v in views.duraton.quantile(
                [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
            ).items()
        },
        "duration_disagreements": int(
            np.count_nonzero(en - st != views.duraton.to_numpy())
        ),
        "nonpositive_intervals": int(np.count_nonzero(en <= st)),
        "starts_at_or_after_test": int(np.count_nonzero(st >= test_start)),
        "cross_test_boundary": int(
            np.count_nonzero((st < test_start) & (en > test_start))
        ),
    }
    print(audit["views"], flush=True)
    en = np.minimum(en, test_start)
    keep = (st < test_start) & (en > st)
    np.savez_compressed(
        OUT / "views.npz",
        user=users[keep].astype(np.int32),
        channel=ch[keep],
        start=st[keep],
        end=en[keep],
    )
    (OUT / "users.json").write_text(json.dumps(user_ids))
    del views
    # Entire source viewing log is before test; never join it against test airings.
    train_sched = sched[
        (sched.end <= test_start) & (~sched.source.str.contains("31-42"))
    ]
    indices = {
        int(channel): AiringIndex(
            g.start.to_numpy(), g.end.to_numpy(), g.airing_id.to_numpy()
        )
        for channel, g in train_sched.groupby("channel_id")
    }
    records = []
    matched = 0
    for i, (u, c, s, e) in enumerate(zip(users, ch, st, en)):
        if not keep[i]:
            continue
        index = indices.get(int(c))
        matches = index.overlaps(int(s), int(e)) if index is not None else []
        matched += bool(matches)
        records.extend((int(u), a, l, r) for a, l, r in matches)
        if (i + 1) % 1000000 == 0:
            print(
                "matched",
                i + 1,
                "view rows;",
                len(records),
                "intersections",
                flush=True,
            )
    audit["matched_view_rows"] = matched
    audit["positive_overlap_segments"] = len(records)
    print("aggregate unioned overlaps", len(records), flush=True)
    a = np.asarray(records, dtype=np.int64)
    del records
    a = a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]
    result = []
    pu = pa = -1
    left = right = total = max_single = segments = 0
    for u, air, l, r in a:
        if u != pu or air != pa:
            if pa != -1:
                result.append((pu, pa, total + right - left, max_single, segments))
            pu, pa, left, right, total, max_single, segments = u, air, l, r, 0, r - l, 1
        else:
            max_single = max(max_single, r - l)
            segments += 1
            if l <= right:
                right = max(right, r)
            else:
                total += right - left
                left, right = l, r
    if pa != -1:
        result.append((pu, pa, total + right - left, max_single, segments))
    events = pd.DataFrame(
        result,
        columns=["user", "airing_id", "seconds", "max_single_seconds", "segments"],
    )
    events = events.merge(
        sched[["airing_id", "tv_show_id", "channel_id", "start", "end", "duration"]],
        on="airing_id",
        validate="many_to_one",
    )
    events["fraction"] = events.seconds / events.duration
    events["single_fraction"] = events.max_single_seconds / events.duration
    assert events.fraction.max() <= 1 and events.fraction.min() > 0
    events.to_pickle(OUT / "events.pkl")
    audit["interactions"] = {
        "user_airings": len(events),
        "full_80pct": int((events.fraction >= 0.8).sum()),
        "single_log_80pct": int((events.single_fraction >= 0.8).sum()),
        "programs": int(events.tv_show_id.nunique()),
        "users": int(events.user.nunique()),
        "multi_segment_user_airings": int((events.segments > 1).sum()),
        "fraction_quantiles": {
            str(k): float(v)
            for k, v in events.fraction.quantile(
                [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
            ).items()
        },
    }
    template = pd.read_csv(RAW / "submission.csv", dtype=str)
    assert template.user_id.is_unique and len(template) == 1260
    audit["template"] = {
        "users": len(template),
        "absent_from_views": len(set(template.user_id) - set(user_ids)),
    }
    audit["elapsed_seconds"] = time.perf_counter() - start
    audit["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (ART / "data-audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
