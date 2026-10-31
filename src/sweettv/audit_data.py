"""Independent direct interval checks and source-data caveats."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    s = pd.read_pickle(ROOT / "data/processed/schedule.pkl")
    e = pd.read_pickle(ROOT / "data/processed/events.pkl")
    v = np.load(ROOT / "data/processed/views.npz")
    rng = np.random.default_rng(20261031)
    sample = e.iloc[rng.choice(len(e), 200, replace=False)]
    # Deliberately does not import AiringIndex or union_intervals.
    maxdiff = 0
    for row in sample.itertuples():
        m = (
            (v["user"] == row.user)
            & (v["channel"] == row.channel_id)
            & (v["start"] < row.end)
            & (v["end"] > row.start)
        )
        left = np.maximum(v["start"][m], row.start)
        right = np.minimum(v["end"][m], row.end)
        boundary = sorted(set(left.tolist() + right.tolist()))
        total = 0
        for a, b in zip(boundary, boundary[1:]):
            if np.any((left <= a) & (right >= b)):
                total += b - a
        maxdiff = max(maxdiff, abs(total - row.seconds))
    assert maxdiff == 0
    train = s[~s.source.str.contains("31-42")].sort_values(
        ["channel_id", "start", "end"]
    )
    overlaps = 0
    examples = []
    for ch, g in train.groupby("channel_id"):
        starts = g.start.to_numpy()
        ends = g.end.to_numpy()
        p = np.maximum.accumulate(ends)
        where = np.flatnonzero(starts[1:] < p[:-1]) + 1
        overlaps += len(where)
        if len(examples) < 8:
            for idx in where[:2]:
                examples.append(
                    {
                        "channel": int(ch),
                        "airing_id": int(g.iloc[idx].airing_id),
                        "start": int(starts[idx]),
                        "earlier_max_end": int(p[idx - 1]),
                    }
                )
    out = {
        "independent_interval_checks": 200,
        "max_seconds_difference": int(maxdiff),
        "passed": True,
        "labelled_training_airings_overlapping_prior_airings_on_same_channel": overlaps,
        "examples": examples,
        "handling": "Every positive intersection is retained; exact duplicate airings removed. Ambiguous overlapping schedule rows are disclosed, not silently resolved.",
    }
    (ROOT / "artifacts/independent-data-checks.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
