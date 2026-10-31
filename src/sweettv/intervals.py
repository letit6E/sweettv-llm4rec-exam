"""Half-open source-clock intervals, without guessing a timezone."""

from bisect import bisect_left, bisect_right
import numpy as np


def union_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def watched_seconds(intervals, airing_start, airing_end):
    return sum(
        e - s
        for s, e in union_intervals(
            (max(s, airing_start), min(e, airing_end)) for s, e in intervals
        )
    )


class AiringIndex:
    def __init__(self, starts, ends, ids):
        order = np.argsort(starts, kind="stable")
        self.starts = np.asarray(starts)[order]
        self.ends = np.asarray(ends)[order]
        self.ids = np.asarray(ids)[order]
        if np.any(self.ends <= self.starts):
            raise ValueError("Nonpositive airing duration")
        self.prefix_end = (
            np.maximum.accumulate(self.ends) if len(order) else np.array([])
        )

    def overlaps(self, start, end):
        if end <= start:
            return []
        lo = bisect_right(self.prefix_end, start)
        hi = bisect_left(self.starts, end)
        return [
            (
                int(self.ids[j]),
                max(start, int(self.starts[j])),
                min(end, int(self.ends[j])),
            )
            for j in range(lo, hi)
            if self.ends[j] > start
        ]
