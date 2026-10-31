import numpy as np
import pytest
from sweettv.metrics import apk, evaluate, topk
from sweettv.intervals import AiringIndex, watched_seconds, union_intervals


def test_ap_hand_example():
    # Hits at ranks 1 and 3: (1 + 2/3) / 2.
    assert apk([7, 9], [7, 2, 9, 4, 5]) == pytest.approx(5 / 6)
    assert apk([7, 9], [7, 7, 9, 4, 5]) == pytest.approx(5 / 6)
    assert apk([], [7, 2, 9, 4, 5]) == 0
    assert apk([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5]) == 1


def test_nonempty_cohort_not_silently_dropped():
    r = evaluate([[1], []], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    assert r["map5_all"] == 0.5 and r["map5_nonempty"] == 1 and r["users"] == 2


def test_ranking_ties_filter_zero_and_nan():
    ids = np.array([0, 9, 8, 7, 6, 5, 4])
    out = topk(np.zeros(7), ids, np.ones(7, bool), [99, 0, 0, 1, 0, 0, 0])
    assert out == [7, 4, 5, 6, 8]
    with pytest.raises(ValueError):
        topk([np.nan] * 7, ids, np.ones(7, bool), np.zeros(7))


def test_disjoint_half_open_overlaps_and_long_session():
    ix = AiringIndex([10, 20, 40], [20, 35, 45], [1, 2, 3])
    assert ix.overlaps(20, 40) == [(2, 20, 35)]
    assert ix.overlaps(5, 50) == [(1, 10, 20), (2, 20, 35), (3, 40, 45)]
    assert ix.overlaps(50, 60) == []


def test_nested_airings_are_not_lost():
    ix = AiringIndex([0, 10, 20], [100, 15, 25], [1, 2, 3])
    assert ix.overlaps(30, 40) == [(1, 30, 40)]


def test_duplicate_views_cannot_create_80_percent_watch():
    assert watched_seconds([(0, 40), (0, 40), (20, 50)], 0, 100) == 50
    assert watched_seconds([(0, 40), (40, 80)], 0, 100) == 80
    assert watched_seconds([(-20, 110)], 0, 100) == 100
    assert union_intervals([(10, 10), (15, 12)]) == []


def test_index_matches_brute_force_random_fixture():
    rng = np.random.default_rng(8)
    starts = rng.integers(0, 1000, 100)
    ends = starts + rng.integers(1, 400, 100)
    ix = AiringIndex(starts, ends, np.arange(100))
    for _ in range(100):
        s = int(rng.integers(0, 1200))
        e = s + int(rng.integers(1, 200))
        expected = sorted(
            (j, max(s, int(starts[j])), min(e, int(ends[j])))
            for j in range(100)
            if starts[j] < e and ends[j] > s
        )
        assert sorted(ix.overlaps(s, e)) == expected


def test_dst_duration_is_elapsed_time():
    from sweettv.experiment import epoch

    # Kyiv advances from UTC+2 to UTC+3 on 29 March 2020.
    assert epoch("2020-03-29 04:30") - epoch("2020-03-29 02:30") == 3600


def tiny_fold():
    import pandas as pd
    from sweettv.experiment import Fold

    f = Fold.__new__(Fold)
    f.start = 1000
    f.items = np.arange(1, 7)
    f.im = {int(i): j for j, i in enumerate(f.items)}
    f.shape = (4, 6)
    f.eval_users = np.arange(4)
    rows = [(0, 1), (0, 2), (1, 1), (1, 2), (1, 3), (2, 4)]
    f.train = pd.DataFrame(rows, columns=["user", "tv_show_id"])
    f.train["fraction"] = 1.0
    f.train["single_fraction"] = 1.0
    f.train["end"] = 900
    f.train["seconds"] = 80
    f.base = f.matrix()
    f.pop = np.asarray((f.base > 0).sum(axis=0)).ravel()
    f.cold = np.array([False, False, False, True])
    f.eligible = np.ones(6, bool)
    return f


def test_ppr_probability_and_dangling_nodes():
    f = tiny_fold()
    s = f.model({"method": "ppr", "alpha": 0.85, "neighbors": 6})
    assert f.last_ppr["converged"]
    assert np.allclose(s[:3].sum(axis=1), 1, atol=2e-6)
    assert np.isclose(s[2, 3], 1)  # Isolated item redistributes to seed.
    assert np.all(s[3] == 0)
    assert f.rank(s)[3] == f.rank(f.model({"method": "popularity"}))[3]


def test_content_cannot_read_future_metadata():
    import pandas as pd

    f = tiny_fold()
    f.past = pd.DataFrame(
        {
            "tv_show_id": [1, 2, 3, 4],
            "start": [1] * 4,
            "tv_show_title": ["a", "b", "c", "d"],
            "tv_show_category": ["film", "film", "news", "kids"],
        }
    )
    for name in [
        "tv_show_genre_1",
        "tv_show_genre_2",
        "tv_show_genre_3",
        "director",
        "actors",
        "year_of_production",
    ]:
        f.past[name] = ""
    f.future = pd.DataFrame(
        {"tv_show_id": [5, 6], "tv_show_title": ["future sentinel", "future sentinel"]}
    )
    s = f.model({"method": "content", "title": True})
    f.future["tv_show_title"] = "changed unseen content"
    assert np.array_equal(s, f.model({"method": "content", "title": True}))
    assert np.all(s[:, 4:] == 0)
