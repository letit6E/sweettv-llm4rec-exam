from types import SimpleNamespace
import numpy as np
import pandas as pd
from sweettv import discovery_context as dc


def test_title_family_keeps_category_and_removes_episode():
    assert dc.title_key("Т/с 'Доктор Кто', 5 сезон, 3 с.", "Сериалы") == dc.title_key(
        "Т/с 'Доктор Кто', 1 сезон, 1 с.", "Сериалы"
    )
    assert dc.title_key("'Сестры'", "Фильмы") != dc.title_key("'Сестры'", "Сериалы")
    assert dc.title_key("Новости, 2 сезон, 3 эп.", "Инфо") == "Инфо:новости"
    assert dc.title_key("", "") == ""


def test_zero_context_excludes_future_unknown_family_and_other_users(monkeypatch):
    cutoff = 20_000_000
    f = SimpleNamespace(
        start=cutoff,
        items=np.array([10, 20]),
        im={10: 0, 20: 1},
        eval_users=np.array([3]),
        past=pd.DataFrame(
            {
                "tv_show_id": [10, 20],
                "tv_show_title": ["Т/с 'Пример'.", "Т/с 'Другой'."],
                "tv_show_category": ["Сериалы", "Сериалы"],
            }
        ),
    )
    key = dc.title_key("Т/с 'Пример'.", "Сериалы")
    events = pd.DataFrame(
        {
            "user": [3, 3, 3, 4, 3],
            "end": [
                cutoff - 1,
                cutoff + 1,
                cutoff - 1,
                cutoff - 1,
                cutoff - 120 * 86400,
            ],
            "family": [key, key, "unknown", key, key],
            "seconds": [90, 999, 999, 999, 999],
            "fraction": [0.9, 1, 1, 1, 1],
        }
    )
    monkeypatch.setattr(dc, "prepare", lambda: events)
    result = dc.context_scores(f)
    np.testing.assert_array_equal(result["zero_28_full"], [[1, 0]])
    np.testing.assert_allclose(result["zero_112_fraction"], [[0.9, 0]])
    np.testing.assert_allclose(result["zero_28_seconds"], [[1.5, 0]])
    # More future evidence cannot affect a historical feature.
    events.loc[1, "seconds"] = 1e9
    np.testing.assert_array_equal(dc.context_scores(f)["zero_28_full"], [[1, 0]])


def test_soft_training_ties_distribute_only_available_top5_slots():
    from sweettv.tie_training import soft_targets

    ids = [10, 20, 30, 40, 50, 60]
    events = pd.DataFrame(
        {"user": [3] * 12, "tv_show_id": np.repeat(ids, [4, 3, 2, 1, 1, 1])}
    )
    f = SimpleNamespace(
        eval_users=np.array([3]),
        im={v: j for j, v in enumerate(ids)},
        truth_events=events,
    )
    y = soft_targets(f, np.zeros(6, dtype=int), np.arange(6))
    np.testing.assert_allclose(y, [1, 1, 1, 2 / 3, 2 / 3, 2 / 3])
    np.testing.assert_allclose(y.sum(), 5)
