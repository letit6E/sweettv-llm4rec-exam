from types import SimpleNamespace
import numpy as np
import pandas as pd
from sweettv.experiment import epoch
from sweettv.propensity import (
    calendar_bins,
    propensity_features,
    repeat_channel_features,
)


def test_calendar_bins_use_local_time_across_dst():
    df = pd.DataFrame(
        {
            "channel_id": [5, 5],
            "start": [epoch("2020-03-28 12:00"), epoch("2020-03-30 12:00")],
        }
    )
    bins, n = calendar_bins(df, {5: 0}, "hour")
    assert bins.tolist() == [6, 6] and n == 12


def fixture(repeat_future=1):
    start = epoch("2020-05-18")
    past = pd.DataFrame(
        {
            "channel_id": [1, 2],
            "tv_show_id": [10, 20],
            "start": [start - 86400] * 2,
            "end": [start - 86000] * 2,
        }
    )
    train = past.iloc[[0]].copy()
    train["user"] = 0
    train["fraction"] = 1.0
    future = pd.DataFrame(
        {
            "channel_id": [1, 2],
            "tv_show_id": [10, 20],
            "start": [start + 86400] * 2,
            "end": [start + 86800] * 2,
        }
    )
    future = pd.concat([future] * repeat_future, ignore_index=True)
    return SimpleNamespace(
        start=start,
        end=start + repeat_future * 5 * 604800,
        past=past,
        train=train,
        future=future,
        eval_users=np.array([0]),
        items=np.array([10, 20]),
        im={10: 0, 20: 1},
    )


def test_channel_propensity_does_not_transfer_to_unwatched_channel():
    f = fixture()
    x, n = propensity_features(f, np.array([0, 0]), np.array([0, 1]))
    assert np.all(x[0] > 0) and np.all(x[1] == 0)
    y, _ = propensity_features(fixture(2), np.array([0, 0]), np.array([0, 1]))
    np.testing.assert_allclose(x, y)


def test_repeat_forecast_responds_to_channel_move():
    f = fixture()
    u = np.array([0])
    i = np.array([0])
    x, n = repeat_channel_features(f, u, i)
    f.future.loc[f.future.tv_show_id == 10, "channel_id"] = 2
    y, _ = repeat_channel_features(f, u, i)
    assert np.all(x > 0) and np.all(y == 0)


def test_neighbor_excludes_self_when_no_shared_history():
    from sweettv.neighbor_cycles import neighbor_features

    f = fixture()
    f.users = ["a", "b"]
    f.shape = (2, 2)
    f.train["item_idx"] = f.train.tv_show_id.map(f.im)
    x, n = neighbor_features(f, np.array([0, 0]), np.array([0, 1]))
    assert np.all(x[:, ::2] == 0)
