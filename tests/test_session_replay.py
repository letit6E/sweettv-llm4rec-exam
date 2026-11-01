import numpy as np
from sweettv.session_replay import accumulate, wall
from sweettv.experiment import epoch


def test_replay_matches_overlap_not_just_start_hour():
    out = np.zeros((4, 1, 2), dtype="float32")
    accumulate(
        np.array([0]),
        np.array([100]),
        np.array([200]),
        np.array([100, 175]),
        np.array([200, 275]),
        np.array([0, 1]),
        np.array([2, 1]),
        86400,
        out,
    )
    np.testing.assert_allclose(out[:, 0, 0], [2, 2, 2, 2])
    np.testing.assert_allclose(out[:, 0, 1], [0, 0, 0, 0.25])


def test_replay_midnight_wrap():
    out = np.zeros((4, 1, 1), dtype="float32")
    accumulate(
        np.array([0]),
        np.array([86300]),
        np.array([86500]),
        np.array([0]),
        np.array([100]),
        np.array([0]),
        np.array([1]),
        86400,
        out,
    )
    np.testing.assert_allclose(out[:, 0, 0], 1)


def test_replay_civil_time_survives_dst():
    vals = wall(np.array([epoch("2020-03-28 12:00"), epoch("2020-03-30 12:00")]))
    assert vals[1] - vals[0] == 2 * 86400
