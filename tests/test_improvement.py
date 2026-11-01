import numpy as np
import torch
from sweettv.sequence import SequenceModel
from sweettv.metrics import apk


def test_causal_sequence_and_padding():
    torch.manual_seed(12)
    model = SequenceModel(12).eval()
    a = torch.tensor([[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]])
    with torch.no_grad():
        h = model(a)
    # Changing a later event must not affect representations of earlier events.
    torch.testing.assert_close(h[0, :3], h[1, :3], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(h).all()
    assert torch.all(h[:, -1] == 0)
    with torch.no_grad():
        b = model(torch.tensor([[1, 2, 3, 4, 0, 0, 0]]))
    torch.testing.assert_close(h[0, :4], b[0, :4], rtol=1e-5, atol=1e-6)


def test_candidate_oracle_upper_bound():
    truth = [1, 2, 3, 4, 5]
    # With m recovered targets in the first m places, AP@5 = m/5.
    for m in range(6):
        pred = truth[:m] + list(range(10, 15 - m))
        assert np.isclose(apk(truth, pred), m / 5)


def test_horizon_scaling_is_identity_at_five_weeks():
    from sweettv.horizon import rescale

    names = [
        "channel_score",
        "future_airings",
        "past_airings",
        "airing_ratio",
        "horizon_weeks",
        "expected_count_14_0.8",
        "future_per_week",
    ]
    x = np.array(
        [
            [
                10,
                np.log1p(20),
                np.log1p(40),
                np.log1p(20) / np.log1p(40),
                5,
                np.log1p(4),
                np.log1p(4),
            ]
        ],
        dtype="float32",
    )
    np.testing.assert_allclose(rescale(x, names, 5), x, rtol=1e-6, atol=1e-6)
    y = rescale(x, names, 10)
    assert np.isclose(y[0, 0], 5)
    assert np.isclose(np.expm1(y[0, 1]), 10)
    assert np.isclose(np.expm1(y[0, 5]), 2)
    assert y[0, 6] == x[0, 6]  # Already-normalized weekly feature must not be rescaled.
