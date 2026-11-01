import numpy as np
import pandas as pd
import torch

from sweettv.temporal_sequence import snapshot, TemporalSequence


def test_snapshot_excludes_future_and_uses_completed_events_only():
    events = pd.DataFrame(
        dict(user=[0, 0, 0, 1], tv_show_id=[10, 20, 30, 10], end=[100, 200, 300, 150])
    )
    seq, count, recent = snapshot(events, 200, [0, 1, 2], {10: 0, 20: 1, 30: 2}, 3)
    assert seq[0, :3].tolist() == [1, 2, 0]
    assert seq[2].sum() == 0
    np.testing.assert_allclose(count.toarray()[0], [np.log(2), np.log(2), 0])
    assert recent[0, 0] < count[0, 0]


def test_sequence_empty_history_and_repeat_prior_are_finite():
    torch.manual_seed(5)
    m = TemporalSequence(4, 8, 1, True).eval()
    x = torch.zeros((2, 100), dtype=torch.long)
    x[0, :2] = torch.tensor([1, 2])
    zeros = torch.zeros((2, 4))
    with torch.no_grad():
        a = m(x, zeros, zeros)
        c = zeros.clone()
        c[0, 2] = 1
        b = m(x, c, zeros)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a[1], m.bias)
    torch.testing.assert_close(
        b[0, 2] - a[0, 2], torch.nn.functional.softplus(m.repeat_weight[0])
    )
    torch.testing.assert_close(b[:, [0, 1, 3]], a[:, [0, 1, 3]])
