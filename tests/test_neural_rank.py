import numpy as np
import torch

from sweettv.neural_rank import query_loss, batch, training_keys, map_swap_weights
from sweettv.improve import WINDOWS


def test_listwise_loss_padding_and_query_weight():
    logits = torch.tensor([[2.0, 0.0, 999.0], [1.0, 1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    loss = query_loss(logits, labels, mask, "listwise")
    expected = (
        torch.logsumexp(logits[0, :2], 0) - 2 + torch.logsumexp(logits[1], 0) - 1
    ) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 2] == 0
    assert logits.grad[0, 0] < 0 < logits.grad[0, 1]


def test_batch_alignment_and_padding_gradients():
    a = dict(
        x=np.arange(10, dtype="float32").reshape(5, 2),
        y=np.array([1, 0, 0, 1, 0]),
        groups=np.array([2, 3]),
        offsets=np.array([0, 2, 5]),
    )
    x, y, mask = batch(a, np.array([1, 0]), "cpu")
    np.testing.assert_array_equal(x[0], a["x"][2:])
    np.testing.assert_array_equal(x[1, :2], a["x"][:2])
    assert mask.tolist() == [[True, True, True], [True, True, False]]
    for objective in ["listwise", "balanced_bce", "lambda_map"]:
        logits = torch.zeros_like(y, requires_grad=True)
        query_loss(logits, y, mask, objective).backward()
        assert logits.grad[1, 2] == 0


def test_training_windows_do_not_cross_cutoff():
    for fold in ["dev", "audit", "final"]:
        assert all(WINDOWS[k][1] <= WINDOWS[fold][0] for k in training_keys(fold))


def test_lambda_map_weights_equal_brute_force_swaps():
    def ap(y):
        return sum(v * sum(y[: i + 1]) / (i + 1) for i, v in enumerate(y[:5])) / 5

    rng = np.random.default_rng(15)
    for _ in range(20):
        labels = np.zeros(12, dtype="float32")
        labels[rng.choice(12, rng.integers(1, 6), replace=False)] = 1
        logits = torch.arange(12, 0, -1, dtype=torch.float32)[None]
        p, q, weights = map_swap_weights(
            logits,
            torch.from_numpy(labels)[None],
            torch.ones_like(logits, dtype=torch.bool),
        )
        for a, positive in enumerate(p[0].tolist()):
            for b, negative in enumerate(q[0].tolist()):
                if labels[positive] == 1 and labels[negative] == 0:
                    changed = labels.copy()
                    changed[positive], changed[negative] = (
                        changed[negative],
                        changed[positive],
                    )
                    assert np.isclose(
                        weights[0, a, b].item(),
                        abs(ap(changed) - ap(labels)),
                        atol=1e-7,
                    )
