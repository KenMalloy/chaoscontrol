import torch
from chaoscontrol.sgns.train import _iter_center_context_batches, train_one_epoch
from chaoscontrol.sgns.model import SGNSModel
from chaoscontrol.sgns.sampler import NegativeSampler


def test_iter_center_context_batches_covers_expected_pairs():
    stream = torch.arange(6, dtype=torch.long)
    observed: list[tuple[int, int]] = []
    for centers, contexts in _iter_center_context_batches(
        stream, window=2, batch_size=2,
    ):
        assert len(centers) == len(contexts)
        assert len(centers) <= 2
        observed.extend(
            (int(c.item()), int(ctx.item()))
            for c, ctx in zip(centers, contexts)
        )

    expected = []
    for offset in (1, 2):
        expected.extend(
            (int(stream[i + offset]), int(stream[i]))
            for i in range(len(stream) - offset)
        )
        expected.extend(
            (int(stream[i]), int(stream[i + offset]))
            for i in range(len(stream) - offset)
        )

    assert sorted(observed) == sorted(expected)


def test_train_one_epoch_loss_decreases():
    torch.manual_seed(0)
    # Synthetic stream of size 200 with 10 tokens, clear co-occurrence pattern
    stream = torch.tensor([i % 10 for i in range(200)], dtype=torch.long)
    counts = torch.bincount(stream, minlength=10).float()
    probs = counts / counts.sum()
    sampler = NegativeSampler(probs)
    model = SGNSModel(vocab_size=10, dim=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_before = train_one_epoch(
        model, stream, sampler, window=2, k=5, batch_size=16, opt=opt, max_batches=1
    )
    for _ in range(10):
        loss_after = train_one_epoch(
            model, stream, sampler, window=2, k=5, batch_size=16, opt=opt, max_batches=5
        )
    assert loss_after < loss_before
