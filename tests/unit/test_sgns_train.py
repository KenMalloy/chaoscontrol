import torch
from chaoscontrol.sgns.train import train_one_epoch
from chaoscontrol.sgns.model import SGNSModel
from chaoscontrol.sgns.sampler import NegativeSampler


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
