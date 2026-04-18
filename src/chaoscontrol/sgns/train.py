from collections.abc import Iterator

import torch
from torch.optim import Optimizer
from chaoscontrol.sgns.model import SGNSModel, nce_loss
from chaoscontrol.sgns.sampler import NegativeSampler


def _iter_center_context_batches(
    stream: torch.Tensor,
    window: int,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield shuffled mini-batches of (center, context) pairs.

    The old implementation materialized *every* pair in two giant tensors and
    then applied a full ``randperm`` over the concatenated result. At the
    documented 50M-token scale that turns into hundreds of millions of pairs
    and a very large hidden memory cliff. This iterator streams one
    offset/direction slice at a time and only randomizes the much smaller batch
    order inside each slice.
    """
    if window <= 0:
        return
    n = int(stream.numel())
    if n <= 1:
        return

    max_offset = min(window, n - 1)
    specs = [(offset, True) for offset in range(1, max_offset + 1)]
    specs.extend((offset, False) for offset in range(1, max_offset + 1))
    spec_order = torch.randperm(len(specs)).tolist()

    for spec_idx in spec_order:
        offset, forward = specs[spec_idx]
        if forward:
            centers = stream[offset:]
            contexts = stream[:-offset]
        else:
            centers = stream[:-offset]
            contexts = stream[offset:]
        pair_count = int(centers.numel())
        if pair_count == 0:
            continue
        n_batches = (pair_count + batch_size - 1) // batch_size
        batch_order = torch.randperm(n_batches).tolist()
        for batch_idx in batch_order:
            start = batch_idx * batch_size
            end = min(start + batch_size, pair_count)
            yield centers[start:end], contexts[start:end]


def train_one_epoch(
    model: SGNSModel,
    stream: torch.Tensor,
    sampler: NegativeSampler,
    window: int,
    k: int,
    batch_size: int,
    opt: Optimizer,
    max_batches: int | None = None,
) -> float:
    """One pass over stream. Returns mean loss over processed batches."""
    total_loss, batches = 0.0, 0
    for c, ctx in _iter_center_context_batches(stream, window, batch_size):
        if max_batches is not None and batches >= max_batches:
            break
        negs = sampler.sample(batch_size=len(c), k=k)
        loss = nce_loss(model, c, ctx, negs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)
