import torch
from torch.optim import Optimizer
from chaoscontrol.sgns.model import SGNSModel, nce_loss
from chaoscontrol.sgns.sampler import NegativeSampler


def _iterate_center_context(
    stream: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (center, context) tensors of all (i, j) pairs with 1 <= |i-j| <= window."""
    centers, contexts = [], []
    for offset in range(1, window + 1):
        centers.append(stream[offset:])
        contexts.append(stream[:-offset])
        centers.append(stream[:-offset])
        contexts.append(stream[offset:])
    return torch.cat(centers), torch.cat(contexts)


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
    centers, contexts = _iterate_center_context(stream, window)
    n = len(centers)
    perm = torch.randperm(n)
    centers = centers[perm]
    contexts = contexts[perm]
    total_loss, batches = 0.0, 0
    for start in range(0, n, batch_size):
        if max_batches is not None and batches >= max_batches:
            break
        end = min(start + batch_size, n)
        c = centers[start:end]
        ctx = contexts[start:end]
        negs = sampler.sample(batch_size=len(c), k=k)
        loss = nce_loss(model, c, ctx, negs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)
