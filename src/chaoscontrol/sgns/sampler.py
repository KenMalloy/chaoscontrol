import torch


def unigram_probs_from_counts(counts: torch.Tensor, power: float = 0.75) -> torch.Tensor:
    """Standard word2vec negative-sampling distribution: counts^power / sum."""
    if power == 0.0:
        return torch.full_like(counts, 1.0 / len(counts))
    weighted = counts.clamp(min=0).float().pow(power)
    return weighted / weighted.sum()


class NegativeSampler:
    """Samples K negatives per positive pair from a unigram distribution."""

    def __init__(self, probs: torch.Tensor):
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
        self.probs = probs

    def sample(self, batch_size: int, k: int) -> torch.Tensor:
        return torch.multinomial(self.probs, batch_size * k, replacement=True).view(batch_size, k)
