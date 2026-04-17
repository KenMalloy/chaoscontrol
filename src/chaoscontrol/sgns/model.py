import torch
import torch.nn as nn


class SGNSModel(nn.Module):
    """Skip-gram with negative sampling. Two separate embedding tables:
    input (W_in) for center words, output (W_out) for contexts/negatives.
    Standard word2vec convention; W_out is discarded after training.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.input_embed = nn.Embedding(vocab_size, dim)
        self.output_embed = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.input_embed.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_embed.weight)

    def score_pairs(self, center: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        w_in = self.input_embed(center)
        w_out = self.output_embed(context)
        return (w_in * w_out).sum(dim=-1)
