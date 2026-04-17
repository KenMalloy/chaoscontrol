import torch
import torch.nn as nn
import torch.nn.functional as F


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


def nce_loss(
    model: SGNSModel,
    center: torch.Tensor,
    context: torch.Tensor,
    negatives: torch.Tensor,
) -> torch.Tensor:
    """Skip-gram NCE loss.
    center:    (B,)
    context:   (B,)
    negatives: (B, K)
    Returns scalar mean loss.
    """
    w_in = model.input_embed(center)  # (B, D)
    w_pos = model.output_embed(context)  # (B, D)
    w_neg = model.output_embed(negatives)  # (B, K, D)

    pos_score = (w_in * w_pos).sum(dim=-1)  # (B,)
    neg_score = torch.einsum("bd,bkd->bk", w_in, w_neg)  # (B, K)

    pos_loss = F.logsigmoid(pos_score).neg()
    neg_loss = F.logsigmoid(-neg_score).neg().sum(dim=-1)
    return (pos_loss + neg_loss).mean()
