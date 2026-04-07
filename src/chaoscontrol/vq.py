"""Vector quantization utilities shared by tokenizer and Wernicke."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def vector_quantize(
    x: torch.Tensor,
    codebook: torch.Tensor,
    beta: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vector quantize x against codebook with straight-through gradient.

    Args:
        x: (..., dim) — continuous representations (any leading dims)
        codebook: (K, dim) — codebook entries
        beta: commitment loss weight

    Returns:
        (quantized, indices, commitment_loss)
        quantized: same shape as x — quantized with straight-through
        indices: same leading dims as x — codebook indices (int64)
        commitment_loss: scalar
    """
    orig_shape = x.shape
    dim = x.size(-1)
    flat_x = x.reshape(-1, dim)  # (N, dim)

    # Distances: (N, K)
    dists = torch.cdist(flat_x.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
    indices_flat = dists.argmin(dim=-1)  # (N,)

    # Look up quantized vectors
    quantized_flat = codebook[indices_flat]  # (N, dim)

    # Commitment loss
    commit_loss = F.mse_loss(flat_x.detach(), quantized_flat) + beta * F.mse_loss(flat_x, quantized_flat.detach())

    # Straight-through: forward uses quantized, backward uses x
    quantized_st = flat_x + (quantized_flat - flat_x).detach()

    # Reshape back
    quantized = quantized_st.reshape(orig_shape)
    indices = indices_flat.reshape(orig_shape[:-1])

    return quantized, indices, commit_loss
