"""Fixed-stride learned tokenizer with VQ codebook and reconstruction loss."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from chaoscontrol.vq import vector_quantize


class FixedStrideTokenizer(nn.Module):
    """Byte-level tokenizer: causal conv downsample → VQ → reconstruct.

    Converts raw byte sequences into a shorter sequence of VQ token embeddings.
    A transposed-conv decoder reconstructs byte logits from the token embeddings,
    providing a reconstruction loss that prevents lossy codebook collapse.

    Args:
        byte_dim: Embedding dimension for individual bytes.
        token_dim: Dimension of each learned token (and codebook entry).
        codebook_size: Number of entries in the VQ codebook (vocabulary size).
        stride: Downsampling factor (token_seq = byte_seq // stride).
        window: Conv kernel size.  Defaults to ``stride * 2`` for overlap.
        beta: VQ commitment-loss weight forwarded to ``vector_quantize``.
    """

    def __init__(
        self,
        byte_dim: int = 64,
        token_dim: int = 128,
        codebook_size: int = 1024,
        stride: int = 4,
        window: int | None = None,
        beta: float = 0.25,
    ):
        super().__init__()
        if window is None:
            window = stride * 2

        self.stride = stride
        self.window = window
        self.beta = beta
        self.codebook_size = codebook_size

        # Byte-level input embedding (0-255)
        self.byte_embed = nn.Embedding(256, byte_dim)

        # Causal downsampling conv: left-pad by (kernel_size - 1) so that
        # each output position depends only on current and past bytes.
        self.downsample = nn.Conv1d(byte_dim, token_dim, kernel_size=window, stride=stride)

        # VQ codebook (the learned vocabulary)
        self.codebook = nn.Parameter(torch.empty(codebook_size, token_dim))
        nn.init.uniform_(self.codebook, -1.0 / codebook_size, 1.0 / codebook_size)

        # Reconstruction decoder: token embeddings → byte logits
        self.decoder = nn.ConvTranspose1d(token_dim, 256, kernel_size=stride * 2, stride=stride)

    def forward(self, byte_ids: torch.Tensor) -> dict:
        """Encode bytes to VQ tokens and compute reconstruction loss.

        Args:
            byte_ids: ``(batch, byte_seq)`` of long values in ``[0, 255]``.

        Returns:
            dict with keys:
                token_embeds: ``(batch, token_seq, token_dim)`` — straight-through embeddings
                token_ids:    ``(batch, token_seq)`` — VQ codebook indices (int64)
                commit_loss:  scalar — VQ commitment loss
                recon_loss:   scalar — reconstruction cross-entropy
                codebook:     reference to ``self.codebook`` parameter
        """
        batch, byte_seq = byte_ids.shape

        # 1. Embed bytes → (batch, byte_seq, byte_dim)
        x = self.byte_embed(byte_ids)

        # 2. Causal conv downsample
        #    Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, byte_dim, byte_seq)

        #    Causal padding: pad left by (kernel_size - 1), right by 0
        x = F.pad(x, (self.window - 1, 0))
        x = self.downsample(x)  # (batch, token_dim, token_seq)

        x = x.transpose(1, 2)  # (batch, token_seq, token_dim)

        # 3. VQ quantize
        token_embeds, token_ids, commit_loss = vector_quantize(x, self.codebook, beta=self.beta)

        # 4. Reconstruction: decode token embeddings back to byte logits
        recon_input = token_embeds.transpose(1, 2)  # (batch, token_dim, token_seq)
        recon_logits = self.decoder(recon_input)     # (batch, 256, recon_len)
        recon_logits = recon_logits.transpose(1, 2)  # (batch, recon_len, 256)

        # Clip to original byte_seq length (transposed conv may produce extra positions)
        recon_logits = recon_logits[:, :byte_seq, :]

        recon_loss = F.cross_entropy(
            recon_logits.reshape(-1, 256),
            byte_ids.reshape(-1),
        )

        return {
            "token_embeds": token_embeds,
            "token_ids": token_ids,
            "commit_loss": commit_loss,
            "recon_loss": recon_loss,
            "codebook": self.codebook,
        }
