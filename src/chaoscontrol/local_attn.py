"""Local attention module and rolling KV cache for hybrid SSM blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RollingKVCache:
    """Fixed-size rolling buffer for K/V pairs."""

    def __init__(self, window: int, dim: int) -> None:
        self.window = window
        self.dim = dim
        self._keys: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []

    def write(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append (batch, dim) key/value pair."""
        self._keys.append(k.detach())
        self._values.append(v.detach())
        if len(self._keys) > self.window:
            self._keys.pop(0)
            self._values.pop(0)

    def last(self, w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return last w entries as (batch, w, dim) with validity mask."""
        n = len(self._keys)
        if n == 0:
            batch = 1
            keys = torch.zeros(batch, w, self.dim)
            values = torch.zeros(batch, w, self.dim)
            mask = torch.zeros(batch, w, dtype=torch.bool)
            return keys, values, mask
        batch = self._keys[0].shape[0]
        device = self._keys[0].device
        dtype = self._keys[0].dtype
        keys = torch.zeros(batch, w, self.dim, device=device, dtype=dtype)
        values = torch.zeros(batch, w, self.dim, device=device, dtype=dtype)
        mask = torch.zeros(batch, w, device=device, dtype=torch.bool)
        fill = min(n, w)
        start = n - fill
        for i in range(fill):
            keys[:, i] = self._keys[start + i]
            values[:, i] = self._values[start + i]
            mask[:, i] = True
        return keys, values, mask

    def reset(self) -> None:
        self._keys.clear()
        self._values.clear()


class LocalAttention(nn.Module):
    """Single-query attention over a bounded KV window."""

    def __init__(self, model_dim: int, attn_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.head_dim = attn_dim // num_heads
        self.q_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, model_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Single-token attention (used by step() path).

        Args:
            query: (batch, model_dim)
            keys: (batch, w, attn_dim) — pre-projected
            values: (batch, w, attn_dim) — pre-projected
            mask: (batch, w) bool
        Returns:
            (batch, model_dim)
        """
        q = self.q_proj(query)  # (batch, attn_dim)
        B, W, _ = keys.shape
        nh, hd = self.num_heads, self.head_dim
        q = q.view(B, nh, 1, hd)
        k = keys.view(B, W, nh, hd).permute(0, 2, 1, 3)  # (B, nh, W, hd)
        v = values.view(B, W, nh, hd).permute(0, 2, 1, 3)
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, W)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)
        scores = scores.masked_fill(~attn_mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # (B, nh, 1, hd)
        out = out.view(B, nh * hd)
        return self.out_proj(out)

    def _get_sliding_window_mask(
        self, seq_len: int, window: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build or return cached causal sliding-window mask.

        Returns (mask, has_valid) where mask is (S, S) bool and
        has_valid is (S,) bool indicating which rows have any valid keys.
        """
        cache_key = (seq_len, window, device)
        if not hasattr(self, "_mask_cache") or self._mask_cache_key != cache_key:
            row = torch.arange(seq_len, device=device)
            col = torch.arange(seq_len, device=device)
            causal = col.unsqueeze(0) < row.unsqueeze(1)
            in_window = row.unsqueeze(1) - col.unsqueeze(0) <= window
            mask = causal & in_window
            has_valid = mask.any(dim=-1)
            self._mask_cache = (mask, has_valid)
            self._mask_cache_key = cache_key
        return self._mask_cache

    def _get_causal_mask(
        self, seq_len: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Strict causal mask (no window constraint). Cached."""
        cache_key = ("causal", seq_len, device)
        if not hasattr(self, "_causal_cache") or self._causal_cache_key != cache_key:
            row = torch.arange(seq_len, device=device)
            col = torch.arange(seq_len, device=device)
            mask = col.unsqueeze(0) < row.unsqueeze(1)  # (S, S)
            has_valid = mask.any(dim=-1)  # (S,)
            self._causal_cache = (mask, has_valid)
            self._causal_cache_key = cache_key
        return self._causal_cache

    def forward_sequence(
        self,
        query_seq: torch.Tensor,
        keys_seq: torch.Tensor,
        values_seq: torch.Tensor,
        window: int,
        topk: int = 0,
    ) -> torch.Tensor:
        """Parallel attention over a full sequence.

        Modes (controlled by window and topk):
            window > 0, topk == 0: causal sliding-window (local)
            window > 0, topk > 0:  top-k by score within causal past (selective)
            window == 0:           no-op (returns zeros)

        Args:
            query_seq: (batch, seq, model_dim) — full sequence queries
            keys_seq: (batch, seq, attn_dim) — pre-projected keys
            values_seq: (batch, seq, attn_dim) — pre-projected values
            window: attention window size (attend to positions [t-w, t-1]).
                    Ignored when topk > 0.
            topk: if > 0, attend to the k highest-scoring causal positions
                  instead of a fixed window. This is selective retrieval.
        Returns:
            (batch, seq, model_dim)
        """
        B, S, _ = query_seq.shape
        nh, hd = self.num_heads, self.head_dim

        q = self.q_proj(query_seq)  # (B, S, attn_dim)
        q = q.view(B, S, nh, hd).permute(0, 2, 1, 3)  # (B, nh, S, hd)
        k = keys_seq.view(B, S, nh, hd).permute(0, 2, 1, 3)
        v = values_seq.view(B, S, nh, hd).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)  # (B, nh, S, S)

        if topk > 0:
            # Selective retrieval: top-k by score within causal past
            causal_mask, has_valid = self._get_causal_mask(S, query_seq.device)
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
            # Keep only top-k scores per query position, mask the rest
            if topk < S:
                kth, _ = scores.topk(topk, dim=-1)
                threshold = kth[..., -1:] # (B, nh, S, 1)
                topk_mask = scores >= threshold
                scores = scores.masked_fill(~topk_mask, -1e9)
        else:
            # Local window
            mask, has_valid = self._get_sliding_window_mask(S, window, query_seq.device)
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # (B, nh, S, hd)

        # Zero out positions with no valid keys (t=0 has no causal history).
        out = out * has_valid.view(1, 1, S, 1)

        out = out.permute(0, 2, 1, 3).reshape(B, S, nh * hd)
        return self.out_proj(out)
