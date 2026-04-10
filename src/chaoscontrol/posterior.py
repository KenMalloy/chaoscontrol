"""Error-driven posterior-state modules for belief correction.

Three causal posterior-state options that store belief updates induced by
past prediction error. All maintain strict forward causality: updates at
step t may only affect predictions from t+1 onward.

- GlobalDelta: one correction vector for the whole document.
- BucketDelta: per-Wernicke-bucket correction vectors.
- ResidualCache: context-keyed cache of (key, correction) pairs with top-k retrieval.

Each class implements the Test-Time Training (TTT) protocol: reset() clears
all learned state between evaluation segments so the model rebuilds its
beliefs from scratch on each new segment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalDelta(nn.Module):
    """Document-level correction vector updated from prediction error.

    Simplest posterior option: one correction vector for the whole document.
    Updated each step from prediction error, added to the stream as a
    constant bias.
    """

    def __init__(self, model_dim: int, lr: float = 0.01) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.lr = lr
        self.register_buffer("delta", torch.zeros(1, model_dim))

    def read(self, batch_size: int) -> torch.Tensor:
        """Return the current correction vector, expanded for the batch."""
        return self.delta.expand(batch_size, -1)

    def update(self, prediction_error_grad: torch.Tensor) -> None:
        """Update delta from prediction error gradient.

        Args:
            prediction_error_grad: (model_dim,) or (1, model_dim) gradient
                direction from prediction error.
        """
        grad = prediction_error_grad.detach()
        if grad.dim() == 1:
            grad = grad.unsqueeze(0)
        self.delta = (self.delta + self.lr * grad).detach()

    def reset(self) -> None:
        """Test-Time Training protocol: clear learned state between evaluation segments."""
        self.delta.zero_()


class BucketDelta(nn.Module):
    """Per-Wernicke-bucket correction vectors updated from prediction error.

    Maintains one correction vector per Wernicke bucket (k_max x model_dim).
    On each training step, after computing loss, updates the delta for the
    dominant bucket. On read, returns the delta for the queried bucket.
    """

    def __init__(self, k_max: int, model_dim: int, lr: float = 0.01) -> None:
        super().__init__()
        self.k_max = k_max
        self.model_dim = model_dim
        self.lr = lr
        self.register_buffer("deltas", torch.zeros(k_max, model_dim))

    def read(self, bucket_id: int, batch_size: int) -> torch.Tensor:
        """Return the correction vector for a specific bucket.

        Args:
            bucket_id: Wernicke bucket index to query.
            batch_size: expand to this batch size.

        Returns:
            (batch_size, model_dim) correction vector.
        """
        return self.deltas[bucket_id].unsqueeze(0).expand(batch_size, -1)

    def update(self, bucket_id: int, prediction_error_grad: torch.Tensor) -> None:
        """Update the delta for the given bucket from prediction error gradient.

        Args:
            bucket_id: Wernicke bucket to update.
            prediction_error_grad: (model_dim,) or (1, model_dim) gradient direction.
        """
        grad = prediction_error_grad.detach()
        if grad.dim() > 1:
            grad = grad.squeeze(0)
        self.deltas[bucket_id] = (self.deltas[bucket_id] + self.lr * grad).detach()

    def reset(self) -> None:
        """Test-Time Training protocol: clear learned state between evaluation segments."""
        self.deltas.zero_()


class ResidualCache(nn.Module):
    """Cached correction traces keyed by context for context-specific recall.

    When a prediction was wrong, stores (context_key, correction_value) pairs.
    On read, retrieves the top-k most similar correction traces by cosine
    similarity to the query context.
    """

    def __init__(self, model_dim: int, k: int = 4, max_entries: int = 256) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.k = k
        self.max_entries = max_entries
        self._keys: list[torch.Tensor] = []    # each (1, model_dim)
        self._values: list[torch.Tensor] = []  # each (1, model_dim)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve top-k most similar correction traces.

        Args:
            query: (batch_size, model_dim) context to match against stored keys.

        Returns:
            (batch_size, model_dim) weighted sum of top-k correction values.
        """
        batch_size = query.shape[0]
        if not self._keys:
            return torch.zeros(batch_size, self.model_dim, device=query.device)

        # Stack stored keys: (num_entries, model_dim)
        key_matrix = torch.cat(self._keys, dim=0)
        val_matrix = torch.cat(self._values, dim=0)

        # Cosine similarity: (batch_size, num_entries)
        query_norm = F.normalize(query.detach(), dim=-1)
        key_norm = F.normalize(key_matrix, dim=-1)
        sim = torch.mm(query_norm, key_norm.T)

        # Top-k retrieval
        actual_k = min(self.k, sim.shape[1])
        topk_sim, topk_idx = sim.topk(actual_k, dim=-1)  # (batch, k)

        # Softmax weights over top-k similarities
        weights = F.softmax(topk_sim, dim=-1)  # (batch, k)

        # Gather top-k values and weight them
        # topk_idx: (batch, k) -> gather from val_matrix: (num_entries, model_dim)
        topk_vals = val_matrix[topk_idx.reshape(-1)].reshape(
            batch_size, actual_k, self.model_dim
        )  # (batch, k, model_dim)
        result = (weights.unsqueeze(-1) * topk_vals).sum(dim=1)  # (batch, model_dim)
        return result

    def store(self, context_key: torch.Tensor, correction_value: torch.Tensor) -> None:
        """Store a (context, correction) pair.

        Args:
            context_key: (model_dim,) or (1, model_dim) context embedding.
            correction_value: (model_dim,) or (1, model_dim) correction direction.
        """
        key = context_key.detach()
        val = correction_value.detach()
        if key.dim() == 1:
            key = key.unsqueeze(0)
        if val.dim() == 1:
            val = val.unsqueeze(0)
        self._keys.append(key)
        self._values.append(val)

        # Evict oldest if over capacity
        if len(self._keys) > self.max_entries:
            self._keys = self._keys[-self.max_entries:]
            self._values = self._values[-self.max_entries:]

    def reset(self) -> None:
        """Test-Time Training protocol: clear learned state between evaluation segments."""
        self._keys.clear()
        self._values.clear()
