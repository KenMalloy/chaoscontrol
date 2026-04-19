from __future__ import annotations

from collections.abc import Sequence
import unicodedata

import torch


def _manual_seed_generator(
    seed: int,
    device: torch.device | None = None,
) -> torch.Generator:
    if device is not None:
        try:
            return torch.Generator(device=device).manual_seed(seed)
        except (RuntimeError, TypeError):
            pass
    return torch.Generator().manual_seed(seed)


def match_row_norm_moments(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rescale `src` per-row so the resulting row-norm distribution has the same
    mean and std as `target`'s. Preserves per-row direction (cosine) exactly.
    """
    src_norms = src.norm(dim=-1)
    target_norms = target.norm(dim=-1)
    # Z-score src norms, then re-scale to target moments
    z = (src_norms - src_norms.mean()) / src_norms.std().clamp(min=1e-8)
    new_norms = z * target_norms.std() + target_norms.mean()
    scale = (new_norms / src_norms.clamp(min=1e-12)).unsqueeze(-1)
    return src * scale


def match_full_covariance(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Whiten `src` then re-color to match `target`'s full row covariance.
    Rows of output have approximately the same covariance as rows of `target`.
    """
    src_centered = src - src.mean(dim=0)
    target_centered = target - target.mean(dim=0)
    cov_src = torch.cov(src_centered.T)
    cov_target = torch.cov(target_centered.T)
    L_src = torch.linalg.cholesky(
        cov_src + 1e-6 * torch.eye(src.shape[1], device=src.device, dtype=src.dtype)
    )
    L_target = torch.linalg.cholesky(
        cov_target
        + 1e-6 * torch.eye(target.shape[1], device=target.device, dtype=target.dtype)
    )
    whitened = torch.linalg.solve_triangular(L_src, src_centered.T, upper=False).T
    recolored = whitened @ L_target.T
    return recolored + target.mean(dim=0)


def shuffle_rows(src: torch.Tensor, seed: int) -> torch.Tensor:
    """Randomly permute rows of `src` under the given seed. Used as a control:
    preserves marginal distribution but destroys ID-to-vector mapping.
    """
    g = _manual_seed_generator(seed, src.device)
    perm = torch.randperm(src.shape[0], generator=g, device=src.device)
    return src[perm].clone()


def sample_with_row_norms(
    reference: torch.Tensor,
    *,
    seed: int,
    shuffle_norms_seed: int | None = None,
) -> torch.Tensor:
    """Sample random unit directions and scale them by ``reference`` row norms.

    With ``shuffle_norms_seed=None``, token ID ``i`` gets row norm ``i`` from
    ``reference``. With a seed, the norm multiset is preserved but the token ID
    to norm assignment is permuted.
    """
    if reference.dim() != 2:
        raise ValueError(f"expected a 2D tensor, got shape={tuple(reference.shape)}")

    g = _manual_seed_generator(seed, reference.device)
    directions = torch.randn(
        reference.shape,
        generator=g,
        device=reference.device,
        dtype=reference.dtype,
    )
    directions = torch.nn.functional.normalize(directions, dim=-1)
    norms = reference.norm(dim=-1)
    if shuffle_norms_seed is not None:
        norm_g = _manual_seed_generator(shuffle_norms_seed, reference.device)
        perm = torch.randperm(norms.shape[0], generator=norm_g, device=reference.device)
        norms = norms[perm]
    return directions * norms.unsqueeze(-1)


def shuffle_rows_within_buckets(
    src: torch.Tensor,
    bucket_ids: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Permute rows only among tokens that share the same bucket ID."""
    if src.dim() != 2:
        raise ValueError(f"expected a 2D tensor, got shape={tuple(src.shape)}")
    if bucket_ids.dim() != 1 or bucket_ids.shape[0] != src.shape[0]:
        raise ValueError(
            "bucket_ids must be a 1D tensor with one entry per source row; "
            f"got bucket_ids={tuple(bucket_ids.shape)} src={tuple(src.shape)}"
        )

    bucket_ids = bucket_ids.to(device=src.device)
    out = src.clone()
    g = _manual_seed_generator(seed, src.device)
    for bucket in torch.unique(bucket_ids):
        idx = torch.nonzero(bucket_ids == bucket, as_tuple=False).flatten()
        if idx.numel() <= 1:
            continue
        perm = torch.randperm(idx.numel(), generator=g, device=src.device)
        out[idx] = src[idx[perm]]
    return out


def frequency_bucket_ids(counts: torch.Tensor, *, num_buckets: int) -> torch.Tensor:
    """Assign token IDs to equal-size buckets by descending frequency rank."""
    if counts.dim() != 1:
        raise ValueError(f"counts must be 1D, got shape={tuple(counts.shape)}")
    if num_buckets <= 0:
        raise ValueError(f"num_buckets must be positive, got {num_buckets}")

    n = counts.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=counts.device)
    order = torch.argsort(counts.float(), descending=True, stable=True)
    rank_bucket_ids = torch.div(
        torch.arange(n, device=counts.device) * num_buckets,
        n,
        rounding_mode="floor",
    ).clamp(max=num_buckets - 1)
    bucket_ids = torch.empty(n, dtype=torch.long, device=counts.device)
    bucket_ids[order] = rank_bucket_ids.long()
    return bucket_ids


def token_class_bucket_ids(pieces: Sequence[str]) -> torch.Tensor:
    """Coarse token classes for row-shuffle controls.

    Buckets are: leading-space/whitespace, punctuation, numeric, alpha, and
    mixed/other. SentencePiece's leading-space marker is recognized via
    ``"\\u2581"``.
    """

    def _is_punctuation(text: str) -> bool:
        return bool(text) and all(
            unicodedata.category(ch).startswith("P") for ch in text
        )

    ids: list[int] = []
    for piece in pieces:
        text = str(piece)
        if text.startswith("\u2581") or text.strip() == "":
            ids.append(0)
        elif _is_punctuation(text):
            ids.append(1)
        elif text.isnumeric():
            ids.append(2)
        elif text.isalpha():
            ids.append(3)
        else:
            ids.append(4)
    return torch.tensor(ids, dtype=torch.long)


def sample_full_covariance(reference: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Sample a synthetic Gaussian cloud matched to ``reference`` mean/cov."""
    if reference.dim() != 2:
        raise ValueError(f"expected a 2D tensor, got shape={tuple(reference.shape)}")
    g = _manual_seed_generator(seed, reference.device)
    noise = torch.randn(
        reference.shape,
        generator=g,
        device=reference.device,
        dtype=reference.dtype,
    )
    return match_full_covariance(noise, reference)
