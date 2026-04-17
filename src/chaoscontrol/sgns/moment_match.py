import torch


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
    L_src = torch.linalg.cholesky(cov_src + 1e-6 * torch.eye(src.shape[1]))
    L_target = torch.linalg.cholesky(cov_target + 1e-6 * torch.eye(target.shape[1]))
    whitened = torch.linalg.solve_triangular(L_src, src_centered.T, upper=False).T
    recolored = whitened @ L_target.T
    return recolored + target.mean(dim=0)


def shuffle_rows(src: torch.Tensor, seed: int) -> torch.Tensor:
    """Randomly permute rows of `src` under the given seed. Used as a control:
    preserves marginal distribution but destroys ID-to-vector mapping.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(src.shape[0], generator=g)
    return src[perm].clone()
