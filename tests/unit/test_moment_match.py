import torch
from chaoscontrol.sgns.moment_match import (
    match_row_norm_moments,
    match_full_covariance,
    shuffle_rows,
)


def test_match_row_norm_moments_matches_target_moments():
    torch.manual_seed(0)
    src = torch.randn(100, 16) * 3.0 + 2.0
    target = torch.randn(100, 16) * 0.5
    out = match_row_norm_moments(src, target)
    src_row_norms = src.norm(dim=-1)
    target_row_norms = target.norm(dim=-1)
    out_row_norms = out.norm(dim=-1)
    assert torch.isclose(out_row_norms.mean(), target_row_norms.mean(), rtol=1e-3)
    assert torch.isclose(out_row_norms.std(), target_row_norms.std(), rtol=1e-3)


def test_match_row_norm_preserves_directions():
    """Rescaling rows preserves pairwise cosine similarity."""
    torch.manual_seed(1)
    src = torch.randn(50, 8)
    target = torch.randn(50, 8) * 0.01
    out = match_row_norm_moments(src, target)
    src_n = torch.nn.functional.normalize(src, dim=-1)
    out_n = torch.nn.functional.normalize(out, dim=-1)
    assert torch.allclose(src_n, out_n, atol=1e-5)


def test_match_full_covariance_matches_cov():
    torch.manual_seed(0)
    src = torch.randn(200, 8) * 3.0
    target = torch.randn(200, 8) * 0.5
    out = match_full_covariance(src, target)
    cov_out = torch.cov(out.T)
    cov_target = torch.cov(target.T)
    assert torch.allclose(cov_out, cov_target, atol=0.05)


def test_shuffle_rows_is_permutation():
    torch.manual_seed(0)
    src = torch.randn(100, 16)
    out = shuffle_rows(src, seed=42)
    # Same set of rows, different order
    sorted_src = src[torch.argsort(src[:, 0])]
    sorted_out = out[torch.argsort(out[:, 0])]
    assert torch.allclose(sorted_src, sorted_out)
    assert not torch.allclose(src, out)
