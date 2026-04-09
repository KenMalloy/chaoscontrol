"""Tests for HierarchicalWernicke two-level routing."""
import torch
from chaoscontrol.wernicke import HierarchicalWernicke, WernickeLayer


def test_hierarchical_init():
    hw = HierarchicalWernicke(dim=128, k_coarse=8, k_fine=8, window=8)
    assert hw.total_buckets == 64


def test_hierarchical_forward_shape():
    hw = HierarchicalWernicke(dim=128, k_coarse=8, k_fine=8, window=8)
    x = torch.randn(2, 32, 128)  # (batch, seq, dim)
    out, bucket_ids, balance_loss = hw(x)
    assert out.shape == (2, 32, 128)
    assert bucket_ids.shape == (2, 32)
    # Bucket ids should be in [0, 64)
    assert bucket_ids.min() >= 0
    assert bucket_ids.max() < 64


def test_hierarchical_bucket_composition():
    """Bucket id = coarse * k_fine + fine."""
    hw = HierarchicalWernicke(dim=128, k_coarse=4, k_fine=8, window=8)
    x = torch.randn(1, 16, 128)
    _, bucket_ids, _ = hw(x)
    # All bucket ids should be < 4 * 8 = 32
    assert bucket_ids.max() < 32


def test_hierarchical_param_budget():
    """Hierarchical should have comparable params to flat at same bucket count."""
    hw = HierarchicalWernicke(dim=128, k_coarse=8, k_fine=8, window=8)
    hier_params = sum(p.numel() for p in hw.parameters())
    flat = WernickeLayer(dim=128, k_max=64, window=8, router_type="moe")
    flat_params = sum(p.numel() for p in flat.parameters())
    # Should be in the same ballpark (within 3x)
    assert hier_params < flat_params * 3


def test_hierarchical_balance_loss_is_scalar():
    hw = HierarchicalWernicke(dim=128, k_coarse=4, k_fine=4, window=8)
    x = torch.randn(2, 16, 128)
    _, _, balance_loss = hw(x)
    assert balance_loss.ndim == 0  # scalar


def test_hierarchical_vq_router():
    hw = HierarchicalWernicke(
        dim=128, k_coarse=4, k_fine=4, window=8, router_type="vq"
    )
    x = torch.randn(1, 16, 128)
    out, bucket_ids, balance_loss = hw(x)
    assert out.shape == (1, 16, 128)
    assert bucket_ids.max() < 16  # 4 * 4


def test_hierarchical_expert_dim():
    """Bottleneck expert_dim should be respected."""
    hw = HierarchicalWernicke(
        dim=128, k_coarse=4, k_fine=4, window=8, expert_dim=32
    )
    x = torch.randn(1, 16, 128)
    out, bucket_ids, _ = hw(x)
    assert out.shape == (1, 16, 128)
