import torch

from experiments._23_fast_path_runner_helpers import compute_ce_minus_entropy_pressure_from_fused


def test_pressure_from_fused_pair_nonnegative_and_ranks_by_innovation():
    ce = torch.tensor([[3.0, 0.5, 2.0]])
    entropy = torch.tensor([[0.1, 2.0, 2.5]])
    # innovation = ce - entropy = [2.9, -1.5, -0.5]; relu -> [2.9, 0, 0]
    pressure = compute_ce_minus_entropy_pressure_from_fused(ce, entropy)
    assert (pressure >= 0).all()
    assert pressure.argmax().item() == 0
    # The two suppressed positions go to zero exactly.
    assert pressure[0, 1].item() == 0.0
    assert pressure[0, 2].item() == 0.0


def test_pressure_from_fused_pair_preserves_shape():
    ce = torch.randn(4, 7).abs()
    entropy = torch.randn(4, 7).abs()
    pressure = compute_ce_minus_entropy_pressure_from_fused(ce, entropy)
    assert pressure.shape == (4, 7)
