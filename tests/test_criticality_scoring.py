import torch
import pytest
from chaoscontrol.optim.criticality import compute_event_mask


def test_event_mask_selects_top_event_frac_positions():
    pressure = torch.arange(100, dtype=torch.float32).reshape(2, 50)
    mask = compute_event_mask(pressure, event_frac=0.1)  # top 10%
    assert mask.shape == pressure.shape
    assert mask.dtype == torch.bool
    # Top 10% of 100 = 10 positions marked True
    assert mask.sum().item() == 10
    # Those 10 positions are the highest-pressure ones
    assert (pressure[mask] >= pressure[~mask].max()).all()


def test_event_mask_handles_all_equal_pressure():
    pressure = torch.ones(2, 10)
    mask = compute_event_mask(pressure, event_frac=0.5)
    # Ties can resolve either way; just assert the count is correct
    assert mask.sum().item() == 10  # 0.5 * 20 total positions


def test_event_mask_empty_at_zero_frac_and_full_at_one():
    # Use strictly-positive pressure so event_frac=1.0 saturates the mask.
    # (compute_event_mask now refuses to mark non-positive positions, so
    # randn — which has negatives — would undercount here by design.)
    pressure = torch.randn(3, 4).abs() + 1e-3
    assert compute_event_mask(pressure, event_frac=0.0).sum().item() == 0
    assert compute_event_mask(pressure, event_frac=1.0).sum().item() == pressure.numel()


def test_compute_event_mask_returns_empty_for_uniform_zero_pressure():
    pressure = torch.zeros(4, 8)
    # Even at event_frac=1.0 (which would normally mark every position),
    # uniform zero pressure must not fabricate events out of zero-valued channels.
    for frac in (0.1, 0.5, 1.0):
        mask = compute_event_mask(pressure, event_frac=frac)
        assert mask.shape == pressure.shape
        assert mask.dtype == torch.bool
        assert mask.sum().item() == 0


def test_compute_event_mask_selects_only_strictly_positive_positions():
    # 20 positions total; only 3 are strictly positive. event_frac=0.5 would
    # nominally pick k=10, but only 3 strictly-positive positions exist,
    # so the mask must have exactly 3 True entries — those 3 positions —
    # and never pad with zero-valued channels.
    pressure = torch.zeros(4, 5)
    pressure[0, 0] = 1.5
    pressure[1, 2] = 0.25
    pressure[3, 4] = 3.0
    mask = compute_event_mask(pressure, event_frac=0.5)
    assert mask.shape == pressure.shape
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 3
    # Every True position is strictly positive.
    assert (pressure[mask] > 0).all()
    # And specifically those three positions.
    assert mask[0, 0].item() is True
    assert mask[1, 2].item() is True
    assert mask[3, 4].item() is True


from chaoscontrol.optim.criticality import compute_future_energy


def test_future_energy_matches_hand_computation_small_case():
    # states shape [B=1, T=4, D=2]
    states = torch.tensor([
        [[1.0, 0.0],
         [2.0, 1.0],
         [3.0, 2.0],
         [4.0, 3.0]]
    ])
    # Horizon H=2. For t=0, future = [t+1:t+3] = [[2,1],[3,2]]
    #   energy = mean([4, 1, 9, 4] grouped by channel) -> [mean(4, 9)=6.5, mean(1, 4)=2.5]
    # For t=1, future = [[3,2],[4,3]] -> [mean(9, 16)=12.5, mean(4, 9)=6.5]
    # For t=2, future = [[4,3]] only (t+H goes past end) -> [16.0, 9.0]
    # For t=3, no future window -> [0, 0] (convention: empty window -> zero)
    out = compute_future_energy(states, horizon_H=2)
    assert out.shape == (1, 4, 2)
    assert torch.allclose(out[0, 0], torch.tensor([6.5, 2.5]))
    assert torch.allclose(out[0, 1], torch.tensor([12.5, 6.5]))
    assert torch.allclose(out[0, 2], torch.tensor([16.0, 9.0]))
    assert torch.allclose(out[0, 3], torch.tensor([0.0, 0.0]))


def test_future_energy_vectorized_matches_reference_on_large_tensor():
    """Vectorized form must match the slow reference on a shape that
    actually matters."""
    torch.manual_seed(0)
    B, T, D, H = 4, 64, 32, 8
    states = torch.randn(B, T, D)
    # Reference (Python loop — the old slow one, re-written locally).
    def _ref(states, H):
        sq = states.pow(2)
        out = torch.zeros_like(sq)
        for t in range(T):
            s, e = t + 1, min(t + 1 + H, T)
            if s < e:
                out[:, t, :] = sq[:, s:e, :].mean(dim=1)
        return out
    expected = _ref(states, H)
    actual = compute_future_energy(states, horizon_H=H)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5), (
        f"max abs diff {(actual - expected).abs().max().item()}"
    )
