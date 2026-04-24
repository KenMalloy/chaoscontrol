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
    pressure = torch.randn(3, 4)
    assert compute_event_mask(pressure, event_frac=0.0).sum().item() == 0
    assert compute_event_mask(pressure, event_frac=1.0).sum().item() == pressure.numel()


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
