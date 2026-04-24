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
