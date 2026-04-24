import torch
import pytest
from chaoscontrol.optim.criticality import CriticalityDistillation


def test_constructs_with_expected_buffer_shapes():
    cd = CriticalityDistillation(
        num_layers=3,
        dim=16,
        trace_ttl_steps=8,
        trace_half_life_steps=4,
        seat_refresh_interval=2,
        criticality_budget_frac=0.25,
        critical_value=0.95,
        min_weighted_events_per_layer=1.0,
        criticality_distill_weight=1e-3,
        baseline_ema_decay=0.99,
    )
    # Evidence bank: [num_layers, trace_ttl_steps, dim]
    assert cd.bank_evidence.shape == (3, 8, 16)
    # Per-slot step counter: -1 means "empty"
    assert cd.bank_step.shape == (3, 8)
    assert cd.bank_event_count.shape == (3, 8)
    # Baseline EMA per layer per channel
    assert cd.baseline_future_energy.shape == (3, 16)
    # Current seat assignment per layer (bool)
    assert cd.seat_mask.shape == (3, 16)
    assert cd.seat_mask.dtype == torch.bool
    # All buffers start zero / empty
    assert torch.equal(cd.bank_evidence, torch.zeros_like(cd.bank_evidence))
    assert torch.equal(cd.bank_step, torch.full_like(cd.bank_step, -1))
    assert torch.equal(cd.bank_event_count, torch.zeros_like(cd.bank_event_count))
    assert not cd.seat_mask.any()


def test_buffers_register_for_state_dict():
    cd = CriticalityDistillation(num_layers=2, dim=4, trace_ttl_steps=3)
    sd = cd.state_dict()
    assert "bank_evidence" in sd
    assert "bank_step" in sd
    assert "bank_event_count" in sd
    assert "baseline_future_energy" in sd
    assert "seat_mask" in sd
