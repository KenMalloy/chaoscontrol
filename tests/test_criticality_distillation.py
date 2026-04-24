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


def test_add_step_evidence_writes_into_correct_slot_and_ttl_wraps():
    cd = CriticalityDistillation(num_layers=1, dim=4, trace_ttl_steps=3)
    e0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    e1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    e2 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    e3 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    cd.add_step_evidence(layer=0, step=0, evidence=e0[0], event_count=10.0)
    cd.add_step_evidence(layer=0, step=1, evidence=e1[0], event_count=11.0)
    cd.add_step_evidence(layer=0, step=2, evidence=e2[0], event_count=12.0)
    # At this point all 3 slots occupied for layer 0.
    assert set(cd.bank_step[0].tolist()) == {0, 1, 2}

    # Wrap: step=3 must evict the oldest (step=0).
    cd.add_step_evidence(layer=0, step=3, evidence=e3[0], event_count=13.0)
    assert set(cd.bank_step[0].tolist()) == {1, 2, 3}
    # Evidence for step=3 present
    slot = (cd.bank_step[0] == 3).nonzero(as_tuple=True)[0].item()
    assert torch.equal(cd.bank_evidence[0, slot], e3[0])
    assert cd.bank_event_count[0, slot].item() == pytest.approx(13.0)


def test_add_step_evidence_rejects_wrong_layer_or_shape():
    cd = CriticalityDistillation(num_layers=2, dim=4, trace_ttl_steps=3)
    with pytest.raises((IndexError, ValueError)):
        cd.add_step_evidence(layer=5, step=0, evidence=torch.zeros(4), event_count=1.0)
    with pytest.raises((RuntimeError, ValueError)):
        cd.add_step_evidence(layer=0, step=0, evidence=torch.zeros(8), event_count=1.0)


def test_score_age_weights_match_hand_computation():
    cd = CriticalityDistillation(
        num_layers=1,
        dim=4,
        trace_ttl_steps=3,
        trace_half_life_steps=2.0,  # half-life = 2 steps for hand math
    )
    # One evidence vector at step=0, another at step=2.
    cd.add_step_evidence(
        layer=0, step=0,
        evidence=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        event_count=1.0,
    )
    cd.add_step_evidence(
        layer=0, step=2,
        evidence=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        event_count=1.0,
    )
    # Score at current_step=4.
    # age_weight uses half-life formulation: w = 2 ** (-age / half_life)
    # age_0 = 4 (entry at step 0) -> w_0 = 2**(-4/2) = 0.25
    # age_1 = 2 (entry at step 2) -> w_1 = 2**(-2/2) = 0.5
    # expected: (0.25 * [1,0,0,0] + 0.5 * [0,1,0,0]) / (0.25 + 0.5)
    #         = [1/3, 2/3, 0, 0]
    score = cd.score(current_step=4)
    expected = torch.tensor([[1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0]])
    assert torch.allclose(score, expected, atol=1e-6), f"{score} != {expected}"


def test_score_returns_zeros_when_bank_empty():
    cd = CriticalityDistillation(num_layers=2, dim=3, trace_ttl_steps=4)
    score = cd.score(current_step=0)
    assert score.shape == (2, 3)
    assert torch.equal(score, torch.zeros_like(score))


def test_state_dict_round_trip_preserves_bank_and_baseline():
    cd1 = CriticalityDistillation(num_layers=1, dim=4, trace_ttl_steps=3, trace_half_life_steps=2.0)
    cd1.add_step_evidence(layer=0, step=0, evidence=torch.tensor([1.0, 0.0, 0.0, 0.0]), event_count=3.0)
    cd1.add_step_evidence(layer=0, step=1, evidence=torch.tensor([0.0, 1.0, 0.0, 0.0]), event_count=5.0)
    cd1.baseline_future_energy.fill_(0.42)
    cd1.seat_mask[0, 0] = True

    sd = cd1.state_dict()

    cd2 = CriticalityDistillation(num_layers=1, dim=4, trace_ttl_steps=3, trace_half_life_steps=2.0)
    cd2.load_state_dict(sd)

    assert torch.equal(cd2.bank_evidence, cd1.bank_evidence)
    assert torch.equal(cd2.bank_step, cd1.bank_step)
    assert torch.equal(cd2.bank_event_count, cd1.bank_event_count)
    assert torch.equal(cd2.baseline_future_energy, cd1.baseline_future_energy)
    assert torch.equal(cd2.seat_mask, cd1.seat_mask)

    # Score must match after round-trip.
    assert torch.allclose(cd2.score(current_step=2), cd1.score(current_step=2))


def test_update_baseline_ema_only_reads_non_event_positions():
    cd = CriticalityDistillation(
        num_layers=1, dim=2, trace_ttl_steps=4,
        baseline_ema_decay=0.5,
    )
    # future_energy shape [B=1, T=4, D=2]
    future_energy = torch.tensor([[
        [1.0, 10.0],  # t=0, event
        [2.0, 20.0],  # t=1, non-event
        [3.0, 30.0],  # t=2, non-event
        [4.0, 40.0],  # t=3, event
    ]])
    event_mask = torch.tensor([[True, False, False, True]])
    # Mean over non-event positions: channel 0 -> (2+3)/2 = 2.5
    #                                channel 1 -> (20+30)/2 = 25.0
    # Baseline starts at zero. One update with decay=0.5:
    #   new = 0.5 * old + 0.5 * obs = 0.5 * 0 + 0.5 * [2.5, 25.0] = [1.25, 12.5]
    cd.update_baseline_ema(layer=0, future_energy=future_energy, event_mask=event_mask)
    assert torch.allclose(cd.baseline_future_energy[0], torch.tensor([1.25, 12.5]))


def test_update_baseline_ema_no_nonevent_positions_is_noop():
    cd = CriticalityDistillation(num_layers=1, dim=2, trace_ttl_steps=4, baseline_ema_decay=0.9)
    cd.baseline_future_energy.fill_(7.0)
    future_energy = torch.randn(1, 4, 2)
    event_mask = torch.ones(1, 4, dtype=torch.bool)  # every position is an event
    cd.update_baseline_ema(layer=0, future_energy=future_energy, event_mask=event_mask)
    assert torch.equal(cd.baseline_future_energy[0], torch.full((2,), 7.0))


def test_ingest_step_writes_one_entry_per_layer_with_events():
    cd = CriticalityDistillation(
        num_layers=2, dim=3, trace_ttl_steps=4,
        baseline_ema_decay=0.0,  # baseline = observation (no smoothing) for easier math
    )
    states_l0 = torch.tensor([[
        [1.0, 0.0, 0.0],  # t=0 event
        [0.0, 1.0, 0.0],  # t=1 non-event
        [0.0, 0.0, 1.0],  # t=2 non-event (future window for event at t=0 covers t=1..end)
    ]])
    states_l1 = torch.tensor([[
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]])
    pressure = torch.tensor([[[10.0, 0.0, 0.0]]]).reshape(1, 3)  # top 1/3 at t=0
    # event_frac chosen so only t=0 is an event; event_mask = [True, False, False]
    cd.ingest_step(
        step=0,
        pressure=pressure,  # [B=1, T=3]
        states_per_layer=[states_l0, states_l1],
        horizon_H=2,
        event_frac=0.34,  # round(0.34 * 3) = 1 position
    )
    # Layer 0: future at t=0 over [t+1:t+3] = rows 1 and 2 -> mean([[0,1,0],[0,0,1]]**2, dim=0) = [0, 0.5, 0.5]
    # Baseline from non-event positions t=1..2: future at t=1 over [t+1:t+3] = [row 2] -> [0, 0, 1]
    #                                            future at t=2 over [t+1:t+3] = [] -> [0, 0, 0]
    # non-event future mean = ([0,0,1] + [0,0,0]) / 2 = [0, 0, 0.5]
    # With decay=0 baseline = observation -> [0, 0, 0.5]
    # excess = relu([0, 0.5, 0.5] - [0, 0, 0.5]) = [0, 0.5, 0]
    # Aggregated over 1 event position = [0, 0.5, 0]
    l0_slot = (cd.bank_step[0] == 0).nonzero(as_tuple=True)[0].item()
    assert torch.allclose(cd.bank_evidence[0, l0_slot], torch.tensor([0.0, 0.5, 0.0]), atol=1e-6)
    assert cd.bank_event_count[0, l0_slot].item() == pytest.approx(1.0)


def test_ingest_step_no_events_writes_nothing():
    cd = CriticalityDistillation(num_layers=1, dim=3, trace_ttl_steps=4)
    states = [torch.randn(1, 3, 3)]
    pressure = torch.zeros(1, 3)
    cd.ingest_step(step=0, pressure=pressure, states_per_layer=states, horizon_H=2, event_frac=0.0)
    assert (cd.bank_step == -1).all()


def test_allocate_seats_respects_evidence_gate():
    cd = CriticalityDistillation(
        num_layers=1, dim=10, trace_ttl_steps=4,
        trace_half_life_steps=1.0,
        criticality_budget_frac=0.3,
        min_weighted_events_per_layer=100.0,  # unreachable with small input
    )
    cd.add_step_evidence(layer=0, step=0, evidence=torch.ones(10), event_count=1.0)
    cd.allocate_seats(current_step=1)
    assert not cd.seat_mask[0].any(), "evidence gate must suppress seat assignment"


def test_allocate_seats_top_k_when_gate_passes():
    cd = CriticalityDistillation(
        num_layers=1, dim=10, trace_ttl_steps=4,
        trace_half_life_steps=100.0,  # slow aging, so recent events count fully
        criticality_budget_frac=0.3,  # 3 seats per layer
        min_weighted_events_per_layer=1.0,
    )
    # Channels 2, 5, 7 have highest evidence.
    evidence = torch.zeros(10)
    evidence[2] = 3.0
    evidence[5] = 5.0
    evidence[7] = 1.0
    evidence[0] = 0.5
    cd.add_step_evidence(layer=0, step=0, evidence=evidence, event_count=10.0)
    cd.allocate_seats(current_step=1)
    assert cd.seat_mask[0].sum().item() == 3
    # Top-3 by score: channels 5, 2, 7 (in order of magnitude)
    assert cd.seat_mask[0, 5].item() is True
    assert cd.seat_mask[0, 2].item() is True
    assert cd.seat_mask[0, 7].item() is True


def test_criticality_loss_is_zero_when_no_seats():
    cd = CriticalityDistillation(num_layers=1, dim=4, trace_ttl_steps=2)
    # seat_mask is all False by default.
    log_a_per_layer = [torch.zeros(4, requires_grad=True)]
    loss = cd.criticality_loss(log_a_per_layer)
    assert loss.item() == 0.0


def test_criticality_loss_values_match_hand_mse_on_seats_only():
    cd = CriticalityDistillation(
        num_layers=1, dim=4, trace_ttl_steps=2,
        critical_value=0.9,
        criticality_distill_weight=1.0,  # isolate per-layer mse
    )
    cd.seat_mask[0] = torch.tensor([True, False, True, False])
    # 1 - sigmoid(log_a=0) = 0.5 on every channel.
    log_a_per_layer = [torch.zeros(4, requires_grad=True)]
    # For seat channels: (0.5 - 0.9)^2 = 0.16. Mean over 2 seats = 0.16.
    loss = cd.criticality_loss(log_a_per_layer)
    assert torch.allclose(loss, torch.tensor(0.16), atol=1e-6)
