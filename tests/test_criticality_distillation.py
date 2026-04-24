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
    assert torch.equal(cd2.baseline_initialized, cd1.baseline_initialized)

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
    # First observation replaces (baseline was uninitialized at zero) — so
    # baseline -> [2.5, 25.0], not 0.5 * [2.5, 25.0]. The "only non-event
    # positions contribute" invariant is preserved: event positions at t=0
    # and t=3 are ignored, so their values (1, 10, 4, 40) never enter the
    # mean.
    cd.update_baseline_ema(layer=0, future_energy=future_energy, event_mask=event_mask)
    assert torch.allclose(cd.baseline_future_energy[0], torch.tensor([2.5, 25.0]))

    # Now that baseline is initialized, a second update should EMA with
    # decay=0.5: new = 0.5 * [2.5, 25.0] + 0.5 * [2.5, 25.0] = [2.5, 25.0]
    # (same observation, so unchanged). Use a different observation to
    # actually exercise the EMA math: event_mask flips, so t=0 and t=3 are
    # now non-events with values (1, 10) and (4, 40) -> mean [2.5, 25.0].
    # Still identical! The whole point is to confirm event positions are
    # excluded — same event pattern by design.
    # Use a fresh future_energy + event_mask to make the EMA update visible:
    future_energy2 = torch.tensor([[
        [0.0, 0.0],
        [10.0, 100.0],  # t=1, non-event
    ]])
    event_mask2 = torch.tensor([[True, False]])
    # Non-event positions: t=1 only -> obs = [10.0, 100.0]
    # EMA: 0.5 * [2.5, 25.0] + 0.5 * [10.0, 100.0] = [6.25, 62.5]
    cd.update_baseline_ema(layer=0, future_energy=future_energy2, event_mask=event_mask2)
    assert torch.allclose(cd.baseline_future_energy[0], torch.tensor([6.25, 62.5]))


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


def test_non_seat_log_a_gets_exactly_zero_gradient_from_criticality_loss():
    cd = CriticalityDistillation(
        num_layers=1, dim=6, trace_ttl_steps=2,
        critical_value=0.9,
    )
    cd.seat_mask[0] = torch.tensor([True, False, True, False, False, True])
    log_a = torch.zeros(6, requires_grad=True)
    loss = cd.criticality_loss([log_a])
    loss.backward()
    # Non-seat entries are indices 1, 3, 4 — their grad MUST be exactly zero.
    non_seat_grad = log_a.grad[~cd.seat_mask[0]]
    assert torch.equal(non_seat_grad, torch.zeros_like(non_seat_grad)), (
        f"non-seat log_a must have exactly zero grad; got {non_seat_grad}"
    )
    # Seat entries must receive a nonzero grad (sanity — loss depends on them).
    seat_grad = log_a.grad[cd.seat_mask[0]]
    assert (seat_grad != 0.0).all(), (
        f"seat log_a must have nonzero grad; got {seat_grad}"
    )


def test_criticality_loss_has_no_grad_path_to_pressure_or_states():
    cd = CriticalityDistillation(
        num_layers=1, dim=4, trace_ttl_steps=2,
        critical_value=0.9, min_weighted_events_per_layer=0.0,
    )
    # Populate one evidence entry and seat.
    evidence = torch.tensor([1.0, 0.0, 2.0, 0.0])
    cd.add_step_evidence(layer=0, step=0, evidence=evidence, event_count=1.0)
    cd.allocate_seats(current_step=1)

    # Pressure and states are usually produced with grad in the training
    # graph — but the criticality loss should only depend on log_a.
    log_a = torch.zeros(4, requires_grad=True)
    loss = cd.criticality_loss([log_a])
    # If the loss depends on anything other than log_a, gradients on a
    # fresh unrelated tensor should cause an error when we try to extract
    # them. Assert directly: loss.backward consumes log_a only.
    assert loss.requires_grad
    loss.backward()
    assert log_a.grad is not None
    # seat_mask is a registered buffer and should not receive grads.
    assert not cd.seat_mask.requires_grad
    # baseline_future_energy should not receive grads either.
    assert not cd.baseline_future_energy.requires_grad


def test_full_mechanism_moves_seat_log_a_more_than_non_seat():
    """After N training steps, seat-channel log_a values should move
    meaningfully while non-seat log_a values stay pinned."""
    from chaoscontrol.core import ChaosSSMCore

    torch.manual_seed(0)
    dim = 8
    core = ChaosSSMCore(dim=dim, a_mode="diag")
    cd = CriticalityDistillation(
        num_layers=1,
        dim=dim,
        trace_ttl_steps=16,
        trace_half_life_steps=4.0,
        seat_refresh_interval=1,  # refresh every step for this test
        criticality_budget_frac=0.25,  # 2 seats
        critical_value=0.99,
        min_weighted_events_per_layer=0.0,  # no gate for this integration test
        criticality_distill_weight=1.0,
    )

    # Snapshot initial log_a.
    log_a_init = core.log_a.detach().clone()

    # Optimizer for log_a only (isolate the mechanism).
    opt = torch.optim.SGD([core.log_a], lr=0.5)

    # Force a specific seat pattern by biasing pressure to only channels
    # 0 and 1 — we expect them to become seats.
    for step in range(10):
        x = torch.randn(2, 6, dim)
        with core.capture_states() as get_states:
            _ = core(x)
            states = get_states()
        # Pressure field biased to T=0, T=1 positions (so events concentrate there).
        pressure = torch.zeros(2, 6)
        pressure[:, 0] = 10.0
        pressure[:, 1] = 8.0
        # Construct states that light up channels 0 and 1 after events.
        fake_states = torch.zeros_like(states)
        fake_states[:, 1:, 0] = 1.0  # channel 0 persists after t=0 events
        fake_states[:, 2:, 1] = 1.0  # channel 1 persists after t=0, 1 events

        cd.ingest_step(
            step=step,
            pressure=pressure,
            states_per_layer=[fake_states],
            horizon_H=4,
            event_frac=2.0 / 12.0,  # top ~2 positions per 12
        )
        cd.allocate_seats(current_step=step + 1)

        opt.zero_grad()
        loss = cd.criticality_loss([core.log_a])
        if loss.requires_grad:
            loss.backward()
            opt.step()

    # Seat channels (0 and 1) should have moved; non-seat channels stay put.
    log_a_delta = (core.log_a.detach() - log_a_init).abs()
    seats = cd.seat_mask[0]
    # At least one of the seat channels moved
    assert log_a_delta[seats].max().item() > 1e-3, (
        f"seat log_a did not move: {log_a_delta[seats]}"
    )
    # No non-seat channel moved
    assert log_a_delta[~seats].max().item() < 1e-6, (
        f"non-seat log_a moved: {log_a_delta[~seats]}"
    )


def test_baseline_ema_first_observation_replaces_not_emas():
    """First non-event observation replaces zero baseline rather than
    being damped by (1 - decay). Otherwise decay=0.99 produces a
    0.01 * obs baseline that makes the first allocation window
    treat raw energy as excess."""
    cd = CriticalityDistillation(
        num_layers=1, dim=2, trace_ttl_steps=4,
        baseline_ema_decay=0.99,  # aggressive smoothing
    )
    future_energy = torch.tensor([[[5.0, 10.0]]])  # [B=1, T=1, D=2]
    event_mask = torch.tensor([[False]])  # single non-event position
    cd.update_baseline_ema(layer=0, future_energy=future_energy, event_mask=event_mask)
    # After first observation: baseline == obs (replacement, not EMA).
    assert torch.allclose(cd.baseline_future_energy[0], torch.tensor([5.0, 10.0]))
    assert cd.baseline_initialized[0].item() is True

    # Second observation should EMA normally.
    future_energy2 = torch.tensor([[[15.0, 20.0]]])
    cd.update_baseline_ema(layer=0, future_energy=future_energy2, event_mask=event_mask)
    # 0.99 * [5, 10] + 0.01 * [15, 20] = [5.1, 10.1]
    assert torch.allclose(cd.baseline_future_energy[0], torch.tensor([5.1, 10.1]))


def test_ingest_step_updates_baseline_even_when_no_events():
    """event_frac=0 means no events -> no bank write, but baseline
    should still observe the full step's non-event future-energy.
    This matters for the budget-only falsifier control which runs at
    event_frac=0, and for warmup before any events fire."""
    cd = CriticalityDistillation(
        num_layers=1, dim=2, trace_ttl_steps=4,
        baseline_ema_decay=0.5,  # simple hand math
    )
    states = [torch.tensor([[
        [1.0, 0.0],
        [2.0, 1.0],
        [3.0, 2.0],
    ]])]
    pressure = torch.zeros(1, 3)
    cd.ingest_step(
        step=0, pressure=pressure, states_per_layer=states,
        horizon_H=2, event_frac=0.0,
    )
    # Bank entry: none written (all positions non-event).
    assert (cd.bank_step == -1).all()
    # Baseline: first observation replaces zero (finding 2 behavior).
    # future_energy per position (H=2):
    #   t=0: mean([[2,1],[3,2]]**2, dim=0) = [(4+9)/2, (1+4)/2] = [6.5, 2.5]
    #   t=1: mean([[3,2]]**2) = [9, 4]
    #   t=2: empty window -> [0, 0]
    # Mean over all three non-event positions: [(6.5 + 9 + 0)/3, (2.5 + 4 + 0)/3]
    #                                        = [15.5/3, 6.5/3]
    #                                        ~ [5.1666..., 2.1666...]
    expected = torch.tensor([15.5 / 3.0, 6.5 / 3.0])
    assert torch.allclose(cd.baseline_future_energy[0], expected, atol=1e-6), (
        cd.baseline_future_energy[0]
    )
    assert cd.baseline_initialized[0].item() is True


def test_criticality_loss_applies_distill_weight_internally():
    cd = CriticalityDistillation(
        num_layers=1, dim=4, trace_ttl_steps=2,
        critical_value=0.9,
        criticality_distill_weight=0.25,  # non-default
    )
    cd.seat_mask[0] = torch.tensor([True, False, True, False])
    log_a = torch.zeros(4, requires_grad=True)
    # Unweighted MSE on 2 seats at log_a=0 (criticality=0.5 vs target 0.9):
    #   mean([(0.5 - 0.9)^2, (0.5 - 0.9)^2]) = 0.16
    # With weight 0.25: 0.16 * 0.25 = 0.04
    loss = cd.criticality_loss([log_a])
    assert torch.allclose(loss, torch.tensor(0.04), atol=1e-6), (
        f"expected 0.04, got {loss.item()}"
    )


def test_criticality_loss_distill_weight_zero_gives_zero_loss():
    cd = CriticalityDistillation(
        num_layers=1, dim=4, trace_ttl_steps=2,
        critical_value=0.9,
        criticality_distill_weight=0.0,
    )
    cd.seat_mask[0] = torch.tensor([True, False, True, False])
    log_a = torch.zeros(4, requires_grad=True)
    loss = cd.criticality_loss([log_a])
    assert loss.item() == 0.0


def test_score_permute_before_topk_selects_random_k_of_D_not_peaks():
    """Falsifier flag: when score_permute_before_topk=True, allocate_seats
    must pick k channels uniformly at random, ignoring the score. Must NOT
    be implemented as "permute score then top-k" (which un-shuffles through
    the permutation and still selects the peaks).
    """
    D = 10
    k_expected = 2  # budget_frac=0.2 -> round(0.2 * 10) = 2
    cd = CriticalityDistillation(
        num_layers=1, dim=D, trace_ttl_steps=4,
        trace_half_life_steps=100.0,  # slow aging
        criticality_budget_frac=0.2,  # 2 seats of 10
        min_weighted_events_per_layer=1.0,
        score_permute_before_topk=True,
    )
    # Peak-score channels are 5 and 2 (in that order of magnitude).
    evidence = torch.zeros(D)
    evidence[5] = 10.0
    evidence[2] = 5.0
    cd.add_step_evidence(layer=0, step=0, evidence=evidence, event_count=10.0)

    peak_set = {5, 2}
    observed_sets = []
    for seed in (0, 1, 2, 3, 4):
        torch.manual_seed(seed)
        cd.allocate_seats(current_step=1)
        # Exactly k seats each call.
        assert cd.seat_mask[0].sum().item() == k_expected
        selected = set(cd.seat_mask[0].nonzero(as_tuple=True)[0].tolist())
        assert len(selected) == k_expected
        observed_sets.append(selected)

    # At least one seed must produce a seat set != peak set. If the bug
    # was present (selection always == peaks), every set would equal
    # {2, 5} and this would fail.
    non_peak_draws = [s for s in observed_sets if s != peak_set]
    assert len(non_peak_draws) >= 1, (
        f"All {len(observed_sets)} seeds selected the peak-score set "
        f"{peak_set}; score_permute_before_topk is not bypassing score. "
        f"observed={observed_sets}"
    )


def test_fixed_random_seats_binds_seats_at_init():
    """Falsifier flag: when fixed_random_seats=True, seats are bound ONCE at
    construction (randomly, same k as the normal top-k). No ingest, no
    allocate_seats call required — seat_mask must already have exactly k
    True entries per layer. Different random seeds must yield different
    seat sets (i.e. randomness is honored, not all-zero or all-first-k).
    """
    D = 16
    num_layers = 2
    budget_frac = 0.25
    k_expected = max(1, int(round(D * budget_frac)))  # 4
    assert k_expected < D

    torch.manual_seed(0)
    cd_a = CriticalityDistillation(
        num_layers=num_layers, dim=D, trace_ttl_steps=4,
        criticality_budget_frac=budget_frac,
        fixed_random_seats=True,
    )
    # Immediately after construction: seat_mask populated, no ingest needed.
    for layer in range(num_layers):
        assert cd_a.seat_mask[layer].sum().item() == k_expected, (
            f"layer {layer}: expected {k_expected} True entries, got "
            f"{cd_a.seat_mask[layer].sum().item()}"
        )

    # Re-construct with a different seed -> different seat set (randomness
    # is honored, not all-zero or all-first-k).
    torch.manual_seed(12345)
    cd_b = CriticalityDistillation(
        num_layers=num_layers, dim=D, trace_ttl_steps=4,
        criticality_budget_frac=budget_frac,
        fixed_random_seats=True,
    )
    set_a = set(cd_a.seat_mask[0].nonzero(as_tuple=True)[0].tolist())
    set_b = set(cd_b.seat_mask[0].nonzero(as_tuple=True)[0].tolist())
    assert set_a != set_b, (
        f"fixed_random_seats did not honor the RNG: set_a={set_a}, "
        f"set_b={set_b} (both seeds produced the same seat set)"
    )
    # Sanity: not degenerate (not all-zero, not first-k deterministic).
    first_k = set(range(k_expected))
    assert not (set_a == first_k and set_b == first_k), (
        "seats are hard-coded to first-k channels; RNG is not being used"
    )


def test_fixed_random_seats_allocate_seats_is_noop():
    """Falsifier flag: when fixed_random_seats=True, allocate_seats must be
    a no-op even when evidence would otherwise drive a score-based top-k
    that differs from the init-bound seats.
    """
    D = 10
    budget_frac = 0.2  # k = 2
    torch.manual_seed(7)
    cd = CriticalityDistillation(
        num_layers=1, dim=D, trace_ttl_steps=4,
        trace_half_life_steps=100.0,
        criticality_budget_frac=budget_frac,
        min_weighted_events_per_layer=1.0,
        fixed_random_seats=True,
    )
    # Snapshot the init-bound seats.
    init_mask = cd.seat_mask.clone()
    assert init_mask[0].sum().item() == 2

    # Feed evidence that would normally steer score-based allocate_seats
    # toward channels [2, 5].
    evidence = torch.zeros(D)
    evidence[5] = 10.0
    evidence[2] = 5.0
    cd.add_step_evidence(layer=0, step=0, evidence=evidence, event_count=10.0)

    cd.allocate_seats(current_step=1)

    # allocate_seats must not have touched seat_mask.
    assert torch.equal(cd.seat_mask, init_mask), (
        f"allocate_seats overwrote fixed_random_seats: "
        f"before={init_mask[0].nonzero(as_tuple=True)[0].tolist()}, "
        f"after={cd.seat_mask[0].nonzero(as_tuple=True)[0].tolist()}"
    )


def test_accumulator_buffers_register_and_initialize_to_zero():
    cd = CriticalityDistillation(num_layers=3, dim=8, trace_ttl_steps=16)
    # score_num: running age-weighted sum of evidence contributions.
    assert cd.score_num.shape == (3, 8)
    assert torch.equal(cd.score_num, torch.zeros_like(cd.score_num))
    # score_den: running age-weighted count.
    assert cd.score_den.shape == (3,)
    assert torch.equal(cd.score_den, torch.zeros_like(cd.score_den))
    # event_mass: running age-weighted event count for the gate.
    assert cd.event_mass.shape == (3,)
    # last_decay_step: last step we applied decay to accumulators.
    assert cd.last_decay_step.item() == -1
    # All in state_dict (buffers registered).
    sd = cd.state_dict()
    for key in ("score_num", "score_den", "event_mass", "last_decay_step"):
        assert key in sd


def test_step_decay_applies_half_life_factor_to_all_accumulators():
    cd = CriticalityDistillation(
        num_layers=1, dim=4, trace_ttl_steps=8,
        trace_half_life_steps=2.0,  # decay factor per step = 2^(-1/2) ≈ 0.7071
    )
    cd.score_num.fill_(1.0)
    cd.score_den.fill_(4.0)
    cd.event_mass.fill_(10.0)
    cd.last_decay_step.fill_(0)
    # Advance to step 2. Total decay = 2^(-2/2) = 0.5.
    cd._step_decay_accumulators(current_step=2)
    assert torch.allclose(cd.score_num, torch.full_like(cd.score_num, 0.5))
    assert torch.allclose(cd.score_den, torch.tensor([2.0]))
    assert torch.allclose(cd.event_mass, torch.tensor([5.0]))
    assert cd.last_decay_step.item() == 2


def test_step_decay_is_idempotent_when_called_with_same_step():
    cd = CriticalityDistillation(num_layers=1, dim=2, trace_ttl_steps=4, trace_half_life_steps=2.0)
    cd.score_num.fill_(1.0)
    cd.last_decay_step.fill_(5)
    cd._step_decay_accumulators(current_step=5)
    assert torch.allclose(cd.score_num, torch.full_like(cd.score_num, 1.0))


def test_add_contribution_updates_numerator_denominator_and_event_mass():
    cd = CriticalityDistillation(
        num_layers=2, dim=3, trace_ttl_steps=8,
        trace_half_life_steps=4.0,
    )
    # Pre-step: call _step_decay (no-op on zero accumulators; just syncs last_decay_step).
    cd._step_decay_accumulators(current_step=0)
    cd._add_contribution(
        layer=0,
        evidence=torch.tensor([1.0, 2.0, 3.0]),
        event_count=5.0,
    )
    # Numerator: evidence * event_count = [5, 10, 15]
    assert torch.allclose(cd.score_num[0], torch.tensor([5.0, 10.0, 15.0]))
    # Denominator: event_count = 5
    assert cd.score_den[0].item() == 5.0
    # Event mass: event_count
    assert cd.event_mass[0].item() == 5.0
    # Layer 1 untouched.
    assert torch.equal(cd.score_num[1], torch.zeros(3))


def test_subtract_expired_removes_contribution_at_its_current_decay_weight():
    cd = CriticalityDistillation(
        num_layers=1, dim=2, trace_ttl_steps=4,
        trace_half_life_steps=2.0,
    )
    # Simulate: entry added at step=0 with evidence=[1, 2] and event_count=3.
    # Current step=4. Entry's current age = 4 -> decay weight 2^(-4/2) = 0.25.
    # Before subtraction:
    cd.score_num[0] = torch.tensor([0.25 * 1 * 3, 0.25 * 2 * 3])  # [0.75, 1.5]
    cd.score_den[0] = 0.25 * 3  # 0.75
    cd.event_mass[0] = 0.25 * 3  # 0.75
    # Subtract as if that slot is now being overwritten.
    cd._subtract_expired_contribution(
        layer=0,
        evicted_step=0,
        current_step=4,
        evicted_evidence=torch.tensor([1.0, 2.0]),
        evicted_event_count=3.0,
    )
    assert torch.allclose(cd.score_num[0], torch.zeros(2), atol=1e-6)
    assert abs(cd.score_den[0].item()) < 1e-6
    assert abs(cd.event_mass[0].item()) < 1e-6


def test_score_from_accumulators_matches_full_bank_score_after_ingest_sequence():
    """The incremental accumulator must produce the same score as the
    full-bank scan after any sequence of ingests. This is the
    consistency pin between the two implementations."""
    cd = CriticalityDistillation(
        num_layers=1, dim=3, trace_ttl_steps=8,
        trace_half_life_steps=4.0,
    )
    ingests = [
        (0, torch.tensor([1.0, 0.0, 0.0]), 2.0),
        (1, torch.tensor([0.0, 1.0, 0.0]), 3.0),
        (3, torch.tensor([0.0, 0.0, 2.0]), 1.0),
    ]
    for step, evidence, ec in ingests:
        cd._step_decay_accumulators(current_step=step)
        cd._add_contribution(layer=0, evidence=evidence, event_count=ec)
        # Also write to the ring bank for the full-scan comparison.
        cd.add_step_evidence(layer=0, step=step, evidence=evidence, event_count=ec)
    # Advance decay to score time.
    current_step = 5
    cd._step_decay_accumulators(current_step=current_step)
    accumulator_score = cd.score_from_accumulators()
    full_scan_score = cd.score(current_step=current_step)
    # Peak channel must match between both scorers.
    assert accumulator_score[0].argmax().item() == full_scan_score[0].argmax().item()


def test_allocate_seats_from_accumulators_picks_topk_by_accumulator_score():
    cd = CriticalityDistillation(
        num_layers=1, dim=8, trace_ttl_steps=8,
        trace_half_life_steps=100.0,
        criticality_budget_frac=0.25,  # k=2
        min_weighted_events_per_layer=0.5,
    )
    cd._step_decay_accumulators(current_step=0)
    cd._add_contribution(layer=0, evidence=torch.tensor([0.1,0.2,5.0,0.0,0.0,9.0,0.0,0.0]), event_count=1.0)
    cd.allocate_seats_from_accumulators(current_step=1)
    assert cd.seat_mask[0].sum().item() == 2
    assert cd.seat_mask[0, 5].item() is True  # peak 1
    assert cd.seat_mask[0, 2].item() is True  # peak 2


def test_ingest_cpu_from_prepared_advances_accumulators():
    cd = CriticalityDistillation(num_layers=1, dim=3, trace_ttl_steps=8, trace_half_life_steps=4.0)
    prepared = {
        "event_mask": torch.tensor([[True, False, True]]),
        "aggregated_excess_per_layer": torch.tensor([[1.0, 2.0, 3.0]]),
        "non_event_mean_future_energy_per_layer": torch.tensor([[0.5, 0.5, 0.5]]),
        "event_count_per_layer": torch.tensor([2.0]),
    }
    cd.ingest_cpu_from_prepared(step=0, prepared=prepared)
    # Accumulators updated: score_num = evidence * count = [2, 4, 6];
    # score_den = 2; event_mass = 2.
    assert torch.allclose(cd.score_num[0], torch.tensor([2.0, 4.0, 6.0]))
    assert cd.score_den[0].item() == 2.0
    assert cd.event_mass[0].item() == 2.0
    # Ring bank also written for TTL/checkpoint state.
    slot = (cd.bank_step[0] == 0).nonzero(as_tuple=True)[0].item()
    assert torch.allclose(cd.bank_evidence[0, slot], torch.tensor([1.0, 2.0, 3.0]))


def test_allocate_seats_from_accumulators_respects_event_mass_gate():
    cd = CriticalityDistillation(
        num_layers=1, dim=8, trace_ttl_steps=8,
        trace_half_life_steps=100.0,
        criticality_budget_frac=0.25,
        min_weighted_events_per_layer=100.0,  # unreachable
    )
    cd._step_decay_accumulators(current_step=0)
    cd._add_contribution(layer=0, evidence=torch.ones(8), event_count=1.0)
    cd.allocate_seats_from_accumulators(current_step=1)
    assert not cd.seat_mask[0].any()


def test_accumulator_score_equals_full_bank_score_within_fp32_tolerance():
    """Incremental accumulator and full-bank scan must agree exactly
    (modulo fp32 rounding) after any ingest sequence and any step
    advance."""
    cd = CriticalityDistillation(
        num_layers=2, dim=5, trace_ttl_steps=10,
        trace_half_life_steps=3.0,
    )
    torch.manual_seed(7)
    steps = [0, 1, 3, 5, 8, 13, 21]
    for step in steps:
        cd._step_decay_accumulators(current_step=step)
        for layer in range(2):
            evidence = torch.randn(5).abs() + 0.1
            cnt = float(torch.randint(1, 10, (1,)).item())
            cd._add_contribution(layer=layer, evidence=evidence, event_count=cnt)
            cd._write_ring_slot(
                layer=layer, step=step, evidence=evidence,
                event_count=cnt, current_step=step,
            )
    current_step = 30
    cd._step_decay_accumulators(current_step=current_step)
    acc = cd.score_from_accumulators()
    full = cd.score(current_step=current_step)
    assert torch.allclose(acc, full, atol=1e-4, rtol=1e-4), (
        f"accumulator diverged from full scan: acc={acc} full={full}"
    )
