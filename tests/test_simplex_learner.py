"""S2: SimplexOnlineLearner — REINFORCE backward over the simplex policy.

The load-bearing test is ``test_simplex_gradient_matches_torch_autograd_reference``:
it compares the C++ kernel's REINFORCE gradient against a torch.autograd
reference computing the same loss on the same forward graph. Same shape as
the C7 parity test that gave us the strongest correctness guarantee in the
per-slot phase.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


N = 16
K_V = 16
K_E = 1
K_S = 4
H = 32


def _build_weights_struct(weights_dict):
    w = _ext.SimplexWeights()
    w.K_v = K_V
    w.K_e = K_E
    w.K_s = K_S
    w.H = H
    w.N = N
    w.n_heads = int(weights_dict.get("n_heads", 0))
    w.W_vp = weights_dict["W_vp"].flatten().tolist()
    w.b_vp = weights_dict["b_vp"].tolist()
    w.W_lh = weights_dict["W_lh"].tolist()
    w.b_lh = float(weights_dict["b_lh"])
    w.W_sb = weights_dict["W_sb"].tolist()
    w.alpha = float(weights_dict["alpha"])
    w.temperature = float(weights_dict.get("temperature", 1.0))
    w.bucket_embed = weights_dict.get(
        "bucket_embed", torch.zeros(8, 8)
    ).flatten().tolist()
    w.lambda_hxh = float(weights_dict.get("lambda_hxh", 0.0))
    if w.n_heads:
        w.W_q = weights_dict["W_q"].flatten().tolist()
        w.W_k = weights_dict["W_k"].flatten().tolist()
        w.W_v = weights_dict["W_v"].flatten().tolist()
        w.W_o = weights_dict["W_o"].flatten().tolist()
        w.W_e = weights_dict["W_e"].flatten().tolist()
    return w


def _random_weights(seed: int = 1337):
    g = torch.Generator().manual_seed(seed)
    return {
        "W_vp": (torch.randn(K_V, H, generator=g) * 0.1),
        "b_vp": (torch.randn(H, generator=g) * 0.05),
        "W_lh": (torch.randn(H, generator=g) * 0.1),
        "b_lh": float(torch.randn((), generator=g) * 0.05),
        "W_sb": (torch.randn(K_S, generator=g) * 0.05),
        "alpha": float(torch.randn((), generator=g) * 0.1),
        "temperature": 1.0,
        "bucket_embed": torch.zeros(8, 8),
        "lambda_hxh": 0.0,
    }


def _random_simplex_inputs(seed: int = 2024):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(N, K_V, generator=g)
    raw = torch.randn(N, K_V, generator=g)
    raw_n = F.normalize(raw, dim=1)
    E = (raw_n @ raw_n.T).clamp(-1.0, 1.0)
    simplex_features = torch.randn(K_S, generator=g) * 0.5
    return V, E, simplex_features


def _replay_outcome_dict(
    *, slot_id: int, gpu_step: int, selection_step: int,
    ce_delta_raw: float, bucket_baseline: float = 0.0,
):
    return {
        "event_type": 3,
        "selected_rank": 0,
        "outcome_status": 0,
        "replay_id": 1,
        "gpu_step": gpu_step,
        "query_event_id": 0,
        "source_write_id": 0,
        "slot_id": slot_id,
        "policy_version": 1,
        "selection_step": selection_step,
        "teacher_score": 0.0,
        "controller_logit": 0.0,
        "ce_before_replay": 0.0,
        "ce_after_replay": 0.0,
        "ce_delta_raw": float(ce_delta_raw),
        "bucket_baseline": float(bucket_baseline),
        "reward_shaped": 0.0,
        "grad_cos_rare": math.nan,
        "grad_cos_total": math.nan,
        "flags": 0,
    }


def _torch_simplex_forward(V, E, simplex_features, weights, *, return_p=True):
    """Pure-torch reference forward; mirrors simplex_policy.cpp."""
    H_ = weights["W_vp"].shape[1]
    pre_gelu = V @ weights["W_vp"] + weights["b_vp"]
    vertex_h = F.gelu(pre_gelu, approximate="none")
    attn_logits = (vertex_h @ vertex_h.T) / math.sqrt(H_) + weights["alpha"] * E
    attn = F.softmax(attn_logits, dim=1)
    attn_out = attn @ vertex_h
    mixed_h = attn_out + vertex_h
    logits = mixed_h @ weights["W_lh"] + weights["b_lh"]
    sb = simplex_features @ weights["W_sb"]
    p = F.softmax((logits + sb) / weights["temperature"], dim=0)
    if return_p:
        return logits, p
    return logits


def test_simplex_online_learner_records_decision_and_telemetry():
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        learning_rate=0.0, sgd_interval=1024, ema_interval=999999,
    )
    learner.initialize_simplex_weights(
        _build_weights_struct(_random_weights(7))
    )
    V, E, sf = _random_simplex_inputs(11)
    learner.record_simplex_decision(
        chosen_slot_id=3, gpu_step=100, policy_version=1,
        chosen_idx=5, p_chosen_decision=0.1,
        V=V.flatten().tolist(),
        E=E.flatten().tolist(),
        simplex_features=sf.tolist(),
        n_actual=11,
        write_bucket=2,
    )
    history = learner.history(3)
    assert len(history) == 1
    e = history[0]
    assert e.gpu_step == 100
    assert e.chosen_idx == 5
    assert e.n_actual == 11
    assert e.write_bucket == 2
    assert e.action_type == 2  # V1 simplex selection
    t = learner.telemetry()
    assert t.history_appends == 1
    assert t.replay_outcomes == 0
    assert t.credited_actions == 0


def test_simplex_gerber_rejects_when_categorical_margin_inactive():
    """Gerber gates simplex credit on log-prob margin, not raw reward size."""
    weights = _random_weights(8)
    # Zero all trainable weights so the current policy is exactly uniform;
    # a high behavior p then has no active current-policy margin to agree with.
    for key in ("W_vp", "b_vp", "W_lh", "W_sb"):
        weights[key] = torch.zeros_like(weights[key])
    weights["b_lh"] = 0.0
    weights["alpha"] = 0.0
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=1.0, learning_rate=1.0, sgd_interval=1, ema_interval=999999,
        gerber_c=0.5,
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights))
    V, E, sf = _random_simplex_inputs(14)
    learner.record_simplex_decision(
        chosen_slot_id=2, gpu_step=10, policy_version=1,
        chosen_idx=3, p_chosen_decision=0.9,
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(), n_actual=N, write_bucket=1,
    )
    learner.on_replay_outcome(
        _replay_outcome_dict(slot_id=2, gpu_step=11, selection_step=10,
                             ce_delta_raw=2.0, bucket_baseline=0.0)
    )
    t = learner.telemetry()
    assert t.backward_ready_actions == 1
    assert t.gerber_rejected_actions == 1
    assert t.gerber_accepted_actions == 0
    assert t.sgd_steps == 0
    assert t.last_current_logprob_margin == pytest.approx(0.0, abs=1e-6)


def test_simplex_gerber_threshold_is_bucket_type_local_not_global():
    """Global margin stats are diagnostic; the actual gate uses bucket/type."""
    weights = _random_weights(9)
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=8,
        gamma=1.0, learning_rate=0.1, sgd_interval=1, ema_interval=999999,
        gerber_c=0.5,
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights))
    V, E, sf = _random_simplex_inputs(18)
    fwd = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    chosen = int(np.argmax(np.asarray(fwd.p)))
    p_chosen = float(fwd.p[chosen])
    # Bucket 2 receives a very different behavior margin, but the replay
    # below belongs to bucket 1. The gate must use bucket 1's stddev only.
    learner.record_simplex_decision(
        chosen_slot_id=4, gpu_step=1, policy_version=1,
        chosen_idx=chosen, p_chosen_decision=0.99,
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(), n_actual=N, write_bucket=2,
    )
    learner.record_simplex_decision(
        chosen_slot_id=3, gpu_step=10, policy_version=1,
        chosen_idx=chosen, p_chosen_decision=p_chosen,
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(), n_actual=N, write_bucket=1,
    )
    learner.on_replay_outcome(
        _replay_outcome_dict(slot_id=3, gpu_step=11, selection_step=10,
                             ce_delta_raw=0.5, bucket_baseline=0.0)
    )
    t = learner.telemetry()
    assert t.gerber_accepted_actions == 1
    assert t.last_bucket_type_stddev == pytest.approx(0.0, abs=1e-8)
    assert t.last_global_type_stddev > 0.0
    assert t.last_gerber_threshold == pytest.approx(0.0, abs=1e-8)


def test_simplex_replay_credits_when_selection_step_matches_producer_gpu_step():
    """Replay credit is keyed by the producer gpu_step, not wall/consumer time."""
    weights = _random_weights(15)
    V, E, sf = _random_simplex_inputs(30)
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=1.0, learning_rate=0.1, sgd_interval=1, ema_interval=999999,
        gerber_c=0.0,
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights))
    fwd = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    chosen = int(np.argmax(np.asarray(fwd.p)))
    learner.record_simplex_decision(
        chosen_slot_id=2, gpu_step=123, policy_version=1,
        chosen_idx=chosen, p_chosen_decision=float(fwd.p[chosen]),
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(), n_actual=N, write_bucket=1,
    )
    learner.on_replay_outcome(
        _replay_outcome_dict(
            slot_id=2, gpu_step=124, selection_step=123,
            ce_delta_raw=0.75, bucket_baseline=0.0,
        )
    )
    assert learner.telemetry().credited_actions == 1
    assert learner.last_advantage != 0.0

    miss = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=1.0, learning_rate=0.1, sgd_interval=1, ema_interval=999999,
        gerber_c=0.0,
    )
    miss.initialize_simplex_weights(_build_weights_struct(weights))
    miss.record_simplex_decision(
        chosen_slot_id=2, gpu_step=123, policy_version=1,
        chosen_idx=chosen, p_chosen_decision=float(fwd.p[chosen]),
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(), n_actual=N, write_bucket=1,
    )
    miss.on_replay_outcome(
        _replay_outcome_dict(
            slot_id=2, gpu_step=124, selection_step=124,
            ce_delta_raw=0.75, bucket_baseline=0.0,
        )
    )
    assert miss.telemetry().credited_actions == 0
    assert miss.last_advantage == 0.0


def test_simplex_learner_does_not_credit_sentinel_padded_slots():
    """Short simplexes must not let zero-padded sentinel slots receive credit."""
    weights = _random_weights(16)
    V, E, sf = _random_simplex_inputs(32)
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=1.0, learning_rate=0.1, sgd_interval=1, ema_interval=999999,
        gerber_c=0.0,
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights))
    fwd = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    chosen = int(np.argmax(np.asarray(fwd.p[:4])))
    chosen_slot = 2 if chosen == 0 else 3
    learner.record_simplex_decision(
        chosen_slot_id=chosen_slot, gpu_step=50, policy_version=1,
        chosen_idx=chosen, p_chosen_decision=float(fwd.p[chosen]),
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(), n_actual=4, write_bucket=2,
    )
    learner.on_replay_outcome(
        _replay_outcome_dict(
            slot_id=0, gpu_step=51, selection_step=50,
            ce_delta_raw=1.0, bucket_baseline=0.0,
        )
    )
    t = learner.telemetry()
    assert t.credited_actions == 0
    assert t.backward_skipped_missing_state == 1


def test_gerber_correction_collapses_to_one_when_behavior_equals_current():
    """On-policy categorical margins should pass Gerber with weight 1."""
    weights = _random_weights(17)
    V, E, sf = _random_simplex_inputs(34)
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=1.0, learning_rate=0.0, sgd_interval=1, ema_interval=999999,
        gerber_c=0.5,
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights))
    fwd = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    chosen = int(np.argmax(np.asarray(fwd.p)))
    learner.record_simplex_decision(
        chosen_slot_id=6, gpu_step=77, policy_version=1,
        chosen_idx=chosen, p_chosen_decision=float(fwd.p[chosen]),
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(), n_actual=N, write_bucket=0,
    )
    learner.on_replay_outcome(
        _replay_outcome_dict(
            slot_id=6, gpu_step=78, selection_step=77,
            ce_delta_raw=0.5, bucket_baseline=0.0,
        )
    )
    assert learner.telemetry().last_gerber_weight == pytest.approx(1.0)
    assert learner.telemetry().gerber_accepted_actions == 1


def test_simplex_standardizes_advantage_per_bucket_before_credit():
    """Bucket baseline centers reward; per-bucket stddev scales magnitude."""
    weights = _random_weights(10)
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=8,
        gamma=1.0, learning_rate=0.0, sgd_interval=1, ema_interval=999999,
        gerber_c=0.0,
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights))
    V, E, sf = _random_simplex_inputs(22)
    fwd = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    chosen = int(np.argmax(np.asarray(fwd.p)))
    p_chosen = float(fwd.p[chosen])
    for step, raw_adv in enumerate([1.0, 3.0, 5.0], start=1):
        learner.record_simplex_decision(
            chosen_slot_id=5, gpu_step=step * 10, policy_version=1,
            chosen_idx=chosen, p_chosen_decision=p_chosen,
            V=V.flatten().tolist(), E=E.flatten().tolist(),
            simplex_features=sf.tolist(), n_actual=N, write_bucket=3,
        )
        learner.on_replay_outcome(
            _replay_outcome_dict(
                slot_id=5,
                gpu_step=step * 10 + 1,
                selection_step=step * 10,
                ce_delta_raw=raw_adv,
                bucket_baseline=0.0,
            )
        )
    t = learner.telemetry()
    assert t.last_advantage_raw == pytest.approx(5.0)
    assert t.last_advantage_stddev > 0.0
    assert abs(t.last_advantage_standardized) < abs(t.last_advantage_raw)


def test_simplex_skip_when_weights_not_initialized():
    learner = _ext.SimplexOnlineLearner(num_slots=4)
    # No initialize_simplex_weights call — replay outcome is a no-op.
    learner.on_replay_outcome(
        _replay_outcome_dict(slot_id=1, gpu_step=10, selection_step=5,
                             ce_delta_raw=0.5)
    )
    t = learner.telemetry()
    assert t.replay_outcomes == 1
    assert t.backward_skipped_missing_weights == 1
    assert t.credited_actions == 0


def test_simplex_skip_when_decision_match_missing():
    learner = _ext.SimplexOnlineLearner(num_slots=4)
    learner.initialize_simplex_weights(
        _build_weights_struct(_random_weights(0))
    )
    # No record_simplex_decision before the replay → skip.
    learner.on_replay_outcome(
        _replay_outcome_dict(slot_id=1, gpu_step=10, selection_step=5,
                             ce_delta_raw=0.5)
    )
    t = learner.telemetry()
    assert t.backward_skipped_missing_state == 1
    assert t.credited_actions == 0


def test_simplex_skip_when_outcome_status_not_ok():
    learner = _ext.SimplexOnlineLearner(num_slots=4)
    learner.initialize_simplex_weights(
        _build_weights_struct(_random_weights(0))
    )
    bad = _replay_outcome_dict(slot_id=1, gpu_step=10, selection_step=5,
                                ce_delta_raw=0.5)
    bad["outcome_status"] = 1
    learner.on_replay_outcome(bad)
    t = learner.telemetry()
    # Outcome is counted but no backward attempted.
    assert t.replay_outcomes == 1
    assert t.credited_actions == 0


def test_simplex_skip_when_slot_id_out_of_range():
    learner = _ext.SimplexOnlineLearner(num_slots=4)
    learner.initialize_simplex_weights(
        _build_weights_struct(_random_weights(0))
    )
    learner.on_replay_outcome(
        _replay_outcome_dict(slot_id=999, gpu_step=10, selection_step=5,
                             ce_delta_raw=0.5)
    )
    t = learner.telemetry()
    assert t.invalid_slot_skips == 1
    assert t.credited_actions == 0


def test_simplex_reinforce_increases_p_chosen_for_positive_advantage():
    """REINFORCE direction sanity. Single event, sgd_interval=1.

    advantage > 0 + chosen vertex selected → push p[chosen] up.
    """
    weights = _random_weights(42)
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=1.0,                # disable recency decay for determinism
        learning_rate=0.5,        # large enough to see the move clearly
        sgd_interval=1,
        ema_interval=999999,      # don't blend slow into fast
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights))
    V, E, sf = _random_simplex_inputs(99)
    chosen = 7
    fwd_before = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    p_before_chosen = fwd_before.p[chosen]

    learner.record_simplex_decision(
        chosen_slot_id=2, gpu_step=10, policy_version=1,
        chosen_idx=chosen, p_chosen_decision=p_before_chosen,
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(),
    )
    learner.on_replay_outcome(
        _replay_outcome_dict(slot_id=2, gpu_step=11, selection_step=10,
                             ce_delta_raw=1.0, bucket_baseline=0.0)
    )

    fwd_after = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    assert fwd_after.p[chosen] > p_before_chosen, (
        f"positive advantage should increase p[chosen]; "
        f"before={p_before_chosen}, after={fwd_after.p[chosen]}"
    )
    assert learner.telemetry().sgd_steps == 1
    assert learner.telemetry().credited_actions == 1


def test_simplex_reinforce_decreases_p_chosen_for_negative_advantage():
    weights = _random_weights(43)
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=1.0, learning_rate=0.5, sgd_interval=1, ema_interval=999999,
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights))
    V, E, sf = _random_simplex_inputs(101)
    chosen = 3
    fwd_before = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    p_before_chosen = fwd_before.p[chosen]

    learner.record_simplex_decision(
        chosen_slot_id=4, gpu_step=20, policy_version=1,
        chosen_idx=chosen, p_chosen_decision=p_before_chosen,
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(),
    )
    learner.on_replay_outcome(
        _replay_outcome_dict(slot_id=4, gpu_step=21, selection_step=20,
                             ce_delta_raw=-1.0, bucket_baseline=0.0)
    )

    fwd_after = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    assert fwd_after.p[chosen] < p_before_chosen, (
        f"negative advantage should decrease p[chosen]; "
        f"before={p_before_chosen}, after={fwd_after.p[chosen]}"
    )


def _assert_simplex_gradient_matches_torch_autograd_reference(
    *,
    atol: float,
    rtol: float,
    entropy_beta: float = 0.0,
) -> None:
    """Run the load-bearing C++ learner vs torch autograd parity check.

    Run one REINFORCE event through the C++ kernel with sgd_interval=1
    and capture the post-update fast_weights. Run the same forward
    through a torch.autograd reference, backprop the same scalar loss,
    apply manual SGD with the same lr. Assert the two weight sets agree
    at atol=1e-4. Same shape as the C7 parity test in the per-slot
    phase — the strongest correctness guarantee we can write without
    running on hardware.
    """
    seed = 7
    weights_init = _random_weights(seed)
    if entropy_beta != 0.0:
        # Non-unit temperature makes the autograd reference verify the
        # softmax-temperature factor in the entropy derivative.
        weights_init["temperature"] = 0.75
    V, E, sf = _random_simplex_inputs(seed * 2)
    chosen = 9
    advantage = 0.7  # ce_delta_raw - bucket_baseline (gamma=1, step_gap=1)
    lr = 0.3

    # ---- Torch autograd reference ----
    W_vp = weights_init["W_vp"].clone().requires_grad_(True)
    b_vp = weights_init["b_vp"].clone().requires_grad_(True)
    W_lh = weights_init["W_lh"].clone().requires_grad_(True)
    b_lh = torch.tensor(weights_init["b_lh"], dtype=torch.float32, requires_grad=True)
    W_sb = weights_init["W_sb"].clone().requires_grad_(True)
    alpha = torch.tensor(weights_init["alpha"], dtype=torch.float32, requires_grad=True)
    weights_torch = {
        "W_vp": W_vp, "b_vp": b_vp, "W_lh": W_lh, "b_lh": b_lh,
        "W_sb": W_sb, "alpha": alpha,
        "temperature": weights_init["temperature"],
    }
    _, p = _torch_simplex_forward(V, E, sf, weights_torch)
    log_p_chosen = torch.log(p[chosen] + 1e-30)
    entropy = -(p * torch.log(p + 1e-30)).sum()
    # Match the C++ scaling: advantage uses gamma=1.0, step_gap=(11-10)=1,
    # so the multiplier reduces to advantage * 1.0 = advantage.
    loss = -advantage * log_p_chosen - entropy_beta * entropy
    loss.backward()

    expected = {
        "W_vp": (weights_init["W_vp"] - lr * W_vp.grad).numpy(),
        "b_vp": (weights_init["b_vp"] - lr * b_vp.grad).numpy(),
        "W_lh": (weights_init["W_lh"] - lr * W_lh.grad).numpy(),
        "b_lh": float(weights_init["b_lh"] - lr * b_lh.grad),
        "W_sb": (weights_init["W_sb"] - lr * W_sb.grad).numpy(),
        "alpha": float(weights_init["alpha"] - lr * alpha.grad),
    }

    # ---- C++ kernel ----
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=1.0, learning_rate=lr,
        sgd_interval=1, ema_interval=999999,
        entropy_beta=entropy_beta,
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights_init))
    fwd_before = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    learner.record_simplex_decision(
        chosen_slot_id=2, gpu_step=10, policy_version=1,
        chosen_idx=chosen, p_chosen_decision=fwd_before.p[chosen],
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(),
    )
    learner.on_replay_outcome(
        _replay_outcome_dict(slot_id=2, gpu_step=11, selection_step=10,
                             ce_delta_raw=advantage, bucket_baseline=0.0)
    )

    fast = learner.fast_weights()
    actual_W_vp = np.array(fast.W_vp).reshape(K_V, H)
    actual_b_vp = np.array(fast.b_vp)
    actual_W_lh = np.array(fast.W_lh)
    actual_b_lh = float(fast.b_lh)
    actual_W_sb = np.array(fast.W_sb)
    actual_alpha = float(fast.alpha)

    np.testing.assert_allclose(
        actual_W_vp, expected["W_vp"], atol=atol, rtol=rtol
    )
    np.testing.assert_allclose(
        actual_b_vp, expected["b_vp"], atol=atol, rtol=rtol
    )
    np.testing.assert_allclose(
        actual_W_lh, expected["W_lh"], atol=atol, rtol=rtol
    )
    assert abs(actual_b_lh - expected["b_lh"]) < atol
    np.testing.assert_allclose(
        actual_W_sb, expected["W_sb"], atol=atol, rtol=rtol
    )
    assert abs(actual_alpha - expected["alpha"]) < atol
    if entropy_beta != 0.0:
        t = learner.telemetry()
        assert t.last_entropy == pytest.approx(float(entropy.detach()), abs=1e-5)
        assert t.last_entropy_bonus_weight == pytest.approx(entropy_beta)


def test_simplex_gradient_matches_torch_autograd_reference():
    """Strict fp32 parity for the scalar/at::matmul learner path."""
    if _ext.amx_bf16_kernel_available() and _ext.has_amx_bf16():
        pytest.skip(
            "AMX BF16 dispatch is live; strict fp32 tolerance is covered by "
            "the bf16-loose learner parity test"
        )
    _assert_simplex_gradient_matches_torch_autograd_reference(
        atol=1e-4,
        rtol=1e-3,
    )


def test_entropy_bonus_matches_torch_autograd_reference():
    """REINFORCE + entropy bonus matches a direct torch autograd loss."""
    if _ext.amx_bf16_kernel_available() and _ext.has_amx_bf16():
        pytest.skip(
            "AMX BF16 dispatch is live; strict fp32 tolerance is covered by "
            "the bf16-loose learner parity test"
        )
    _assert_simplex_gradient_matches_torch_autograd_reference(
        atol=1e-4,
        rtol=1e-3,
        entropy_beta=0.05,
    )


def test_simplex_gradient_matches_torch_autograd_reference_amx_bf16_path():
    """Hardware-gated parity for AMX-dispatched simplex backward GEMMs."""
    if not (_ext.amx_bf16_kernel_available() and _ext.has_amx_bf16()):
        pytest.skip("AMX BF16 hardware/OS state and compiled kernel are required")
    _assert_simplex_gradient_matches_torch_autograd_reference(
        atol=3e-2,
        rtol=3e-2,
    )


def test_simplex_sgd_apply_and_slow_ema_blend_cadences():
    """Multi-event run: verify sgd_steps and ema_blends counters increment
    on the configured cadence; fast and slow diverge then converge.
    """
    weights = _random_weights(13)
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=1.0, learning_rate=0.1,
        sgd_interval=2, ema_interval=4,  # SGD every 2 events; EMA every 4
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights))
    V, E, sf = _random_simplex_inputs(2024)

    for step in range(8):
        slot = step % 4
        chosen = step % N
        fwd = _ext.simplex_forward(
            learner.fast_weights(),
            V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
        )
        learner.record_simplex_decision(
            chosen_slot_id=slot, gpu_step=10 * step, policy_version=1,
            chosen_idx=chosen, p_chosen_decision=fwd.p[chosen],
            V=V.flatten().tolist(), E=E.flatten().tolist(),
            simplex_features=sf.tolist(),
            n_actual=N,
            write_bucket=step % 4,
        )
        learner.on_replay_outcome(
            _replay_outcome_dict(slot_id=slot, gpu_step=10 * step + 1,
                                 selection_step=10 * step, ce_delta_raw=0.5)
        )

    t = learner.telemetry()
    # 8 credited actions, sgd_interval=2 → 4 SGD steps.
    assert t.sgd_steps == 4
    # ema_interval=4 means tick_event() == 4 yields one blend; 8 events → 2 blends.
    assert t.ema_blends == 2

    # After SGD the fast weights have moved; after EMA the slow weights
    # have absorbed some of the fast movement. Both should differ from
    # the original initialization.
    fast = np.array(learner.fast_weights().W_vp)
    slow = np.array(learner.slow_weights().W_vp)
    init_W_vp = weights["W_vp"].flatten().numpy()
    assert not np.allclose(fast, init_W_vp)
    assert not np.allclose(slow, init_W_vp)


def test_simplex_trace_writes_ndjson_per_replay_event(tmp_path):
    """One NDJSON line per replay event that reached simplex_backward.

    Operationalizes the design doc's "A stop that is not logged is a hidden
    experimental confound" for the simplex head. Scope for this commit is
    events that reached `simplex_backward` (including the entropy-bonus
    rejection branch where advantage is zeroed but the backward still fires);
    pure early returns (missing weights, missing decision, sentinel slot,
    zero-advantage zero-beta short-circuit) do not emit traces yet.
    """
    import json
    weights = _random_weights(99)
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=1.0, learning_rate=0.1,
        sgd_interval=1, ema_interval=999999,
        gerber_c=0.0,  # accept everything so we exercise the post-backward path
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights))

    trace_path = tmp_path / "simplex_trace.ndjson"
    learner.set_simplex_trace_path(str(trace_path))

    V, E, sf = _random_simplex_inputs(123)
    fwd_before = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    chosen = int(np.argmax(np.asarray(fwd_before.p)))
    learner.record_simplex_decision(
        chosen_slot_id=2, gpu_step=10, policy_version=1,
        chosen_idx=chosen, p_chosen_decision=float(fwd_before.p[chosen]),
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(),
        n_actual=N, write_bucket=1,
    )
    learner.on_replay_outcome(
        _replay_outcome_dict(
            slot_id=2, gpu_step=11, selection_step=10,
            ce_delta_raw=0.5, bucket_baseline=0.0,
        )
    )

    text = trace_path.read_text()
    lines = [ln for ln in text.splitlines() if ln]
    assert len(lines) == 1, f"expected exactly one NDJSON line, got {len(lines)}"

    rec = json.loads(lines[0])
    expected_fields = {
        "gpu_step",
        "chosen_idx",
        "p_chosen",
        "entropy",
        "advantage_raw",
        "advantage_after_gerber",
        "gerber_gate",
        "gerber_threshold",
        "write_bucket",
    }
    assert set(rec.keys()) == expected_fields, (
        f"unexpected fields: {set(rec.keys()) ^ expected_fields}"
    )

    # Integer fields: no formatting, exact ints
    assert isinstance(rec["gpu_step"], int)
    assert isinstance(rec["chosen_idx"], int)
    assert isinstance(rec["write_bucket"], int)
    assert rec["gpu_step"] == 10
    assert rec["chosen_idx"] == chosen
    assert rec["write_bucket"] == 1

    # Float fields: must round-trip through float and stay finite
    for fname in (
        "p_chosen", "entropy", "advantage_raw",
        "advantage_after_gerber", "gerber_gate", "gerber_threshold",
    ):
        assert isinstance(rec[fname], (int, float)), fname
        assert math.isfinite(float(rec[fname])), fname

    # entropy in [0, ln(N)]
    assert 0.0 <= float(rec["entropy"]) <= math.log(N) + 1e-5
    # gerber_gate in [0, 1]
    assert 0.0 <= float(rec["gerber_gate"]) <= 1.0 + 1e-6
    # p_chosen in (0, 1]
    assert 0.0 < float(rec["p_chosen"]) <= 1.0 + 1e-6


def test_simplex_trace_disabled_when_path_empty(tmp_path):
    """Empty path means tracing off; no file should be created or grown."""
    weights = _random_weights(99)
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=1.0, learning_rate=0.1,
        sgd_interval=1, ema_interval=999999,
        gerber_c=0.0,
    )
    learner.initialize_simplex_weights(_build_weights_struct(weights))

    # Open-then-close: a previously enabled trace must be disabled by
    # calling with an empty string. The file may or may not exist.
    trace_path = tmp_path / "simplex_trace.ndjson"
    learner.set_simplex_trace_path(str(trace_path))
    learner.set_simplex_trace_path("")

    V, E, sf = _random_simplex_inputs(123)
    fwd = _ext.simplex_forward(
        learner.fast_weights(),
        V.flatten().tolist(), E.flatten().tolist(), sf.tolist(),
    )
    chosen = int(np.argmax(np.asarray(fwd.p)))
    learner.record_simplex_decision(
        chosen_slot_id=2, gpu_step=10, policy_version=1,
        chosen_idx=chosen, p_chosen_decision=float(fwd.p[chosen]),
        V=V.flatten().tolist(), E=E.flatten().tolist(),
        simplex_features=sf.tolist(),
        n_actual=N, write_bucket=1,
    )
    learner.on_replay_outcome(
        _replay_outcome_dict(
            slot_id=2, gpu_step=11, selection_step=10,
            ce_delta_raw=0.5, bucket_baseline=0.0,
        )
    )

    if trace_path.exists():
        # Open-then-close-then-replay must not have appended anything.
        assert trace_path.read_text() == ""
