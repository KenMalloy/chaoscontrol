import torch
import torch.nn.functional as F

from chaoscontrol.eval_stream.temporal_heads import (
    PreviousChunkPriorityGate,
    TemporalHeadConfig,
    make_same_horizon_virtual_depth_config,
    score_temporal_heads_chunk,
    uniform_logprob_mixture,
    weighted_logprob_mixture,
)
from chaoscontrol.model import ChaosStudentLM


def test_uniform_logprob_mixture_one_head_is_identity():
    logp = torch.log_softmax(torch.randn(2, 3, 5), dim=-1)

    mixed = uniform_logprob_mixture([logp])

    assert torch.equal(mixed, logp)


def test_uniform_logprob_mixture_matches_probability_average():
    logits_a = torch.tensor([[[2.0, 0.0]]])
    logits_b = torch.tensor([[[0.0, 2.0]]])
    logp_a = torch.log_softmax(logits_a, dim=-1)
    logp_b = torch.log_softmax(logits_b, dim=-1)

    mixed = uniform_logprob_mixture([logp_a, logp_b])
    expected = torch.log((logp_a.exp() + logp_b.exp()) / 2.0)

    assert torch.allclose(mixed, expected)


def test_weighted_logprob_mixture_matches_probability_average():
    logits_a = torch.tensor([[[4.0, 0.0]]])
    logits_b = torch.tensor([[[0.0, 4.0]]])
    logp_a = torch.log_softmax(logits_a, dim=-1)
    logp_b = torch.log_softmax(logits_b, dim=-1)

    mixed = weighted_logprob_mixture([logp_a, logp_b], weights=(0.8, 0.2))
    expected = torch.log(logp_a.exp() * 0.8 + logp_b.exp() * 0.2)

    assert torch.allclose(mixed, expected)


def test_weighted_logprob_mixture_rejects_bad_weights():
    logp = torch.log_softmax(torch.randn(1, 2, 3), dim=-1)

    try:
        weighted_logprob_mixture([logp, logp], weights=(1.0,))
    except ValueError as exc:
        assert "weights length" in str(exc)
    else:
        raise AssertionError("expected bad weight length to fail")


def test_single_zero_shift_matches_direct_model():
    torch.manual_seed(0)
    model = ChaosStudentLM(
        vocab_size=64,
        dim=16,
        num_layers=2,
        block_type="ssm",
        a_mode="diag",
    )
    chunk = torch.randint(0, 64, (1, 12))
    cfg = TemporalHeadConfig(horizon_shifts=(0.0,))

    result = score_temporal_heads_chunk(
        model,
        chunk,
        states_by_shift={0.0: None},
        cfg=cfg,
    )

    with torch.no_grad():
        out = model(chunk)
        direct_log_probs = F.log_softmax(out["logits"], dim=-1)
        direct_loss = F.cross_entropy(
            out["logits"][:, :-1].reshape(-1, 64),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        )
    assert torch.equal(result.mixed_log_probs, direct_log_probs)
    assert torch.isclose(torch.tensor(result.loss_nats), direct_loss)
    assert result.tokens_scored == chunk.size(1) - 1
    assert set(result.final_states_by_shift) == {0.0}


def test_identical_uniform_heads_match_direct_model_with_independent_states():
    torch.manual_seed(0)
    model = ChaosStudentLM(
        vocab_size=64,
        dim=16,
        num_layers=2,
        block_type="ssm",
        a_mode="diag",
    )
    chunk = torch.randint(0, 64, (1, 12))
    cfg = TemporalHeadConfig(
        horizon_shifts=(0.0, 0.0, 0.0),
        head_ids=("same_a", "same_b", "same_c"),
    )

    result = score_temporal_heads_chunk(
        model,
        chunk,
        states_by_shift={head_id: None for head_id in cfg.head_ids},
        cfg=cfg,
    )

    with torch.no_grad():
        out = model(chunk)
        direct_log_probs = F.log_softmax(out["logits"], dim=-1)
        direct_loss = F.cross_entropy(
            out["logits"][:, :-1].reshape(-1, 64),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        )

    assert torch.allclose(result.mixed_log_probs, direct_log_probs)
    assert torch.isclose(torch.tensor(result.loss_nats), direct_loss)
    assert set(result.final_states_by_shift) == {"same_a", "same_b", "same_c"}
    assert result.winner_counts_by_shift == {
        "same_a": result.tokens_scored,
        "same_b": 0,
        "same_c": 0,
    }
    for layer_idx in range(2):
        assert (
            result.final_states_by_shift["same_a"][layer_idx].data_ptr()
            != result.final_states_by_shift["same_b"][layer_idx].data_ptr()
        )
        assert (
            result.final_states_by_shift["same_b"][layer_idx].data_ptr()
            != result.final_states_by_shift["same_c"][layer_idx].data_ptr()
        )


def test_base_prior_mixer_protects_base_probability():
    slow = torch.log(torch.tensor([[[0.01, 0.99]]]))
    base = torch.log(torch.tensor([[[0.90, 0.10]]]))
    fast = torch.log(torch.tensor([[[0.01, 0.99]]]))

    mixed = weighted_logprob_mixture(
        [slow, base, fast],
        weights=(0.1, 0.8, 0.1),
    )

    assert mixed.exp()[0, 0, 0] >= 0.8 * base.exp()[0, 0, 0]


def test_temporal_heads_keep_independent_states():
    torch.manual_seed(0)
    model = ChaosStudentLM(
        vocab_size=64,
        dim=16,
        num_layers=2,
        block_type="ssm",
        a_mode="diag",
    )
    chunk = torch.randint(0, 64, (1, 12))
    cfg = TemporalHeadConfig(horizon_shifts=(-0.5, 0.0, 0.5))

    result = score_temporal_heads_chunk(
        model,
        chunk,
        states_by_shift={shift: None for shift in cfg.horizon_shifts},
        cfg=cfg,
    )

    states = result.final_states_by_shift
    assert set(states) == {-0.5, 0.0, 0.5}
    assert len(states[-0.5]) == len(states[0.0]) == len(states[0.5]) == 2
    for layer_idx in range(2):
        assert states[-0.5][layer_idx].data_ptr() != states[0.0][layer_idx].data_ptr()
        assert states[0.0][layer_idx].data_ptr() != states[0.5][layer_idx].data_ptr()


def test_score_temporal_heads_exposes_analysis_only_diagnostics():
    torch.manual_seed(0)
    model = ChaosStudentLM(
        vocab_size=64,
        dim=16,
        num_layers=2,
        block_type="ssm",
        a_mode="diag",
    )
    chunk = torch.randint(0, 64, (1, 12))
    cfg = TemporalHeadConfig(horizon_shifts=(-0.5, 0.0, 0.5))

    result = score_temporal_heads_chunk(
        model,
        chunk,
        states_by_shift={shift: None for shift in cfg.horizon_shifts},
        cfg=cfg,
    )

    assert set(result.winner_counts_by_shift) == {-0.5, 0.0, 0.5}
    assert sum(result.winner_counts_by_shift.values()) == result.tokens_scored
    assert set(result.half_life_stats_by_shift) == {-0.5, 0.0, 0.5}
    assert len(result.half_life_stats_by_shift[0.0]) == 2
    assert {
        "layer",
        "p10",
        "median",
        "p90",
        "separated_fraction_vs_base",
    }.issubset(result.half_life_stats_by_shift[-0.5][0])
    assert set(result.state_divergence_by_shift) == {-0.5, 0.5}
    assert len(result.state_divergence_by_shift[-0.5]) == 2
    assert {
        "layer",
        "l2_vs_base",
        "cosine_vs_base",
    }.issubset(result.state_divergence_by_shift[-0.5][0])


def test_temporal_heads_state_values_diverge_across_chunks():
    torch.manual_seed(0)
    model = ChaosStudentLM(
        vocab_size=64,
        dim=16,
        num_layers=2,
        block_type="ssm",
        a_mode="diag",
    )
    chunk_a = torch.randint(0, 64, (1, 12))
    chunk_b = torch.randint(0, 64, (1, 12))
    cfg = TemporalHeadConfig(horizon_shifts=(-0.5, 0.0, 0.5))

    first = score_temporal_heads_chunk(
        model,
        chunk_a,
        states_by_shift={shift: None for shift in cfg.horizon_shifts},
        cfg=cfg,
    )
    second = score_temporal_heads_chunk(
        model,
        chunk_b,
        states_by_shift=first.final_states_by_shift,
        cfg=cfg,
    )

    for layer_idx in range(2):
        assert not torch.allclose(
            second.final_states_by_shift[-0.5][layer_idx],
            second.final_states_by_shift[0.5][layer_idx],
        )


def test_primary_gate_decays_stale_head_disagreement_after_base_only_chunk():
    gate = PreviousChunkPriorityGate(
        threshold=10.0,
        entropy_weight=0.0,
        loss_spike_weight=0.0,
        state_delta_weight=0.0,
        use_disagreement_ema=True,
        disagreement_weight=1.0,
        disagreement_decay=0.5,
    )

    gate.update_after_chunk(
        entropy=0.0,
        loss_spike=0.0,
        state_delta_norm=0.0,
        head_disagreement=40.0,
        temporal_heads_ran=True,
    )
    assert gate.should_run(extra_cost_seconds=0.1, slack_remaining_seconds=1.0)

    gate.update_after_chunk(
        entropy=0.0,
        loss_spike=0.0,
        state_delta_norm=0.0,
        head_disagreement=999.0,
        temporal_heads_ran=False,
    )

    assert not gate.should_run(extra_cost_seconds=0.1, slack_remaining_seconds=1.0)


def test_same_horizon_virtual_depth_config_uses_all_layers_when_no_group():
    cfg = {
        "vocab_size": 64,
        "dim": 16,
        "num_layers": 3,
        "block_type": "ssm",
        "a_mode": "diag",
    }

    out = make_same_horizon_virtual_depth_config(cfg, depth_recurrence_count=3)

    assert out["depth_recurrence_shared_layers"] == [0, 1, 2]
    assert out["depth_recurrence_count"] == 3


def test_same_horizon_virtual_depth_config_preserves_existing_group():
    cfg = {
        "vocab_size": 64,
        "dim": 16,
        "num_layers": 4,
        "block_type": "ssm",
        "a_mode": "diag",
        "depth_recurrence_shared_layers": [1, 2],
    }

    out = make_same_horizon_virtual_depth_config(cfg, depth_recurrence_count=2)

    assert out["depth_recurrence_shared_layers"] == [1, 2]
    assert out["depth_recurrence_count"] == 2
