import torch
import torch.nn.functional as F

from chaoscontrol.eval_stream.temporal_heads import (
    PreviousChunkPriorityGate,
    TemporalHeadConfig,
    make_same_horizon_virtual_depth_config,
    score_temporal_heads_chunk,
    uniform_logprob_mixture,
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
        direct_loss = F.cross_entropy(
            out["logits"][:, :-1].reshape(-1, 64),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        )
    assert torch.isclose(torch.tensor(result.loss_nats), direct_loss)
    assert result.tokens_scored == chunk.size(1) - 1
    assert set(result.final_states_by_shift) == {0.0}


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


def test_primary_gate_ignores_stale_head_disagreement_after_base_only_chunk():
    gate = PreviousChunkPriorityGate(
        threshold=1.0,
        entropy_weight=1.0,
        loss_spike_weight=0.0,
        state_delta_weight=0.0,
    )

    gate.update_after_chunk(
        entropy=0.5,
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
