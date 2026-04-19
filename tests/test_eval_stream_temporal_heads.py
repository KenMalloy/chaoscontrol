import torch
import torch.nn.functional as F

from chaoscontrol.eval_stream.temporal_heads import (
    TemporalHeadConfig,
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
