import importlib.util

import pytest
import torch
from chaoscontrol.eval_stream.ttt_runner import select_adapt_params
from chaoscontrol.model import ChaosStudentLM


def _tiny_ssm_lm():
    return ChaosStudentLM(
        vocab_size=32, dim=16, num_layers=2, block_type="ssm", a_mode="diag",
    )


def test_log_a_selection_is_small_and_correct():
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="log_a")
    # 2 layers * dim=16 each = 32 scalars; count parameters not tensors
    total = sum(p.numel() for p in params)
    assert total == 32
    # Every selected param's name should contain "log_a"
    names = {n for n, p in m.named_parameters() if any(p is q for q in params)}
    assert all("log_a" in n for n in names)


def test_delta_proj_selection():
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="delta_proj")
    # dim=16, delta_proj is Linear(dim, dim) -> 256 params per layer × 2 layers
    assert sum(p.numel() for p in params) == 2 * 16 * 16


def test_lm_head_selection():
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="lm_head")
    # vocab=32, dim=16 -> 32*16=512
    assert sum(p.numel() for p in params) == 32 * 16


def test_none_selection_is_empty():
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="none")
    assert params == []


def test_all_selection_covers_every_param():
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="all")
    expected = sum(p.numel() for p in m.parameters())
    assert sum(p.numel() for p in params) == expected


def test_embed_rows_seen_is_exact_match():
    """Must match 'embed.weight' exactly — no collision with any hypothetical
    embed_norm / embedding_* future sibling params.
    """
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="embed_rows_seen")
    names = {n for n, p in m.named_parameters() if any(p is q for q in params)}
    assert names == {"embed.weight"}


@pytest.mark.skipif(
    importlib.util.find_spec("chaoscontrol.eval_stream.persistence") is None,
    reason="persistence module lands in Task 7",
)
def test_trainable_h0_pattern(tmp_path):
    from chaoscontrol.eval_stream.persistence import attach_trainable_h0
    m = _tiny_ssm_lm()
    attach_trainable_h0(m)
    params = select_adapt_params(m, adapt_set="trainable_h0")
    names = {n for n, p in m.named_parameters() if any(p is q for q in params)}
    assert len(names) == 2  # 2 layers
    assert all("_trainable_h0" in n for n in names)
