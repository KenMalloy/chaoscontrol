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


def test_log_a_plus_delta_proj_union_selection():
    """`log_a+delta_proj` selects the union of both; no param is double-counted."""
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="log_a+delta_proj")
    names = {n for n, p in m.named_parameters() if any(p is q for q in params)}
    # log_a (2 layers × 16 = 32) + delta_proj.weight (2 × 16 × 16 = 512) = 544 params
    # 2 log_a tensors + 2 delta_proj.weight tensors = 4 tensors total
    assert len(params) == 4, f"expected 4 tensors, got {len(params)}: {names}"
    assert sum(p.numel() for p in params) == 32 + 512
    assert any("log_a" in n for n in names)
    assert any("delta_proj" in n for n in names)


def test_B_side_selection_matches_in_and_select_proj():
    """B_side = input → state projections (in_proj + select_proj)."""
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="B_side")
    names = {n for n, p in m.named_parameters() if any(p is q for q in params)}
    # 2 layers × (in_proj.weight + select_proj.weight) = 4 tensors
    # Each is Linear(16, 16) = 256 params → 4 × 256 = 1024
    assert sum(p.numel() for p in params) == 4 * 16 * 16
    # Every selected name is either in_proj or select_proj
    assert all(("in_proj" in n) or ("select_proj" in n) for n in names), names
    # Must NOT include out_proj, gate_proj, delta_proj, log_a, lm_head, embed.
    assert not any(("out_proj" in n) or ("gate_proj" in n) or ("delta_proj" in n)
                   or ("log_a" in n) or ("lm_head" in n) or (n == "embed.weight")
                   for n in names), f"B_side leaked into non-B names: {names}"


def test_C_side_selection_matches_out_and_gate_proj():
    """C_side = state → residual projections (out_proj + gate_proj)."""
    m = _tiny_ssm_lm()
    params = select_adapt_params(m, adapt_set="C_side")
    names = {n for n, p in m.named_parameters() if any(p is q for q in params)}
    assert sum(p.numel() for p in params) == 4 * 16 * 16
    assert all(("out_proj" in n) or ("gate_proj" in n) for n in names), names
    assert not any(("in_proj" in n) or ("select_proj" in n) or ("delta_proj" in n)
                   or ("log_a" in n) or ("lm_head" in n) or (n == "embed.weight")
                   for n in names), f"C_side leaked into non-C names: {names}"


def test_unknown_adapt_set_raises():
    """Typos should fail loudly, not silently produce an empty list."""
    m = _tiny_ssm_lm()
    with pytest.raises((ValueError, KeyError)):
        select_adapt_params(m, adapt_set="log_A")  # note typo case


def test_selections_are_disjoint_where_expected():
    """B_side and C_side must not overlap — they target different projections."""
    m = _tiny_ssm_lm()
    b = set(id(p) for p in select_adapt_params(m, adapt_set="B_side"))
    c = set(id(p) for p in select_adapt_params(m, adapt_set="C_side"))
    assert b.isdisjoint(c), "B_side and C_side overlap"
