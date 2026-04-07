import torch
from chaoscontrol.data import (
    batch_from_starts, build_lm_starts, resolve_device, resolve_param_dtype,
    maybe_autocast, maybe_sync_cuda, choose_eval_starts,
)

def test_build_lm_starts():
    starts = build_lm_starts(100, seq_len=10, stride=5)
    assert len(starts) > 0
    assert all(s + 10 + 1 <= 100 for s in starts)

def test_batch_from_starts():
    tokens = torch.arange(100)
    starts = [0, 10, 20]
    inputs, targets = batch_from_starts(tokens, starts, seq_len=8, device=torch.device("cpu"))
    assert inputs.shape == (3, 8)
    assert targets.shape == (3, 8)

def test_resolve_device():
    d = resolve_device("cpu")
    assert d == torch.device("cpu")

def test_resolve_param_dtype():
    d = resolve_param_dtype("fp32", torch.device("cpu"))
    assert d == torch.float32

def test_choose_eval_starts():
    starts = list(range(0, 100, 5))
    selected = choose_eval_starts(starts, batch_size=4, eval_batches=2, seed=42)
    assert len(selected) == 8
