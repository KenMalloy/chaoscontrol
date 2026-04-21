"""Tests for the Exp 23 batch prefetch helper."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments" / "23_fast_path" / "fast_path.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("exp23_fast_path_prefetch", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_batch_prefetcher_primes_and_advances_one_batch_ahead():
    mod = _load_module()
    tokens = torch.arange(128, dtype=torch.int16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)

    prefetcher = mod.Exp23BatchPrefetcher(
        tokens=tokens,
        seq_len=4,
        stride=3,
        batch_size=2,
        rank=1,
        world_size=4,
        device=torch.device("cpu"),
        generator=generator,
    )

    prefetcher.prime()
    first_inputs, first_targets = prefetcher.next()
    second_inputs, second_targets = prefetcher.next()

    ref_generator = torch.Generator(device="cpu")
    ref_generator.manual_seed(7)

    first_starts = mod.sample_sharded_lm_starts(
        num_tokens=tokens.numel(),
        seq_len=4,
        stride=3,
        batch_size=2,
        rank=1,
        world_size=4,
        generator=ref_generator,
    )
    ref_first_inputs, ref_first_targets = mod.batch_from_start_tensor(
        tokens=tokens,
        starts=first_starts,
        seq_len=4,
        device=torch.device("cpu"),
    )
    second_starts = mod.sample_sharded_lm_starts(
        num_tokens=tokens.numel(),
        seq_len=4,
        stride=3,
        batch_size=2,
        rank=1,
        world_size=4,
        generator=ref_generator,
    )
    ref_second_inputs, ref_second_targets = mod.batch_from_start_tensor(
        tokens=tokens,
        starts=second_starts,
        seq_len=4,
        device=torch.device("cpu"),
    )

    assert torch.equal(first_inputs, ref_first_inputs)
    assert torch.equal(first_targets, ref_first_targets)
    assert torch.equal(second_inputs, ref_second_inputs)
    assert torch.equal(second_targets, ref_second_targets)
