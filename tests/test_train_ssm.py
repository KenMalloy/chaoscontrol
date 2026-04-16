"""Integration-level equivalence test for the lean ``train_ssm`` step.

One optimizer step through ``train_ssm.train_ssm_step`` must produce
the same parameter gradients as the bare-SSM path through
``training.py`` — ``forward() + chunked_cross_entropy + backward()`` —
within bf16 numerical tolerance. This is the correctness gate before
any of the throughput-first work (bs=1024 at seq=512, bs=4096 at
seq=2048) goes to a pod.

The comparison only makes sense on a bare-SSM config: ``train_ssm`` is
narrower than ``training.py`` and rejects every feature
(metabolic / MC / criticality / wernicke / outer_model / posterior)
that the old loop knows how to run.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.training import chunked_cross_entropy
from chaoscontrol.train_ssm import train_ssm_for_budget, train_ssm_step


@pytest.fixture
def bare_ssm_model() -> ChaosStudentLM:
    torch.manual_seed(2026)
    m = ChaosStudentLM(
        vocab_size=64,
        dim=16,
        num_layers=2,
        ff_mult=2,
        a_mode="diag",
    )
    m.train()
    return m


def _make_batch(batch: int, seq: int, vocab: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    inputs = torch.randint(0, vocab, (batch, seq), generator=g)
    targets = torch.randint(0, vocab, (batch, seq), generator=g)
    return inputs, targets


def _old_path_grads(
    model: ChaosStudentLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run one bare-SSM training step the old way and snapshot param grads.

    Mirrors ``training.py`` at the level that applies to a bare-SSM
    config: forward → chunked CE → backward. No criticality, no MC,
    no aux losses (all wired to ``None`` in this config).
    """
    vocab = model.vocab_size
    model.zero_grad(set_to_none=True)
    out = model(inputs)
    loss = chunked_cross_entropy(
        out["logits"].reshape(-1, vocab),
        targets.reshape(-1),
    )
    loss.backward()
    return {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


class TestTrainSSMStepEquivalence:
    """Parameter-gradient parity between train_ssm_step and the old path."""

    def test_grads_match_bare_ssm_old_path(self, bare_ssm_model: ChaosStudentLM) -> None:
        inputs, targets = _make_batch(batch=2, seq=16, vocab=64, seed=11)

        # Two independent copies of the same initial weights
        model_old = bare_ssm_model
        model_new = copy.deepcopy(model_old)

        old_grads = _old_path_grads(model_old, inputs, targets)

        # New path: train_ssm step, chunk_size large enough that chunking
        # doesn't alter reduction order meaningfully (tight fp32 tolerance).
        model_new.zero_grad(set_to_none=True)
        _loss = train_ssm_step(
            model=model_new,
            inputs=inputs,
            targets=targets,
            chunk_size=32,  # >= seq, one chunk → same reduction order as old path
        )
        new_grads = {
            name: p.grad.detach().clone()
            for name, p in model_new.named_parameters()
            if p.grad is not None
        }

        assert set(old_grads.keys()) == set(new_grads.keys()), (
            f"param grad coverage differs; "
            f"old-only: {set(old_grads) - set(new_grads)}, "
            f"new-only: {set(new_grads) - set(old_grads)}"
        )
        max_diff_by_param: dict[str, float] = {}
        for name in old_grads:
            d = (old_grads[name] - new_grads[name]).abs().max().item()
            max_diff_by_param[name] = d
        worst = max(max_diff_by_param.values())
        assert worst < 1e-5, (
            f"gradient mismatch between old path and train_ssm_step. "
            f"worst param: {max(max_diff_by_param, key=max_diff_by_param.get)!r}, "
            f"max-abs diff {worst}. all diffs: {max_diff_by_param}"
        )

    def test_grads_match_with_small_chunks(self, bare_ssm_model: ChaosStudentLM) -> None:
        # Exercise actual chunking: seq=16, chunk=4 → 4 chunks.
        # Reduction order differs from the old path (per-chunk sum / N then
        # backward) so tolerance is looser but still well inside
        # fp32 noise floor.
        inputs, targets = _make_batch(batch=2, seq=16, vocab=64, seed=13)
        model_old = bare_ssm_model
        model_new = copy.deepcopy(model_old)

        old_grads = _old_path_grads(model_old, inputs, targets)

        model_new.zero_grad(set_to_none=True)
        train_ssm_step(
            model=model_new,
            inputs=inputs,
            targets=targets,
            chunk_size=4,
        )
        new_grads = {
            name: p.grad.detach().clone()
            for name, p in model_new.named_parameters()
            if p.grad is not None
        }
        for name in old_grads:
            d = (old_grads[name] - new_grads[name]).abs().max().item()
            assert d < 1e-5, f"fp32 chunk=4 grad mismatch on {name!r}: {d}"


class TestTrainSSMStepRejectsUnsupportedConfigs:
    """Configs that ``training.py`` supports but ``train_ssm`` does not
    must fail loudly at the step entry rather than produce silently
    wrong results.
    """

    def test_rejects_wernicke_enabled_model(self) -> None:
        torch.manual_seed(1)
        model = ChaosStudentLM(
            vocab_size=32, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag",
            wernicke_enabled=True, wernicke_k_max=4, wernicke_window=4,
        )
        inputs, targets = _make_batch(batch=1, seq=4, vocab=32, seed=14)
        with pytest.raises(ValueError, match="wernicke"):
            train_ssm_step(model=model, inputs=inputs, targets=targets, chunk_size=4)

    def test_rejects_outer_model_enabled(self) -> None:
        torch.manual_seed(2)
        model = ChaosStudentLM(
            vocab_size=32, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag",
            outer_model_dim=8, outer_model_type="single",
        )
        inputs, targets = _make_batch(batch=1, seq=4, vocab=32, seed=15)
        with pytest.raises(ValueError, match="outer_model"):
            train_ssm_step(model=model, inputs=inputs, targets=targets, chunk_size=4)


class TestTrainSSMForBudget:
    """Smoke test for the budget-bounded training loop wrapper.

    The wall-clock loop itself is just the step function plus batch
    construction, optimizer step, and DDP plumbing — the step function
    already has per-gradient parity coverage above. This test is for
    "the loop actually runs and makes progress", not for numeric
    equivalence against the old path.
    """

    def test_single_process_loop_reduces_loss(self, bare_ssm_model: ChaosStudentLM) -> None:
        # Deterministic fake corpus — enough tokens to form a handful of
        # batches of short sequences.
        g = torch.Generator().manual_seed(17)
        vocab = bare_ssm_model.vocab_size
        train_tokens = torch.randint(0, vocab, (256,), generator=g)
        train_starts = list(range(0, 256 - 16, 4))  # overlapping windows of 16

        optimizer = torch.optim.AdamW(bare_ssm_model.parameters(), lr=1e-3)
        result = train_ssm_for_budget(
            model=bare_ssm_model,
            train_tokens=train_tokens,
            train_starts=train_starts,
            seq_len=16,
            batch_size=2,
            device=torch.device("cpu"),
            optimizer=optimizer,
            budget_seconds=2.0,  # small but non-zero
            chunk_size=16,
            seed=0,
        )
        assert result["steps"] > 1, (
            f"loop should run multiple steps in 2s, got {result['steps']}"
        )
        assert result["world_size"] == 1
        assert result["rank"] == 0
        # Result schema matches training.py's elapsed_s field name so any
        # caller can read both paths with the same key.
        assert "elapsed_s" in result and result["elapsed_s"] > 0
        # Loss should generally decrease — not strictly monotonic on such a
        # tiny model/corpus, but the final EMA should beat the initial loss.
        losses = [r["loss"] for r in result["history"]]
        first_quarter = sum(losses[: len(losses) // 4]) / max(1, len(losses) // 4)
        last_quarter = sum(losses[-len(losses) // 4 :]) / max(1, len(losses) // 4)
        assert last_quarter < first_quarter, (
            f"loss should drop across the loop; "
            f"first-quarter mean {first_quarter}, last-quarter mean {last_quarter}"
        )
