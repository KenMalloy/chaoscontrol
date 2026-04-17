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


class _DeterministicClock:
    """Monotonic fake clock for deterministic budget-loop tests."""

    def __init__(self, *, start: float = 0.0, step: float = 0.25) -> None:
        self._t = start
        self._step = step

    def __call__(self) -> float:
        current = self._t
        self._t += self._step
        return current


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

    def test_fused_grad_clip_matches_stdlib_trajectory(
        self, bare_ssm_model: ChaosStudentLM,
    ) -> None:
        """End-to-end loss trajectory parity with fused_grad_clip toggled.

        Runs train_ssm_for_budget with the same seed and deterministic
        clock under both paths (stdlib clip_grad_norm_ vs fused). The
        fused path recomputes the global L2 norm via a single flat
        reduction rather than the per-tensor-norm-then-stack reduction
        the stdlib helper uses, so under fp32/CPU the loss trajectory
        agrees to fp reduction-order tolerance, not bit-identity. The
        diff is within the ~1e-6 relative noise of the norm computation
        propagated through a handful of optimizer steps.
        """
        g = torch.Generator().manual_seed(29)
        vocab = bare_ssm_model.vocab_size
        train_tokens = torch.randint(0, vocab, (256,), generator=g)
        train_starts = list(range(0, 256 - 16, 4))

        model_ref = bare_ssm_model
        model_new = copy.deepcopy(model_ref)

        optimizer_ref = torch.optim.AdamW(model_ref.parameters(), lr=1e-3)
        result_ref = train_ssm_for_budget(
            model=model_ref,
            train_tokens=train_tokens,
            train_starts=train_starts,
            seq_len=16,
            batch_size=2,
            device=torch.device("cpu"),
            optimizer=optimizer_ref,
            budget_seconds=1e9,  # max_steps gates the loop
            chunk_size=16,
            grad_clip_norm=0.1,  # small enough to fire clipping
            fused_grad_clip=False,
            seed=0,
            time_fn=_DeterministicClock(step=0.25),
            max_steps=5,
        )

        optimizer_new = torch.optim.AdamW(model_new.parameters(), lr=1e-3)
        result_new = train_ssm_for_budget(
            model=model_new,
            train_tokens=train_tokens,
            train_starts=train_starts,
            seq_len=16,
            batch_size=2,
            device=torch.device("cpu"),
            optimizer=optimizer_new,
            budget_seconds=1e9,
            chunk_size=16,
            grad_clip_norm=0.1,
            fused_grad_clip=True,
            seed=0,
            time_fn=_DeterministicClock(step=0.25),
            max_steps=5,
        )

        assert result_ref["steps"] == result_new["steps"] == 5
        losses_ref = [row["loss"] for row in result_ref["history"]]
        losses_new = [row["loss"] for row in result_new["history"]]
        # Step 0: both paths see identical grads (clip happens after
        # backward), so loss[0] is bit-identical regardless of clip path.
        assert losses_ref[0] == losses_new[0], (
            f"loss[0] must match pre-clip: ref={losses_ref[0]} new={losses_new[0]}"
        )
        # Subsequent steps: clipped grads differ by fp reduction-order
        # (see test_distributed.py::TestClipGradNormFused). That
        # propagates through the optimizer into parameter values and
        # onto the next forward's loss. The drift is bounded by the
        # ~1e-6 relative norm difference compounded over a handful of
        # steps — well under 1e-4 absolute on these scales.
        for i, (lr, ln) in enumerate(zip(losses_ref, losses_new)):
            assert abs(lr - ln) < 1e-4, (
                f"loss[{i}] drift too large: ref={lr} new={ln} diff={lr - ln}"
            )

    def test_fused_muon_matches_unfused_trajectory(
        self, bare_ssm_model: ChaosStudentLM,
    ) -> None:
        """End-to-end loss trajectory parity with fused_muon toggled.

        Builds two Muon optimizers — one with ``fused=False`` iterating
        params one-by-one, one with ``fused=True`` batching NS calls per
        shape-group and coalescing AdamW over flat non-matrix buffers —
        and drives five training steps through ``train_ssm_for_budget``
        at the same seed. Losses agree within ``1e-4`` per step; the
        drift comes from the batched NS's intra-shape reduction order
        vs the per-param loop, same fp-reduction-noise regime as the
        fused grad-clip trajectory test above.
        """
        from chaoscontrol.optim.muon import Muon

        g = torch.Generator().manual_seed(31)
        vocab = bare_ssm_model.vocab_size
        train_tokens = torch.randint(0, vocab, (256,), generator=g)
        train_starts = list(range(0, 256 - 16, 4))

        model_ref = bare_ssm_model
        model_new = copy.deepcopy(model_ref)

        optimizer_ref = Muon(
            list(model_ref.parameters()),
            lr=0.005, adamw_lr=0.001,
            compute_dtype=torch.float32,
            fused=False,
        )
        optimizer_ref.bind_param_names(list(model_ref.named_parameters()))
        result_ref = train_ssm_for_budget(
            model=model_ref,
            train_tokens=train_tokens,
            train_starts=train_starts,
            seq_len=16,
            batch_size=2,
            device=torch.device("cpu"),
            optimizer=optimizer_ref,
            budget_seconds=1e9,
            chunk_size=16,
            grad_clip_norm=0.0,
            seed=0,
            time_fn=_DeterministicClock(step=0.25),
            max_steps=5,
        )

        optimizer_new = Muon(
            list(model_new.parameters()),
            lr=0.005, adamw_lr=0.001,
            compute_dtype=torch.float32,
            fused=True,
        )
        optimizer_new.bind_param_names(list(model_new.named_parameters()))
        result_new = train_ssm_for_budget(
            model=model_new,
            train_tokens=train_tokens,
            train_starts=train_starts,
            seq_len=16,
            batch_size=2,
            device=torch.device("cpu"),
            optimizer=optimizer_new,
            budget_seconds=1e9,
            chunk_size=16,
            grad_clip_norm=0.0,
            seed=0,
            time_fn=_DeterministicClock(step=0.25),
            max_steps=5,
        )

        assert result_ref["steps"] == result_new["steps"] == 5
        losses_ref = [row["loss"] for row in result_ref["history"]]
        losses_new = [row["loss"] for row in result_new["history"]]
        # Step 0: both paths see identical grads BEFORE opt.step runs
        # (forward/backward is fused-path-agnostic), so loss[0] is
        # bit-identical regardless of optimizer batching.
        assert losses_ref[0] == losses_new[0], (
            f"loss[0] must match pre-step: ref={losses_ref[0]} new={losses_new[0]}"
        )
        for i, (lr_loss, ln_loss) in enumerate(zip(losses_ref, losses_new)):
            assert abs(lr_loss - ln_loss) < 1e-4, (
                f"fused-Muon loss[{i}] drift too large: "
                f"ref={lr_loss} new={ln_loss} diff={lr_loss - ln_loss}"
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
    """Deterministic test for the budget-bounded training loop wrapper.

    The step function already has per-gradient parity coverage above.
    Here we care about loop bookkeeping and optimizer integration, so
    the budget is driven by a fake monotonic clock rather than host
    wall-clock speed.
    """

    def test_single_process_loop_has_deterministic_history(self, bare_ssm_model: ChaosStudentLM) -> None:
        # Deterministic fake corpus — enough tokens to form a handful of
        # batches of short sequences.
        g = torch.Generator().manual_seed(17)
        vocab = bare_ssm_model.vocab_size
        train_tokens = torch.randint(0, vocab, (256,), generator=g)
        train_starts = list(range(0, 256 - 16, 4))  # overlapping windows of 16
        clock = _DeterministicClock(step=0.25)
        initial_flat = torch.cat([
            p.detach().flatten().clone() for p in bare_ssm_model.parameters()
        ])

        optimizer = torch.optim.AdamW(bare_ssm_model.parameters(), lr=1e-3)
        result = train_ssm_for_budget(
            model=bare_ssm_model,
            train_tokens=train_tokens,
            train_starts=train_starts,
            seq_len=16,
            batch_size=2,
            device=torch.device("cpu"),
            optimizer=optimizer,
            budget_seconds=1.1,
            chunk_size=16,
            seed=0,
            time_fn=clock,
        )
        assert result["steps"] == 4
        assert result["world_size"] == 1
        assert result["rank"] == 0
        assert result["elapsed_s"] == pytest.approx(1.5)
        assert len(result["history"]) == result["steps"]
        assert [row["step"] for row in result["history"]] == [0.0, 1.0, 2.0, 3.0]
        losses = [row["loss"] for row in result["history"]]
        assert all(torch.isfinite(torch.tensor(losses)))
        assert all(loss > 0.0 for loss in losses)

        final_flat = torch.cat([p.detach().flatten() for p in bare_ssm_model.parameters()])
        max_param_delta = (final_flat - initial_flat).abs().max().item()
        assert max_param_delta > 0.0, "optimizer steps should update model parameters"

    def test_max_steps_caps_loop_below_budget(self, bare_ssm_model: ChaosStudentLM) -> None:
        g = torch.Generator().manual_seed(23)
        vocab = bare_ssm_model.vocab_size
        train_tokens = torch.randint(0, vocab, (256,), generator=g)
        train_starts = list(range(0, 256 - 16, 4))
        clock = _DeterministicClock(step=0.25)

        optimizer = torch.optim.AdamW(bare_ssm_model.parameters(), lr=1e-3)
        result = train_ssm_for_budget(
            model=bare_ssm_model,
            train_tokens=train_tokens,
            train_starts=train_starts,
            seq_len=16,
            batch_size=2,
            device=torch.device("cpu"),
            optimizer=optimizer,
            budget_seconds=1000.0,
            chunk_size=16,
            seed=0,
            time_fn=clock,
            max_steps=3,
        )
        assert result["steps"] == 3


class TestLossAccumulatorParity:
    """Stacked end-of-loop loss transfer must match per-step ``.cpu()``.

    The 2026-04-17 switch from per-step ``float(loss.detach().cpu())`` to
    ``torch.stack(losses).cpu()`` is purely a throughput win (collapses N
    GPU->CPU syncs into one). It must not drift the reported history
    values — downstream summarizers, bpb conversions, and the paper's
    figures read those numbers. This test locks the numerical equivalence
    so a future edit to the accumulator contract can't silently skew
    reported losses.
    """

    @staticmethod
    def _per_step_history(
        losses: list[torch.Tensor],
    ) -> list[dict[str, float]]:
        """The pre-2026-04-17 pattern — one ``.cpu()`` per step."""
        return [
            {"step": float(i), "loss": float(loss.detach().cpu())}
            for i, loss in enumerate(losses)
        ]

    @staticmethod
    def _stacked_history(
        losses: list[torch.Tensor],
    ) -> list[dict[str, float]]:
        """The current train_ssm.train_ssm_for_budget pattern.

        Mirrors lines 308-328 of ``src/chaoscontrol/train_ssm.py``: detach
        per step, stack+transfer once at loop exit, build history from the
        CPU-side tensor.
        """
        loss_tensors = [loss.detach() for loss in losses]
        if not loss_tensors:
            return []
        stacked = torch.stack(loss_tensors).cpu()
        return [
            {"step": float(i), "loss": float(stacked[i])}
            for i in range(len(loss_tensors))
        ]

    def test_stacked_matches_per_step_on_synthetic_losses(self) -> None:
        """Arbitrary CPU scalars: stacked path must equal per-step path."""
        torch.manual_seed(0)
        losses = [torch.randn(()) for _ in range(25)]
        per_step = self._per_step_history(losses)
        stacked = self._stacked_history(losses)
        assert per_step == stacked

    def test_stacked_matches_per_step_with_degenerate_values(self) -> None:
        """Edge cases the production loop can encounter.

        Zero, huge, and tiny loss values all need to round-trip identically.
        The CPU transfer path does not do any dtype promotion, so this is
        primarily a check that ``float(tensor)`` of a stacked row equals
        ``float(tensor)`` of the standalone row.
        """
        losses = [
            torch.tensor(0.0),
            torch.tensor(1e-30),
            torch.tensor(1e30),
            torch.tensor(4.2),
        ]
        per_step = self._per_step_history(losses)
        stacked = self._stacked_history(losses)
        assert per_step == stacked

    def test_empty_losses_produce_empty_history(self) -> None:
        """``train_ssm_for_budget`` returns ``history=[]`` if max_steps=0.

        Pins the zero-training short-circuit against an edit that would
        try to ``torch.stack([])`` (which raises).
        """
        assert self._stacked_history([]) == []

    def test_detach_severs_autograd_graph(self) -> None:
        """``loss.detach()`` must not keep a backward reference alive.

        Test-coverage rationale: the loss accumulator holds one detached
        scalar per training step. If ``.detach()`` ever stopped severing
        the graph (extremely unlikely, but load-bearing for memory),
        activations would pile up and the long-run OOM pattern would
        return at scale. Cheap unit check; catches the regression without
        needing a training run to blow up.
        """
        x = torch.randn(4, requires_grad=True)
        loss = (x ** 2).sum()
        detached = loss.detach()
        assert detached.grad_fn is None
        assert detached.requires_grad is False
