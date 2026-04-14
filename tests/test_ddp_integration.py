"""Tests for the DDP integration in chaoscontrol.training.

These tests deliberately avoid any GPU requirement — the DDP path is
exercised on CPU with the gloo backend. A small bare ``ChaosStudentLM`` is
used throughout so the model construction is cheap and the spawn overhead
dominates the test time instead of compute.

The load-bearing test is :meth:`TestDDPIntegration.test_single_device_parity`
— it confirms the refactor did not drift the single-device code path.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from chaoscontrol.core import verify_diag_recurrence
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.training import train_chaoscontrol_for_budget

# Pre-warm the diag recurrence backend at import time so the first test's
# budget isn't consumed by the torch.compile resolution path. On CPU the
# compile either succeeds and is cached, or it fails-fast to the python
# fallback which is what our determinism assertions expect.
verify_diag_recurrence(torch.device("cpu"))


# ---------------------------------------------------------------------------
# Shared training configuration for every test
# ---------------------------------------------------------------------------

MODEL_KWARGS = dict(
    vocab_size=64,
    dim=16,
    num_layers=2,
    ff_mult=2,
    a_mode="diag",
    rich_b_mode="none",
    outer_model_dim=0,
)

TRAIN_KWARGS = dict(
    seq_len=16,
    batch_size=4,
    device=torch.device("cpu"),
    param_dtype=torch.float32,
    # Caller overrides per-test; see _train_fixed_steps docstring. The CPU
    # training loop completes hundreds of steps/s on a small model, so a
    # 2s budget is more than enough for the 10-step parity checks and
    # absorbs one-time torch.compile resolution overhead in the first test.
    budget_seconds=2.0,
    base_lr=1e-3,
    weight_decay=0.0,
    grad_clip_norm=1.0,
    crit_reg_alpha=0.0,
    crit_reg_beta=0.0,
)

N_TRAIN_TOKENS = 1024
TRAIN_STARTS = list(range(0, 800, 8))  # 100 starts — safe for ws <= 4


def _build_model(seed: int) -> ChaosStudentLM:
    """Deterministic model construction under a fixed seed."""
    torch.manual_seed(seed)
    return ChaosStudentLM(**MODEL_KWARGS)


def _build_tokens(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, MODEL_KWARGS["vocab_size"], (N_TRAIN_TOKENS,), generator=g)


def _train_fixed_steps(
    model: ChaosStudentLM,
    tokens: torch.Tensor,
    *,
    seed: int,
    n_steps: int,
    rank: int | None = None,
    world_size: int | None = None,
    budget_seconds: float = 60.0,
) -> dict:
    """Train for a fixed step count by using a massive time budget and then
    slicing the loss history. This makes the test deterministic regardless
    of wall clock; the existing loop exits when ``budget_seconds`` expires
    AND ``steps > 0``, so we set a budget that comfortably exceeds what the
    test machine can finish in n_steps.

    For deterministic step counts we cap via ``budget_seconds`` and then
    assert the history length is at least ``n_steps``; the test compares
    the first ``n_steps`` entries.
    """
    result = train_chaoscontrol_for_budget(
        model,
        train_tokens=tokens,
        train_starts=list(TRAIN_STARTS),
        seed=seed,
        rank=rank,
        world_size=world_size,
        **{**TRAIN_KWARGS, "budget_seconds": budget_seconds},
    )
    assert result["steps"] >= n_steps, (
        f"Training finished before reaching {n_steps} steps "
        f"(got {result['steps']}). Increase budget_seconds."
    )
    return result


# ---------------------------------------------------------------------------
# Worker functions for multi-process DDP tests
# ---------------------------------------------------------------------------

def _ddp_worker_grad_sync(
    rank: int,
    world_size: int,
    master_port: int,
    param_dump_paths: list[str],
    stdout_log_paths: list[str],
    output_json_paths: list[str],
    barrier_path: str,
    seed: int,
) -> None:
    """Worker for multi-rank DDP tests.

    Every worker:
        1. Initializes the process group on gloo with a shared master port.
        2. Redirects stdout to a per-rank log file so the parent can inspect
           whether non-rank-0 printed anything.
        3. Pre-warms the diag-recurrence backend (matches the runner_exp17
           and runner_exp18 pattern of verify_diag_recurrence per-rank).
        4. Builds an identical model (same init seed) and sharded training
           data.
        5. Runs ``train_chaoscontrol_for_budget`` with a short time budget
           (typically dozens of steps) and dumps parameters + loss trace.
        6. Writes a per-rank parameter dump + (for rank 0 only) an output
           JSON file whose creation the parent inspects.
        7. Destroys the process group before returning.

    Using file-backed output instead of queues keeps the test resilient on
    systems where mp.Queue occasionally hangs at shutdown.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # warn_only=True determinism: belt-and-suspenders for the
    # seed-reproducibility test. The actual determinism comes from the
    # loss-trace comparison (not final-param comparison, which can differ
    # by +/- 1 step due to wall-clock jitter), but leaving this on keeps
    # future contributors from tripping over obscure CPU nondeterminism in
    # backward passes that would otherwise not surface until a flaky CI run.
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Capture stdout so the parent test can assert non-rank-0 never prints.
    # We keep this exclusive to ranks > 0 to avoid clobbering pytest's
    # captured output on rank 0 when running embedded; rank 0 routes to the
    # same log path so the check is symmetric.
    with open(stdout_log_paths[rank], "w") as log_file:
        sys.stdout = log_file

        try:
            dist.init_process_group(
                backend="gloo", rank=rank, world_size=world_size
            )

            model = _build_model(seed)
            tokens = _build_tokens(seed)

            # Pre-warm the diag backend inside each worker subprocess so
            # the first training step isn't consumed by a compile attempt.
            verify_diag_recurrence(torch.device("cpu"))

            result = train_chaoscontrol_for_budget(
                model,
                train_tokens=tokens,
                train_starts=list(TRAIN_STARTS),
                seed=seed,
                rank=rank,
                world_size=world_size,
                # Short budget — we only need >= 5 steps worth of training
                # to verify DDP gradient sync. CPU is fast enough at the
                # test model size (~150 params) that 1s gives dozens of
                # steps; keeping it short minimizes total test wall time.
                **{**TRAIN_KWARGS, "budget_seconds": 1.0},
            )

            # Optional rank-0-only output file — the rank-0-guard test checks
            # that non-rank-0 never creates this path.
            if rank == 0:
                Path(output_json_paths[rank]).write_text(
                    json.dumps({"steps": result["steps"]})
                )
                # Rank 0 *does* print a tiny marker line so the rank-0 guard
                # test can distinguish "wrote nothing" from "log file did
                # not get created".
                print(f"rank0 marker: steps={result['steps']}", flush=True)

            # Dump flat parameter vector for parent comparisons.
            flat = torch.cat([p.detach().flatten() for p in model.parameters()])
            torch.save(flat, param_dump_paths[rank])

            # Also dump the first 20 loss values. The step count depends on
            # wall-clock jitter (different across runs) so the final param
            # vector is not directly comparable — but the first N loss
            # values are identical across runs with the same seed, so the
            # seed-reproducibility test uses this slice.
            loss_trace = [
                float(h["loss"]) for h in result["history"][:20]
            ]
            Path(param_dump_paths[rank] + ".losses.json").write_text(
                json.dumps(loss_trace)
            )

            # Barrier + file marker so the parent can wait on this file.
            dist.barrier()
            Path(barrier_path + f".rank{rank}").write_text("ok")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            sys.stdout = sys.__stdout__


def _run_multiprocess(
    worker,
    *,
    world_size: int,
    seed: int,
    tmpdir: Path,
) -> dict:
    """Spawn ``world_size`` processes running ``worker`` and collect outputs.

    Returns a dict with:
        - ``param_dumps``: list[Tensor] of flattened parameters per rank
        - ``stdout_logs``: list[str] of captured stdout per rank
        - ``output_json_paths``: list[str] the workers *may* have written
    """
    param_dump_paths = [str(tmpdir / f"params_rank{r}.pt") for r in range(world_size)]
    stdout_log_paths = [str(tmpdir / f"stdout_rank{r}.log") for r in range(world_size)]
    output_json_paths = [str(tmpdir / f"out_rank{r}.json") for r in range(world_size)]
    barrier_path = str(tmpdir / "barrier")

    # Pick a fresh port per invocation to avoid collisions between back-to-back
    # multi-process tests in the same pytest run.
    master_port = _pick_free_port()

    mp.spawn(
        worker,
        args=(
            world_size,
            master_port,
            param_dump_paths,
            stdout_log_paths,
            output_json_paths,
            barrier_path,
            seed,
        ),
        nprocs=world_size,
        join=True,
    )

    param_dumps = [torch.load(p, weights_only=True) for p in param_dump_paths]
    stdout_logs = [Path(p).read_text() if Path(p).exists() else "" for p in stdout_log_paths]
    loss_traces = []
    for p in param_dump_paths:
        loss_path = Path(p + ".losses.json")
        if loss_path.exists():
            loss_traces.append(json.loads(loss_path.read_text()))
        else:
            loss_traces.append([])
    return {
        "param_dumps": param_dumps,
        "stdout_logs": stdout_logs,
        "output_json_paths": output_json_paths,
        "loss_traces": loss_traces,
    }


def _pick_free_port() -> int:
    """Ask the OS for an ephemeral TCP port and immediately release it.

    There's a race between close() and the DDP init using the port, but in
    practice it's reliable enough for CPU tests and is the standard idiom in
    torch's own DDP test suite.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleDevicePath(unittest.TestCase):
    """Load-bearing regression: the DDP refactor must not drift the
    single-device training trajectory at rank=0, world_size=1.
    """

    def test_single_device_parity(self) -> None:
        """Default call (rank/world unset) must match explicit rank=0/ws=1.

        The training loop exits on a wall-clock budget, so the two runs can
        end at slightly different step counts. We deliberately compare the
        *first* ``n_steps`` entries of each history — the loop takes the
        same branches at every step for those first entries regardless of
        when the budget expires, so they must match exactly.
        """
        seed = 12345
        n_steps = 10
        # 2s gives ~800 steps on CPU for this tiny model and absorbs any
        # one-off torch.compile resolution cost on the first test in the
        # session.
        budget = 2.0

        # Reference: no DDP args — exercises the None/None path that resolves
        # to the single-device fallback.
        model_ref = _build_model(seed)
        tokens_ref = _build_tokens(seed)
        result_ref = _train_fixed_steps(
            model_ref, tokens_ref, seed=seed, n_steps=n_steps, budget_seconds=budget,
        )

        # New path: explicit rank=0, world_size=1. Must take the same branch
        # (ddp_active = world_size > 1 = False) and produce identical output.
        model_new = _build_model(seed)
        tokens_new = _build_tokens(seed)
        result_new = _train_fixed_steps(
            model_new,
            tokens_new,
            seed=seed,
            n_steps=n_steps,
            rank=0,
            world_size=1,
            budget_seconds=budget,
        )

        ref_losses = [h["loss"] for h in result_ref["history"][:n_steps]]
        new_losses = [h["loss"] for h in result_new["history"][:n_steps]]

        self.assertEqual(len(ref_losses), n_steps)
        self.assertEqual(len(new_losses), n_steps)
        for i, (a, b) in enumerate(zip(ref_losses, new_losses)):
            # Bit-equal comparison — the two runs exercise identical code
            # paths (ddp_active = False) with identical seeds. Any drift is
            # a real regression in the refactor, not floating-point jitter.
            self.assertEqual(
                a,
                b,
                msg=f"Single-device loss trajectory drift at step {i}: ref={a} new={b}",
            )

    def test_ddp_ctx_helper_resolves_defaults(self) -> None:
        """_resolve_ddp_context must not pick up stale env vars when the
        caller is using the default single-device path."""
        # Import the private helper directly — this is internal but stable
        # enough for tests to reach in and poke at.
        from chaoscontrol.training import _resolve_ddp_context

        # Explicit args take precedence.
        self.assertEqual(_resolve_ddp_context(0, 1), (0, 1))
        self.assertEqual(_resolve_ddp_context(3, 8), (3, 8))

        # Partial args are an error.
        with self.assertRaises(ValueError):
            _resolve_ddp_context(0, None)
        with self.assertRaises(ValueError):
            _resolve_ddp_context(None, 4)

        # Fallback to env vars. Stash any existing state so we don't clobber
        # a real torchrun launch context (not expected in pytest, but be safe).
        old_rank = os.environ.pop("RANK", None)
        old_world = os.environ.pop("WORLD_SIZE", None)
        try:
            self.assertEqual(_resolve_ddp_context(None, None), (0, 1))
            os.environ["RANK"] = "1"
            os.environ["WORLD_SIZE"] = "4"
            self.assertEqual(_resolve_ddp_context(None, None), (1, 4))
        finally:
            os.environ.pop("RANK", None)
            os.environ.pop("WORLD_SIZE", None)
            if old_rank is not None:
                os.environ["RANK"] = old_rank
            if old_world is not None:
                os.environ["WORLD_SIZE"] = old_world


class TestSingleDeviceOptimizerBranches(unittest.TestCase):
    """One-step smoke per optimizer choice on the single-device code path.

    These confirm the optimizer kwarg takes the expected branch when the
    DDP path resolves to world_size=1 (no wrapping, no barriers). The
    multi-process DDP optimizer smoke is deferred to real 8xH100 runs
    because CPU gloo adds process-spawning overhead that is not worth
    paying for a 3x Muon + LAMB + AdamW check when the branches are
    already exercised here.
    """

    def _run_one_opt(self, opt_name: str) -> dict:
        seed = 321
        model = _build_model(seed)
        tokens = _build_tokens(seed)
        return train_chaoscontrol_for_budget(
            model,
            train_tokens=tokens,
            train_starts=list(TRAIN_STARTS),
            seed=seed,
            rank=0,
            world_size=1,
            optimizer=opt_name,
            **{**TRAIN_KWARGS, "budget_seconds": 2.0},
        )

    def test_adamw_branch_single_device(self) -> None:
        result = self._run_one_opt("adamw")
        self.assertEqual(result["optimizer_type"], "AdamW")
        self.assertEqual(result["optimizer_name"], "adamw")
        self.assertGreaterEqual(result["steps"], 1)
        first_loss = float(result["history"][0]["loss"])
        self.assertFalse(first_loss != first_loss, msg="NaN loss on adamw step 0")

    def test_muon_branch_single_device(self) -> None:
        result = self._run_one_opt("muon")
        self.assertEqual(result["optimizer_type"], "Muon")
        self.assertEqual(result["optimizer_name"], "muon")
        self.assertGreaterEqual(result["steps"], 1)
        first_loss = float(result["history"][0]["loss"])
        self.assertFalse(first_loss != first_loss, msg="NaN loss on muon step 0")

    def test_lamb_branch_single_device(self) -> None:
        result = self._run_one_opt("lamb")
        self.assertEqual(result["optimizer_type"], "LAMB")
        self.assertEqual(result["optimizer_name"], "lamb")
        self.assertGreaterEqual(result["steps"], 1)
        first_loss = float(result["history"][0]["loss"])
        self.assertFalse(first_loss != first_loss, msg="NaN loss on lamb step 0")


class TestMultiProcessDDP(unittest.TestCase):
    """CPU+gloo multi-process DDP smoke tests.

    These tests are skipped on systems where gloo is not available or when
    running under an environment that blocks process spawning (e.g., CUDA
    tests mode where mp.spawn is restricted).
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not dist.is_available():
            raise unittest.SkipTest("torch.distributed not available")
        if not dist.is_gloo_available():
            raise unittest.SkipTest("gloo backend not available")

    def _run(self, *, world_size: int, seed: int, tmppath: Path) -> dict:
        """Spawn DDP workers inside an externally-managed tmpdir.

        The caller owns the tmpdir lifetime so test assertions can inspect
        per-rank files on disk (e.g. "rank 1 didn't create a JSON file")
        before cleanup.
        """
        return _run_multiprocess(
            _ddp_worker_grad_sync,
            world_size=world_size,
            seed=seed,
            tmpdir=tmppath,
        )

    def test_gradient_sync_across_ranks(self) -> None:
        """After training steps with DDP, both ranks must hold identical
        parameters (DDP all-reduce averages grads so the optimizer step
        applies the same update everywhere)."""
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self._run(world_size=world_size, seed=2024, tmppath=Path(tmpdir))

            params_rank0 = out["param_dumps"][0]
            params_rank1 = out["param_dumps"][1]
            self.assertEqual(params_rank0.shape, params_rank1.shape)
            # After DDP all-reduce of gradients + identical optimizer state,
            # the post-step params should be bit-identical across ranks.
            max_diff = (params_rank0 - params_rank1).abs().max().item()
            self.assertLess(
                max_diff,
                1e-6,
                f"DDP gradient sync did not produce identical params across ranks "
                f"(max |diff| = {max_diff:.3e}).",
            )

    def test_rank0_guard_no_writes_or_prints_on_non_rank0(self) -> None:
        """Non-rank-0 processes must not touch the rank-0 output path nor
        print the rank-0 marker line."""
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self._run(world_size=world_size, seed=7, tmppath=Path(tmpdir))

            # Only rank-0 output file should exist.
            rank0_out = Path(out["output_json_paths"][0])
            rank1_out = Path(out["output_json_paths"][1])
            self.assertTrue(rank0_out.exists(), "rank-0 output file missing")
            self.assertFalse(
                rank1_out.exists(), "rank-1 created a file it shouldn't have"
            )

            # Rank 0 stdout must contain the marker; rank 1 must not.
            self.assertIn("rank0 marker", out["stdout_logs"][0])
            self.assertNotIn("rank0 marker", out["stdout_logs"][1])
            # Rank 1 should be literally empty since the training loop
            # itself emits no prints and the runner_exp17 prints are
            # rank-0-guarded.
            self.assertEqual(
                out["stdout_logs"][1].strip(),
                "",
                f"rank-1 unexpectedly printed: {out['stdout_logs'][1]!r}",
            )

    def test_seed_reproducibility_across_two_runs(self) -> None:
        """Two independent 2-rank runs with the same seed must produce the
        same per-step loss trajectory.

        Why loss trajectory and not final params: the training loop exits
        on a wall-clock budget, so two independent runs can finish at
        different step counts (jitter of +/- 1 step is normal under mp.spawn
        on macOS). Comparing final params would false-positive drift because
        of differing step counts. Comparing the first N loss values — which
        are computed at deterministic steps regardless of when the budget
        later expires — is the correct shape-invariant comparison.
        """
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir_a:
            out_a = self._run(
                world_size=world_size, seed=99, tmppath=Path(tmpdir_a)
            )
            trace_a_rank0 = list(out_a["loss_traces"][0])
            trace_a_rank1 = list(out_a["loss_traces"][1])
        with tempfile.TemporaryDirectory() as tmpdir_b:
            out_b = self._run(
                world_size=world_size, seed=99, tmppath=Path(tmpdir_b)
            )
            trace_b_rank0 = list(out_b["loss_traces"][0])
            trace_b_rank1 = list(out_b["loss_traces"][1])

        # Both runs must have produced at least a few loss values.
        self.assertGreaterEqual(len(trace_a_rank0), 5)
        self.assertGreaterEqual(len(trace_b_rank0), 5)

        n = min(len(trace_a_rank0), len(trace_b_rank0), 10)
        for i in range(n):
            # Deterministic DDP on CPU with the same seed must produce
            # exactly identical loss values at each step — any drift here
            # indicates DDP sync or seeding is broken.
            self.assertAlmostEqual(
                trace_a_rank0[i],
                trace_b_rank0[i],
                places=5,
                msg=(
                    f"Rank-0 loss trajectory diverged at step {i}: "
                    f"run A={trace_a_rank0[i]} run B={trace_b_rank0[i]}"
                ),
            )
            # Rank 1 must also agree with rank 0 since DDP all-reduces loss
            # computations contribute to grads, and the main loss (ce_loss)
            # on each rank comes from that rank's shard — different shards
            # can produce different per-step losses. The stronger assertion
            # is run-over-run (A vs B) for the same rank.
            self.assertAlmostEqual(
                trace_a_rank1[i],
                trace_b_rank1[i],
                places=5,
                msg=(
                    f"Rank-1 loss trajectory diverged at step {i}: "
                    f"run A={trace_a_rank1[i]} run B={trace_b_rank1[i]}"
                ),
            )


class TestMetabolicDDPGuard(unittest.TestCase):
    """Guard: metabolic_fork/micro_mcts + DDP is unsafe and must raise fast.

    Under DDP, metabolic_fork and micro_mcts take the raw (unwrapped) model,
    which bypasses DDP's gradient all-reduce hook — so a fork step would
    either hang waiting for an all-reduce that never triggers, or silently
    skip gradient sync on that step. The guard at the top of
    train_chaoscontrol_for_budget raises NotImplementedError before any
    expensive setup so the incompatibility is caught at call time, not
    discovered mid-training as a deadlock on real hardware.
    """

    def _call(self, *, metabolic_gate: bool, metabolic_mode: str) -> None:
        seed = 42
        model = _build_model(seed)
        tokens = _build_tokens(seed)
        train_chaoscontrol_for_budget(
            model,
            train_tokens=tokens,
            train_starts=list(TRAIN_STARTS),
            seed=seed,
            rank=0,
            world_size=2,
            metabolic_gate=metabolic_gate,
            metabolic_mode=metabolic_mode,
            **TRAIN_KWARGS,
        )

    def test_metabolic_fork_under_ddp_raises(self) -> None:
        """world_size > 1 + metabolic_gate + mode='fork' must raise fast."""
        with self.assertRaises(NotImplementedError) as ctx:
            self._call(metabolic_gate=True, metabolic_mode="fork")
        msg = str(ctx.exception)
        self.assertIn("metabolic_gate", msg)
        self.assertIn("DDP", msg)
        self.assertIn("fork", msg)

    def test_metabolic_mcts_under_ddp_raises(self) -> None:
        """world_size > 1 + metabolic_gate + mode='mcts' must raise fast."""
        with self.assertRaises(NotImplementedError) as ctx:
            self._call(metabolic_gate=True, metabolic_mode="mcts")
        msg = str(ctx.exception)
        self.assertIn("metabolic_gate", msg)
        self.assertIn("DDP", msg)
        self.assertIn("mcts", msg)

    def test_metabolic_gate_false_under_ddp_does_not_hit_guard(self) -> None:
        """Sanity: the metabolic guard fires only when metabolic_gate=True.

        With gate=False + world_size=2, our guard is skipped and the code
        proceeds to the next guard (DDP init check) which raises RuntimeError
        about torch.distributed not being initialized. This confirms the
        metabolic guard is narrowly scoped to the bad combo and not
        over-firing on metabolic_mode alone.
        """
        with self.assertRaises(RuntimeError) as ctx:
            self._call(metabolic_gate=False, metabolic_mode="fork")
        # Must be the torch.distributed-not-initialized error, not our guard.
        self.assertIn("torch.distributed", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, NotImplementedError)

    def test_single_device_with_metabolic_fork_does_not_hit_guard(self) -> None:
        """Sanity: the guard fires only when ddp_active=True.

        With world_size=1 + metabolic_gate=True + mode='fork', the guard
        must NOT fire — single-device metabolic fork is the legal path.
        The training call may fail later for other reasons (e.g. the tiny
        test model doesn't support the fork path cleanly), but it must not
        raise our specific NotImplementedError.
        """
        seed = 42
        model = _build_model(seed)
        tokens = _build_tokens(seed)
        # Use a very short budget so we don't actually complete training.
        kwargs = {**TRAIN_KWARGS, "budget_seconds": 0.05}
        try:
            train_chaoscontrol_for_budget(
                model,
                train_tokens=tokens,
                train_starts=list(TRAIN_STARTS),
                seed=seed,
                rank=0,
                world_size=1,
                metabolic_gate=True,
                metabolic_mode="fork",
                **kwargs,
            )
        except NotImplementedError as exc:
            self.fail(
                f"Guard incorrectly fired on single-device metabolic_fork: {exc}"
            )
        except Exception:
            # Any other exception is fine — the guard didn't over-fire.
            pass


class TestDDPAuxModuleGuard(unittest.TestCase):
    """Guard: only the main model is DDP-wrapped, so trainable auxiliary
    modules (structured_proj, tokenizer) are incompatible with DDP.

    Under DDP, only parameters inside the wrapped module get gradient
    all-reduce hooks. structured_proj is created outside the wrap and
    tokenizer parameters are added to the optimizer without being wrapped
    — both would receive rank-local gradients and drift across ranks
    silently. The Exp 18 bare-SSM path has both disabled, but a future
    caller with a trainable tokenizer or structured generation under DDP
    would hit this bug. The guard raises NotImplementedError at function
    entry so the incompatibility is caught at call time, not discovered
    as a silent quality drop.
    """

    def _call_with_aux(
        self,
        *,
        generation_mode: str = "noise",
        tokenizer: object = None,
        world_size: int = 2,
        budget_seconds: float = 2.0,
    ) -> None:
        seed = 42
        model = _build_model(seed)
        tokens = _build_tokens(seed)
        kwargs = {**TRAIN_KWARGS, "budget_seconds": budget_seconds}
        train_chaoscontrol_for_budget(
            model,
            train_tokens=tokens,
            train_starts=list(TRAIN_STARTS),
            seed=seed,
            rank=0,
            world_size=world_size,
            generation_mode=generation_mode,
            tokenizer=tokenizer,
            **kwargs,
        )

    def test_ddp_with_structured_generation_raises(self) -> None:
        """world_size > 1 + generation_mode='structured' must raise fast."""
        with self.assertRaises(NotImplementedError) as ctx:
            self._call_with_aux(generation_mode="structured")
        msg = str(ctx.exception)
        self.assertIn("structured", msg)
        self.assertIn("DDP", msg)

    def test_ddp_with_trainable_tokenizer_raises(self) -> None:
        """world_size > 1 + trainable tokenizer must raise fast.

        A minimal stand-in with .parameters() is enough to trip the
        "tokenizer is not None" check. The guard does not introspect the
        tokenizer's structure — it checks the caller's intent signal.
        """
        fake_tokenizer = torch.nn.Linear(4, 4)
        with self.assertRaises(NotImplementedError) as ctx:
            self._call_with_aux(tokenizer=fake_tokenizer)
        msg = str(ctx.exception)
        self.assertIn("tokenizer", msg)
        self.assertIn("DDP", msg)

    def test_ddp_with_both_aux_modules_names_both_in_error(self) -> None:
        """Both aux modules at once — error message cites both causes."""
        fake_tokenizer = torch.nn.Linear(4, 4)
        with self.assertRaises(NotImplementedError) as ctx:
            self._call_with_aux(
                generation_mode="structured", tokenizer=fake_tokenizer
            )
        msg = str(ctx.exception)
        self.assertIn("structured", msg)
        self.assertIn("tokenizer", msg)

    def test_single_device_with_structured_generation_does_not_hit_guard(self) -> None:
        """Sanity: the guard is DDP-specific.

        generation_mode='structured' at world_size=1 must not trigger this
        guard. The training call may fail later for other reasons (e.g.
        structured projection setup on the toy model), but it must not
        raise our *specific* DDP+structured NotImplementedError.
        """
        try:
            self._call_with_aux(
                generation_mode="structured",
                world_size=1,
                budget_seconds=0.05,
            )
        except NotImplementedError as exc:
            msg = str(exc)
            if "DDP" in msg and "structured" in msg:
                self.fail(f"Guard incorrectly fired at world_size=1: {exc}")
        except Exception:
            # Any non-guard exception is fine — the guard didn't over-fire.
            pass

    def test_ddp_without_aux_modules_does_not_hit_guard(self) -> None:
        """Sanity: bare SSM at world_size=2 must not hit this guard.

        With generation_mode='noise' and tokenizer=None (both Exp 18
        defaults), this guard is skipped and the next guard fires — the
        existing RuntimeError about torch.distributed not being
        initialized.
        """
        with self.assertRaises(RuntimeError) as ctx:
            self._call_with_aux()
        # Must be the torch.distributed init error, not our guard's NotImplementedError.
        self.assertIn("torch.distributed", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, NotImplementedError)


if __name__ == "__main__":
    unittest.main()
