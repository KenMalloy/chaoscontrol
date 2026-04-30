"""Unit tests for runner_persistent_ddp helpers.

The warmup-restore helper must leave the model byte-equivalent to its
pre-warmup state so that reported bpb is free of warmup-phase weight
updates — matches the Parameter Golf submission harness contract
(``baselines/parameter_golf/train_gpt.py`` lines 935-961): warmup primes
compile/kernels/allocator, restore loads initial weights back, timer
starts from the restored state.

The config-match helper guards the idempotent-skip path against silent
cross-matrix contamination — filename-only skip would reuse stale
results when the matrix is edited (LR change etc.) and the output dir
is reused.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "19_prereqs"))
sys.path.insert(0, str(REPO / "experiments" / "18_throughput_levers"))
sys.path.insert(0, str(REPO / "experiments" / "17_local_attn_sidecar"))
sys.path.insert(0, str(REPO / "src"))

from chaoscontrol.model import CareStudentLM  # noqa: E402
from runner_persistent_ddp import _require_config_match, _warmup_and_restore  # noqa: E402


def _tiny_model() -> CareStudentLM:
    torch.manual_seed(42)
    return CareStudentLM(
        vocab_size=32, dim=8, num_layers=1, ff_mult=2, a_mode="diag",
    )


class TestWarmupAndRestore:
    def test_model_weights_roundtrip_exactly(self) -> None:
        model = _tiny_model()
        initial_state = {
            name: tensor.clone() for name, tensor in model.state_dict().items()
        }

        def warmup_call() -> None:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p))

        built_count = [0]

        def build_optimizer() -> torch.optim.Optimizer:
            built_count[0] += 1
            return torch.optim.AdamW(model.parameters(), lr=1e-3)

        new_optimizer = _warmup_and_restore(
            model=model,
            warmup_call_fn=warmup_call,
            build_optimizer_fn=build_optimizer,
            device=torch.device("cpu"),
            ddp_active=False,
        )

        restored_state = model.state_dict()
        for name, initial_tensor in initial_state.items():
            assert torch.equal(restored_state[name], initial_tensor), (
                f"param {name!r} not restored to initial state"
            )
        assert built_count[0] == 1
        assert isinstance(new_optimizer, torch.optim.AdamW)

    def test_warmup_runs_before_optimizer_rebuild(self) -> None:
        model = _tiny_model()
        events: list[str] = []

        def warmup_call() -> None:
            events.append("warmup")
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1.0)

        def build_optimizer() -> torch.optim.Optimizer:
            events.append("build_optimizer")
            return torch.optim.AdamW(model.parameters(), lr=1e-3)

        _warmup_and_restore(
            model=model,
            warmup_call_fn=warmup_call,
            build_optimizer_fn=build_optimizer,
            device=torch.device("cpu"),
            ddp_active=False,
        )
        assert events == ["warmup", "build_optimizer"]

    def test_grads_cleared_after_restore(self) -> None:
        """Matches PG reference at baselines/parameter_golf/train_gpt.py:955.

        Warmup leaves per-param grads populated (optimizer.step consumes them
        but never zeros them; zero_grad fires at the START of the next iter).
        If the helper returns without clearing, the real timed pass starts
        with a full set of grad tensors allocated — perturbing first-step
        timing and peak-memory accounting relative to a true submission run.
        """
        model = _tiny_model()

        def warmup_call() -> None:
            with torch.no_grad():
                for p in model.parameters():
                    p.grad = torch.ones_like(p)

        def build_optimizer() -> torch.optim.Optimizer:
            return torch.optim.AdamW(model.parameters(), lr=1e-3)

        _warmup_and_restore(
            model=model,
            warmup_call_fn=warmup_call,
            build_optimizer_fn=build_optimizer,
            device=torch.device("cpu"),
            ddp_active=False,
        )

        for name, p in model.named_parameters():
            assert p.grad is None, (
                f"param {name!r} still has grads after _warmup_and_restore; "
                "timer-start state diverges from PG reference harness"
            )


class TestRequireConfigMatch:
    """Config-sensitive idempotent skip. Critical scientific-validity guard.

    A matrix rerun that reuses the output dir after editing (e.g., LR,
    seq_len, optimizer) must not silently count the stale outputs as
    "resumed." Filename-only skip was the pre-2026-04-17 default; this
    check raises instead.
    """

    @staticmethod
    def _stored(
        tmp_path: Path, name: str, seed: int, payload: dict,
    ) -> Path:
        path = tmp_path / f"{name}_s{seed}.json"
        path.write_text(json.dumps(payload))
        return path

    def test_exact_match_does_not_raise(self, tmp_path: Path) -> None:
        """Byte-identical configs: the whole point of idempotent skip."""
        entry = {"name": "bf16", "seed": 1337, "base_lr": 0.064}
        path = self._stored(tmp_path, "bf16", 1337, {
            "config": entry,
            "train": {"steps": 1000, "final_loss": 4.2},
            "eval": {"bpb": 1.49},
        })
        _require_config_match(path, entry)  # must not raise

    def test_lr_change_raises(self, tmp_path: Path) -> None:
        """Canonical silent-contamination scenario."""
        stored_entry = {"name": "bf16", "seed": 1337, "base_lr": 0.064}
        path = self._stored(tmp_path, "bf16", 1337, {
            "config": stored_entry,
            "train": {"steps": 1000, "final_loss": 4.2},
            "eval": {"bpb": 1.49},
        })
        requested = {"name": "bf16", "seed": 1337, "base_lr": 0.128}
        with pytest.raises(RuntimeError, match="config mismatch"):
            _require_config_match(path, requested)

    def test_added_or_removed_field_raises(self, tmp_path: Path) -> None:
        """Extra / missing fields count as mismatch — strict equality."""
        stored = {"name": "bf16", "seed": 1337}
        path = self._stored(tmp_path, "bf16", 1337, {
            "config": stored,
            "eval": {"bpb": 1.49},
        })
        with pytest.raises(RuntimeError, match="config mismatch"):
            _require_config_match(
                path, {**stored, "activation_checkpoint": True},
            )

    def test_mismatch_error_includes_both_hashes(self, tmp_path: Path) -> None:
        """Operator must be able to confirm which stored run wrote the
        file and which matrix entry asked for the skip.

        The two 8-char hashes are the most compact invariant that
        survives config edits — grep-able, diff-able, uniquely
        identifies the run if combined with the timestamped log.
        """
        stored = {"name": "bf16", "seed": 1337, "base_lr": 0.064}
        path = self._stored(tmp_path, "bf16", 1337, {"config": stored})
        requested = {"name": "bf16", "seed": 1337, "base_lr": 0.128}
        with pytest.raises(RuntimeError) as ctx:
            _require_config_match(path, requested)
        msg = str(ctx.value)
        assert "hash=" in msg
        # Both hashes appear, and they differ (since configs differ).
        assert msg.count("hash=") == 2

    def test_malformed_stored_json_raises(self, tmp_path: Path) -> None:
        """Unreadable stored output is treated as a hard failure.

        Silently running the entry would overwrite the broken file, but
        it would also mask the fact that something went wrong before
        the rerun — the operator should be able to tell the difference
        between "rerunning a legitimate stale entry" and "my output dir
        is corrupted." Fail loudly, let the human decide.
        """
        bad_path = tmp_path / "bf16_s1337.json"
        bad_path.write_text("{this is not valid json")
        with pytest.raises(RuntimeError, match="could not read"):
            _require_config_match(
                bad_path, {"name": "bf16", "seed": 1337},
            )

    def test_stored_json_missing_config_field_raises(
        self, tmp_path: Path,
    ) -> None:
        """A JSON without a ``config`` key cannot be confirmed to match.

        Not something the runner would ever write, but defends against
        a hand-written or externally-generated file landing in the
        output dir.
        """
        path = self._stored(tmp_path, "bf16", 1337, {
            "train": {"steps": 1000},
            "eval": {"bpb": 1.49},
        })
        with pytest.raises(RuntimeError, match="config mismatch"):
            _require_config_match(path, {"name": "bf16", "seed": 1337})

    def test_error_marker_with_matching_config_does_not_raise(
        self, tmp_path: Path,
    ) -> None:
        """A prior error marker for the same config is a legitimate skip.

        If the user is OK re-reading the same error, they can; if they
        want to retry, they delete the marker. Either way, *silently
        resuming* with a different config is the bug, not "was this a
        success or error?"
        """
        entry = {"name": "fp8", "seed": 1337, "precision": "fp8"}
        path = self._stored(tmp_path, "fp8", 1337, {
            "config": entry,
            "error": "skipped: transformer_engine unavailable on pod",
        })
        _require_config_match(path, entry)  # must not raise
