"""Regression tests for ``experiments/18_throughput_levers/_harness.py``.

Targeted at the ``runner_script`` override threaded through
``build_launch_cmd`` and ``run_parallel_ddp_matrix``. The override was
added on 2026-04-16 so Test 5b (and later tests) can route through
``runner_exp18_ssm.py`` while pre-Test-5b callers continue to hit the
frozen ``runner_exp18.py`` unchanged.

The first pre-commit review of Test 5b caught a bug here: the DDP
branch of ``build_launch_cmd`` ignored the new kwarg and hardcoded
``RUNNER_SCRIPT``, which would have silently routed Test 5b back to
the frozen runner and OOMed at bs=1024. This file locks in the
correct behavior so no future edit can re-introduce that class of
silent regression.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "18_throughput_levers"))

from _harness import RUNNER_SCRIPT, build_launch_cmd  # noqa: E402


def _call_site_kwargs(override: Path | None = None, num_gpus: int = 1) -> list[str]:
    """Minimal valid call-site shape; the rest of the kwargs are
    positional-only in the sense that they don't interact with the
    runner-selection decision this test locks in.
    """
    return build_launch_cmd(
        num_gpus=num_gpus,
        cfg_path=Path("/tmp/cfg.yaml"),
        data_path="/tmp/data",
        sp_model_path="/tmp/sp.model",
        budget=600.0,
        out_path=Path("/tmp/out.json"),
        rdzv_port=12345,
        runner_script=override,
    )


class TestRunnerScriptOverride:
    """Every pre-Test-5b caller passes ``runner_script=None`` implicitly
    and MUST continue to hit the frozen runner. Test 5b passes an
    explicit override and MUST land on that runner in both launch
    branches (direct-python and torchrun).
    """

    def test_single_gpu_default_hits_frozen_runner(self) -> None:
        cmd = _call_site_kwargs(override=None, num_gpus=1)
        assert str(RUNNER_SCRIPT) in cmd, (
            "default (no override) single-GPU launch must use the frozen runner"
        )
        assert "runner_exp18_ssm.py" not in " ".join(cmd), (
            "default launch must NOT pick up the ssm runner"
        )

    def test_ddp_default_hits_frozen_runner(self) -> None:
        # The ship-blocker we fixed lived in this branch — the DDP
        # argv was hardcoding RUNNER_SCRIPT regardless of the kwarg.
        # This test locks in the fix.
        cmd = _call_site_kwargs(override=None, num_gpus=2)
        assert str(RUNNER_SCRIPT) in cmd, (
            "default (no override) DDP launch must use the frozen runner"
        )
        assert "runner_exp18_ssm.py" not in " ".join(cmd)

    def test_single_gpu_override_routes_to_ssm_runner(self) -> None:
        override = Path("/fake/runner_exp18_ssm.py")
        cmd = _call_site_kwargs(override=override, num_gpus=1)
        assert str(override) in cmd, (
            "single-GPU launch must honor the runner_script override"
        )
        assert str(RUNNER_SCRIPT) not in cmd, (
            "frozen runner must NOT appear when override is set"
        )

    def test_ddp_override_routes_to_ssm_runner(self) -> None:
        # The exact failure mode the reviewer caught — this is the
        # path Test 5b hits on every DDP run.
        override = Path("/fake/runner_exp18_ssm.py")
        cmd = _call_site_kwargs(override=override, num_gpus=2)
        assert str(override) in cmd, (
            "DDP launch must honor the runner_script override — "
            "this is the ship-blocker the reviewer caught pre-commit"
        )
        assert str(RUNNER_SCRIPT) not in cmd

    def test_ddp_override_emits_torchrun_argv_shape(self) -> None:
        # Belt-and-suspenders: verify the DDP cmd structure around the
        # runner slot is as expected, so an edit that reorders argv
        # doesn't pass the runner-identity checks above while corrupting
        # the launch.
        override = Path("/fake/runner_exp18_ssm.py")
        cmd = _call_site_kwargs(override=override, num_gpus=2)
        assert cmd[1] == "-m", f"expected 'python -m torch.distributed.run ...', got {cmd[:3]}"
        assert cmd[2] == "torch.distributed.run"
        assert "--nproc_per_node=2" in cmd
        # Runner path appears immediately after the rdzv-id argument.
        rdzv_id_idx = next(
            i for i, a in enumerate(cmd) if a.startswith("--rdzv-id=")
        )
        assert cmd[rdzv_id_idx + 1] == str(override), (
            f"runner script should immediately follow --rdzv-id; "
            f"got cmd[{rdzv_id_idx+1}] = {cmd[rdzv_id_idx+1]!r}"
        )
