"""Shared launch/validation helpers for Exp 18 test orchestrators.

Purpose of this module is to avoid repeating the same subprocess cmd
construction and path preflighting across ``run_exp18_test{3,4,5,6,9}.py``.
The existing ``run_exp18.py`` (Test 2) and ``run_exp18_test7.py`` (Test 7)
predate this helper and keep their own copies on purpose — refactoring
them is deferred until after Tests 3-9 have been run on real hardware.

Two orthogonal launch patterns covered here:

  - **Parallel single-GPU:** multiple conditions run concurrently with
    one GPU per child process (Test 2's pattern, and Test 3's). Suitable
    for throughput or bpb comparisons that don't need gradient sync.

  - **Sequential DDP:** one condition at a time under torchrun with
    nproc_per_node=N (Test 7's pattern, and Tests 4/5/6/9's). Each run
    seizes all N GPUs; the next run can only start after teardown.

Both patterns share a ``build_launch_cmd`` helper that picks ``torchrun``
vs direct python invocation based on ``num_gpus``. The direct-python path
degrades ``runner_exp18.py`` to its world_size=1 code branch (bit-identical
to the pre-DDP training loop) when ``WORLD_SIZE`` is unset.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
RUNNER_SCRIPT = EXPERIMENT / "runner_exp18.py"


def pick_free_port() -> int:
    """Ask the OS for an ephemeral TCP port and release it immediately.

    There's a technical race between close() and the subsequent bind, but
    in practice this is the standard idiom for multi-DDP-group launches
    and is reliable enough on a dedicated pod where nothing else competes
    for the ephemeral port range.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def build_launch_cmd(
    *,
    num_gpus: int,
    cfg_path: Path,
    data_path: str,
    sp_model_path: str,
    budget: float,
    out_path: Path,
    rdzv_port: int | None = None,
) -> list[str]:
    """Construct the subprocess argv for one runner_exp18.py invocation.

    num_gpus <= 1:
        Direct python invocation. runner_exp18 reads the absence of
        WORLD_SIZE and takes its single-device branch. Used for both
        local CPU smokes and 1-GPU pods.

    num_gpus > 1:
        ``python -m torch.distributed.run --nproc_per_node=N
        --rdzv-endpoint=localhost:<port>``  against runner_exp18.py.
        torchrun spawns N ranks on this host, each binding to its own
        LOCAL_RANK GPU. runner_exp18 wraps the model in DDP internally.

        ``rdzv_port`` is required when ``num_gpus > 1`` and two or more
        DDP groups share a host — each concurrent torchrun needs a unique
        rendezvous port or the second one fails to bind. Pass a port from
        ``pick_free_port()`` per slot.

    Rationale for ``python -m torch.distributed.run`` over the ``torchrun``
    CLI: the module form is version-resilient and doesn't depend on
    whatever PATH the pod happens to have torchrun on.
    """
    common_tail = [
        str(RUNNER_SCRIPT),
        "--config", str(cfg_path),
        "--data-path", data_path,
        "--sp-model-path", sp_model_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
    ]
    if num_gpus <= 1:
        return [sys.executable, "-u"] + common_tail
    if rdzv_port is None:
        rdzv_port = pick_free_port()
    return [
        sys.executable,
        "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        f"--rdzv-endpoint=localhost:{rdzv_port}",
        f"--rdzv-backend=c10d",
        f"--rdzv-id=cc_exp18_{rdzv_port}",
        str(RUNNER_SCRIPT),
        "--config", str(cfg_path),
        "--data-path", data_path,
        "--sp-model-path", sp_model_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
    ]


def build_env_with_gpu_mask(gpu_ids: list[int]) -> dict[str, str]:
    """Clone os.environ and pin CUDA_VISIBLE_DEVICES to the given GPU list.

    Used when launching multiple concurrent DDP groups on the same host:
    each slot sees only its own 2 GPUs via this mask, so torchrun's
    LOCAL_RANK maps to the correct physical device within the mask.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    return env


def validate_data_paths(data_path: str, sp_model_path: str) -> None:
    """Fail fast on missing data or tokenizer files before launching."""
    data_dir = Path(data_path)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir {data_dir} does not exist")
    model_file = Path(sp_model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"SP model file {model_file} does not exist")


def read_result_json(out_path: Path) -> dict[str, Any]:
    """Parse a runner_exp18 result JSON. Returns {} if the file is missing."""
    if not out_path.exists():
        return {}
    return json.loads(out_path.read_text())


def _cleanup_active(active: list) -> None:
    for entry in active:
        proc, cfg_path = entry[0], entry[1]
        if proc.poll() is None:
            proc.terminate()
        cfg_path.unlink(missing_ok=True)
    for entry in active:
        proc = entry[0]
        if proc.poll() is None:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        if len(entry) > 5:
            entry[5].close()


def run_parallel_ddp_matrix(
    *,
    conditions: dict[str, dict[str, Any]],
    seeds: list[int],
    ws_per_slot: int,
    num_slots: int,
    data_path: str,
    sp_model_path: str,
    budget: float,
    results_dir: Path,
    timeout_multiplier: float = 2.5,
) -> None:
    """Launch ``conditions x seeds`` under parallel DDP groups.

    Each slot owns a disjoint set of ``ws_per_slot`` GPUs pinned via
    ``CUDA_VISIBLE_DEVICES`` and uses its own torchrun rendezvous port.
    ``num_slots * ws_per_slot`` must not exceed the pod's GPU count;
    the caller is responsible for validating this.

    Idempotent: any ``{condition}_s{seed}.json`` that already exists in
    ``results_dir`` is skipped, so re-launches after partial completion
    only run the missing seeds.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    run_timeout = max(budget * timeout_multiplier, 60.0)

    queue: list[tuple[str, int, Path]] = []
    for condition_name, cfg in conditions.items():
        for seed in seeds:
            out_path = results_dir / f"{condition_name}_s{seed}.json"
            if out_path.exists():
                continue
            seed_cfg = dict(cfg, seed=seed)
            tmp = Path(tempfile.mkstemp(prefix=f"{condition_name}_s{seed}_", suffix=".yaml")[1])
            tmp.write_text(yaml.safe_dump(seed_cfg, sort_keys=False))
            queue.append((condition_name, seed, tmp))

    active: list = []
    slot_cursor = 0

    while queue or active:
        while queue and len(active) < num_slots:
            condition_name, seed, cfg_path = queue.pop(0)
            out_path = results_dir / f"{condition_name}_s{seed}.json"
            log_path = results_dir / f"{condition_name}_s{seed}.log"
            log_fh = open(log_path, "w")
            slot_id = slot_cursor % num_slots
            slot_cursor += 1
            gpu_ids = [slot_id * ws_per_slot + i for i in range(ws_per_slot)]
            env = build_env_with_gpu_mask(gpu_ids)
            rdzv = pick_free_port()
            cmd = build_launch_cmd(
                num_gpus=ws_per_slot,
                cfg_path=cfg_path,
                data_path=data_path,
                sp_model_path=sp_model_path,
                budget=budget,
                out_path=out_path,
                rdzv_port=rdzv,
            )
            print(
                f"Launching {condition_name} seed={seed} gpus={gpu_ids} rdzv=:{rdzv}",
                flush=True,
            )
            proc = subprocess.Popen(
                cmd, env=env, text=True, stdout=log_fh, stderr=subprocess.STDOUT,
            )
            active.append((proc, cfg_path, condition_name, seed, time.monotonic(), log_fh))

        next_active: list = []
        for i, (proc, cfg_path, condition_name, seed, t0, log_fh) in enumerate(active):
            ret = proc.poll()
            elapsed = time.monotonic() - t0
            if ret is None and elapsed < run_timeout:
                next_active.append((proc, cfg_path, condition_name, seed, t0, log_fh))
                continue
            log_fh.close()
            if ret is None:
                print(
                    f"TIMEOUT: {condition_name} seed={seed} after {elapsed:.0f}s",
                    flush=True,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                cfg_path.unlink(missing_ok=True)
                _cleanup_active(next_active + list(active[i + 1:]))
                raise RuntimeError(
                    f"{condition_name} seed={seed} TIMEOUT after {elapsed:.0f}s"
                )
            cfg_path.unlink(missing_ok=True)
            if ret != 0:
                log_path = results_dir / f"{condition_name}_s{seed}.log"
                tail = ""
                if log_path.exists():
                    lines = log_path.read_text().splitlines()
                    tail = "\n".join(lines[-20:])
                _cleanup_active(next_active + list(active[i + 1:]))
                raise RuntimeError(
                    f"{condition_name} seed={seed} failed with exit code {ret}\n"
                    f"--- last 20 lines of {log_path} ---\n{tail}"
                )
        active = next_active
        if active:
            time.sleep(2.0)
