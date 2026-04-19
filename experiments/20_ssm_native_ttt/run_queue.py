#!/usr/bin/env python3
"""Run Exp20 config JSONs across a fixed list of local GPUs.

Each config is a complete single-GPU invocation of ``scripts/run_exp20_eval.py``.
The queue sets ``CUDA_VISIBLE_DEVICES`` to one physical GPU per subprocess and
``LOCAL_RANK=0`` so the driver pins itself correctly.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _name(path: Path, cfg: dict[str, Any]) -> str:
    return str(cfg.get("name") or path.stem)


def _config_paths(config_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in config_dir.glob("*.json")
        if path.name != "manifest.json"
    )


def _summary_exists(cfg: dict[str, Any]) -> bool:
    summary_path = cfg.get("summary_path")
    return bool(summary_path) and Path(str(summary_path)).exists()


def _pending_configs(config_dir: Path, *, resume: bool) -> tuple[list[Path], list[tuple[str, Path]]]:
    pending: list[Path] = []
    skipped: list[tuple[str, Path]] = []
    for path in _config_paths(config_dir):
        cfg = _load(path)
        name = _name(path, cfg)
        if resume and _summary_exists(cfg):
            skipped.append((name, path))
        else:
            pending.append(path)
    return pending, skipped


def _run_one(
    *,
    config_path: Path,
    gpu_queue: queue.Queue[str],
    runner: Path,
    python: str,
    log_dir: Path,
) -> tuple[str, int, str]:
    cfg = _load(config_path)
    name = _name(config_path, cfg)
    gpu = gpu_queue.get()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["LOCAL_RANK"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [python, str(runner), "--config", str(config_path)]
        with log_path.open("w") as log:
            log.write(f"$ {' '.join(cmd)}\n")
            log.write(f"CUDA_VISIBLE_DEVICES={gpu} LOCAL_RANK=0\n\n")
            log.flush()
            result = subprocess.run(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return name, result.returncode, str(log_path)
    finally:
        gpu_queue.put(gpu)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--runner", type=Path, default=Path("scripts/run_exp20_eval.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--log-dir", type=Path, default=Path("experiments/20_ssm_native_ttt/logs"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pending, skipped = _pending_configs(args.config_dir, resume=args.resume)

    for name, _path in skipped:
        print(f"SKIP {name}")

    if args.dry_run:
        for idx, path in enumerate(pending):
            cfg = _load(path)
            print(f"RUN {_name(path, cfg)} gpu={args.gpus[idx % len(args.gpus)]} config={path}")
        return

    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in args.gpus:
        gpu_queue.put(str(gpu))

    failures: list[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(
                _run_one,
                config_path=path,
                gpu_queue=gpu_queue,
                runner=args.runner,
                python=args.python,
                log_dir=args.log_dir,
            )
            for path in pending
        ]
        for future in as_completed(futures):
            name, returncode, log_path = future.result()
            if returncode == 0:
                print(f"PASS {name} log={log_path}", flush=True)
            else:
                print(f"FAIL {name} rc={returncode} log={log_path}", flush=True)
                failures.append((name, returncode, log_path))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
