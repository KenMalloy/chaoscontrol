"""Shared launch utilities for Exp 23."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_torchrun_cmd(
    *,
    runner_path: Path,
    config_path: Path,
    data_path: str,
    sp_model_path: str,
    output_json: Path,
    world_size: int,
    rdzv_port: int | None = None,
    output_ckpt: Path | None = None,
    budget_seconds: float | None = None,
) -> list[str]:
    if rdzv_port is None:
        rdzv_port = pick_free_port()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={int(world_size)}",
        f"--rdzv-endpoint=localhost:{rdzv_port}",
        "--rdzv-backend=c10d",
        f"--rdzv-id=cc_exp23_{rdzv_port}",
        str(runner_path),
        "--config",
        str(config_path),
        "--data-path",
        data_path,
        "--sp-model-path",
        sp_model_path,
        "--output-json",
        str(output_json),
    ]
    if output_ckpt is not None:
        cmd += ["--output-ckpt", str(output_ckpt)]
    if budget_seconds is not None:
        cmd += ["--budget", str(float(budget_seconds))]
    return cmd


def write_entry_config(entry: dict[str, Any], config_dir: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / f"{entry['name']}.yaml"
    cfg_path.write_text(yaml.safe_dump(entry, sort_keys=True))
    return cfg_path


def summarize_result_dir(results_dir: Path) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if not results_dir.exists():
        return {"ranked": ranked, "errors": errors}
    for path in sorted(results_dir.glob("*.json")):
        if path.name in {"matrix.json", "summary.json"}:
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        cfg = data.get("config") or {}
        name = str(cfg.get("name", path.stem))
        if "error" in data:
            errors.append({"name": name, "error": str(data["error"])})
            continue
        train = data.get("train") or {}
        artifact = data.get("artifact") or {}
        exp24 = data.get("exp24") or {}
        ranked.append({
            "name": name,
            "tokens_per_sec": float(train.get("aggregate_tokens_per_sec", 0.0)),
            "per_gpu_tokens_per_sec": float(train.get("per_gpu_tokens_per_sec", 0.0)),
            "steps": int(train.get("steps", 0)),
            "final_loss": float(train.get("final_loss", float("nan"))),
            "peak_vram_mb": float(train.get("peak_vram_mb", 0.0)),
            "artifact_impact": artifact.get("artifact_impact"),
            "submit_valid": artifact.get("submit_valid"),
            "exp24_phase": exp24.get("phase"),
            "exp24_mechanism": exp24.get("mechanism"),
            "path": str(path),
        })
    ranked.sort(key=lambda row: row["tokens_per_sec"], reverse=True)
    return {"ranked": ranked, "errors": errors}


def run_matrix_entries(
    *,
    entries: list[dict[str, Any]],
    runner_path: Path,
    data_path: str,
    sp_model_paths: dict[int, str],
    results_dir: Path,
    world_size: int,
    limit: int | None = None,
    dry_run: bool = False,
    skip_existing: bool = True,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    config_dir = results_dir / "configs"
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    selected = entries[:limit] if limit is not None else entries
    commands: list[list[str]] = []
    for entry in selected:
        name = str(entry["name"])
        out_json = results_dir / f"{name}.json"
        if skip_existing and out_json.exists():
            continue
        vocab = int(entry["vocab_size"])
        if vocab not in sp_model_paths:
            raise KeyError(f"missing SentencePiece model path for vocab {vocab}")
        cfg_path = write_entry_config(entry, config_dir)
        output_ckpt = None
        if checkpoint_dir is not None:
            output_ckpt = checkpoint_dir / f"{name}.pt"
        cmd = build_torchrun_cmd(
            runner_path=runner_path,
            config_path=cfg_path,
            data_path=data_path,
            sp_model_path=sp_model_paths[vocab],
            output_json=out_json,
            output_ckpt=output_ckpt,
            world_size=world_size,
            budget_seconds=float(entry.get("budget_seconds", 90.0)),
        )
        commands.append(cmd)
        if dry_run:
            continue
        log_path = logs_dir / f"{name}.log"
        with tempfile.TemporaryDirectory(prefix="exp23_") as _tmpdir:
            with log_path.open("w") as log:
                proc = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        if proc.returncode != 0:
            payload = {
                "config": entry,
                "error": f"returncode={proc.returncode}",
                "log_path": str(log_path),
            }
            out_json.write_text(json.dumps(payload, indent=2, default=str))
    summary = summarize_result_dir(results_dir)
    summary["commands"] = commands if dry_run else []
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
