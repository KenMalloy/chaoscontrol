#!/usr/bin/env python3
"""Phase 0 throughput and stability harness for Experiment 18."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "15_chaospiece"))
sys.path.insert(0, str(EXPERIMENT))

from chaoscontrol.data import batch_from_starts, maybe_autocast, maybe_sync_cuda, resolve_device, resolve_param_dtype  # noqa: E402
from runner_exp15 import build_model, load_sp_data  # noqa: E402
from runner_exp18 import (  # noqa: E402
    DEFAULT_BASE_LR,
    DEFAULT_RESCORE_BUDGET_S,
    DEFAULT_SELECTION_BUDGET_S,
    DEFAULT_SWEEP_BUDGET_S,
    DEFAULT_TOTAL_BUDGET_S,
    assert_runtime_compatibility,
    build_child_env,
    build_coverage_plan,
    choose_tokenizer,
    make_smoke_summary,
    make_smoke_tokens,
    validate_gpu_concurrency,
)


VOCAB_VARIANTS = {
    8192: ("fineweb10B_sp8192", "fineweb_8192_bpe.model", 1.9673904486794673),
    16384: ("fineweb10B_sp16384", "fineweb_16384_bpe.model", 1.9594690598775102),
}


def resolve_variant_paths(data_root: str, vocab_size: int) -> tuple[str, str, float]:
    suffix, model_name, prior_mean_bpb = VOCAB_VARIANTS[vocab_size]
    return (
        str(Path(data_root) / "datasets" / suffix),
        str(Path(data_root) / "tokenizers" / model_name),
        prior_mean_bpb,
    )


def _throughput_steps(
    model: torch.nn.Module,
    *,
    train_tokens: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    param_dtype: torch.dtype,
    steps: int,
    base_lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    ) -> dict[str, float]:
    starts = list(range(0, max(int(train_tokens.numel()) - seq_len - 1, 1), seq_len))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    autocast_dtype = next(model.parameters()).dtype if device.type == "cuda" else torch.float32

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    maybe_sync_cuda(device)
    begin = time.perf_counter()
    actual_steps = 0
    total_tokens = 0
    cursor = 0
    for _ in range(steps):
        if cursor >= len(starts):
            cursor = 0
        batch_starts = starts[cursor : cursor + batch_size]
        if not batch_starts:
            break
        cursor += len(batch_starts)
        inputs, targets = batch_from_starts(train_tokens, batch_starts, seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(device, autocast_dtype if device.type == "cuda" else param_dtype):
            out = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                out["logits"].reshape(-1, model.vocab_size),
                targets.reshape(-1),
            )
        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), grad_clip_norm)
        optimizer.step()
        total_tokens += int(targets.numel())
        actual_steps += 1
    maybe_sync_cuda(device)
    elapsed = time.perf_counter() - begin
    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    step_time_s = elapsed / max(actual_steps, 1)
    tokens_per_s = total_tokens / max(elapsed, 1e-9)
    return {
        "steps": float(actual_steps),
        "elapsed_s": float(elapsed),
        "step_time_s": float(step_time_s),
        "tokens_per_s": float(tokens_per_s),
        "peak_vram_gb": float(peak_vram_gb),
    }


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    if isinstance(exc, oom_type):
        return True
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _cleanup_cuda_state(device: torch.device, *objects: Any) -> None:
    for obj in objects:
        del obj
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _lr_screen(
    model_config: dict[str, Any],
    *,
    train_tokens: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    param_dtype: torch.dtype,
    steps: int,
    lr_candidates: dict[str, float],
) -> list[dict[str, Any]]:
    starts = list(range(0, max(int(train_tokens.numel()) - seq_len - 1, 1), seq_len))
    results: list[dict[str, Any]] = []
    for label, lr in lr_candidates.items():
        model = None
        optimizer = None
        losses: list[float] = []
        cursor = 0
        failed = False
        oom = False
        try:
            model = build_model(model_config, device, param_dtype)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            autocast_dtype = next(model.parameters()).dtype if device.type == "cuda" else torch.float32
            for _ in range(steps):
                if cursor >= len(starts):
                    cursor = 0
                batch_starts = starts[cursor : cursor + batch_size]
                if not batch_starts:
                    break
                cursor += len(batch_starts)
                inputs, targets = batch_from_starts(train_tokens, batch_starts, seq_len, device)
                optimizer.zero_grad(set_to_none=True)
                with maybe_autocast(device, autocast_dtype if device.type == "cuda" else param_dtype):
                    out = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        out["logits"].reshape(-1, model.vocab_size),
                        targets.reshape(-1),
                    )
                if not torch.isfinite(loss):
                    failed = True
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            oom = True
            failed = True
        finally:
            _cleanup_cuda_state(device, optimizer, model)
        if oom:
            results.append(
                {
                    "label": label,
                    "lr": float(lr),
                    "stable": False,
                    "failed": True,
                    "oom": True,
                    "loss_start": float(losses[0]) if losses else None,
                    "loss_end": float(losses[-1]) if losses else None,
                }
            )
            continue
        if not losses and not failed:
            failed = True
        decreasing = len(losses) >= 2 and losses[-1] <= losses[0] * 1.02
        stable = (not failed) and bool(losses) and decreasing
        results.append(
            {
                "label": label,
                "lr": float(lr),
                "stable": stable,
                "failed": failed,
                "oom": False,
                "loss_start": float(losses[0]) if losses else None,
                "loss_end": float(losses[-1]) if losses else None,
            }
        )
    return results


def benchmark_tokenizer(
    *,
    vocab_size: int,
    data_path: str | None,
    sp_model_path: str | None,
    device_name: str,
    smoke: bool,
    batch_sizes: list[int],
    throughput_steps: int,
    lr_steps: int,
    sweep_budget_s: float,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    param_dtype = resolve_param_dtype("bf16", device)
    runtime = assert_runtime_compatibility(
        smoke=smoke,
        device=device,
        require_sentencepiece=not smoke,
        sp_model_path=sp_model_path if not smoke else None,
    )
    if smoke:
        train_tokens, _ = make_smoke_tokens(vocab_size=64, total_tokens=1024)
        prior_mean_bpb = 1.9 if vocab_size == 16384 else 2.0
    else:
        if data_path is None:
            raise ValueError("data_path required for real benchmark runs")
        train_tokens, _, _ = load_sp_data(data_path, vocab_size)
        prior_mean_bpb = VOCAB_VARIANTS[vocab_size][2]

    model_config = {
        "model_type": "transformer" if smoke else "ssm",
        "vocab_size": 64 if smoke else vocab_size,
        "model_dim": 16 if smoke else 256,
        "num_layers": 1 if smoke else 4,
        "ff_mult": 2,
        "seq_len": 16 if smoke else 512,
        "stride": 8 if smoke else 256,
        "batch_size": batch_sizes[0],
        "base_lr": DEFAULT_BASE_LR,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
    }
    seq_len = model_config["seq_len"]
    candidates: list[dict[str, Any]] = []
    oom_batches: list[int] = []
    for batch_size in batch_sizes:
        cfg = dict(model_config, batch_size=batch_size)
        model = None
        try:
            model = build_model(cfg, device, param_dtype)
            throughput = _throughput_steps(
                model,
                train_tokens=train_tokens,
                seq_len=seq_len,
                batch_size=batch_size,
                device=device,
                param_dtype=param_dtype,
                steps=throughput_steps,
                base_lr=DEFAULT_BASE_LR,
                weight_decay=1e-2,
                grad_clip_norm=1.0,
            )
            coverage = build_coverage_plan(
                int(train_tokens.numel()),
                seq_len=seq_len,
                projected_windows=int((throughput["tokens_per_s"] * sweep_budget_s) // max(seq_len, 1)),
            )
            candidates.append(
                {
                    "batch_size": int(batch_size),
                    "step_time_s": throughput["step_time_s"],
                    "tokens_per_s": throughput["tokens_per_s"],
                    "peak_vram_gb": throughput["peak_vram_gb"],
                    "projected_unique_windows": coverage.planned_windows,
                    "projected_unique_targets": coverage.unique_targets,
                    "projected_coverage_frac": coverage.coverage_frac,
                    "low_coverage_regime": coverage.low_coverage_regime,
                }
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            oom_batches.append(int(batch_size))
            break
        finally:
            _cleanup_cuda_state(device, model)

    feasible = sorted(candidates, key=lambda row: row["batch_size"])
    if not feasible:
        raise RuntimeError(
            f"No feasible batch sizes for sp{vocab_size}; first tried batch {batch_sizes[0]} and hit OOM."
        )
    best_candidate = feasible[-1]
    scale = best_candidate["batch_size"] / 32.0
    lr_candidates = {
        "linear": DEFAULT_BASE_LR * scale,
        "sqrt": DEFAULT_BASE_LR * math.sqrt(scale),
        "fixed": DEFAULT_BASE_LR,
    }
    lr_screen = _lr_screen(
        model_config,
        train_tokens=train_tokens,
        seq_len=seq_len,
        batch_size=int(best_candidate["batch_size"]),
        device=device,
        param_dtype=param_dtype,
        steps=lr_steps,
        lr_candidates=lr_candidates,
    )
    stable_lrs = [row for row in lr_screen if row["stable"]]
    chosen_lr = stable_lrs[0]["lr"] if stable_lrs else DEFAULT_BASE_LR
    return {
        "tokenizer": f"sp{vocab_size}",
        "vocab_size": int(model_config["vocab_size"]),
        "data_path": data_path,
        "sp_model_path": sp_model_path,
        "prior_mean_bpb": prior_mean_bpb,
        "stable": bool(stable_lrs),
        "batch_candidates": feasible,
        "oom_batches": oom_batches,
        "selected_candidate": dict(best_candidate, selected_lr=chosen_lr, base_lr=DEFAULT_BASE_LR),
        "lr_screen": lr_screen,
        "model_config": model_config,
        "runtime": runtime,
    }


def _worker_main(args: argparse.Namespace) -> None:
    result = benchmark_tokenizer(
        vocab_size=args.vocab_size,
        data_path=args.data_path,
        sp_model_path=args.sp_model_path,
        device_name=args.device,
        smoke=args.smoke,
        batch_sizes=[int(x) for x in args.batch_sizes.split(",") if x],
        throughput_steps=args.throughput_steps,
        lr_steps=args.lr_steps,
        sweep_budget_s=args.sweep_budget,
    )
    Path(args.output_json).write_text(json.dumps(result, indent=2))


def _launch_worker(
    *,
    vocab_size: int,
    data_path: str | None,
    sp_model_path: str | None,
    device_id: int,
    output_json: str,
    batch_sizes: str,
    throughput_steps: int,
    lr_steps: int,
    sweep_budget: float,
    smoke: bool,
) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--vocab-size",
        str(vocab_size),
        "--device",
        "cuda" if not smoke else "cpu",
        "--batch-sizes",
        batch_sizes,
        "--throughput-steps",
        str(throughput_steps),
        "--lr-steps",
        str(lr_steps),
        "--sweep-budget",
        str(sweep_budget),
        "--output-json",
        output_json,
    ]
    if data_path is not None:
        cmd.extend(["--data-path", data_path])
    if sp_model_path is not None:
        cmd.extend(["--sp-model-path", sp_model_path])
    if smoke:
        cmd.append("--smoke")
    env = build_child_env(gpu_slot=device_id, smoke=smoke)
    return subprocess.Popen(cmd, env=env, text=True)


def orchestrate_phase0(
    *,
    data_root: str | None,
    num_gpus: int,
    batch_sizes: list[int],
    throughput_steps: int,
    lr_steps: int,
    output_json: str | None,
    smoke: bool,
) -> dict[str, Any]:
    if smoke:
        results = [
            benchmark_tokenizer(
                vocab_size=8192,
                data_path=None,
                sp_model_path=None,
                device_name="cpu",
                smoke=True,
                batch_sizes=[2, 4, 8],
                throughput_steps=2,
                lr_steps=3,
                sweep_budget_s=10.0,
            ),
            benchmark_tokenizer(
                vocab_size=16384,
                data_path=None,
                sp_model_path=None,
                device_name="cpu",
                smoke=True,
                batch_sizes=[2, 4, 8],
                throughput_steps=2,
                lr_steps=3,
                sweep_budget_s=10.0,
            ),
        ]
    else:
        if data_root is None:
            raise ValueError("--data-root is required unless --smoke is used")
        validate_gpu_concurrency(max(num_gpus, 1))
        tmp_dir = Path(tempfile.mkdtemp(prefix="exp18_phase0_"))
        jobs: list[tuple[subprocess.Popen[str], Path]] = []
        for idx, vocab_size in enumerate((8192, 16384)):
            data_path, sp_model_path, _ = resolve_variant_paths(data_root, vocab_size)
            if not Path(data_path).exists():
                raise FileNotFoundError(data_path)
            if not Path(sp_model_path).exists():
                raise FileNotFoundError(sp_model_path)
            out = tmp_dir / f"sp{vocab_size}.json"
            proc = _launch_worker(
                vocab_size=vocab_size,
                data_path=data_path,
                sp_model_path=sp_model_path,
                device_id=idx % max(num_gpus, 1),
                output_json=str(out),
                batch_sizes=",".join(str(x) for x in batch_sizes),
                throughput_steps=throughput_steps,
                lr_steps=lr_steps,
                sweep_budget=DEFAULT_SWEEP_BUDGET_S,
                smoke=False,
            )
            jobs.append((proc, out))
        for proc, out in jobs:
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"Phase 0 worker failed with exit code {ret}: {out}")
        results = [json.loads(out.read_text()) for _, out in jobs]

    candidates: list[dict[str, Any]] = []
    for row in results:
        selected = row["selected_candidate"]
        candidates.append(
            {
                "tokenizer": row["tokenizer"],
                "stable": row["stable"],
                "prior_mean_bpb": row["prior_mean_bpb"],
                "projected_coverage_frac": selected["projected_coverage_frac"],
                "tokens_per_s": selected["tokens_per_s"],
                "low_coverage_regime": selected["low_coverage_regime"],
                "selected_lr": selected["selected_lr"],
                "batch_size": selected["batch_size"],
                "data_path": row.get("data_path"),
                "model_config": row["model_config"],
            }
        )
    chosen = choose_tokenizer(candidates)
    winner_details = next(row for row in results if row["tokenizer"] == chosen["tokenizer"])
    selected = dict(winner_details["selected_candidate"])
    selected["tokenizer"] = chosen["tokenizer"]
    selected["prior_mean_bpb"] = winner_details["prior_mean_bpb"]
    selected["model_config"] = dict(winner_details["model_config"], batch_size=int(selected["batch_size"]), base_lr=DEFAULT_BASE_LR)
    selected["data_path"] = winner_details.get("data_path")
    selected["runtime"] = winner_details.get("runtime", {})
    if not smoke and data_root is not None:
        _, sp_model_path, _ = resolve_variant_paths(data_root, int(chosen["tokenizer"].replace("sp", "")))
        selected["sp_model_path"] = sp_model_path
    else:
        selected["sp_model_path"] = None
    selected["sweep_budget_s"] = DEFAULT_SWEEP_BUDGET_S
    selected["rescore_budget_s"] = DEFAULT_RESCORE_BUDGET_S
    selected["selection_budget_s"] = DEFAULT_SELECTION_BUDGET_S
    selected["retarget_budget_s"] = DEFAULT_TOTAL_BUDGET_S - DEFAULT_SWEEP_BUDGET_S - DEFAULT_RESCORE_BUDGET_S - DEFAULT_SELECTION_BUDGET_S
    summary = {
        "phase": "phase0",
        "phase0_results": results,
        "selected": selected,
        "low_coverage_reconsideration": bool(selected["projected_coverage_frac"] < 0.25),
        "runtime_preflight": winner_details.get("runtime", {}),
    }
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Exp 18 Phase 0 benchmark harness")
    p.add_argument("--data-root", help="Root containing datasets/ and tokenizers/")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--batch-sizes", default="32,128,256,512,1024,2048")
    p.add_argument("--throughput-steps", type=int, default=20)
    p.add_argument("--lr-steps", type=int, default=200)
    p.add_argument("--output-json")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--vocab-size", type=int)
    p.add_argument("--data-path")
    p.add_argument("--sp-model-path")
    p.add_argument("--device", default="auto")
    p.add_argument("--sweep-budget", type=float, default=DEFAULT_SWEEP_BUDGET_S)
    args = p.parse_args()

    if args.worker:
        if args.vocab_size is None or args.output_json is None:
            raise SystemExit("--worker requires --vocab-size and --output-json")
        _worker_main(args)
        return

    summary = orchestrate_phase0(
        data_root=args.data_root,
        num_gpus=args.num_gpus,
        batch_sizes=[int(x) for x in args.batch_sizes.split(",") if x],
        throughput_steps=args.throughput_steps,
        lr_steps=args.lr_steps,
        output_json=args.output_json,
        smoke=args.smoke,
    )
    print(
        f"Selected {summary['selected']['tokenizer']} "
        f"batch={summary['selected']['batch_size']} "
        f"coverage={summary['selected']['projected_coverage_frac']:.1%}"
    )


if __name__ == "__main__":
    main()
