#!/usr/bin/env python3
"""Phase 1A throughput microbenchmark.

Measures per-step throughput, peak VRAM, and final loss for every
combination of the three Track 1A levers at the Exp 18 submission-regime
SSM config:

    {fused_grad_clip: 0/1} x {fused_muon: 0/1} x {compile_full_path: 0/1}

Each (lever_combo, seed) pair runs as one "measurement": N warmup steps
with warmup-restore (to prime inductor caches + CUDA autotune without
contaminating optimizer state) then N_timed steps split into K blocks
timed individually so we can report a within-measurement tok/s std.

Output
------
``<output-dir>/results.jsonl``
    One JSON object per measurement, appended. Skipped and re-run safely
    via ``config_hash`` idempotence.
``<output-dir>/summary.md``
    Written at the end of a full matrix run. Paired-delta tables per
    lever (tok/s, peak_vram, final_loss) across the 4 "other-lever"
    settings, so each lever's marginal contribution has 4-point
    statistical leverage per seed.

CLI
---
See ``--help``. The plan's ``--n-steps 200`` is renamed
``--n-timed-steps`` here (and ``--warmup-steps`` is its own knob) to
make the warmup-vs-measurement split explicit — the plan's original
phrasing conflated "total" and "timed-after-warmup".

Not run in this harness: no eval pass, no bpb — final_loss is read from
``train_result['history'][-1]['loss']`` so the loss gate uses the same
numerator as the (plan step 3) success test. Add an eval pass later if
we decide bpb is the gate metric, but throughput microbenching doesn't
need it.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CPU-thread pinning BEFORE torch import. The pod is a one-socket slice
# of a 2x Xeon 8480+ with 28 vCPUs visible; torch's default intra-op pool
# grabs all of them and the 28 threads fight for dispatch work this
# benchmark never actually needs, inflating per-step variance. Pin to 4
# so measurement noise tracks real GPU throughput, not CPU scheduler
# contention.
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import gc
import itertools
import json
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

torch.set_num_threads(4)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "17_local_attn_sidecar"))
sys.path.insert(0, str(REPO / "experiments" / "18_throughput_levers"))
sys.path.insert(0, str(REPO / "experiments" / "19_prereqs"))

# Reuse every helper already shipped in the persistent-DDP runner. See
# ``runner_persistent_ddp.py`` — we call train_ssm_for_budget exactly the
# way it does (same warmup-restore, same optimizer wiring, same data
# sharding) so the bench measures the actual submission path, not a
# microbench fiction.
from chaoscontrol.core import verify_diag_recurrence  # noqa: E402
from chaoscontrol.data import (  # noqa: E402
    build_lm_starts,
    resolve_device,
    resolve_param_dtype,
)
from chaoscontrol.train_ssm import (  # noqa: E402
    _reject_unsupported,
    train_ssm_for_budget,
)
from runner_exp17 import (  # noqa: E402
    build_model,
    load_sp_data,
)
from runner_exp18_ssm import (  # noqa: E402
    _init_distributed,
    _pick_device,
    _shard_train_starts,
)
from runner_persistent_ddp import (  # noqa: E402
    _apply_seed,
    _build_optimizer_with_fused_muon,
    _config_hash,
    _warmup_and_restore,
)


# ---------------------------------------------------------------------------
# Submission-regime base config (matches run_exp18_test4b._base + Test 5b
# winner LR=0.064 at global_batch=2048). The bench does NOT sweep
# optimization hyperparameters — it measures three throughput levers at
# the regime we will actually submit under, at ws=1 by default (1 H100).
# World-size > 1 is supported but the LR defaults are anchored at
# global_batch = ws * batch_size * seq_len and are not re-scaled here;
# if you need a different global-batch LR, pass --base-lr.
# ---------------------------------------------------------------------------
BASE_CONFIG: dict[str, Any] = {
    "model_type": "ssm",
    "vocab_size": 16384,
    "model_dim": 256,
    "num_layers": 4,
    "ff_mult": 2,
    "seq_len": 512,
    "stride": 256,
    "batch_size": 1024,
    "a_mode": "diag",
    "activation_checkpoint": True,
    "optimizer": "muon",
    "chunk_size": 64,
    "base_lr": 0.064,
    "weight_decay": 1e-2,
    "grad_clip_norm": 1.0,
    "precision": "bf16",
    "dtype": "bf16",
    "device": "auto",
    "local_attn_window": 0,
    "local_attn_heads": 1,
    "local_attn_dim": 64,
}

LEVER_NAMES: tuple[str, str, str] = (
    "fused_grad_clip",
    "fused_muon",
    "compile_full_path",
)


def _all_lever_combos() -> list[dict[str, bool]]:
    """Enumerate the 2**3 = 8 lever combinations in stable order."""
    combos: list[dict[str, bool]] = []
    for bits in itertools.product((False, True), repeat=len(LEVER_NAMES)):
        combos.append(dict(zip(LEVER_NAMES, bits)))
    return combos


def _combo_name(combo: dict[str, bool]) -> str:
    """Short deterministic name, e.g. ``fgc1_fm0_cfp1``."""
    return "_".join(
        f"{abbr}{int(combo[name])}"
        for abbr, name in zip(("fgc", "fm", "cfp"), LEVER_NAMES)
    )


def _build_measurement_config(
    *,
    combo: dict[str, bool],
    seed: int,
    world_size: int,
    n_timed_steps: int,
    warmup_steps: int,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Return the full config dict used for config_hash + run_one."""
    cfg = dict(BASE_CONFIG)
    cfg.update(overrides)
    cfg.update(combo)
    cfg["seed"] = int(seed)
    cfg["world_size"] = int(world_size)
    cfg["n_timed_steps"] = int(n_timed_steps)
    cfg["warmup_steps"] = int(warmup_steps)
    cfg["n_blocks"] = int(N_BLOCKS)
    return cfg


# ---------------------------------------------------------------------------
# Results.jsonl I/O — appendable, idempotent by config_hash so a relaunch
# re-measures only the missing rows.
# ---------------------------------------------------------------------------
def _read_existing_hashes(results_path: Path) -> set[str]:
    """Return config_hashes for SUCCESSFUL rows only.

    Error rows (those with an ``"error"`` key) are deliberately excluded
    so that a relaunch retries failed measurements rather than silently
    skipping them. A clean rerun of a previously-errored config is the
    expected recovery path; successful rows still deduplicate normally.
    """
    if not results_path.exists():
        return set()
    hashes: set[str] = set()
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in row:
            continue
        h = row.get("config_hash")
        if isinstance(h, str):
            hashes.add(h)
    return hashes


def _append_jsonl(results_path: Path, row: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str))
        fh.write("\n")


# ---------------------------------------------------------------------------
# Per-measurement timed run. Uses the shared warmup-restore helper from
# runner_persistent_ddp.py so warmup behavior is bit-equivalent to the
# real runner. The timed region itself calls train_ssm_for_budget K
# times (K = N_BLOCKS) with max_steps=block_size so we collect
# per-block elapsed times for a within-measurement tok/s std.
# ---------------------------------------------------------------------------
N_BLOCKS = 5  # 5 blocks x 40 steps = 200 default timed steps


def run_one_measurement(
    *,
    combo: dict[str, bool],
    seed: int,
    config: dict[str, Any],
    device: torch.device,
    param_dtype: torch.dtype,
    rank: int,
    world_size: int,
    ddp_active: bool,
    n_timed_steps: int,
    warmup_steps: int,
    # Pre-loaded, reused across measurements:
    train_tokens: torch.Tensor,
) -> dict[str, Any]:
    """Run one (combo, seed) measurement and return a results.jsonl row.

    Follows ``runner_persistent_ddp.run_one_seed`` for the build-model,
    build-optimizer, warmup-restore sequence so the measured path matches
    the real submission path. Diverges from run_one_seed only in
    replacing the single ``train_ssm_for_budget(budget_seconds=...)``
    call with K bounded calls (``max_steps=block_size``) so per-block
    timings are visible and we can report a within-measurement tok/s std.

    RNG caveat: each of the K blocks passes a distinct seed
    (``seed + 7919 * (block_idx + 1)``) to ``train_ssm_for_budget``, so
    ``final_loss`` is the loss of the last 40-step block under an
    independent batch-RNG draw — not the loss along a single 200-step
    trajectory. The ±0.02 loss gate is comfortable for this spread, but
    this metric is a sanity check, not a convergence comparison.
    """
    _apply_seed(seed)
    is_rank0 = rank == 0

    seq_len = int(config["seq_len"])
    batch_size = int(config["batch_size"])
    stride = int(config.get("stride", seq_len // 2))

    train_starts_all = build_lm_starts(int(train_tokens.numel()), seq_len, stride)
    train_starts = _shard_train_starts(
        train_starts_all, rank=rank, world_size=world_size,
    )

    model = build_model(config, device, param_dtype)
    _reject_unsupported(model)

    optimizer_name = str(config["optimizer"])
    base_lr = float(config["base_lr"])
    weight_decay = float(config["weight_decay"])
    fused_muon = bool(combo["fused_muon"])
    fused_grad_clip = bool(combo["fused_grad_clip"])
    compile_full_path = bool(combo["compile_full_path"])
    precision = str(config["precision"])
    chunk_size = int(config["chunk_size"])
    grad_clip_norm = float(config["grad_clip_norm"])

    optimizer = _build_optimizer_with_fused_muon(
        optimizer_name, model, base_lr=base_lr, weight_decay=weight_decay,
        fused_muon=fused_muon,
    )

    # Warmup-restore — matches Parameter Golf's reference harness. This
    # is where inductor does its multi-second cold compile, where CUDA
    # autotune picks kernels, and where the allocator settles. We keep
    # the helper call byte-identical to runner_persistent_ddp so any
    # future fix to warmup semantics flows into both paths.
    if warmup_steps > 0:
        if is_rank0:
            print(
                f"[rank 0] warmup: {warmup_steps} steps before timer "
                f"(combo={_combo_name(combo)}, seed={seed})",
                flush=True,
            )

        def _run_warmup() -> None:
            train_ssm_for_budget(
                model,
                train_tokens=train_tokens,
                train_starts=train_starts,
                seq_len=seq_len,
                batch_size=batch_size,
                device=device,
                optimizer=optimizer,
                budget_seconds=3600.0,  # unbounded; max_steps gates us
                chunk_size=chunk_size,
                grad_clip_norm=grad_clip_norm,
                fused_grad_clip=fused_grad_clip,
                seed=seed,
                rank=rank,
                world_size=world_size,
                precision=precision,
                compile_full_path=compile_full_path,
                max_steps=warmup_steps,
            )

        def _build_fresh_optimizer() -> torch.optim.Optimizer:
            return _build_optimizer_with_fused_muon(
                optimizer_name, model, base_lr=base_lr, weight_decay=weight_decay,
                fused_muon=fused_muon,
            )

        optimizer = _warmup_and_restore(
            model=model,
            warmup_call_fn=_run_warmup,
            build_optimizer_fn=_build_fresh_optimizer,
            device=device,
            ddp_active=ddp_active,
        )

    # ---- Timed region ----
    # Peak-VRAM accounting: ``train_ssm_for_budget`` internally resets
    # ``torch.cuda.max_memory_allocated`` on each call, so any external
    # pre-loop reset would be clobbered after the first block. We read
    # the per-block peak after each block and take the running max, which
    # equals the true timed-region peak because the workload is identical
    # per block (same seq_len, batch_size, chunk_size, and model state).
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Split n_timed_steps into N_BLOCKS approximately-equal chunks. If
    # n_timed_steps < N_BLOCKS we fall back to single-step blocks.
    block_sizes = _split_steps(n_timed_steps, N_BLOCKS)
    block_elapsed: list[float] = []
    block_steps: list[int] = []
    peak_vram_bytes = 0
    final_loss = float("nan")
    wall_clock_start = time.perf_counter()

    for block_idx, block_steps_target in enumerate(block_sizes):
        if block_steps_target <= 0:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        train_result = train_ssm_for_budget(
            model,
            train_tokens=train_tokens,
            train_starts=train_starts,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            optimizer=optimizer,
            budget_seconds=3600.0,
            chunk_size=chunk_size,
            grad_clip_norm=grad_clip_norm,
            fused_grad_clip=fused_grad_clip,
            seed=seed + 7919 * (block_idx + 1),  # advance RNG across blocks
            rank=rank,
            world_size=world_size,
            precision=precision,
            compile_full_path=compile_full_path,
            max_steps=block_steps_target,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        if device.type == "cuda":
            # train_ssm_for_budget reset peak stats at its entry; this
            # reading is the peak for THIS block only. Fold it into the
            # running max so peak_vram_bytes ends up being max over
            # per-block peaks == true timed-region peak.
            block_peak = torch.cuda.max_memory_allocated(device)
            if block_peak > peak_vram_bytes:
                peak_vram_bytes = block_peak
        steps_done = int(train_result["steps"])
        if steps_done == 0:
            continue
        block_elapsed.append(t1 - t0)
        block_steps.append(steps_done)
        history = train_result.get("history", [])
        if history:
            final_loss = float(history[-1]["loss"])

    wall_clock_s = time.perf_counter() - wall_clock_start
    peak_vram_mb = peak_vram_bytes / (1024 * 1024) if device.type == "cuda" else 0.0

    total_steps = sum(block_steps)
    tokens_per_step = world_size * batch_size * seq_len
    # Per-block tok/s values drive the within-measurement std. Mean
    # weighted by step count across blocks is equivalent to total
    # tokens / total elapsed.
    block_tok_per_s: list[float] = [
        (s * tokens_per_step) / max(e, 1e-9)
        for s, e in zip(block_steps, block_elapsed)
    ]
    if block_tok_per_s:
        tok_per_s_mean = (
            sum(s * tokens_per_step for s in block_steps)
            / max(sum(block_elapsed), 1e-9)
        )
        tok_per_s_std = (
            statistics.stdev(block_tok_per_s) if len(block_tok_per_s) > 1 else 0.0
        )
    else:
        tok_per_s_mean = 0.0
        tok_per_s_std = 0.0

    # Free before returning so the next measurement sees a clean allocator.
    del optimizer
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    row: dict[str, Any] = {
        "seed": int(seed),
        "fused_grad_clip": fused_grad_clip,
        "fused_muon": fused_muon,
        "compile_full_path": compile_full_path,
        "n_timed_steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
        "tokens_per_sec_mean": float(tok_per_s_mean),
        "tokens_per_sec_std": float(tok_per_s_std),
        "peak_vram_mb": float(peak_vram_mb),
        "final_loss": float(final_loss),
        "wall_clock_s": float(wall_clock_s),
        "config_hash": _config_hash(config),
        "block_elapsed_s": block_elapsed,
        "block_steps": block_steps,
        "world_size": int(world_size),
        "tokens_per_step": int(tokens_per_step),
    }
    return row


def _split_steps(n_timed_steps: int, n_blocks: int) -> list[int]:
    """Split ``n_timed_steps`` into ``n_blocks`` roughly-equal chunks."""
    if n_timed_steps <= 0:
        return []
    k = max(1, min(n_blocks, n_timed_steps))
    base, rem = divmod(n_timed_steps, k)
    sizes = [base + (1 if i < rem else 0) for i in range(k)]
    return sizes


# ---------------------------------------------------------------------------
# Summary — paired-delta tables.
# ---------------------------------------------------------------------------
def _paired_deltas(
    rows: list[dict[str, Any]],
    lever: str,
    metric: str,
) -> dict[str, Any]:
    """Mean and std of (on - off) deltas paired on (seed, other_levers).

    For each (seed, other_lever_settings) 4-tuple we look up the two rows
    with ``lever=False`` and ``lever=True`` and compute their delta. With
    2 seeds and 4 "other-lever" settings there are up to 2*4 = 8 paired
    deltas per lever per metric.
    """
    other_levers = [ln for ln in LEVER_NAMES if ln != lever]
    # Index rows by (seed, lever, other_lever_vals).
    index: dict[tuple[int, bool, tuple[bool, ...]], dict[str, Any]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            bool(row[lever]),
            tuple(bool(row[ol]) for ol in other_levers),
        )
        index[key] = row

    deltas: list[float] = []
    for seed in sorted({int(r["seed"]) for r in rows}):
        for other_vals in itertools.product((False, True), repeat=len(other_levers)):
            off_row = index.get((seed, False, other_vals))
            on_row = index.get((seed, True, other_vals))
            if off_row is None or on_row is None:
                continue
            on_val = on_row.get(metric)
            off_val = off_row.get(metric)
            if on_val is None or off_val is None:
                continue
            if not (math.isfinite(on_val) and math.isfinite(off_val)):
                continue
            deltas.append(float(on_val) - float(off_val))

    if not deltas:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "deltas": []}
    return {
        "n": len(deltas),
        "mean": statistics.mean(deltas),
        "std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "deltas": deltas,
    }


def _fmt_delta(d: dict[str, Any], unit: str, relative_to: float | None = None) -> str:
    if d["n"] == 0:
        return "n=0 (missing rows)"
    mean = d["mean"]
    std = d["std"]
    parts = [f"Δ mean={mean:+.3f} {unit}", f"std={std:.3f}", f"n={d['n']}"]
    if relative_to is not None and relative_to > 0:
        parts.append(f"({100.0 * mean / relative_to:+.2f}%)")
    return ", ".join(parts)


def write_summary(results_path: Path, summary_path: Path) -> None:
    """Write the paired-delta summary.md; all numbers read from JSONL."""
    if not results_path.exists():
        summary_path.write_text("# Phase 1A Bench Summary\n\nNo results yet.\n")
        return

    rows: list[dict[str, Any]] = []
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not rows:
        summary_path.write_text("# Phase 1A Bench Summary\n\nNo valid rows.\n")
        return

    # Baseline tok/s for relative-% framing: mean across all-levers-off rows.
    baseline_rows = [
        r for r in rows
        if not r["fused_grad_clip"] and not r["fused_muon"] and not r["compile_full_path"]
    ]
    baseline_tps = (
        statistics.mean(float(r["tokens_per_sec_mean"]) for r in baseline_rows)
        if baseline_rows else None
    )
    baseline_vram = (
        statistics.mean(float(r["peak_vram_mb"]) for r in baseline_rows)
        if baseline_rows else None
    )

    lines: list[str] = []
    lines.append("# Phase 1A Bench Summary\n")
    lines.append(
        "Paired-delta tables — each lever's marginal contribution computed "
        "as (on - off) across the 4 other-lever combinations, per seed. "
        "Each delta is the on-vs-off difference at a matched seed + "
        "matched settings of the other two levers.\n"
    )
    lines.append(f"- Rows analyzed: {len(rows)}")
    if baseline_tps is not None:
        lines.append(
            f"- Baseline tok/s (all levers off, mean across seeds): {baseline_tps:,.0f}"
        )
    if baseline_vram is not None:
        lines.append(
            f"- Baseline peak VRAM (all levers off, mean across seeds): {baseline_vram:,.1f} MB"
        )
    lines.append("")

    for lever in LEVER_NAMES:
        lines.append(f"## Lever: `{lever}`")
        for metric, unit in (
            ("tokens_per_sec_mean", "tok/s"),
            ("peak_vram_mb", "MB"),
            ("final_loss", "loss"),
        ):
            d = _paired_deltas(rows, lever, metric)
            rel = None
            if metric == "tokens_per_sec_mean":
                rel = baseline_tps
            elif metric == "peak_vram_mb":
                rel = baseline_vram
            lines.append(f"- **{metric}**: {_fmt_delta(d, unit, rel)}")

        # Gate readout: +5% tok/s, ±5% VRAM, Δloss ≤ 0.02.
        tps_delta = _paired_deltas(rows, lever, "tokens_per_sec_mean")
        vram_delta = _paired_deltas(rows, lever, "peak_vram_mb")
        loss_delta = _paired_deltas(rows, lever, "final_loss")
        gates: list[str] = []
        if baseline_tps and tps_delta["n"] > 0:
            rel_tps = tps_delta["mean"] / baseline_tps
            pass_tps = rel_tps >= 0.05 and tps_delta["std"] <= max(1e-9, tps_delta["mean"] / 2)
            gates.append(
                f"tok/s +5%: {'PASS' if pass_tps else 'FAIL'} "
                f"(mean={100*rel_tps:+.2f}%, std/|mean|={tps_delta['std']/max(abs(tps_delta['mean']),1e-9):.2f})"
            )
        elif baseline_tps is None:
            gates.append(
                'tok/s gate skipped (no all-off baseline row; run --levers "" to include it)'
            )
        if baseline_vram and vram_delta["n"] > 0:
            rel_vram = vram_delta["mean"] / baseline_vram
            pass_vram = abs(rel_vram) <= 0.05
            gates.append(
                f"VRAM ±5%: {'PASS' if pass_vram else 'FAIL'} (mean={100*rel_vram:+.2f}%)"
            )
        if loss_delta["n"] > 0:
            pass_loss = abs(loss_delta["mean"]) <= 0.02
            gates.append(
                f"Δloss ≤ 0.02: {'PASS' if pass_loss else 'FAIL'} "
                f"(mean={loss_delta['mean']:+.4f})"
            )
        if gates:
            lines.append("- Gates: " + "; ".join(gates))
        lines.append("")

    summary_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def _parse_levers_arg(levers_arg: list[str] | None) -> list[dict[str, bool]]:
    """Return the set of combos to run.

    Accepted forms:
      - None / empty: all 8 combos.
      - List of comma-separated lever-names, one per combo: each string
        is the set of levers enabled; levers not listed are off. Empty
        string = baseline (all off). Example: ``"fused_grad_clip"`` or
        ``"fused_grad_clip,compile_full_path"`` or ``""``.
    """
    if not levers_arg:
        return _all_lever_combos()
    combos: list[dict[str, bool]] = []
    for entry in levers_arg:
        combo = {name: False for name in LEVER_NAMES}
        if entry.strip():
            for token in entry.split(","):
                token = token.strip()
                if token not in LEVER_NAMES:
                    raise ValueError(
                        f"--levers entry {entry!r} references unknown lever "
                        f"{token!r}. Known: {LEVER_NAMES}"
                    )
                combo[token] = True
        combos.append(combo)
    return combos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1A throughput microbench (1xH100 baseline)"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for results.jsonl + summary.md.",
    )
    parser.add_argument("--n-timed-steps", type=int, default=200)
    parser.add_argument(
        "--warmup-steps", type=int, default=20,
        help="Warmup steps BEFORE the timer starts. With compile_full_path "
             "on, the first step is multi-second cold; 20 is the floor "
             "that lets inductor + CUDA autotune stabilize.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2674])
    parser.add_argument(
        "--levers", type=str, nargs="+", default=None,
        help=(
            "Restrict to explicit combos. Pass one string per combo, each "
            "a comma-separated set of enabled lever names (empty string = "
            "all off). Default: run all 2^3=8 combos."
        ),
    )
    parser.add_argument(
        "--world-size", type=int, default=1,
        help="1 = single H100 (default). Match --nproc_per_node if using torchrun.",
    )
    parser.add_argument(
        "--base-lr", type=float, default=None,
        help="Override BASE_CONFIG['base_lr']. Default 0.064 (Test 5b winner at global_batch=2048).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override per-rank batch size (default 1024).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the measurement matrix + config hashes, then exit.",
    )
    args = parser.parse_args(argv)

    combos = _parse_levers_arg(args.levers)
    seeds = list(args.seeds)

    overrides: dict[str, Any] = {}
    if args.base_lr is not None:
        overrides["base_lr"] = float(args.base_lr)
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)

    output_dir = Path(args.output_dir)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.md"

    # Build the full measurement matrix up front so --dry-run and the run
    # path share one source of truth.
    matrix: list[tuple[dict[str, bool], int, dict[str, Any]]] = []
    for combo in combos:
        for seed in seeds:
            cfg = _build_measurement_config(
                combo=combo, seed=seed, world_size=args.world_size,
                n_timed_steps=args.n_timed_steps,
                warmup_steps=args.warmup_steps,
                overrides=overrides,
            )
            matrix.append((combo, seed, cfg))

    if args.dry_run:
        print(f"[dry-run] combos={len(combos)} seeds={len(seeds)} "
              f"total_measurements={len(matrix)}")
        print(f"[dry-run] output_dir={output_dir}")
        print(f"[dry-run] results_path={results_path}")
        existing = _read_existing_hashes(results_path)
        for combo, seed, cfg in matrix:
            h = _config_hash(cfg)
            state = "skip" if h in existing else "run"
            print(
                f"[dry-run] {_combo_name(combo):>14s} seed={seed} "
                f"hash={h} ({state})"
            )
        return 0

    # ----- One-time process setup -----
    if args.world_size > 1 and os.environ.get("WORLD_SIZE") != str(args.world_size):
        raise RuntimeError(
            f"--world-size {args.world_size} requires launch via "
            f"`torchrun --nproc_per_node {args.world_size}`; got "
            f"WORLD_SIZE env={os.environ.get('WORLD_SIZE')!r}"
        )
    rank, world_size, local_rank = _init_distributed(args.world_size)
    is_rank0 = rank == 0
    ddp_active = world_size > 1

    device = _pick_device(local_rank, BASE_CONFIG.get("device", "auto"))
    param_dtype = resolve_param_dtype(BASE_CONFIG.get("dtype", "bf16"), device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Warm the chunked-scan backend's torch.compile cache once per process;
    # otherwise the first measurement pays a cold-compile tax the rest
    # don't pay and the paired-delta is garbage.
    verify_diag_recurrence(device)

    vocab_size = int(BASE_CONFIG["vocab_size"])
    train_tokens, _val_tokens, _ = load_sp_data(args.data_path, vocab_size)

    if is_rank0:
        print(
            f"[rank {rank}/{world_size}] phase1a-bench ready: "
            f"device={device} dtype={param_dtype} measurements={len(matrix)} "
            f"n_timed_steps={args.n_timed_steps} warmup_steps={args.warmup_steps}",
            flush=True,
        )

    existing_hashes = _read_existing_hashes(results_path)
    completed = 0
    skipped = 0
    errored = 0
    t_matrix_start = time.monotonic()

    for i, (combo, seed, cfg) in enumerate(matrix):
        h = _config_hash(cfg)
        if h in existing_hashes:
            if is_rank0:
                print(
                    f"[rank 0] skip {_combo_name(combo)} seed={seed} "
                    f"hash={h} (already in results.jsonl)",
                    flush=True,
                )
            skipped += 1
            continue

        t_entry = time.monotonic()
        try:
            row = run_one_measurement(
                combo=combo, seed=seed, config=cfg,
                device=device, param_dtype=param_dtype,
                rank=rank, world_size=world_size, ddp_active=ddp_active,
                n_timed_steps=args.n_timed_steps,
                warmup_steps=args.warmup_steps,
                train_tokens=train_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — per-measurement isolation
            errored += 1
            if is_rank0:
                print(
                    f"[rank 0] ERROR on {_combo_name(combo)} seed={seed}: "
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    flush=True,
                )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Record the error so a relaunch doesn't endlessly retry.
            if is_rank0:
                _append_jsonl(results_path, {
                    "seed": int(seed),
                    "fused_grad_clip": bool(combo["fused_grad_clip"]),
                    "fused_muon": bool(combo["fused_muon"]),
                    "compile_full_path": bool(combo["compile_full_path"]),
                    "config_hash": h,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                existing_hashes.add(h)
            continue

        if is_rank0:
            _append_jsonl(results_path, row)
            existing_hashes.add(h)
            elapsed = time.monotonic() - t_entry
            print(
                f"[rank 0] done {_combo_name(combo)} seed={seed} "
                f"tok/s={row['tokens_per_sec_mean']:,.0f}±{row['tokens_per_sec_std']:,.0f} "
                f"peak_vram={row['peak_vram_mb']:.1f} MB "
                f"loss={row['final_loss']:.4f} "
                f"entry_wall={elapsed:.1f}s "
                f"({i+1}/{len(matrix)})",
                flush=True,
            )
        completed += 1

    t_matrix_elapsed = time.monotonic() - t_matrix_start
    if is_rank0:
        print(
            f"[rank 0] matrix done: completed={completed} skipped={skipped} "
            f"errored={errored} elapsed={t_matrix_elapsed:.1f}s",
            flush=True,
        )
        write_summary(results_path, summary_path)
        print(f"[rank 0] summary written: {summary_path}", flush=True)

    if ddp_active and dist.is_initialized():
        dist.destroy_process_group()

    return 0 if errored == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
