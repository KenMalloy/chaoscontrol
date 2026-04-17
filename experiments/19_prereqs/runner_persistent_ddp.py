#!/usr/bin/env python3
"""Persistent-DDP multi-seed worker for Exp 19 and beyond.

Replaces the per-seed ``torchrun`` spawn pattern used by Exp 18 Tests
3-10 (``experiments/18_throughput_levers/runner_exp18_ssm.py`` launched
once per (condition, seed)). One ``torchrun`` call starts N workers;
each worker reads a matrix JSON, loops over its entries, and writes one
result JSON per entry. The process / CUDA context / DDP process group /
torch.compile cache / FUSE mmap stay warm across the loop, so the
second seed onward pays only model-build + optimizer-build +
compile-cache-hit + training time.

Design contract (matches frozen runner_exp18_ssm.py per-entry):

    One entry's run_one_seed() call is behaviorally equivalent to one
    invocation of runner_exp18_ssm.run_ddp — same seed plumbing, same
    optimizer construction, same train_ssm_for_budget call, same rank-0
    eval, same JSON schema. The only difference is that the process
    group is NOT destroyed after each entry, only once on exit.

Usage:

    torchrun --standalone --nproc_per_node=N \\
        experiments/19_prereqs/runner_persistent_ddp.py \\
        --data-path <fineweb-dir> \\
        --sp-model-path <sp.model> \\
        --config-matrix /tmp/matrix.json \\
        --output-dir results_persistent_test/ \\
        --budget 600

The matrix JSON is a list of entry dicts; each entry is a complete
config (not a base+override) plus a ``name`` field used to key the
output JSON filename (``{name}_s{seed}.json``). See the launcher for
construction.

Error isolation:
    - Config errors and symmetric OOMs (same batch → both ranks OOM) are
      caught per-entry; an error-marker JSON is written and the loop
      proceeds to the next entry.
    - Rank-0-only failures (e.g., eval OOM on rank 0 while others wait
      at the post-eval barrier) CANNOT be isolated — they deadlock until
      torchrun's outer timeout. This is inherent to manual all-reduce
      semantics; don't try to engineer around it.

Idempotence:
    Entries whose output JSON already exists are skipped, so a relaunch
    after partial completion only runs the missing entries. The skip is
    **config-sensitive**: the stored JSON's ``config`` field must byte-
    match the requested entry. If they differ (e.g., the matrix was
    edited and the output dir reused), the runner raises rather than
    silently reusing stale results — filename-only idempotence would
    otherwise contaminate the new matrix with results from a prior
    different run.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "17_local_attn_sidecar"))
sys.path.insert(0, str(REPO / "experiments" / "18_throughput_levers"))

from chaoscontrol.core import verify_diag_recurrence  # noqa: E402
from chaoscontrol.data import (  # noqa: E402
    build_lm_starts,
    choose_eval_starts,
    resolve_device,
    resolve_param_dtype,
)
from chaoscontrol.train_ssm import (  # noqa: E402
    _reject_unsupported,
    train_ssm_for_budget,
)

# Reuse the frozen runner's helpers rather than copy-paste — keeps
# persistent-DDP behavior bit-equivalent to runner_exp18_ssm.run_ddp
# per entry. Any future fix in those helpers flows into both paths.
from runner_exp17 import (  # noqa: E402
    build_model,
    build_sentencepiece_luts,
    evaluate_bpb_sp,
    load_sp_data,
)
from runner_exp18_ssm import (  # noqa: E402
    _build_optimizer,
    _env_int,
    _init_distributed,
    _pick_device,
    _shard_train_starts,
)


def _build_optimizer_with_fused_muon(
    optimizer_name: str,
    model: torch.nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    fused_muon: bool,
) -> torch.optim.Optimizer:
    """Build Muon with ``fused=True`` if requested; else delegate.

    ``_build_optimizer`` in ``runner_exp18_ssm`` is frozen (Exp 18
    submission regime); editing it would break bit-equivalence between
    persistent-DDP entries and the Exp 18 runs they inherit from. The
    fused-path kwargs below MIRROR that frozen helper — if it is ever
    unfrozen, update both paths together so the classifier path stays
    identical. The non-fused / non-Muon path delegates straight to the
    frozen helper, which is authoritative.
    """
    if fused_muon and optimizer_name == "muon":
        from chaoscontrol.optim.muon import Muon
        optimizer = Muon(
            list(model.parameters()),
            lr=base_lr,
            weight_decay=weight_decay,
            adamw_lr=base_lr,
            adamw_weight_decay=weight_decay,
            fused=True,
        )
        optimizer.bind_param_names(list(model.named_parameters()))
        return optimizer
    # Non-fused path OR non-Muon optimizer: frozen helper is authoritative.
    return _build_optimizer(
        optimizer_name, model, base_lr=base_lr, weight_decay=weight_decay,
    )


def _config_hash(config: dict[str, Any]) -> str:
    """Short stable hash of a config, for dry-run printout."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:8]


def _require_config_match(
    out_path: Path,
    requested: dict[str, Any],
) -> None:
    """Raise if the stored output's config differs from what's requested.

    Idempotent skip honors prior outputs by filename. Filename alone does
    not encode the config, so a matrix edit (e.g., LR change) reusing
    the same ``--output-dir`` would silently skip entries whose stale
    JSONs answer a different question — scientific-validity poison that
    would still look like a clean "successfully resumed" run.

    Every rank reads the file independently; the pod's shared filesystem
    guarantees every rank sees the same bytes, so every rank decides the
    same way and either all raise or all proceed. No collective required.

    Raises
    ------
    RuntimeError
        If the stored JSON is unreadable or its ``config`` field differs
        from ``requested``. The message names the file and the two
        config hashes so ``rm`` / ``diff`` / fresh-output-dir recovery is
        unambiguous.
    """
    try:
        stored = json.loads(out_path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"idempotent skip aborted: could not read stored JSON at "
            f"{out_path}: {exc}. Delete or fix the file to re-run."
        ) from exc
    stored_config = stored.get("config")
    if stored_config != requested:
        stored_hash = _config_hash(stored_config or {})
        requested_hash = _config_hash(requested)
        raise RuntimeError(
            f"idempotent skip aborted: config mismatch at {out_path}. "
            f"Stored config hash={stored_hash} does not match requested "
            f"hash={requested_hash}. A prior run with a different config "
            f"wrote this filename. Delete the stale output or use a "
            f"fresh --output-dir to re-run this entry."
        )


def _apply_seed(seed: int) -> None:
    """Re-seed every RNG the training path touches.

    Called at the START of every entry iteration — the previous
    entry's training loop will have advanced torch / numpy / random
    global state, so carry-over would silently correlate seed N+1's
    trajectory with seed N. Matches runner_exp18_ssm.run_ddp's seeding
    block (rows 213-216) verbatim so per-entry seeding is identical.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_result_json(out_path: Path, payload: dict[str, Any]) -> None:
    """Atomic write (tmp + rename) matching runner_exp18_ssm's pattern."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str))
    tmp_path.rename(out_path)


def _sync_error_flag(
    local_error: bool,
    device: torch.device,
    ddp_active: bool,
) -> bool:
    """all_reduce(MAX) an error flag across ranks.

    If any rank hit an exception during the entry, every rank must take
    the same abort-or-continue branch — otherwise the next iteration's
    broadcast_params / all-reduce will desync the same way bug #3 in
    project_ddp_manual_allreduce_2026-04-16.md desynced the stop flag.
    """
    if not ddp_active:
        return local_error
    flag = torch.tensor([1.0 if local_error else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return flag.item() > 0.5


def _warmup_and_restore(
    *,
    model: torch.nn.Module,
    warmup_call_fn: Callable[[], Any],
    build_optimizer_fn: Callable[[], torch.optim.Optimizer],
    device: torch.device,
    ddp_active: bool,
) -> torch.optim.Optimizer:
    """Snapshot model, run warmup, restore model, build fresh optimizer.

    Matches Parameter Golf's submission harness contract
    (``baselines/parameter_golf/train_gpt.py`` lines 935-961): the warmup
    iterations prime ``torch.compile``, CUDA kernel autotune, the memory
    allocator, and NCCL broadcast paths. After restore the model is
    byte-equivalent to its pre-warmup state, and the returned optimizer
    is freshly built with no accumulated moments. The caller's real
    ``train_ssm_for_budget`` timer then starts from a state functionally
    identical to a fresh submission run's timer-start state.
    """
    model_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    warmup_call_fn()
    model.load_state_dict(model_state, strict=True)
    # Clear any grads the warmup pass left on the parameters. Matches PG's
    # reference harness (baselines/parameter_golf/train_gpt.py:955) which
    # zeros grads after restore so the measured pass starts with grad=None
    # on every param. Without this, first-step timing and peak-memory
    # accounting inherit the full grad-tensor allocation from warmup.
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if ddp_active:
        dist.barrier()
    return build_optimizer_fn()


def run_one_seed(
    *,
    config: dict[str, Any],
    out_path: Path,
    data_path: str,
    sp_model_path: str,
    budget_seconds: float,
    device: torch.device,
    param_dtype: torch.dtype,
    rank: int,
    world_size: int,
    ddp_active: bool,
    # Pre-loaded, reused across entries:
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
) -> dict[str, Any]:
    """Run one (precision, seed, config) entry.

    Mirrors the body of ``runner_exp18_ssm.run_ddp`` from the seed
    block through the JSON write — same model build, same optimizer,
    same train_ssm_for_budget, same rank-0 eval, same atomic write.

    Differences from run_ddp:
        - Assumes the process group + device + data + LUTs are already
          initialized. Doesn't touch dist init or teardown.
        - Writes to ``out_path`` (per-entry), not to a single output
          configured by CLI args.
        - Returns a summary dict so the caller can log per-entry status
          without re-reading the JSON.
    """
    is_rank0 = rank == 0
    seed = int(config["seed"])
    _apply_seed(seed)

    # Data split / sharding is per-entry because seed drives the eval
    # sample AND the stride-shard. The tokens / LUTs themselves are
    # the expensive bit and those are reused from the caller.
    seq_len = int(config["seq_len"])
    stride = int(config.get("stride", seq_len // 2))
    batch_size = int(config["batch_size"])
    eval_batches = int(config.get("eval_batches", 16))

    train_starts_all = build_lm_starts(int(train_tokens.numel()), seq_len, stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), seq_len, stride)
    eval_starts = choose_eval_starts(
        val_starts, batch_size=batch_size, eval_batches=eval_batches, seed=seed,
    )
    train_starts = _shard_train_starts(
        train_starts_all, rank=rank, world_size=world_size,
    )

    model = build_model(config, device, param_dtype)
    _reject_unsupported(model)
    precision = str(config.get("precision", "bf16"))
    if precision == "fp8":
        # The launcher pre-flight already dropped fp8 entries if TE is
        # unavailable; this branch runs only when TE is present. Still
        # guard with the promoter's own skip-on-missing-TE for safety.
        from chaoscontrol.precision import maybe_promote_linears_to_te
        n_promoted = maybe_promote_linears_to_te(model, enabled=True)
        if is_rank0:
            print(
                f"[rank 0] promoted {n_promoted} nn.Linear -> te.Linear for fp8",
                flush=True,
            )
    model_params = sum(p.numel() for p in model.parameters())

    if is_rank0:
        print(
            f"[rank {rank}/{world_size}] entry seed={seed} "
            f"precision={precision} dim={config['model_dim']} "
            f"layers={config['num_layers']} params={model_params:,}",
            flush=True,
        )

    optimizer_name = str(config.get("optimizer", "")).strip()
    if not optimizer_name:
        raise ValueError(
            "runner_persistent_ddp requires config['optimizer'] to be "
            "set (one of {'adamw', 'muon', 'lamb'})."
        )
    base_lr = float(config.get("base_lr", 2e-3))
    weight_decay = float(config.get("weight_decay", 1e-2))
    fused_muon = bool(config.get("fused_muon", False))
    optimizer = _build_optimizer_with_fused_muon(
        optimizer_name, model, base_lr=base_lr, weight_decay=weight_decay,
        fused_muon=fused_muon,
    )
    chunk_size = int(config.get("chunk_size", 64))
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    fused_grad_clip = bool(config.get("fused_grad_clip", False))

    # Warmup-restore phase — matches Parameter Golf's pre-timer warmup
    # (baselines/parameter_golf/train_gpt.py lines 935-961). Runs
    # `warmup_steps` full fwd/bwd/opt.step iterations to prime
    # torch.compile, CUDA kernel autotune, the memory allocator, and
    # NCCL broadcast paths; restores the model to its pre-warmup weights;
    # rebuilds the optimizer fresh. The subsequent `train_ssm_for_budget`
    # call's internal timer starts from a state equivalent to a
    # submission-run's timer-start state, so reported bpb is free of
    # compile-cost contamination. `warmup_steps=0` disables the phase.
    warmup_steps = int(config.get("warmup_steps", 20))
    if warmup_steps > 0:
        if is_rank0:
            print(
                f"[rank 0] warmup: {warmup_steps} steps before timer",
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
                budget_seconds=300.0,
                chunk_size=chunk_size,
                grad_clip_norm=grad_clip_norm,
                fused_grad_clip=fused_grad_clip,
                seed=seed,
                rank=rank,
                world_size=world_size,
                precision=precision,
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
        if is_rank0:
            print("[rank 0] warmup complete, real timer starts now", flush=True)

    train_result = train_ssm_for_budget(
        model,
        train_tokens=train_tokens,
        train_starts=train_starts,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        optimizer=optimizer,
        budget_seconds=budget_seconds,
        chunk_size=chunk_size,
        grad_clip_norm=grad_clip_norm,
        fused_grad_clip=fused_grad_clip,
        seed=seed,
        rank=rank,
        world_size=world_size,
        precision=precision,
    )

    if ddp_active:
        dist.barrier()

    if is_rank0:
        eval_result = evaluate_bpb_sp(
            model,
            tokens=val_tokens,
            eval_starts=eval_starts,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
    else:
        eval_result = {}

    if ddp_active:
        dist.barrier()

    history_final = train_result["history"][-1] if train_result["history"] else {}
    train_summary = {
        "steps": int(train_result["steps"]),
        "elapsed_s": float(train_result["elapsed_s"]),
        "steps_per_second": float(
            train_result["steps"] / max(train_result["elapsed_s"], 1e-9)
        ),
        "final_loss": float(history_final.get("loss", float("nan"))),
        "peak_vram_mb": float(train_result.get("peak_vram_mb", 0.0)),
        "ddp_rank": int(train_result.get("rank", rank)),
        "ddp_world_size": int(train_result.get("world_size", world_size)),
    }

    result = {
        "config": config,
        "params": model_params,
        "train": train_summary,
        "eval": eval_result,
    }

    # Fail closed on non-finite results — same gate as runner_exp18_ssm
    # so result_is_finite() treats both paths' outputs uniformly.
    if is_rank0:
        violations: list[str] = []
        if not math.isfinite(train_summary["final_loss"]):
            violations.append(
                f"train.final_loss={train_summary['final_loss']} is not finite"
            )
        for key in ("bpb", "loss"):
            if key in eval_result:
                val = float(eval_result[key])
                if not math.isfinite(val):
                    violations.append(f"eval.{key}={val} is not finite")
        if violations:
            # Write an error-marker JSON so partial-success visibility is
            # preserved; don't RAISE the way the frozen runner does,
            # because a raise here would abort the remaining seeds.
            err = {
                "config": config,
                "error": "non_finite_result: " + "; ".join(violations),
                "train": train_summary,
                "eval": eval_result,
            }
            _write_result_json(out_path, err)
        else:
            _write_result_json(out_path, result)

    if is_rank0:
        bpb_display = eval_result.get("bpb", float("nan"))
        print(
            f"[rank {rank}/{world_size}] done entry seed={seed}: "
            f"bpb={bpb_display:.4f} steps={train_summary['steps']} "
            f"steps/s={train_summary['steps_per_second']:.2f} "
            f"peak_vram={train_summary['peak_vram_mb']:.1f} MB",
            flush=True,
        )

    # Tear down per-entry state. The model, optimizer, and train_starts
    # lists can be substantial; explicit del + empty_cache before the
    # next iteration avoids carrying a dead graph through the next
    # build_model call.
    del optimizer
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Barrier so a fast rank doesn't enter the NEXT entry's build_model /
    # broadcast_params while a slow rank is still freeing memory. This
    # is the analog of train_ssm_for_budget's teardown barrier — bug
    # class #2/#3 from project_ddp_manual_allreduce_2026-04-16.md.
    if ddp_active:
        dist.barrier()

    return train_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Persistent-DDP multi-seed worker for Exp 19+"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument(
        "--config-matrix",
        required=True,
        help="Path to a JSON file listing entries to run sequentially.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for per-entry result JSONs and error markers.",
    )
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Override WORLD_SIZE env var set by torchrun (1 forces single-device).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Parse the matrix and print (name, seed, config-hash) tuples, "
            "then exit. Bypasses dist init and model build so it runs on "
            "CPU without a GPU or torchrun launch."
        ),
    )
    args = parser.parse_args(argv)

    matrix_path = Path(args.config_matrix)
    entries = json.loads(matrix_path.read_text())
    if not isinstance(entries, list):
        raise ValueError(
            f"matrix JSON must be a list of entry dicts, got {type(entries).__name__}"
        )

    output_dir = Path(args.output_dir)

    if args.dry_run:
        # Dry-run path prints the matrix it would execute. Used as a
        # CPU-only sanity check that matrix JSON parses and the expected
        # fields are present. Does NOT touch dist / CUDA / the model.
        print(f"[dry-run] matrix={matrix_path} entries={len(entries)}")
        print(f"[dry-run] output_dir={output_dir}")
        for i, entry in enumerate(entries):
            name = str(entry.get("name", f"entry{i}"))
            seed = int(entry.get("seed", 0))
            precision = str(entry.get("precision", "bf16"))
            out_path = output_dir / f"{name}_s{seed}.json"
            exists = "existing" if out_path.exists() else "new"
            print(
                f"[dry-run] {i:3d}: name={name} seed={seed} "
                f"precision={precision} hash={_config_hash(entry)} "
                f"out={out_path} ({exists})"
            )
        # Required-field check so a malformed matrix gets caught on CPU.
        required = {"name", "seed", "precision", "model_dim", "num_layers",
                    "seq_len", "batch_size", "vocab_size", "optimizer",
                    "base_lr", "chunk_size"}
        for i, entry in enumerate(entries):
            missing = required - set(entry.keys())
            if missing:
                raise ValueError(
                    f"entry {i} (name={entry.get('name')!r}) missing "
                    f"required fields: {sorted(missing)}"
                )
        print(f"[dry-run] all {len(entries)} entries have required fields")
        return 0

    # ----- One-time process setup (persistent across entries) -----
    rank, world_size, local_rank = _init_distributed(args.world_size)
    is_rank0 = rank == 0
    ddp_active = world_size > 1

    # device/config-device resolution: every entry in the matrix must
    # agree on device. Pick from the first entry; validate the rest.
    first_device_name = str(entries[0].get("device", "auto"))
    for i, entry in enumerate(entries):
        ed = str(entry.get("device", "auto"))
        if ed != first_device_name:
            raise ValueError(
                f"entry {i} has device={ed!r} but entry 0 has "
                f"device={first_device_name!r}; matrix must agree on device "
                f"for persistent DDP."
            )
    device = _pick_device(local_rank, first_device_name)
    param_dtype = resolve_param_dtype(
        str(entries[0].get("dtype", "bf16")), device,
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # verify_diag_recurrence warms the chunked-scan backend ONCE per
    # process. Subsequent entries reuse the torch.compile cache entry,
    # so only seed 1 pays the compile cost.
    verify_diag_recurrence(device)

    # All entries must agree on vocab_size (data/tokenizer is shared).
    first_vocab = int(entries[0]["vocab_size"])
    for i, entry in enumerate(entries):
        ev = int(entry["vocab_size"])
        if ev != first_vocab:
            raise ValueError(
                f"entry {i} vocab_size={ev} differs from entry 0 "
                f"vocab_size={first_vocab}; persistent-DDP requires "
                f"a single vocabulary across the whole matrix."
            )

    train_tokens, val_tokens, _ = load_sp_data(args.data_path, first_vocab)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(args.sp_model_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, first_vocab, device)
    )

    if is_rank0:
        print(
            f"[rank {rank}/{world_size}] persistent-DDP ready: "
            f"device={device} dtype={param_dtype} entries={len(entries)} "
            f"vocab={first_vocab}",
            flush=True,
        )

    # ----- Per-entry loop -----
    completed = 0
    skipped = 0
    errored = 0
    t_matrix_start = time.monotonic()

    for i, entry in enumerate(entries):
        name = str(entry.get("name"))
        seed = int(entry.get("seed"))
        if not name:
            raise ValueError(f"entry {i} missing 'name' field")
        out_path = output_dir / f"{name}_s{seed}.json"

        # Idempotent skip — a prior partial run's JSONs carry over.
        # All ranks should see the same filesystem state on a pod, but
        # reduce the exists-flag across ranks so a straggler rank can't
        # diverge into the training path while the rest skip.
        local_exists = out_path.exists()
        any_exists = local_exists
        if ddp_active:
            flag = torch.tensor([1.0 if local_exists else 0.0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            any_exists = flag.item() > 0.5
        if any_exists:
            # Config-sensitive skip: refuse to honor a stored JSON whose
            # config differs from the current entry. Filename match alone
            # would silently reuse stale results across matrix edits.
            _require_config_match(out_path, entry)
            if is_rank0:
                print(
                    f"[rank 0] skip {name}_s{seed} (output exists at {out_path})",
                    flush=True,
                )
            skipped += 1
            continue

        t_entry = time.monotonic()
        local_error_msg: str | None = None

        try:
            run_one_seed(
                config=entry,
                out_path=out_path,
                data_path=args.data_path,
                sp_model_path=args.sp_model_path,
                budget_seconds=args.budget,
                device=device,
                param_dtype=param_dtype,
                rank=rank,
                world_size=world_size,
                ddp_active=ddp_active,
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
        except Exception as exc:  # noqa: BLE001 — per-entry isolation
            local_error_msg = f"{type(exc).__name__}: {exc}"
            if is_rank0:
                print(
                    f"[rank 0] ERROR on {name}_s{seed}: {local_error_msg}\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )
            # Free whatever the failed build left behind so the next
            # entry sees a clean CUDA allocator.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Cross-rank error-flag reduction. Any rank that raised → every
        # rank takes the "write error marker, continue" branch together.
        any_error = _sync_error_flag(
            local_error_msg is not None, device, ddp_active,
        )
        if any_error:
            errored += 1
            if is_rank0:
                err_payload = {
                    "config": entry,
                    "error": local_error_msg or "peer_rank_error",
                }
                _write_result_json(out_path, err_payload)
            # Barrier so writers and readers stay lockstep before the
            # next entry's broadcast_params.
            if ddp_active:
                dist.barrier()
            continue

        completed += 1
        if is_rank0:
            elapsed = time.monotonic() - t_entry
            print(
                f"[rank 0] completed {name}_s{seed} in {elapsed:.1f}s "
                f"(entry {i+1}/{len(entries)})",
                flush=True,
            )

    t_matrix_elapsed = time.monotonic() - t_matrix_start
    if is_rank0:
        print(
            f"[rank 0] matrix done: completed={completed} skipped={skipped} "
            f"errored={errored} elapsed={t_matrix_elapsed:.1f}s",
            flush=True,
        )

    # Process-group teardown ONLY at process exit — never between entries.
    if ddp_active and dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
