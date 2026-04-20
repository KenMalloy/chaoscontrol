#!/usr/bin/env python3
"""Cache-backed score-only Exp20 validation runner.

This is the performance-oriented sibling of ``scripts/run_exp20_eval.py``.
It starts with the simplest record-facing semantic: reset state at each doc
boundary, score all cached docs, and write exact BPB bookkeeping. TTT and
distributed sharding layer on top after this path is parity-tested.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from chaoscontrol.eval_stream.budget import BudgetTracker
from chaoscontrol.eval_stream.val_cache import CachedDoc, ValCache, load_val_cache
from chaoscontrol.evaluation import compute_bpb


DEFAULT_MAX_FORWARD_TOKENS = 524_288


class _DocWork(NamedTuple):
    doc: CachedDoc
    output_index: int


class _DocScore(NamedTuple):
    doc: CachedDoc
    output_index: int
    ce_nats: float
    tokens_scored: int
    chunk_count: int
    loss_before: float
    wall_ms: float
    state_norm: float


def prepare_doc_work(docs: list[CachedDoc], *, sort_by_length: bool) -> list[_DocWork]:
    work = [_DocWork(doc=doc, output_index=idx) for idx, doc in enumerate(docs)]
    if sort_by_length:
        return sorted(work, key=lambda item: (-item.doc.token_len, item.output_index))
    return work


def resolve_doc_batch_size(
    *,
    requested_doc_batch_size: int,
    chunk_size: int,
    max_forward_tokens: int,
) -> int:
    requested = max(1, int(requested_doc_batch_size))
    if chunk_size < 0 or max_forward_tokens <= 0:
        return requested
    token_limited = max(1, int(max_forward_tokens) // max(1, int(chunk_size)))
    return min(requested, token_limited)


def resolve_max_forward_tokens(
    *,
    max_forward_tokens: str | int,
    requested_doc_batch_size: int,
    chunk_size: int,
    device: torch.device,
    probe_fn: Callable[[int], int] | None = None,
) -> int:
    if isinstance(max_forward_tokens, int) or str(max_forward_tokens).lower() != "auto":
        value = int(max_forward_tokens)
        if value <= 0:
            raise ValueError(f"max_forward_tokens must be positive or 'auto', got {max_forward_tokens!r}")
        return value

    if chunk_size < 0:
        return 0
    requested_forward_tokens = max(1, int(requested_doc_batch_size)) * max(1, int(chunk_size))
    if device.type != "cuda":
        return requested_forward_tokens
    if probe_fn is None:
        return min(requested_forward_tokens, DEFAULT_MAX_FORWARD_TOKENS)
    probed = int(probe_fn(requested_forward_tokens))
    return max(1, min(requested_forward_tokens, probed))


def expected_scored_tokens(
    *,
    token_len: int,
    chunk_size: int,
    score_boundary_targets: bool,
) -> int:
    token_len = int(token_len)
    if token_len < 2:
        return 0
    if chunk_size < 0 or score_boundary_targets:
        return token_len - 1
    return sum(end - start - 1 for start, end in _token_chunk_ranges(token_len, chunk_size))


def validate_doc_score_coverage(
    score: _DocScore,
    *,
    chunk_size: int,
    score_boundary_targets: bool,
) -> None:
    expected = expected_scored_tokens(
        token_len=score.doc.token_len,
        chunk_size=chunk_size,
        score_boundary_targets=score_boundary_targets,
    )
    if int(score.tokens_scored) != expected:
        raise RuntimeError(
            "token coverage mismatch "
            f"doc_id={score.doc.doc_id} expected={expected} actual={score.tokens_scored} "
            f"token_len={score.doc.token_len} chunk_size={chunk_size} "
            f"score_boundary_targets={score_boundary_targets}"
        )


def doc_range_for_rank(*, num_docs: int, rank: int, world_size: int) -> tuple[int, int]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    start = (int(num_docs) * int(rank)) // int(world_size)
    end = (int(num_docs) * (int(rank) + 1)) // int(world_size)
    return start, end


def resolve_distributed_context(env: dict[str, str] | None = None) -> dict[str, int | bool]:
    source = os.environ if env is None else env
    rank = int(source.get("RANK", "0"))
    world_size = int(source.get("WORLD_SIZE", "1"))
    local_rank = int(source.get("LOCAL_RANK", "0"))
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "distributed": world_size > 1,
    }


def _build_model(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    from chaoscontrol.model import ChaosStudentLM

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = ChaosStudentLM(**cfg)
    model.load_state_dict(blob["model"], strict=True)
    return model, cfg


def _token_chunk_ranges(token_len: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size < 0:
        return [(0, token_len)] if token_len >= 2 else []
    return [
        (start, min(start + chunk_size, token_len))
        for start in range(0, token_len, chunk_size)
        if min(start + chunk_size, token_len) - start >= 2
    ]


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return device


def _model_logits(out: object) -> torch.Tensor:
    return out["logits"] if isinstance(out, dict) else out  # type: ignore[index,return-value]


def _model_final_states(out: object) -> list[torch.Tensor] | None:
    if isinstance(out, dict) and out.get("final_states"):
        return list(out["final_states"])
    return None


def _probe_max_forward_tokens(
    model: torch.nn.Module,
    *,
    requested_forward_tokens: int,
    chunk_size: int,
    device: torch.device,
) -> int:
    if device.type != "cuda" or chunk_size < 0:
        return requested_forward_tokens
    candidate_docs = max(1, int(requested_forward_tokens) // max(1, int(chunk_size)))
    model_was_training = model.training
    model.eval()
    with torch.inference_mode():
        while candidate_docs >= 1:
            try:
                torch.cuda.empty_cache()
                chunk = torch.zeros(
                    (candidate_docs, chunk_size),
                    dtype=torch.long,
                    device=device,
                )
                out = model(chunk)
                logits = _model_logits(out)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    chunk[:, 1:].reshape(-1),
                    reduction="sum",
                )
                del loss, logits, out, chunk
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                if model_was_training:
                    model.train()
                return candidate_docs * chunk_size
            except torch.cuda.OutOfMemoryError:
                candidate_docs //= 2
                torch.cuda.empty_cache()
    if model_was_training:
        model.train()
    return max(1, min(DEFAULT_MAX_FORWARD_TOKENS, requested_forward_tokens))


def _run_score_warmup(
    model: torch.nn.Module,
    *,
    doc_batch_size: int,
    chunk_size: int,
    device: torch.device,
    steps: int,
) -> float:
    if steps <= 0 or doc_batch_size <= 0 or chunk_size < 2:
        return 0.0
    warmup_t0 = time.monotonic()
    model_was_training = model.training
    model.eval()
    prev_states: list[torch.Tensor] | None = None
    with torch.inference_mode():
        chunk = torch.zeros((doc_batch_size, chunk_size), dtype=torch.long, device=device)
        for _ in range(int(steps)):
            kwargs = {"initial_states": prev_states} if prev_states is not None else {}
            out = model(chunk, **kwargs)
            logits = _model_logits(out)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                chunk[:, 1:].reshape(-1),
                reduction="sum",
            )
            prev_states = _model_final_states(out)
            del loss, logits, out
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del chunk, prev_states
    if model_was_training:
        model.train()
    return time.monotonic() - warmup_t0


def _rank_output_path(output_path: Path, rank: int, world_size: int) -> Path:
    if world_size <= 1:
        return output_path
    return output_path.with_name(f"{output_path.stem}.rank{rank}{output_path.suffix}")


def _score_doc(
    *,
    model: torch.nn.Module,
    cache: ValCache,
    doc: CachedDoc,
    chunk_size: int,
    device: torch.device,
    score_boundary_targets: bool,
    device_tokens: torch.Tensor | None = None,
) -> tuple[float, int, int, float, float, float]:
    prev_states: list[torch.Tensor] | None = None
    doc_ce = torch.zeros((), device=device, dtype=torch.float64)
    tokens_scored = 0
    chunk_count = 0
    t0 = time.monotonic()
    doc_tokens_np = None if device_tokens is not None else cache.tokens_for_doc(doc)

    for start, end in _token_chunk_ranges(int(doc.token_len), chunk_size):
        if end - start < 2:
            continue
        if device_tokens is not None:
            chunk = device_tokens[doc.token_start + start:doc.token_start + end].unsqueeze(0)
        else:
            assert doc_tokens_np is not None
            chunk_np = doc_tokens_np[start:end]
            chunk = torch.tensor(chunk_np, dtype=torch.long, device=device).unsqueeze(0)
        kwargs = {"initial_states": prev_states} if prev_states else {}
        out = model(chunk, **kwargs)
        logits = _model_logits(out)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        )
        if score_boundary_targets and end < int(doc.token_len):
            if device_tokens is not None:
                next_token = device_tokens[doc.token_start + end:doc.token_start + end + 1]
            else:
                assert doc_tokens_np is not None
                next_token = torch.tensor(
                    [int(doc_tokens_np[end])],
                    dtype=torch.long,
                    device=device,
                )
            loss = loss + F.cross_entropy(
                logits[:, -1, :],
                next_token,
                reduction="sum",
            )
            tokens_scored += 1
        doc_ce += loss.to(torch.float64)
        tokens_scored += int(chunk.size(1) - 1)
        chunk_count += 1
        final_states = _model_final_states(out)
        if final_states:
            prev_states = final_states

    wall_ms = (time.monotonic() - t0) * 1000.0
    state_norm = 0.0
    if prev_states:
        state_norm = sum(float(s.norm().item()) for s in prev_states) / len(prev_states)
    ce_nats = float(doc_ce.item())
    loss_before = ce_nats / max(chunk_count, 1)
    return ce_nats, tokens_scored, chunk_count, loss_before, wall_ms, state_norm


def _score_docs_reset_batch(
    *,
    model: torch.nn.Module,
    work_items: list[_DocWork],
    device_tokens: torch.Tensor,
    chunk_size: int,
    device: torch.device,
    score_boundary_targets: bool,
) -> list[_DocScore]:
    """Score a document batch while preserving reset-mode semantics.

    Each document owns its own recurrent state trajectory. We only batch
    together chunks with the same true length, which avoids padding tokens from
    contaminating final-state diagnostics or future carried chunks.
    """
    if not work_items:
        return []
    if chunk_size < 0:
        raise ValueError("batched reset scoring requires chunk_size >= 0")
    work_items = sorted(work_items, key=lambda item: (-item.doc.token_len, item.output_index))
    batch_t0 = time.monotonic()
    batch_size = len(work_items)
    token_starts = torch.tensor(
        [item.doc.token_start for item in work_items],
        dtype=torch.long,
        device=device,
    )
    token_lens = [int(item.doc.token_len) for item in work_items]
    ce_sums = torch.zeros(batch_size, device=device, dtype=torch.float64)
    tokens_scored = [0 for _ in work_items]
    chunk_counts = [0 for _ in work_items]
    states: list[torch.Tensor] | None = None
    has_state = [False for _ in work_items]
    arange_cache: dict[int, torch.Tensor] = {}
    row_index_cache: dict[tuple[int, ...], torch.Tensor] = {}
    max_token_len = max(token_lens)

    def offsets_for(seq_len: int) -> torch.Tensor:
        offsets = arange_cache.get(seq_len)
        if offsets is None:
            offsets = torch.arange(seq_len, dtype=torch.long, device=device)
            arange_cache[seq_len] = offsets
        return offsets

    def row_index_for(doc_indices: list[int]) -> torch.Tensor:
        key = tuple(doc_indices)
        row_index = row_index_cache.get(key)
        if row_index is None:
            row_index = torch.tensor(doc_indices, dtype=torch.long, device=device)
            row_index_cache[key] = row_index
        return row_index

    def process_group(doc_indices: list[int], *, chunk_start: int, seq_len: int) -> None:
        nonlocal states
        if not doc_indices:
            return
        contiguous_prefix = doc_indices == list(range(len(doc_indices)))
        row_index: torch.Tensor | None = None
        if contiguous_prefix:
            starts = token_starts[:len(doc_indices)] + chunk_start
            if states is None or not all(has_state[idx] for idx in doc_indices):
                initial_states = None
            else:
                initial_states = [state[:len(doc_indices)] for state in states]
        else:
            row_index = row_index_for(doc_indices)
            starts = token_starts.index_select(0, row_index) + chunk_start
            if states is None or not all(has_state[idx] for idx in doc_indices):
                initial_states = None
            else:
                initial_states = [state.index_select(0, row_index) for state in states]

        chunk = device_tokens[starts.unsqueeze(1) + offsets_for(seq_len).unsqueeze(0)]
        kwargs = {"initial_states": initial_states} if initial_states is not None else {}
        out = model(chunk, **kwargs)
        logits = _model_logits(out)
        losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(len(doc_indices), seq_len - 1).sum(dim=1)

        boundary_token_count = [0 for _ in doc_indices]
        if score_boundary_targets:
            boundary_rows: list[int] = []
            boundary_doc_indices: list[int] = []
            next_token_pos = chunk_start + seq_len
            for row_idx, doc_idx in enumerate(doc_indices):
                if next_token_pos < token_lens[doc_idx]:
                    boundary_rows.append(row_idx)
                    boundary_doc_indices.append(doc_idx)
                    boundary_token_count[row_idx] = 1
            if boundary_rows:
                boundary_row_index = torch.tensor(boundary_rows, dtype=torch.long, device=device)
                boundary_docs = row_index_for(boundary_doc_indices)
                targets = device_tokens[token_starts.index_select(0, boundary_docs) + next_token_pos]
                boundary_losses = F.cross_entropy(
                    logits.index_select(0, boundary_row_index)[:, -1, :],
                    targets,
                    reduction="none",
                )
                losses.index_add_(0, boundary_row_index, boundary_losses)

        final_states = _model_final_states(out)
        if states is None and final_states is not None:
            states = [
                torch.empty(
                    (batch_size, state.size(1)),
                    dtype=state.dtype,
                    device=device,
                )
                for state in final_states
            ]
        if states is not None and final_states is not None:
            if contiguous_prefix:
                for state, final_state in zip(states, final_states):
                    state[:len(doc_indices)].copy_(final_state)
            else:
                assert row_index is not None
                for state, final_state in zip(states, final_states):
                    state.index_copy_(0, row_index, final_state)

        if contiguous_prefix:
            ce_sums[:len(doc_indices)] += losses.to(torch.float64)
        else:
            assert row_index is not None
            ce_sums.index_add_(0, row_index, losses.to(torch.float64))
        for row_idx, doc_idx in enumerate(doc_indices):
            tokens_scored[doc_idx] += seq_len - 1 + boundary_token_count[row_idx]
            chunk_counts[doc_idx] += 1
            if final_states is not None:
                has_state[doc_idx] = True

    for chunk_start in range(0, max_token_len, chunk_size):
        active_count = 0
        full_count = 0
        full_end = chunk_start + chunk_size
        for token_len in token_lens:
            remaining = token_len - chunk_start
            if remaining < 2:
                break
            active_count += 1
            if token_len >= full_end:
                full_count += 1
        process_group(list(range(full_count)), chunk_start=chunk_start, seq_len=chunk_size)

        tail_groups: dict[int, list[int]] = {}
        for doc_idx in range(full_count, active_count):
            seq_len = token_lens[doc_idx] - chunk_start
            tail_groups.setdefault(seq_len, []).append(doc_idx)
        for seq_len, doc_indices in tail_groups.items():
            process_group(doc_indices, chunk_start=chunk_start, seq_len=seq_len)

    batch_wall_ms = (time.monotonic() - batch_t0) * 1000.0
    total_wall_weight = sum(chunk_counts)
    ce_values = ce_sums.cpu().tolist()
    if states is None:
        state_norm_values = [0.0 for _ in work_items]
    else:
        state_norm_tensor = torch.zeros(batch_size, dtype=torch.float64, device=device)
        for state in states:
            state_norm_tensor += state.norm(dim=1).to(torch.float64)
        state_norm_values = (state_norm_tensor / len(states)).cpu().tolist()
    scores = []
    for doc_idx, item in enumerate(work_items):
        ce_nats = float(ce_values[doc_idx])
        if total_wall_weight and chunk_counts[doc_idx]:
            measured_wall_ms = batch_wall_ms * chunk_counts[doc_idx] / total_wall_weight
        else:
            measured_wall_ms = 0.0
        scores.append(_DocScore(
            doc=item.doc,
            output_index=item.output_index,
            ce_nats=ce_nats,
            tokens_scored=tokens_scored[doc_idx],
            chunk_count=chunk_counts[doc_idx],
            loss_before=ce_nats / max(chunk_counts[doc_idx], 1),
            wall_ms=measured_wall_ms,
            state_norm=float(state_norm_values[doc_idx]),
        ))
    return scores


def run(args: argparse.Namespace) -> dict:
    if args.persistence_mode != "reset":
        raise NotImplementedError(
            "run_exp20_fast_score.py currently implements only persistence_mode='reset'"
        )

    ctx = resolve_distributed_context()
    rank = int(ctx["rank"])
    world_size = int(ctx["world_size"])
    local_rank = int(ctx["local_rank"])
    distributed = bool(ctx["distributed"])
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and args.device != "cpu" else "gloo"
        dist.init_process_group(backend=backend)

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    setup_t0 = time.monotonic()
    cache = load_val_cache(args.cache_dir)
    model, _ckpt_cfg = _build_model(args.checkpoint_path)
    model.to(device)
    model.eval()
    if args.torch_compile_mode != "none":
        compile_kwargs = {}
        if args.torch_compile_mode != "default":
            compile_kwargs["mode"] = args.torch_compile_mode
        model = torch.compile(model, **compile_kwargs)

    rank_output_path = _rank_output_path(args.output_path, rank, world_size)
    rank_output_path.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    budget = BudgetTracker(
        total_budget_seconds=args.budget_seconds,
        score_floor_seconds=0.0,
        safety_margin_seconds=args.safety_margin_seconds,
    )
    total_ce = 0.0
    total_raw_bytes = 0
    docs_scored = 0
    chunks_scored = 0
    tokens_scored = 0
    timed_out = False

    docs = list(cache.iter_docs())
    doc_start, doc_end = doc_range_for_rank(
        num_docs=len(docs),
        rank=rank,
        world_size=world_size,
    )

    rank_docs = docs[doc_start:doc_end]
    doc_work = prepare_doc_work(rank_docs, sort_by_length=args.sort_docs_by_length)
    device_tokens = torch.tensor(cache.tokens, dtype=torch.long, device=device)
    doc_records: list[str | None] = [None for _ in rank_docs]
    requested_doc_batch_size = max(1, int(args.doc_batch_size))
    max_forward_tokens = resolve_max_forward_tokens(
        max_forward_tokens=args.max_forward_tokens,
        requested_doc_batch_size=requested_doc_batch_size,
        chunk_size=args.chunk_size,
        device=device,
        probe_fn=(
            lambda limit: _probe_max_forward_tokens(
                model,
                requested_forward_tokens=limit,
                chunk_size=args.chunk_size,
                device=device,
            )
        ),
    )
    doc_batch_size = resolve_doc_batch_size(
        requested_doc_batch_size=requested_doc_batch_size,
        chunk_size=args.chunk_size,
        max_forward_tokens=max_forward_tokens,
    )
    score_warmup_seconds = _run_score_warmup(
        model,
        doc_batch_size=doc_batch_size,
        chunk_size=args.chunk_size,
        device=device,
        steps=args.score_warmup_steps,
    )
    pre_eval_setup_seconds = time.monotonic() - setup_t0
    run_t0 = time.monotonic()
    with torch.inference_mode():
        for batch_start in range(0, len(doc_work), doc_batch_size):
            if time.monotonic() - run_t0 > args.budget_seconds:
                timed_out = True
                break
            batch_work = doc_work[batch_start:batch_start + doc_batch_size]
            score_t0 = time.monotonic()
            if len(batch_work) == 1 or args.chunk_size < 0:
                scores = []
                for work_item in batch_work:
                    doc = work_item.doc
                    ce_nats, doc_tokens, doc_chunks, loss_before, wall_ms, state_norm = _score_doc(
                        model=model,
                        cache=cache,
                        doc=doc,
                        chunk_size=args.chunk_size,
                        device=device,
                        score_boundary_targets=args.score_boundary_targets,
                        device_tokens=device_tokens,
                    )
                    scores.append(_DocScore(
                        doc=doc,
                        output_index=work_item.output_index,
                        ce_nats=ce_nats,
                        tokens_scored=doc_tokens,
                        chunk_count=doc_chunks,
                        loss_before=loss_before,
                        wall_ms=wall_ms,
                        state_norm=state_norm,
                    ))
            else:
                scores = _score_docs_reset_batch(
                    model=model,
                    work_items=batch_work,
                    device_tokens=device_tokens,
                    chunk_size=args.chunk_size,
                    device=device,
                    score_boundary_targets=args.score_boundary_targets,
                )
            budget.add_score_time(time.monotonic() - score_t0)
            for score in scores:
                validate_doc_score_coverage(
                    score,
                    chunk_size=args.chunk_size,
                    score_boundary_targets=args.score_boundary_targets,
                )
                if score.tokens_scored <= 0:
                    continue
                doc_bpb = compute_bpb(score.ce_nats, score.doc.raw_bytes)
                total_ce += score.ce_nats
                total_raw_bytes += int(score.doc.raw_bytes)
                tokens_scored += int(score.tokens_scored)
                chunks_scored += int(score.chunk_count)
                docs_scored += 1
                doc_records[score.output_index] = json.dumps({
                    "doc_id": score.doc.doc_id,
                    "bpb": doc_bpb,
                    "tokens": score.tokens_scored,
                    "loss_before": score.loss_before,
                    "loss_after": None,
                    "step_count": 0,
                    "wall_ms": score.wall_ms,
                    "grad_norm": 0.0,
                    "state_norm": score.state_norm,
                })
            if time.monotonic() - run_t0 > args.budget_seconds and batch_start + doc_batch_size < len(doc_work):
                timed_out = True
                break

    with rank_output_path.open("w", encoding="utf-8") as out_fh:
        for record in doc_records:
            if record is not None:
                out_fh.write(record + "\n")

    elapsed = time.monotonic() - run_t0
    if distributed:
        sums = torch.tensor(
            [total_ce, float(total_raw_bytes), float(docs_scored), float(chunks_scored), float(tokens_scored)],
            device=device,
            dtype=torch.float64,
        )
        maxima = torch.tensor(
            [elapsed, 1.0 if timed_out else 0.0],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        total_ce = float(sums[0].item())
        total_raw_bytes = int(sums[1].item())
        docs_scored = int(sums[2].item())
        chunks_scored = int(sums[3].item())
        tokens_scored = int(sums[4].item())
        elapsed = float(maxima[0].item())
        timed_out = bool(maxima[1].item())
        budget.score_wall_seconds = elapsed

    max_docs = int(cache.manifest.get("max_docs", cache.num_docs))
    summary = budget.summary(
        docs_scored=docs_scored,
        chunks_scored=chunks_scored,
        tokens_scored=tokens_scored,
        adapt_steps=0,
        timed_out=timed_out,
        collapsed=False,
        score_only_mode=True,
        elapsed_seconds=elapsed,
        stream_seed=0,
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        chunk_size=args.chunk_size,
        max_docs=max_docs,
    )
    summary.update({
        "aggregate_bpb": compute_bpb(total_ce, total_raw_bytes),
        "aggregate_ce_nats": total_ce,
        "aggregate_raw_bytes": total_raw_bytes,
        "cache_dir": str(args.cache_dir),
        "persistence_mode": args.persistence_mode,
        "parallel_semantics": args.persistence_mode,
        "world_size": world_size,
        "rank": rank,
        "rank_doc_start": doc_start,
        "rank_doc_end": doc_end,
        "rank_output_path": str(rank_output_path),
        "doc_batch_size": doc_batch_size,
        "requested_doc_batch_size": requested_doc_batch_size,
        "max_forward_tokens": int(max_forward_tokens),
        "max_forward_tokens_request": str(args.max_forward_tokens),
        "max_batch_tokens": int(max_forward_tokens),
        "score_boundary_targets": bool(args.score_boundary_targets),
        "doc_ordering": "token_len_desc" if args.sort_docs_by_length else "source_order",
        "device_tokens_staged": True,
        "device_token_dtype": str(device_tokens.dtype),
        "torch_compile_mode": args.torch_compile_mode,
        "score_warmup_steps": int(args.score_warmup_steps),
        "score_warmup_seconds": score_warmup_seconds,
        "pre_eval_setup_seconds": pre_eval_setup_seconds,
    })
    if world_size == 4:
        summary["projected_8x_wall_seconds"] = elapsed * 4.0 / 8.0
    if rank == 0:
        args.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"docs_scored={docs_scored} tokens_scored={tokens_scored} "
            f"aggregate_bpb={summary['aggregate_bpb']:.8f} elapsed_seconds={elapsed:.3f}"
        )
    if distributed:
        dist.destroy_process_group()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument("--safety-margin-seconds", type=float, default=0.0)
    parser.add_argument("--persistence-mode", choices=["reset"], default="reset")
    parser.add_argument("--doc-batch-size", type=int, default=1)
    parser.add_argument(
        "--max-forward-tokens",
        default="auto",
        help=(
            "Maximum fixed-shape token positions per forward, or 'auto' to probe "
            "the requested CUDA shape and back off on OOM. The deprecated --max-batch-tokens "
            "spelling is accepted as an alias."
        ),
    )
    parser.add_argument("--max-batch-tokens", dest="max_forward_tokens", help=argparse.SUPPRESS)
    parser.add_argument("--score-boundary-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sort-docs-by-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--torch-compile-mode",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        default="none",
    )
    parser.add_argument("--score-warmup-steps", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
