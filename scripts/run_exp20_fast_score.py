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
from pathlib import Path
from typing import NamedTuple

import numpy as np

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


class _DocScore(NamedTuple):
    doc: CachedDoc
    ce_nats: float
    tokens_scored: int
    chunk_count: int
    loss_before: float
    wall_ms: float
    state_norm: float


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
) -> tuple[float, int, int, float, float, float]:
    prev_states: list[torch.Tensor] | None = None
    doc_ce = torch.zeros((), device=device, dtype=torch.float64)
    tokens_scored = 0
    chunk_count = 0
    t0 = time.monotonic()
    doc_tokens_np = cache.tokens_for_doc(doc)

    for start, end in _token_chunk_ranges(int(doc.token_len), chunk_size):
        chunk_np = doc_tokens_np[start:end]
        if len(chunk_np) < 2:
            continue
        chunk = torch.tensor(chunk_np, dtype=torch.long, device=device).unsqueeze(0)
        kwargs = {"initial_states": prev_states} if prev_states else {}
        out = model(chunk, **kwargs)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        )
        if score_boundary_targets and end < int(doc.token_len):
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
        if isinstance(out, dict) and out.get("final_states"):
            prev_states = list(out["final_states"])

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
    cache: ValCache,
    docs: list[CachedDoc],
    chunk_size: int,
    device: torch.device,
    score_boundary_targets: bool,
) -> list[_DocScore]:
    """Score a document batch while preserving reset-mode semantics.

    Each document owns its own recurrent state trajectory. We only batch
    together chunks with the same true length, which avoids padding tokens from
    contaminating final-state diagnostics or future carried chunks.
    """
    if not docs:
        return []
    batch_t0 = time.monotonic()
    token_arrays = [cache.tokens_for_doc(doc) for doc in docs]
    chunk_ranges = [
        _token_chunk_ranges(int(doc.token_len), chunk_size)
        for doc in docs
    ]
    progress = [0 for _ in docs]
    states_by_doc: list[list[torch.Tensor] | None] = [None for _ in docs]
    ce_sums = torch.zeros(len(docs), device=device, dtype=torch.float64)
    tokens_scored = [0 for _ in docs]
    chunk_counts = [0 for _ in docs]
    wall_ms = [0.0 for _ in docs]

    while True:
        groups: dict[int, list[int]] = {}
        for doc_idx, ranges in enumerate(chunk_ranges):
            if progress[doc_idx] >= len(ranges):
                continue
            start, end = ranges[progress[doc_idx]]
            groups.setdefault(end - start, []).append(doc_idx)
        if not groups:
            break

        for seq_len, doc_indices in groups.items():
            group_t0 = time.monotonic()
            rows = []
            for doc_idx in doc_indices:
                start, end = chunk_ranges[doc_idx][progress[doc_idx]]
                rows.append(token_arrays[doc_idx][start:end])
            chunk_np = np.stack(rows, axis=0)
            chunk = torch.tensor(chunk_np, dtype=torch.long, device=device)

            prev_states = states_by_doc[doc_indices[0]]
            if prev_states is None:
                initial_states = None
            else:
                initial_states = [
                    torch.cat(
                        [
                            states_by_doc[doc_idx][layer_idx]  # type: ignore[index]
                            for doc_idx in doc_indices
                        ],
                        dim=0,
                    )
                    for layer_idx in range(len(prev_states))
                ]
            kwargs = {"initial_states": initial_states} if initial_states is not None else {}
            out = model(chunk, **kwargs)
            logits = out["logits"] if isinstance(out, dict) else out
            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                chunk[:, 1:].reshape(-1),
                reduction="none",
            ).reshape(len(doc_indices), seq_len - 1).sum(dim=1)
            boundary_token_count = [0 for _ in doc_indices]
            if score_boundary_targets:
                boundary_targets: list[int] = []
                boundary_rows: list[int] = []
                for row_idx, doc_idx in enumerate(doc_indices):
                    _start, end = chunk_ranges[doc_idx][progress[doc_idx]]
                    if end < int(docs[doc_idx].token_len):
                        boundary_rows.append(row_idx)
                        boundary_targets.append(int(token_arrays[doc_idx][end]))
                        boundary_token_count[row_idx] = 1
                if boundary_rows:
                    row_index = torch.tensor(boundary_rows, dtype=torch.long, device=device)
                    targets = torch.tensor(boundary_targets, dtype=torch.long, device=device)
                    boundary_losses = F.cross_entropy(
                        logits.index_select(0, row_index)[:, -1, :],
                        targets,
                        reduction="none",
                    )
                    losses.index_add_(0, row_index, boundary_losses)

            final_states = (
                list(out["final_states"])
                if isinstance(out, dict) and out.get("final_states")
                else None
            )
            group_wall_ms = (time.monotonic() - group_t0) * 1000.0
            wall_share_ms = group_wall_ms / max(len(doc_indices), 1)
            for row_idx, doc_idx in enumerate(doc_indices):
                ce_sums[doc_idx] += losses[row_idx].to(torch.float64)
                tokens_scored[doc_idx] += seq_len - 1 + boundary_token_count[row_idx]
                chunk_counts[doc_idx] += 1
                progress[doc_idx] += 1
                wall_ms[doc_idx] += wall_share_ms
                if final_states is not None:
                    states_by_doc[doc_idx] = [
                        state[row_idx:row_idx + 1]
                        for state in final_states
                    ]

    batch_wall_ms = (time.monotonic() - batch_t0) * 1000.0
    total_wall_weight = sum(chunk_counts)
    scores = []
    for doc_idx, doc in enumerate(docs):
        states = states_by_doc[doc_idx]
        state_norm = 0.0
        if states:
            state_norm = sum(float(s.norm().item()) for s in states) / len(states)
        ce_nats = float(ce_sums[doc_idx].item())
        if total_wall_weight and chunk_counts[doc_idx]:
            measured_wall_ms = batch_wall_ms * chunk_counts[doc_idx] / total_wall_weight
        else:
            measured_wall_ms = wall_ms[doc_idx]
        scores.append(_DocScore(
            doc=doc,
            ce_nats=ce_nats,
            tokens_scored=tokens_scored[doc_idx],
            chunk_count=chunk_counts[doc_idx],
            loss_before=ce_nats / max(chunk_counts[doc_idx], 1),
            wall_ms=measured_wall_ms,
            state_norm=state_norm,
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

    cache = load_val_cache(args.cache_dir)
    model, _ckpt_cfg = _build_model(args.checkpoint_path)
    model.to(device)
    model.eval()

    rank_output_path = _rank_output_path(args.output_path, rank, world_size)
    rank_output_path.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    budget = BudgetTracker(
        total_budget_seconds=args.budget_seconds,
        score_floor_seconds=0.0,
        safety_margin_seconds=args.safety_margin_seconds,
    )
    run_t0 = time.monotonic()
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
    doc_batch_size = max(1, int(args.doc_batch_size))
    with rank_output_path.open("w", encoding="utf-8") as out_fh, torch.inference_mode():
        for batch_start in range(0, len(rank_docs), doc_batch_size):
            if time.monotonic() - run_t0 > args.budget_seconds:
                timed_out = True
                break
            batch_docs = rank_docs[batch_start:batch_start + doc_batch_size]
            score_t0 = time.monotonic()
            if len(batch_docs) == 1:
                doc = batch_docs[0]
                ce_nats, doc_tokens, doc_chunks, loss_before, wall_ms, state_norm = _score_doc(
                    model=model,
                    cache=cache,
                    doc=doc,
                    chunk_size=args.chunk_size,
                    device=device,
                    score_boundary_targets=args.score_boundary_targets,
                )
                scores = [_DocScore(
                    doc=doc,
                    ce_nats=ce_nats,
                    tokens_scored=doc_tokens,
                    chunk_count=doc_chunks,
                    loss_before=loss_before,
                    wall_ms=wall_ms,
                    state_norm=state_norm,
                )]
            else:
                scores = _score_docs_reset_batch(
                    model=model,
                    cache=cache,
                    docs=batch_docs,
                    chunk_size=args.chunk_size,
                    device=device,
                    score_boundary_targets=args.score_boundary_targets,
                )
            budget.add_score_time(time.monotonic() - score_t0)
            for score in scores:
                if score.tokens_scored <= 0:
                    continue
                doc_bpb = compute_bpb(score.ce_nats, score.doc.raw_bytes)
                total_ce += score.ce_nats
                total_raw_bytes += int(score.doc.raw_bytes)
                tokens_scored += int(score.tokens_scored)
                chunks_scored += int(score.chunk_count)
                docs_scored += 1
                out_fh.write(json.dumps({
                    "doc_id": score.doc.doc_id,
                    "bpb": doc_bpb,
                    "tokens": score.tokens_scored,
                    "loss_before": score.loss_before,
                    "loss_after": None,
                    "step_count": 0,
                    "wall_ms": score.wall_ms,
                    "grad_norm": 0.0,
                    "state_norm": score.state_norm,
                }) + "\n")
            out_fh.flush()
            if time.monotonic() - run_t0 > args.budget_seconds and batch_start + doc_batch_size < len(rank_docs):
                timed_out = True
                break

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
        "score_boundary_targets": bool(args.score_boundary_targets),
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
    parser.add_argument("--score-boundary-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
