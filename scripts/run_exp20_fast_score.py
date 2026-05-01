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
from chaoscontrol.eval_stream.legality import LegalityController
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


class _ScoreGraphStats:
    def __init__(self) -> None:
        self.replay_count = 0
        self.fallback_count = 0


def chunk_count_for_doc(doc: CachedDoc, *, chunk_size: int) -> int:
    if chunk_size < 0:
        return 1 if int(doc.token_len) >= 2 else 0
    return max(1, (int(doc.token_len) + int(chunk_size) - 1) // int(chunk_size))


def padded_token_work(doc: CachedDoc, *, chunk_size: int) -> int:
    if chunk_size < 0:
        return int(doc.token_len)
    return chunk_count_for_doc(doc, chunk_size=chunk_size) * int(chunk_size)


def resolve_doc_packing(
    *,
    doc_packing: str | None,
    sort_docs_by_length: bool | None,
) -> str:
    if doc_packing:
        return doc_packing
    if sort_docs_by_length is False:
        return "source_order"
    if sort_docs_by_length is True:
        return "token_len_desc"
    return "chunk_count_tail"


def _doc_packing_key(item: _DocWork, *, doc_packing: str, chunk_size: int) -> tuple[int, ...]:
    n_tokens = int(item.doc.token_len)
    n_chunks = chunk_count_for_doc(item.doc, chunk_size=chunk_size)
    if chunk_size < 0:
        tail = 0
        tail_bucket = 0
    else:
        tail = n_tokens % int(chunk_size)
        tail_bucket = tail // min(32, max(1, int(chunk_size)))
    if doc_packing == "source_order":
        return (item.output_index,)
    if doc_packing == "token_len_desc":
        return (-n_tokens, item.output_index)
    if doc_packing == "chunk_count_desc":
        return (-n_chunks, -n_tokens, item.output_index)
    if doc_packing == "chunk_count_tail":
        return (-n_chunks, tail_bucket, -n_tokens, item.output_index)
    raise ValueError(f"unsupported doc_packing: {doc_packing}")


def prepare_doc_work(
    docs: list[CachedDoc],
    *,
    sort_by_length: bool | None = None,
    doc_packing: str | None = None,
    chunk_size: int = 256,
    start_index: int = 0,
) -> list[_DocWork]:
    packing = resolve_doc_packing(
        doc_packing=doc_packing,
        sort_docs_by_length=sort_by_length,
    )
    work = [
        _DocWork(doc=doc, output_index=start_index + idx)
        for idx, doc in enumerate(docs)
    ]
    if packing != "source_order":
        return sorted(work, key=lambda item: _doc_packing_key(
            item,
            doc_packing=packing,
            chunk_size=chunk_size,
        ))
    return work


def prepare_rank_doc_work(
    docs: list[CachedDoc],
    *,
    rank: int,
    world_size: int,
    doc_packing: str,
    chunk_size: int,
    doc_batch_size: int,
) -> list[_DocWork]:
    if doc_packing == "source_order":
        start, end = doc_range_for_rank(num_docs=len(docs), rank=rank, world_size=world_size)
        return prepare_doc_work(
            docs[start:end],
            doc_packing="source_order",
            chunk_size=chunk_size,
            start_index=start,
        )
    work = prepare_doc_work(docs, doc_packing=doc_packing, chunk_size=chunk_size)
    if world_size <= 1:
        return work

    batch_size = max(1, int(doc_batch_size))
    batches = [work[idx:idx + batch_size] for idx in range(0, len(work), batch_size)]
    batches.sort(
        key=lambda batch: sum(padded_token_work(item.doc, chunk_size=chunk_size) for item in batch),
        reverse=True,
    )
    rank_loads = [0 for _ in range(world_size)]
    rank_batches: list[list[_DocWork]] = [[] for _ in range(world_size)]
    for batch in batches:
        target_rank = min(range(world_size), key=lambda idx: (rank_loads[idx], idx))
        rank_batches[target_rank].extend(batch)
        rank_loads[target_rank] += sum(
            padded_token_work(item.doc, chunk_size=chunk_size)
            for item in batch
        )
    return rank_batches[rank]


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


def record_order_safe_reason(*, persistence_mode: str, score_only_mode: bool, doc_packing: str) -> str:
    if doc_packing == "source_order":
        return "source_order_preserved"
    if persistence_mode == "reset" and score_only_mode:
        return "reset_score_only_commutative_ce_reduction"
    return "not_order_safe_for_stateful_or_adaptive_eval"


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


def _fast_score_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1).to(torch.long),
        reduction="sum",
    )


_ONLINE_REPLAY_EVAL_ERROR = (
    "This checkpoint carries CRCT replay-eviction online state, but "
    "run_exp20_fast_score.py does not run the CPU control plane + GPU3 memory "
    "oracle. Use the distributed fast-path eval/runner for this artifact."
)


def _requires_online_replay_eval(blob: dict, cfg: dict) -> bool:
    online_eval_state = blob.get("online_eval_state")
    return bool(cfg.get("replay_eviction_enabled", False)) or (
        isinstance(online_eval_state, dict)
        and isinstance(online_eval_state.get("replay_eviction"), dict)
    )


def _build_model_with_blob(
    ckpt_path: Path,
    *,
    allow_online_replay_checkpoint: bool = False,
) -> tuple[torch.nn.Module, dict, dict]:
    from chaoscontrol.model import CareStudentLM

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    if (
        _requires_online_replay_eval(blob, cfg)
        and not bool(allow_online_replay_checkpoint)
    ):
        raise RuntimeError(_ONLINE_REPLAY_EVAL_ERROR)
    model = CareStudentLM(**cfg)
    model.load_state_dict(blob["model"], strict=True)
    online_eval_state = blob.get("online_eval_state")
    if isinstance(online_eval_state, dict):
        model._online_eval_state = online_eval_state
    return model, cfg, blob


def _build_model(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    model, cfg, _blob = _build_model_with_blob(ckpt_path)
    return model, cfg


def _load_episodic_cache_from_ckpt(blob: dict):
    """Construct an EpisodicCache from a checkpoint payload when present."""
    from chaoscontrol.optim.episodic_cache import EpisodicCache

    payload = blob.get("episodic_cache")
    if payload is None:
        return None
    return EpisodicCache.from_dict(payload)


def _make_fresh_episodic_cache(args: argparse.Namespace, model_dim: int):
    """Build the train-no-cache / eval-fresh-cache fallback shape."""
    from chaoscontrol.optim.episodic_cache import EpisodicCache

    key_rep_dim = (
        int(model_dim)
        if int(args.episodic_key_rep_dim) == -1
        else int(args.episodic_key_rep_dim)
    )
    return EpisodicCache(
        capacity=int(args.episodic_cache_capacity),
        span_length=int(args.episodic_span_length),
        key_rep_dim=key_rep_dim,
        grace_steps=int(args.episodic_grace_steps),
        fingerprint_window=int(args.episodic_fingerprint_window),
    )


def _build_legality_controller(
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    ckpt_cfg: dict,
    ckpt_blob: dict,
) -> tuple[LegalityController, object | None, str]:
    """Mirror run_exp20_eval.py's eval-side episodic cache construction."""
    episodic_cache = None
    cache_source = "disabled"
    requested_source = str(getattr(args, "episodic_cache_source", "auto"))
    if bool(args.episodic_cache_enabled):
        model_dim = int(ckpt_cfg["dim"])
        if requested_source == "fresh":
            episodic_cache = _make_fresh_episodic_cache(args, model_dim)
            cache_source = "fresh_forced"
        elif requested_source == "checkpoint":
            episodic_cache = _load_episodic_cache_from_ckpt(ckpt_blob)
            if episodic_cache is None:
                raise RuntimeError(
                    "--episodic-cache-source=checkpoint requires "
                    "ckpt['episodic_cache']; payload absent. The trainer "
                    "must serialize the cache (F1 warm-cache arms depend "
                    "on this contract)."
                )
            cache_source = "loaded"
        else:
            episodic_cache = _load_episodic_cache_from_ckpt(ckpt_blob)
            if episodic_cache is None:
                episodic_cache = _make_fresh_episodic_cache(args, model_dim)
                cache_source = "fresh"
            else:
                cache_source = "loaded"

    if episodic_cache is not None:
        controller_fp_window = int(episodic_cache.fingerprint_window)
    else:
        controller_fp_window = int(args.episodic_fingerprint_window)
    controller = LegalityController(
        model,
        loss_fn=_fast_score_ce,
        cache=episodic_cache,
        fingerprint_window=controller_fp_window,
    )
    return controller, episodic_cache, cache_source


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


class _CudaGraphScoreRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        score_boundary_targets: bool,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("--score-graph-mode cuda requires CUDA")
        if batch_size <= 0 or seq_len < 2:
            raise ValueError("CUDA graph scoring requires batch_size > 0 and seq_len >= 2")

        self.model = model
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.device = device
        self.score_boundary_targets = bool(score_boundary_targets)
        self.static_chunk = torch.empty(
            (self.batch_size, self.seq_len),
            dtype=torch.long,
            device=device,
        )
        self.static_boundary_targets = torch.empty(
            self.batch_size,
            dtype=torch.long,
            device=device,
        )
        self.static_boundary_mask = torch.empty(
            self.batch_size,
            dtype=torch.float32,
            device=device,
        )
        self.static_chunk.zero_()
        self.static_boundary_targets.zero_()
        self.static_boundary_mask.fill_(1.0)
        self.static_prev_states = self._make_static_prev_states()
        self.graph = torch.cuda.CUDAGraph()
        self.zero_graph = torch.cuda.CUDAGraph()
        self.static_losses: torch.Tensor | None = None
        self.static_final_states: list[torch.Tensor] | None = None
        self.static_zero_losses: torch.Tensor | None = None
        self.static_zero_final_states: list[torch.Tensor] | None = None
        self._capture()

    def _make_static_prev_states(self) -> list[torch.Tensor]:
        with torch.inference_mode():
            out = self.model(self.static_chunk)
            final_states = _model_final_states(out)
            if not final_states:
                raise RuntimeError("CUDA graph scoring requires model final_states")
            prev_states = [torch.empty_like(state) for state in final_states]
            for state in prev_states:
                # Capture only needs stable input addresses; replay overwrites
                # these tensors before use. Zero-fill avoids undefined warmup data.
                state.zero_()
            del out, final_states
        return prev_states

    def _forward_static(
        self,
        *,
        initial_states: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        kwargs = {"initial_states": initial_states} if initial_states is not None else {}
        out = self.model(self.static_chunk, **kwargs)
        logits = _model_logits(out)
        losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            self.static_chunk[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(self.batch_size, self.seq_len - 1).sum(dim=1)
        if self.score_boundary_targets:
            boundary_losses = F.cross_entropy(
                logits[:, -1, :],
                self.static_boundary_targets,
                reduction="none",
            )
            losses = losses + boundary_losses * self.static_boundary_mask.to(boundary_losses.dtype)
        final_states = _model_final_states(out)
        if not final_states:
            raise RuntimeError("CUDA graph scoring requires model final_states")
        return losses, final_states

    def _capture(self) -> None:
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                losses, final_states = self._forward_static(initial_states=None)
                del losses, final_states
                losses, final_states = self._forward_static(initial_states=self.static_prev_states)
                del losses, final_states
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        with torch.cuda.graph(self.zero_graph), torch.inference_mode():
            self.static_zero_losses, self.static_zero_final_states = self._forward_static(
                initial_states=None,
            )
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_losses, self.static_final_states = self._forward_static(
                initial_states=self.static_prev_states,
            )

    def replay_zero_initial(
        self,
        *,
        chunk: torch.Tensor,
        boundary_targets: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if tuple(chunk.shape) != tuple(self.static_chunk.shape):
            raise ValueError(f"graph chunk shape {tuple(chunk.shape)} != {tuple(self.static_chunk.shape)}")
        self.static_chunk.copy_(chunk, non_blocking=True)
        if self.score_boundary_targets:
            self.static_boundary_targets.copy_(boundary_targets, non_blocking=True)
            self.static_boundary_mask.copy_(boundary_mask, non_blocking=True)
        self.zero_graph.replay()
        if self.static_zero_losses is None or self.static_zero_final_states is None:
            raise RuntimeError("CUDA zero-state graph replay used before capture")
        return self.static_zero_losses, self.static_zero_final_states

    def replay(
        self,
        *,
        chunk: torch.Tensor,
        initial_states: list[torch.Tensor],
        boundary_targets: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if tuple(chunk.shape) != tuple(self.static_chunk.shape):
            raise ValueError(f"graph chunk shape {tuple(chunk.shape)} != {tuple(self.static_chunk.shape)}")
        if len(initial_states) != len(self.static_prev_states):
            raise ValueError("graph initial state count mismatch")
        self.static_chunk.copy_(chunk, non_blocking=True)
        for static_state, state in zip(self.static_prev_states, initial_states):
            static_state.copy_(state, non_blocking=True)
        if self.score_boundary_targets:
            self.static_boundary_targets.copy_(boundary_targets, non_blocking=True)
            self.static_boundary_mask.copy_(boundary_mask, non_blocking=True)
        self.graph.replay()
        if self.static_losses is None or self.static_final_states is None:
            raise RuntimeError("CUDA graph replay used before capture")
        return self.static_losses, self.static_final_states


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
            chunk[:, 1:].reshape(-1).to(torch.long),
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
                next_token.to(torch.long),
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
    score_graph_runner: _CudaGraphScoreRunner | None = None,
    graph_stats: _ScoreGraphStats | None = None,
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
    token_lens_tensor = torch.tensor(token_lens, dtype=torch.long, device=device)
    ce_sums = torch.zeros(batch_size, device=device, dtype=torch.float64)
    tokens_scored = [0 for _ in work_items]
    chunk_counts = [0 for _ in work_items]
    states: list[torch.Tensor] | None = None
    has_state = [False for _ in work_items]
    arange_cache: dict[int, torch.Tensor] = {}
    row_index_cache: dict[tuple[int, ...], torch.Tensor] = {}
    max_token_len = max(token_lens)

    def arange_for(length: int) -> torch.Tensor:
        values = arange_cache.get(length)
        if values is None:
            values = torch.arange(length, dtype=torch.long, device=device)
            arange_cache[length] = values
        return values

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
        # LOAD-BEARING: work_items is sorted by (-token_len, output_index), so
        # full-width active docs are always the leading rows for a chunk step.
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

        boundary_token_count = [0 for _ in doc_indices]
        graph_boundary_targets: torch.Tensor | None = None
        graph_boundary_mask: torch.Tensor | None = None
        if score_boundary_targets:
            boundary_rows: list[int] = []
            boundary_doc_indices: list[int] = []
            next_token_pos = chunk_start + seq_len
            for row_idx, doc_idx in enumerate(doc_indices):
                if next_token_pos < token_lens[doc_idx]:
                    boundary_rows.append(row_idx)
                    boundary_doc_indices.append(doc_idx)
                    boundary_token_count[row_idx] = 1
            if score_graph_runner is not None and len(doc_indices) == score_graph_runner.batch_size:
                if row_index is None:
                    doc_index_tensor = arange_for(len(doc_indices))
                else:
                    doc_index_tensor = row_index
                has_boundary = token_lens_tensor.index_select(0, doc_index_tensor) > next_token_pos
                safe_positions = torch.where(
                    has_boundary,
                    token_starts.index_select(0, doc_index_tensor) + next_token_pos,
                    token_starts.index_select(0, doc_index_tensor),
                )
                graph_boundary_targets = device_tokens[safe_positions]
                graph_boundary_mask = has_boundary.to(torch.float32)

        chunk = device_tokens[starts.unsqueeze(1) + arange_for(seq_len).unsqueeze(0)]
        can_replay_graph = (
            score_graph_runner is not None
            and contiguous_prefix
            and len(doc_indices) == score_graph_runner.batch_size
            and seq_len == score_graph_runner.seq_len
        )
        if can_replay_graph:
            if graph_boundary_targets is None:
                graph_boundary_targets = torch.zeros(len(doc_indices), dtype=torch.long, device=device)
                graph_boundary_mask = torch.zeros(len(doc_indices), dtype=torch.float32, device=device)
            assert graph_boundary_mask is not None
            if initial_states is None:
                losses, final_states = score_graph_runner.replay_zero_initial(
                    chunk=chunk,
                    boundary_targets=graph_boundary_targets,
                    boundary_mask=graph_boundary_mask,
                )
            else:
                losses, final_states = score_graph_runner.replay(
                    chunk=chunk,
                    initial_states=initial_states,
                    boundary_targets=graph_boundary_targets,
                    boundary_mask=graph_boundary_mask,
                )
            if graph_stats is not None:
                graph_stats.replay_count += 1
        else:
            if graph_stats is not None:
                graph_stats.fallback_count += 1
            kwargs = {"initial_states": initial_states} if initial_states is not None else {}
            out = model(chunk, **kwargs)
            logits = _model_logits(out)
            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                chunk[:, 1:].reshape(-1).to(torch.long),
                reduction="none",
            ).reshape(len(doc_indices), seq_len - 1).sum(dim=1)
            if score_boundary_targets and boundary_doc_indices:
                boundary_row_index = torch.tensor(boundary_rows, dtype=torch.long, device=device)
                boundary_docs = row_index_for(boundary_doc_indices)
                targets = device_tokens[token_starts.index_select(0, boundary_docs) + next_token_pos].to(torch.long)
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
    if args.score_graph_mode == "cuda" and device.type != "cuda":
        raise RuntimeError("--score-graph-mode cuda requires CUDA")

    setup_t0 = time.monotonic()
    cache = load_val_cache(args.cache_dir)
    model, ckpt_cfg, ckpt_blob = _build_model_with_blob(
        args.checkpoint_path,
        allow_online_replay_checkpoint=bool(
            args.allow_online_replay_checkpoint
        ),
    )
    model.to(device)
    model.eval()
    if args.torch_compile_mode != "none":
        compile_kwargs = {}
        if args.torch_compile_mode != "default":
            compile_kwargs["mode"] = args.torch_compile_mode
        model = torch.compile(model, **compile_kwargs)
    legality_controller, episodic_cache, episodic_cache_source = (
        _build_legality_controller(
            model,
            args=args,
            ckpt_cfg=ckpt_cfg,
            ckpt_blob=ckpt_blob,
        )
    )
    if episodic_cache_source in {"fresh", "fresh_forced"}:
        forced = " (forced via --episodic-cache-source=fresh)" if (
            episodic_cache_source == "fresh_forced"
        ) else ""
        print(
            "[exp20_fast_score] episodic_cache: fresh empty cache"
            f"{forced} (capacity={episodic_cache.capacity}, "
            f"span_length={episodic_cache.span_length}, "
            f"key_rep_dim={episodic_cache.key_rep_dim})",
            flush=True,
        )
    elif episodic_cache_source == "loaded":
        print(
            "[exp20_fast_score] episodic_cache: loaded from checkpoint "
            f"(capacity={episodic_cache.capacity}, "
            f"occupied={int(episodic_cache.occupied.sum().item())})",
            flush=True,
        )
    if bool(getattr(args, "controller_train_online", False)):
        print(
            "[exp20_fast_score] WARNING controller_train_online=True but "
            "the CPU SSM controller's online learning loop is not yet "
            "wired into the eval path; this run is equivalent to "
            "trained-frozen until that lands.",
            flush=True,
        )
    if episodic_cache_source in {"fresh_forced", "loaded"}:
        # Pre-existing fast-score limitation (not introduced by the F1
        # eval-arg wiring): the optimized scoring loop calls _score_doc /
        # _score_docs_reset_batch directly and bypasses the cache-aware
        # LegalityController path. The episodic_cache is constructed,
        # ckpt-loaded if requested, and ticked via mark_new_epoch /
        # reset, but it does NOT affect CE numbers. Treat F1's cold/warm
        # distinction as metadata-only here; cache-aware eval is the
        # run_exp20_eval.py path, not this fast scorer.
        print(
            "[exp20_fast_score] WARNING --episodic-cache-source="
            f"{getattr(args, 'episodic_cache_source', 'auto')!r} loaded "
            "the cache, but this fast-score path's CE hot path bypasses "
            "the cache-aware controller. cold/warm differs only in "
            "summary metadata, not in scored CE.",
            flush=True,
        )

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
    doc_packing = resolve_doc_packing(
        doc_packing=args.doc_packing,
        sort_docs_by_length=args.sort_docs_by_length,
    )
    order_safe_reason = record_order_safe_reason(
        persistence_mode=args.persistence_mode,
        score_only_mode=True,
        doc_packing=doc_packing,
    )
    if order_safe_reason == "not_order_safe_for_stateful_or_adaptive_eval":
        raise RuntimeError(
            f"doc_packing={doc_packing!r} is not order-safe for persistence_mode={args.persistence_mode!r}"
        )
    device_tokens = torch.tensor(cache.tokens, dtype=torch.int32, device=device)
    doc_records: dict[int, str] = {}
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
    doc_work = prepare_rank_doc_work(
        docs,
        rank=rank,
        world_size=world_size,
        doc_packing=doc_packing,
        chunk_size=args.chunk_size,
        doc_batch_size=doc_batch_size,
    )
    if doc_work:
        rank_doc_start = min(item.output_index for item in doc_work)
        rank_doc_end = max(item.output_index for item in doc_work) + 1
    else:
        rank_doc_start = 0
        rank_doc_end = 0
    score_warmup_seconds = _run_score_warmup(
        model,
        doc_batch_size=doc_batch_size,
        chunk_size=args.chunk_size,
        device=device,
        steps=args.score_warmup_steps,
    )
    graph_stats = _ScoreGraphStats() if args.score_graph_mode == "cuda" else None
    score_graph_runner = None
    if args.score_graph_mode == "cuda" and doc_batch_size > 1 and args.chunk_size >= 2:
        score_graph_runner = _CudaGraphScoreRunner(
            model,
            batch_size=doc_batch_size,
            seq_len=args.chunk_size,
            device=device,
            score_boundary_targets=args.score_boundary_targets,
        )
    pre_eval_setup_seconds = time.monotonic() - setup_t0
    run_t0 = time.monotonic()
    with torch.inference_mode():
        for batch_start in range(0, len(doc_work), doc_batch_size):
            if time.monotonic() - run_t0 > args.budget_seconds:
                timed_out = True
                break
            batch_work = doc_work[batch_start:batch_start + doc_batch_size]
            # Keep the cache-aware LegalityController's per-doc scratch state
            # aligned with reset-mode scoring. The optimized scorer below owns
            # the CE hot path; the controller construction is still load-bearing
            # because cache-aware eval arms must load the checkpoint cache and
            # carry the trainer's fingerprint window into the eval substrate.
            if episodic_cache is not None:
                for work_item in batch_work:
                    legality_controller.mark_new_epoch()
                    if args.episodic_cache_reset_per_doc:
                        episodic_cache.reset()
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
                    score_graph_runner=score_graph_runner,
                    graph_stats=graph_stats,
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
        for _doc_idx, record in sorted(doc_records.items()):
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
        "rank_doc_start": rank_doc_start,
        "rank_doc_end": rank_doc_end,
        "rank_doc_count": len(doc_work),
        "rank_output_path": str(rank_output_path),
        "doc_batch_size": doc_batch_size,
        "requested_doc_batch_size": requested_doc_batch_size,
        "max_forward_tokens": int(max_forward_tokens),
        "max_forward_tokens_request": str(args.max_forward_tokens),
        "max_batch_tokens": int(max_forward_tokens),
        "score_boundary_targets": bool(args.score_boundary_targets),
        "doc_ordering": doc_packing,
        "doc_packing": doc_packing,
        "record_order_safe": order_safe_reason != "not_order_safe_for_stateful_or_adaptive_eval",
        "record_order_safe_reason": order_safe_reason,
        "score_reduction_order_invariant": args.persistence_mode == "reset",
        "rank_assignment": (
            "contiguous_source_range"
            if doc_packing == "source_order"
            else ("single_rank" if world_size == 1 else "lpt_padded_tokens")
        ),
        "device_tokens_staged": True,
        "device_token_dtype": str(device_tokens.dtype),
        "torch_compile_mode": args.torch_compile_mode,
        "score_warmup_steps": int(args.score_warmup_steps),
        "score_warmup_seconds": score_warmup_seconds,
        "pre_eval_setup_seconds": pre_eval_setup_seconds,
        "score_graph_mode": args.score_graph_mode,
        "score_graph_enabled": score_graph_runner is not None,
        "graph_replay_count": graph_stats.replay_count if graph_stats is not None else 0,
        "graph_fallback_count": graph_stats.fallback_count if graph_stats is not None else 0,
        "episodic_cache_enabled": bool(args.episodic_cache_enabled),
        "episodic_cache_source": episodic_cache_source,
        "episodic_cache_capacity": (
            int(episodic_cache.capacity) if episodic_cache is not None else None
        ),
        "episodic_cache_occupied": (
            int(episodic_cache.occupied.sum().item())
            if episodic_cache is not None
            else None
        ),
        "episodic_fingerprint_window": int(
            legality_controller.fingerprint_window
        ),
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
        "--doc-packing",
        choices=["source_order", "token_len_desc", "chunk_count_desc", "chunk_count_tail"],
        default=None,
        help=(
            "Document scheduling policy. Defaults to chunk_count_tail; "
            "--no-sort-docs-by-length remains an alias for source_order."
        ),
    )
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
    parser.add_argument("--sort-docs-by-length", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--torch-compile-mode",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        default="none",
    )
    parser.add_argument("--score-warmup-steps", type=int, default=0)
    parser.add_argument("--score-graph-mode", choices=["none", "cuda"], default="none")
    parser.add_argument("--episodic-cache-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--episodic-cache-reset-per-doc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--episodic-cache-capacity", type=int, default=4096)
    parser.add_argument("--episodic-span-length", type=int, default=4)
    parser.add_argument("--episodic-key-rep-dim", type=int, default=-1)
    parser.add_argument("--episodic-grace-steps", type=int, default=1000)
    parser.add_argument("--episodic-fingerprint-window", type=int, default=8)
    parser.add_argument(
        "--episodic-cache-source",
        choices=("auto", "fresh", "checkpoint"),
        default="auto",
        help=(
            "auto = load from checkpoint if present else fresh (current "
            "default); fresh = force-fresh regardless of checkpoint payload "
            "(F1 cold-cache arms); checkpoint = require the payload, error "
            "if absent (F1 warm-cache arms)."
        ),
    )
    parser.add_argument(
        "--controller-train-online",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "F1 controller-online arms set this. The eval scorer logs a "
            "warning when set: the CPU SSM controller's online learning "
            "loop is not yet wired into the eval path. Until that lands, "
            "trained-online arms behave identically to trained-frozen."
        ),
    )
    parser.add_argument(
        "--allow-online-replay-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow frozen-weight score-only eval of checkpoints that carry "
            "CRCT replay-eviction online state. This is an explicit diagnostic "
            "escape hatch: the CPU/GPU memory control plane is not run."
        ),
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
