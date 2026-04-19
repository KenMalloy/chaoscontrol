"""Exp 22 temporal-head eval driver."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn.functional as F

from chaoscontrol.eval_stream.budget import BudgetTracker, EvalDeadline
from chaoscontrol.eval_stream.doc_stream import DocStreamer
from chaoscontrol.eval_stream.temporal_heads import (
    TemporalHeadConfig,
    make_same_horizon_virtual_depth_config,
    score_temporal_heads_chunk,
)
from chaoscontrol.evaluation import compute_bpb


@dataclass
class Exp22RunConfig:
    condition: Literal[
        "score_only",
        "single_horizon",
        "temporal_heads",
        "gated_temporal_heads",
        "same_horizon_virtual_depth",
    ] = "temporal_heads"
    horizon_shifts: tuple[float, ...] = (-0.5, 0.0, 0.5)
    chunk_size: int = 256
    max_docs: int = 50_000
    seed: int = 0
    budget_seconds: float = 600.0
    score_floor_seconds: float = 0.0
    safety_margin_seconds: float = 0.0
    checkpoint_path: str = ""
    output_path: str = ""
    summary_path: str = ""
    analysis_path: str = ""
    evidence_label: str = "exploratory"
    depth_recurrence_count: int = 3
    mixer: Literal["uniform_logprob", "base_prior_logprob"] = "uniform_logprob"
    mixer_weights: tuple[float, ...] | None = None


def _iter_chunks(tokens: list[int], chunk_size: int):
    if chunk_size < 0:
        yield tokens
        return
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i + chunk_size]


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk_size), b""):
            h.update(buf)
    return h.hexdigest()


def _hash_cfg(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _normalize_float_tuple(values) -> tuple[float, ...] | None:
    if values is None:
        return None
    return tuple(float(value) for value in values)


def _json_shift_map(values: dict[float, object]) -> dict[str, object]:
    return {str(shift): value for shift, value in values.items()}


def _average_layer_summaries(
    samples_by_shift: dict[float, list[list[dict[str, float | int | None]]]],
) -> dict[float, list[dict[str, float | int | None]]]:
    averaged: dict[float, list[dict[str, float | int | None]]] = {}
    for shift, samples in samples_by_shift.items():
        if not samples:
            averaged[shift] = []
            continue
        layer_count = max((len(sample) for sample in samples), default=0)
        layers: list[dict[str, float | int | None]] = []
        for layer_idx in range(layer_count):
            metric_names = sorted(
                {
                    key
                    for sample in samples
                    if layer_idx < len(sample)
                    for key in sample[layer_idx]
                    if key != "layer"
                }
            )
            layer_summary: dict[str, float | int | None] = {"layer": layer_idx}
            for name in metric_names:
                numeric_values = [
                    float(sample[layer_idx][name])
                    for sample in samples
                    if layer_idx < len(sample)
                    and sample[layer_idx].get(name) is not None
                ]
                layer_summary[name] = (
                    sum(numeric_values) / len(numeric_values)
                    if numeric_values
                    else None
                )
            layers.append(layer_summary)
        averaged[shift] = layers
    return averaged


def _build_model(ckpt_path: Path, cfg: Exp22RunConfig) -> tuple[torch.nn.Module, dict]:
    from chaoscontrol.model import ChaosStudentLM

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = dict(blob["config"])
    model_cfg = ckpt_cfg
    if cfg.condition == "same_horizon_virtual_depth":
        model_cfg = make_same_horizon_virtual_depth_config(
            ckpt_cfg,
            depth_recurrence_count=cfg.depth_recurrence_count,
        )
    model = ChaosStudentLM(**model_cfg)
    model.load_state_dict(blob["model"], strict=True)
    return model, ckpt_cfg


def _score_direct_chunk(
    model: torch.nn.Module,
    chunk: torch.Tensor,
    states: list[torch.Tensor] | None,
) -> tuple[float, list[torch.Tensor]]:
    kwargs = {}
    if states:
        kwargs["initial_states"] = states
    with torch.no_grad():
        out = model(chunk, **kwargs)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        )
        final_states = (
            [state.detach().clone() for state in out["final_states"]]
            if isinstance(out, dict) and "final_states" in out
            else []
        )
    return float(loss.item()), final_states


def run(cfg: Exp22RunConfig, *, jsonl_paths: list[str], sp_model_path: str) -> None:
    if cfg.condition == "gated_temporal_heads":
        raise NotImplementedError(
            "gated_temporal_heads is not wired into this runner yet; run "
            "temporal_heads for always-on heads or add a pre-registered "
            "previous-chunk gate before using the gated condition."
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(cfg.checkpoint_path)
    model, ckpt_cfg = _build_model(ckpt_path, cfg)
    model.to(device)
    model.eval()

    streamer = DocStreamer(
        jsonl_paths=[Path(path) for path in jsonl_paths],
        sp_model_path=Path(sp_model_path),
        max_docs=cfg.max_docs,
    )
    budget = BudgetTracker(
        total_budget_seconds=cfg.budget_seconds,
        score_floor_seconds=cfg.score_floor_seconds,
        safety_margin_seconds=cfg.safety_margin_seconds,
    )
    deadline = EvalDeadline(cfg.budget_seconds)
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    docs_scored = 0
    chunks_scored = 0
    tokens_scored = 0
    total_loss_nats = 0.0
    total_raw_bytes = 0
    timed_out = False
    mixer_weights = _normalize_float_tuple(cfg.mixer_weights)
    temporal_cfg = TemporalHeadConfig(
        horizon_shifts=tuple(float(x) for x in cfg.horizon_shifts),
        mixer=cfg.mixer,
        mixer_weights=mixer_weights,
    )
    temporal_condition = cfg.condition in ("single_horizon", "temporal_heads")
    if cfg.condition == "single_horizon" and len(temporal_cfg.horizon_shifts) != 1:
        raise ValueError("single_horizon requires exactly one horizon shift")

    analysis_fh = None
    if cfg.analysis_path:
        analysis_path = Path(cfg.analysis_path)
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_fh = analysis_path.open("w")

    try:
        with out_path.open("w") as out_fh:
            for doc in streamer:
                if deadline.is_expired():
                    timed_out = True
                    break

                doc_t0 = time.monotonic()
                doc_loss_nats = 0.0
                doc_tokens = 0
                per_head_loss_nats = {shift: 0.0 for shift in temporal_cfg.horizon_shifts}
                doc_chunks_scored = 0
                doc_winner_counts = {shift: 0 for shift in temporal_cfg.horizon_shifts}
                doc_half_life_samples: dict[
                    float,
                    list[list[dict[str, float | int | None]]],
                ] = {shift: [] for shift in temporal_cfg.horizon_shifts}
                doc_state_divergence_samples: dict[
                    float,
                    list[list[dict[str, float | int | None]]],
                ] = {}

                if temporal_condition:
                    states_by_shift: dict[float, list[torch.Tensor] | None] = {
                        shift: None for shift in temporal_cfg.horizon_shifts
                    }
                else:
                    states: list[torch.Tensor] | None = None

                for chunk_list in _iter_chunks(doc.tokens, cfg.chunk_size):
                    if len(chunk_list) < 2:
                        continue
                    chunk = torch.tensor(
                        chunk_list,
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(0)
                    score_t0 = time.monotonic()
                    if temporal_condition:
                        result = score_temporal_heads_chunk(
                            model,
                            chunk,
                            states_by_shift=states_by_shift,
                            cfg=temporal_cfg,
                        )
                        states_by_shift = result.final_states_by_shift
                        doc_loss_nats += result.loss_nats
                        doc_tokens += result.tokens_scored
                        for shift, loss_nats in result.per_head_loss_nats.items():
                            per_head_loss_nats[shift] += loss_nats
                        for shift, count in result.winner_counts_by_shift.items():
                            doc_winner_counts[shift] = (
                                doc_winner_counts.get(shift, 0) + count
                            )
                        for shift, summaries in result.half_life_stats_by_shift.items():
                            doc_half_life_samples.setdefault(shift, []).append(summaries)
                        for shift, summaries in result.state_divergence_by_shift.items():
                            doc_state_divergence_samples.setdefault(shift, []).append(summaries)
                    else:
                        loss_nats, states = _score_direct_chunk(model, chunk, states)
                        doc_loss_nats += loss_nats
                        doc_tokens += chunk.size(1) - 1
                    budget.add_score_time(time.monotonic() - score_t0)
                    chunks_scored += 1
                    doc_chunks_scored += 1

                    if deadline.is_expired():
                        timed_out = True
                        break

                if timed_out:
                    break
                if doc_tokens <= 0:
                    continue

                docs_scored += 1
                tokens_scored += doc_tokens
                total_loss_nats += doc_loss_nats
                total_raw_bytes += doc.raw_bytes
                bpb = compute_bpb(doc_loss_nats, doc.raw_bytes)
                record = {
                    "doc_id": doc.doc_id,
                    "condition": cfg.condition,
                    "horizon_shifts": list(temporal_cfg.horizon_shifts),
                    "bpb": bpb,
                    "tokens": doc_tokens,
                    "loss_nats": doc_loss_nats,
                    "wall_ms": (time.monotonic() - doc_t0) * 1000.0,
                }
                if temporal_condition:
                    record["per_head_bpb"] = {
                        str(shift): compute_bpb(loss, doc.raw_bytes)
                        for shift, loss in per_head_loss_nats.items()
                    }
                out_fh.write(json.dumps(record, sort_keys=True) + "\n")
                if temporal_condition and analysis_fh is not None:
                    analysis_record = {
                        "analysis_only": True,
                        "doc_id": doc.doc_id,
                        "condition": cfg.condition,
                        "mixer": temporal_cfg.mixer,
                        "mixer_weights": (
                            list(temporal_cfg.mixer_weights)
                            if temporal_cfg.mixer_weights is not None
                            else None
                        ),
                        "horizon_shifts": list(temporal_cfg.horizon_shifts),
                        "chunks": doc_chunks_scored,
                        "tokens": doc_tokens,
                        "winner_counts_by_shift": _json_shift_map(doc_winner_counts),
                        "half_life_stats_by_shift": _json_shift_map(
                            _average_layer_summaries(doc_half_life_samples)
                        ),
                        "state_divergence_by_shift": _json_shift_map(
                            _average_layer_summaries(doc_state_divergence_samples)
                        ),
                    }
                    analysis_fh.write(
                        json.dumps(analysis_record, sort_keys=True) + "\n"
                    )
    finally:
        if analysis_fh is not None:
            analysis_fh.close()

    if cfg.summary_path:
        summary_path = Path(cfg.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = budget.summary(
            docs_scored=docs_scored,
            chunks_scored=chunks_scored,
            tokens_scored=tokens_scored,
            adapt_steps=0,
            timed_out=timed_out,
            collapsed=False,
            score_only_mode=cfg.condition == "score_only",
            elapsed_seconds=deadline.elapsed(),
            ckpt_sha256=_sha256_file(ckpt_path),
            ckpt_cfg_hash=_hash_cfg(ckpt_cfg),
            stream_seed=cfg.seed,
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            chunk_size=cfg.chunk_size,
            max_docs=cfg.max_docs,
        )
        summary.update(
            {
                "condition": cfg.condition,
                "evidence_label": cfg.evidence_label,
                "aggregate_bpb": compute_bpb(total_loss_nats, total_raw_bytes),
                "aggregate_loss_nats": total_loss_nats,
                "aggregate_raw_bytes": total_raw_bytes,
                "horizon_shifts": list(temporal_cfg.horizon_shifts),
                "mixer": temporal_cfg.mixer,
                "mixer_weights": (
                    list(temporal_cfg.mixer_weights)
                    if temporal_cfg.mixer_weights is not None
                    else None
                ),
                "analysis_path": cfg.analysis_path or None,
                "temporal_head_count": (
                    len(temporal_cfg.horizon_shifts)
                    if temporal_condition
                    else 1
                ),
                "depth_recurrence_count": (
                    cfg.depth_recurrence_count
                    if cfg.condition == "same_horizon_virtual_depth"
                    else 1
                ),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    raw = json.loads(Path(args.config).read_text())
    jsonl_paths = raw.pop("jsonl_paths")
    sp_model_path = raw.pop("sp_model_path")
    cfg_fields = {field.name for field in fields(Exp22RunConfig)}
    unknown_keys = sorted(set(raw) - cfg_fields)
    if unknown_keys:
        raise ValueError(f"unknown Exp 22 config key(s): {', '.join(unknown_keys)}")
    cfg = Exp22RunConfig(**{key: value for key, value in raw.items() if key in cfg_fields})
    run(cfg, jsonl_paths=jsonl_paths, sp_model_path=sp_model_path)


if __name__ == "__main__":
    main()
