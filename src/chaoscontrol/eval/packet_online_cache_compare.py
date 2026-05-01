"""Helpers for running and persisting the packet-online-cache compare.

The direct compare path is a live seeded-vs-empty A/B over the same
validation cache.  The core calc_type already returns the metric bundle;
this module packages the two runs and writes a durable summary JSON when
asked.
"""
from __future__ import annotations

import json
from dataclasses import asdict
import copy
from pathlib import Path
from typing import Any

import torch

from chaoscontrol.artifact import load_artifact
from chaoscontrol.eval.calc_types.packet_online_cache import packet_online_cache
from chaoscontrol.eval.ttt_eval import CalcTypeContext
from chaoscontrol.eval_stream.val_cache import ValCache, load_val_cache


def _lookup_tables_for(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab = int(getattr(model, "vocab_size", 0) or 0)
    if vocab <= 0:
        vocab = int(getattr(getattr(model, "lm_head", None), "out_features", 0) or 0)
    if vocab <= 0:
        raise ValueError("model must expose a positive vocab_size or lm_head.out_features")
    return (
        torch.zeros(vocab, dtype=torch.long),
        torch.zeros(vocab, dtype=torch.bool),
        torch.zeros(vocab, dtype=torch.bool),
    )


def _make_ctx(
    model: torch.nn.Module,
    val_cache: ValCache,
    *,
    device: torch.device,
    config: dict[str, Any],
) -> CalcTypeContext:
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = _lookup_tables_for(model)
    return CalcTypeContext(
        model=model,
        val_cache=val_cache,
        device=device,
        base_bytes_lut=base_bytes_lut.to(device=device),
        has_leading_space_lut=has_leading_space_lut.to(device=device),
        is_boundary_token_lut=is_boundary_token_lut.to(device=device),
        config=dict(config),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    if isinstance(result, dict):
        return dict(result)
    raise TypeError(f"unexpected calc_type result type: {type(result)!r}")


def run_packet_online_cache_compare(
    *,
    model: torch.nn.Module,
    val_cache: ValCache,
    device: torch.device,
    compare_config: dict[str, Any] | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run seeded and empty packet-cache passes and optionally persist JSON."""
    compare_config = dict(compare_config or {})
    if getattr(model, "_compare_artifact_path", None) is None:
        # Snapshot the untouched model before the seeded pass mutates its
        # episodic cache. The empty comparison must start from the same
        # pre-seeded state, not from the post-seeded model.
        empty_model = copy.deepcopy(model)
    else:
        empty_model = None

    seeded_ctx = _make_ctx(
        model,
        val_cache,
        device=device,
        config={**compare_config, "seeded": True},
    )
    seeded = packet_online_cache(seeded_ctx)

    # Re-load the model path if available so the empty-cache pass is truly
    # independent from the seeded pass's online writes.
    artifact_path = getattr(model, "_compare_artifact_path", None)
    if artifact_path is not None:
        empty_model, _tokenizer, _config = load_artifact(artifact_path, device)
        empty_model._compare_artifact_path = artifact_path
    elif empty_model is None:
        empty_model = copy.deepcopy(model)
    empty_ctx = _make_ctx(
        empty_model,
        val_cache,
        device=device,
        config={**compare_config, "seeded": False},
    )
    empty = packet_online_cache(empty_ctx)

    seeded_dict = _result_to_dict(seeded)
    empty_dict = _result_to_dict(empty)
    out = {
        "compare_type": "packet_online_cache",
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "val_cache_dir": str(val_cache.cache_dir),
        "compare_config": dict(compare_config),
        "seeded": seeded_dict,
        "empty": empty_dict,
        "delta_bpb": float(empty_dict["bpb"]) - float(seeded_dict["bpb"]),
        "delta_loss": float(empty_dict["loss"]) - float(seeded_dict["loss"]),
    }

    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, sort_keys=True, default=_json_default))
        tmp.rename(path)

    return out


def load_and_run_packet_online_cache_compare(
    *,
    artifact_path: str | Path,
    val_cache_dir: str | Path,
    device: str | torch.device = "cpu",
    compare_config: dict[str, Any] | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Load a serialized artifact + ValCache and run the compare."""
    device = torch.device(device) if isinstance(device, str) else device
    model, _tokenizer, _config = load_artifact(artifact_path, device)
    model._compare_artifact_path = str(artifact_path)
    val_cache = load_val_cache(Path(val_cache_dir))
    return run_packet_online_cache_compare(
        model=model,
        val_cache=val_cache,
        device=device,
        compare_config=compare_config,
        output_json=output_json,
    )
