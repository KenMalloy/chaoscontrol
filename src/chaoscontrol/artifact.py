"""16MB artifact pipeline: serialize, load, and evaluate ChaosControl models.

Quantizes weight matrices to int8 (or int6 if needed), compresses episodic
memory slots, and LZMA-compresses the whole bundle to fit under the 16MB
competition budget. Inverse operations restore a runnable model from the
artifact with minimal quality loss.
"""
from __future__ import annotations

import io
import lzma
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from chaoscontrol.config import ChaosControlConfig
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.memory import MultiSlotOuterModel, SemanticTier
from chaoscontrol.regret import RegretTable


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _quantize_int8(tensor: torch.Tensor) -> dict[str, Any]:
    """Per-tensor symmetric int8 quantization for a 2-D weight matrix."""
    scale = tensor.abs().max() / 127
    if scale == 0:
        scale = torch.tensor(1.0, dtype=tensor.dtype)
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return {"q": quantized, "scale": scale.float()}


def _quantize_int6(tensor: torch.Tensor) -> dict[str, Any]:
    """Per-tensor symmetric int6 quantization (range -32..31) for tighter packing."""
    scale = tensor.abs().max() / 31
    if scale == 0:
        scale = torch.tensor(1.0, dtype=tensor.dtype)
    quantized = (tensor / scale).round().clamp(-32, 31).to(torch.int8)
    return {"q": quantized, "scale": scale.float()}


def _dequantize(entry: dict[str, Any]) -> torch.Tensor:
    """Restore a float tensor from a quantized entry ``{"q": int8, "scale": float}``."""
    return entry["q"].float() * entry["scale"]


def _quantize_state_dict(sd: dict[str, Any], *, use_int6: bool = False) -> dict[str, Any]:
    """Quantize a model/tokenizer state dict in-place (returns the same dict).

    2-D tensors -> int8/int6 quantized dicts.  Everything else -> fp16.
    """
    quantize_fn = _quantize_int6 if use_int6 else _quantize_int8
    for key in list(sd.keys()):
        tensor = sd[key]
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 2:
            sd[key] = quantize_fn(tensor.float())
        else:
            sd[key] = tensor.half()
    return sd


def _dequantize_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    """Dequantize a state dict, returning float32 tensors."""
    for key in list(sd.keys()):
        val = sd[key]
        if isinstance(val, dict) and "q" in val and "scale" in val:
            sd[key] = _dequantize(val)
        elif isinstance(val, torch.Tensor):
            sd[key] = val.float()
    return sd


def _quantize_slots(slots: list[torch.Tensor], *, use_int6: bool = False) -> list[dict[str, Any]]:
    """Quantize episodic slot vectors to int8/int6."""
    quantize_fn = _quantize_int6 if use_int6 else _quantize_int8
    result = []
    for s in slots:
        if s.ndim < 2:
            s = s.unsqueeze(0)
        result.append(quantize_fn(s.float()))
    return result


def _dequantize_slots(entries: list[dict[str, Any]]) -> list[torch.Tensor]:
    """Restore slot tensors from quantized entries."""
    return [_dequantize(e) for e in entries]


# ---------------------------------------------------------------------------
# Serialize
# ---------------------------------------------------------------------------

def serialize_artifact(
    model: ChaosStudentLM,
    tokenizer: nn.Module | None,
    config: ChaosControlConfig,
    path: str | Path,
    *,
    target_bytes: int = 16_777_216,
    regret_table: RegretTable | None = None,
    lzma_preset: int = 6,
) -> dict[str, Any]:
    """Quantize, compress, and write a ChaosControl model to a single file.

    Returns a metadata dict with path, size, quantization level, and slot stats.
    """
    path = Path(path)

    # 1. Collect state -------------------------------------------------------
    state: dict[str, Any] = {}
    state["model"] = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in model.state_dict().items()
    }
    if tokenizer is not None:
        state["tokenizer"] = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in tokenizer.state_dict().items()
        }
    else:
        state["tokenizer"] = None

    # Config dict (needed to reconstruct model architecture)
    state["config"] = asdict(config)

    # Episodic memory state
    om = getattr(model, "outer_model", None)
    original_slot_count = 0
    if om is not None and isinstance(om, MultiSlotOuterModel):
        state["episodic_slots"] = [s.detach().cpu() for s in om._slots]
        state["episodic_survival"] = list(om._survival)
        state["episodic_buckets"] = list(om._slot_buckets)
        state["latent_traces"] = [
            {"bucket_id": t["bucket_id"], "centroid_contrib": t["centroid_contrib"].detach().cpu()}
            for t in om._latent_traces
        ]
        original_slot_count = len(om._slots)
    else:
        state["episodic_slots"] = []
        state["episodic_survival"] = []
        state["episodic_buckets"] = []
        state["latent_traces"] = []

    # Semantic tier bases
    st = getattr(model, "semantic_tier", None)
    if st is not None and isinstance(st, SemanticTier):
        state["semantic_bases"] = st.bases.detach().cpu()
    else:
        state["semantic_bases"] = None

    # Regret table
    if regret_table is not None:
        state["regret_table"] = regret_table.cumulative_regret.detach().cpu()
    else:
        state["regret_table"] = None

    # 2. Quantize weights ----------------------------------------------------
    def _try_pack(state_dict: dict[str, Any], quantization: str, slots_to_keep: int | None = None) -> tuple[bytes, str, int, int]:
        """Quantize, compress, return (compressed_bytes, quant_name, kept, pruned)."""
        import copy
        sd = copy.deepcopy(state_dict)

        use_int6 = quantization == "int6"
        _quantize_state_dict(sd["model"], use_int6=use_int6)
        if sd["tokenizer"] is not None:
            _quantize_state_dict(sd["tokenizer"], use_int6=use_int6)

        # Quantize episodic slots
        if sd["episodic_slots"]:
            if slots_to_keep is not None and slots_to_keep < len(sd["episodic_slots"]):
                # Prune lowest-survival first
                survival = sd["episodic_survival"]
                indices = sorted(range(len(survival)), key=lambda i: survival[i])
                keep_indices = sorted(indices[len(indices) - slots_to_keep:])
                sd["episodic_slots"] = [sd["episodic_slots"][i] for i in keep_indices]
                sd["episodic_survival"] = [sd["episodic_survival"][i] for i in keep_indices]
                sd["episodic_buckets"] = [sd["episodic_buckets"][i] for i in keep_indices]
            sd["episodic_slots"] = _quantize_slots(sd["episodic_slots"], use_int6=use_int6)

        # Quantize latent traces
        if sd["latent_traces"]:
            for trace in sd["latent_traces"]:
                t = trace["centroid_contrib"]
                if t.ndim < 2:
                    t = t.unsqueeze(0)
                trace["centroid_contrib"] = (_quantize_int6 if use_int6 else _quantize_int8)(t.float())

        # Semantic bases -> fp16
        if sd["semantic_bases"] is not None:
            sd["semantic_bases"] = sd["semantic_bases"].half()

        # Regret table -> fp16
        if sd["regret_table"] is not None:
            sd["regret_table"] = sd["regret_table"].half()

        # Pack with torch.save into BytesIO
        buf = io.BytesIO()
        torch.save(sd, buf)
        raw = buf.getvalue()

        # LZMA compress
        compressed = lzma.compress(raw, preset=lzma_preset)

        kept = len(sd["episodic_slots"]) if isinstance(sd["episodic_slots"], list) else 0
        pruned = original_slot_count - kept
        return compressed, quantization, kept, pruned

    # Try int8 first
    compressed, quant, kept, pruned = _try_pack(state, "int8")

    if len(compressed) > target_bytes:
        # Try int6
        compressed, quant, kept, pruned = _try_pack(state, "int6")

    if len(compressed) > target_bytes and original_slot_count > 0:
        # Prune episodic slots progressively
        for keep_n in range(original_slot_count - 1, -1, -1):
            compressed, quant, kept, pruned = _try_pack(state, "int6", slots_to_keep=keep_n)
            if len(compressed) <= target_bytes:
                break

    if len(compressed) > target_bytes:
        raise ValueError(
            f"Artifact too large after all compression attempts: "
            f"{len(compressed):,} bytes > {target_bytes:,} byte budget"
        )

    # 3. Write ---------------------------------------------------------------
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)

    size = len(compressed)
    print(f"Artifact written: {path}")
    print(f"  Size: {size:,} / {target_bytes:,} bytes ({100*size/target_bytes:.1f}%)")
    print(f"  Quantization: {quant}")
    print(f"  Episodic slots: {kept} kept, {pruned} pruned")

    return {
        "path": str(path),
        "size_bytes": size,
        "quantization": quant,
        "slots_kept": kept,
        "slots_pruned": pruned,
    }


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_artifact(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ChaosStudentLM, nn.Module | None, ChaosControlConfig]:
    """Load a serialized artifact, returning (model, tokenizer, config).

    Dequantizes weights and restores all episodic/semantic/regret state.
    """
    path = Path(path)
    device = torch.device(device) if isinstance(device, str) else device

    # 1. Read and decompress -------------------------------------------------
    raw_compressed = path.read_bytes()
    raw = lzma.decompress(raw_compressed)
    buf = io.BytesIO(raw)
    state = torch.load(buf, map_location="cpu", weights_only=False)

    # 2. Reconstruct config --------------------------------------------------
    cfg_dict = state["config"]
    # Filter to only fields that ChaosControlConfig accepts
    valid_fields = {f.name for f in fields(ChaosControlConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    config = ChaosControlConfig(**cfg_dict)

    # 3. Build model ---------------------------------------------------------
    from chaoscontrol.runner import build_model
    from chaoscontrol.data import resolve_param_dtype

    param_dtype = resolve_param_dtype(config.dtype, device)
    model = build_model(config, device, param_dtype)

    # 4. Dequantize and load model weights -----------------------------------
    model_sd = _dequantize_state_dict(state["model"])
    # Cast to device dtype
    for k in model_sd:
        if isinstance(model_sd[k], torch.Tensor):
            model_sd[k] = model_sd[k].to(device=device)
    model.load_state_dict(model_sd, strict=False)

    # 5. Build and load tokenizer --------------------------------------------
    tokenizer = None
    if state["tokenizer"] is not None:
        from chaoscontrol.tokenizer import FixedStrideTokenizer
        tokenizer = FixedStrideTokenizer(
            byte_dim=config.tokenizer_byte_dim,
            token_dim=config.tokenizer_token_dim,
            codebook_size=config.tokenizer_codebook_size,
            stride=config.tokenizer_stride,
            beta=config.tokenizer_beta,
        ).to(device)
        tok_sd = _dequantize_state_dict(state["tokenizer"])
        for k in tok_sd:
            if isinstance(tok_sd[k], torch.Tensor):
                tok_sd[k] = tok_sd[k].to(device=device)
        tokenizer.load_state_dict(tok_sd, strict=False)

    # 6. Restore episodic memory ---------------------------------------------
    om = getattr(model, "outer_model", None)
    if om is not None and isinstance(om, MultiSlotOuterModel):
        if state["episodic_slots"]:
            om._slots = _dequantize_slots(state["episodic_slots"])
            om._slots = [s.to(device) for s in om._slots]
        else:
            om._slots = []
        om._survival = list(state.get("episodic_survival", []))
        om._slot_buckets = list(state.get("episodic_buckets", []))

        # Restore latent traces
        raw_traces = state.get("latent_traces", [])
        om._latent_traces = []
        for trace in raw_traces:
            centroid = trace["centroid_contrib"]
            if isinstance(centroid, dict) and "q" in centroid:
                centroid = _dequantize(centroid)
            elif isinstance(centroid, torch.Tensor):
                centroid = centroid.float()
            om._latent_traces.append({
                "bucket_id": trace["bucket_id"],
                "centroid_contrib": centroid.to(device),
            })

    # 7. Restore semantic tier bases -----------------------------------------
    st = getattr(model, "semantic_tier", None)
    if st is not None and isinstance(st, SemanticTier) and state["semantic_bases"] is not None:
        st.bases = state["semantic_bases"].float().to(device)

    # 8. Restore regret table (as attribute on model for convenience) --------
    if state.get("regret_table") is not None:
        rt = state["regret_table"].float()
        model._restored_regret_table = rt.to(device)

    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Eval convenience
# ---------------------------------------------------------------------------

def eval_artifact(
    path: str | Path,
    data_path: str,
    device: str | torch.device = "cpu",
    *,
    eval_batches: int = 32,
    batch_size: int | None = None,
    pretrain_model: nn.Module | None = None,
    pretrain_tokenizer: nn.Module | None = None,
) -> dict[str, Any]:
    """Load an artifact and evaluate it, producing three bpb measurements.

    1. **bpb_pretrain** — the training-time model (bf16), before compression.
       Only computed if pretrain_model is provided.
    2. **bpb_artifact** — the model loaded from the compressed artifact.
    3. **bpb_ttt** — the artifact model after test-time training (forward pass
       over training data with episodic memory writes), then fresh eval.

    Returns:
        Dict with bpb_pretrain, bpb_artifact, bpb_ttt,
        quant_degradation (2 - 1), ttt_recovery (2 - 3).
    """
    from chaoscontrol.data import (
        prepare_fineweb_splits, build_lm_starts, choose_eval_starts,
    )
    from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb

    device = torch.device(device) if isinstance(device, str) else device

    train_tokens, val_tokens, _test = prepare_fineweb_splits(data_path, device=device)

    # Load artifact
    model, tokenizer, config = load_artifact(path, device)
    bs = batch_size or config.batch_size

    val_starts = build_lm_starts(int(val_tokens.numel()), config.seq_len, config.seq_len)
    eval_starts = choose_eval_starts(
        val_starts, batch_size=bs, eval_batches=eval_batches, seed=config.seed,
    )

    result: dict[str, Any] = {
        "artifact_path": str(path),
        "artifact_size_bytes": Path(path).stat().st_size,
    }

    # 1. bpb_pretrain — training-time model, before compression
    if pretrain_model is not None:
        pretrain_eval = evaluate_chaoscontrol_bpb(
            pretrain_model, tokens=val_tokens, eval_starts=eval_starts,
            batch_size=bs, seq_len=config.seq_len, device=device,
            tokenizer=pretrain_tokenizer,
        )
        result["bpb_pretrain"] = pretrain_eval["bpb"]

    # 2. bpb_artifact — loaded from compressed artifact
    artifact_eval = evaluate_chaoscontrol_bpb(
        model, tokens=val_tokens, eval_starts=eval_starts,
        batch_size=bs, seq_len=config.seq_len, device=device,
        tokenizer=tokenizer,
    )
    result["bpb_artifact"] = artifact_eval["bpb"]

    # 3. bpb_ttt — test-time training: forward pass over training data
    # with episodic writes, then fresh eval on validation
    train_starts = build_lm_starts(int(train_tokens.numel()), config.seq_len, config.seq_len)
    ttt_eval_starts = choose_eval_starts(
        train_starts, batch_size=bs, eval_batches=min(eval_batches, 16), seed=config.seed,
    )
    ttt_eval = evaluate_chaoscontrol_bpb(
        model, tokens=val_tokens, eval_starts=eval_starts,
        batch_size=bs, seq_len=config.seq_len, device=device,
        tokenizer=tokenizer,
        warmup=True,
        warmup_write_mode="full_sequence",
        warmup_latent=True,
        warmup_cold_start=False,
    )
    result["bpb_ttt"] = ttt_eval["bpb"]

    # Derived metrics
    if "bpb_pretrain" in result:
        result["quant_degradation"] = result["bpb_artifact"] - result["bpb_pretrain"]
    result["ttt_recovery"] = result["bpb_artifact"] - result["bpb_ttt"]

    return result
