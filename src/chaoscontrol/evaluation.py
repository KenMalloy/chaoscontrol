"""Evaluation utilities for ChaosControl models."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from chaoscontrol.data import batch_from_starts, maybe_autocast
from chaoscontrol.metabolic import metabolic_fork, StructuredProjections
from chaoscontrol.memory import MultiSlotOuterModel


WARMING_CURVE_STEPS = [0, 100, 500, 1000, 5000]


def evaluate_chaoscontrol_bpb(
    model: Any,
    *,
    tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    metabolic_gate: bool = False,
    metabolic_k: int = 4,
    metabolic_score: str = "memory_consistency",
    metabolic_noise_std: float = 0.01,
    generation_mode: str = "noise",
    structured_proj: Any = None,
) -> dict[str, float]:
    """Evaluate ChaosStudentLM, returning loss and bits-per-byte.

    When metabolic_gate=True, runs both a plain forward pass and a gate-aware
    pass, returning both so experiments can compare.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_loss_gated = 0.0
    total_tokens = 0
    vocab_size = model.vocab_size
    try:
        with torch.no_grad():
            for idx in range(0, len(eval_starts), batch_size):
                batch_starts = eval_starts[idx : idx + batch_size]
                inputs, targets = batch_from_starts(tokens, batch_starts, seq_len, device)
                autocast_dtype = next(model.parameters()).dtype if device.type == "cuda" else torch.float32
                with maybe_autocast(device, autocast_dtype):
                    # Standard deterministic eval
                    logits = model(inputs)["logits"]

                    # Gate-aware eval (if metabolic gate is active)
                    if metabolic_gate:
                        gated_out = metabolic_fork(
                            model, inputs,
                            k=metabolic_k,
                            noise_std=metabolic_noise_std,
                            score_mode=metabolic_score,
                            generation_mode=generation_mode,
                            structured_proj=structured_proj,
                        )
                        gated_logits = gated_out["logits"]
                        total_loss_gated += float(
                            F.cross_entropy(
                                gated_logits.float().reshape(-1, vocab_size),
                                targets.reshape(-1), reduction="sum",
                            ).item()
                        )

                total_loss += float(
                    F.cross_entropy(logits.float().reshape(-1, vocab_size), targets.reshape(-1), reduction="sum").item()
                )
                total_tokens += int(targets.numel())
    finally:
        if was_training:
            model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    result = {
        "loss": float(mean_loss),
        "bpb": float(mean_loss / math.log(2.0)),
        "tokens": float(total_tokens),
    }
    if metabolic_gate:
        mean_loss_gated = total_loss_gated / max(total_tokens, 1)
        result["loss_gated"] = float(mean_loss_gated)
        result["bpb_gated"] = float(mean_loss_gated / math.log(2.0))
    return result


def _reset_model_state(model: Any) -> None:
    """Reset all stateful components: SSM state, buffer, prototypes, semantic tier, posterior.

    This implements the TTT evaluation contract: between segments, all
    runtime state is cleared so the buffer rebuilds from scratch.
    """
    # Reset SSM recurrence state (hidden states in ChaosSSMCore)
    for layer in getattr(model, "layers", []):
        core = getattr(layer, "core", None)
        if core is not None and hasattr(core, "state"):
            core.state = None

    # Reset multi-slot buffer: slots, survival scores, bucket assignments
    om = getattr(model, "outer_model", None)
    if om is not None and isinstance(om, MultiSlotOuterModel):
        om._slots.clear()
        om._survival.clear()
        om._slot_buckets.clear()
        om._retrieval_weights = None
        om._compression_consequences.clear()
        om.loss_ema.fill_(2.0)
    elif om is not None:
        # Single-slot OuterModel
        if hasattr(om, "state"):
            om.state.zero_()
        if hasattr(om, "loss_ema"):
            om.loss_ema.fill_(2.0)

    # Reset BucketPrototypes if present
    bpm = getattr(model, "bucket_prototypes_module", None)
    if bpm is not None and hasattr(bpm, "prototypes"):
        bpm.prototypes.zero_()

    # Reset SemanticTier bases if present
    st = getattr(model, "semantic_tier", None)
    if st is not None and hasattr(st, "bases"):
        st.bases.zero_()

    # Reset posterior state if present
    posterior = getattr(model, "posterior", None)
    if posterior is not None and hasattr(posterior, "reset"):
        posterior.reset()


def evaluate_warming_curve(
    model: Any,
    *,
    tokens: torch.Tensor,
    segment_starts: list[int],
    score_len: int = 1024,
    warming_steps: list[int] | None = None,
    device: torch.device,
) -> dict[int, float]:
    """Evaluate bpb warming curve following the TTT evaluation contract.

    For each warming step count N and each segment:
      1. Reset all model state (SSM, buffer, prototypes, semantic tier, posterior)
      2. Feed N warm-up tokens with writes enabled but no scoring
      3. Score the next score_len tokens
      4. Reset again before the next segment

    Args:
        model: ChaosStudentLM instance.
        tokens: Full token tensor (1D, long).
        segment_starts: Start indices for evaluation segments. Each segment
            must have at least max(warming_steps) + score_len tokens available.
        score_len: Number of tokens to score after warming.
        warming_steps: List of N values (default: WARMING_CURVE_STEPS).
        device: Device to run on.

    Returns:
        {N: mean_bpb} for each N in warming_steps.
    """
    if warming_steps is None:
        warming_steps = list(WARMING_CURVE_STEPS)

    was_training = model.training
    model.eval()
    vocab_size = model.vocab_size

    results: dict[int, float] = {}

    try:
        with torch.no_grad():
            for n_warmup in warming_steps:
                total_loss = 0.0
                total_tokens_scored = 0

                for seg_start in segment_starts:
                    _reset_model_state(model)

                    # Warm-up phase: feed N tokens, writes enabled, no scoring
                    if n_warmup > 0:
                        warmup_end = seg_start + n_warmup
                        # Process in chunks to avoid excessive memory
                        chunk_size = 256
                        for chunk_start in range(seg_start, warmup_end, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, warmup_end)
                            chunk_len = chunk_end - chunk_start
                            if chunk_start + chunk_len + 1 > tokens.numel():
                                break
                            inp = tokens[chunk_start:chunk_start + chunk_len].unsqueeze(0).to(device)
                            out = model(inp)
                            # Buffer write happens in forward pass for append-only mode
                            # For consolidation-based models, trigger a write
                            om = getattr(model, "outer_model", None)
                            if om is not None and hasattr(out, "__getitem__") and "hidden" in out:
                                hidden = out["hidden"][:, -1, :].detach()
                                logits = out["logits"]
                                target = tokens[chunk_start + 1:chunk_start + chunk_len + 1].unsqueeze(0).to(device)
                                if target.shape[1] == logits.shape[1]:
                                    ce = float(F.cross_entropy(
                                        logits.reshape(-1, vocab_size),
                                        target.reshape(-1),
                                    ).item())
                                    om.consolidation_step(hidden, current_loss=ce)

                    # Scoring phase: score the next score_len tokens
                    score_start = seg_start + n_warmup
                    score_end = score_start + score_len
                    if score_end + 1 > tokens.numel():
                        continue

                    inp = tokens[score_start:score_end].unsqueeze(0).to(device)
                    target = tokens[score_start + 1:score_end + 1].unsqueeze(0).to(device)
                    out = model(inp)
                    logits = out["logits"]

                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        target.reshape(-1),
                        reduction="sum",
                    )
                    total_loss += float(loss.item())
                    total_tokens_scored += int(target.numel())

                if total_tokens_scored > 0:
                    mean_loss = total_loss / total_tokens_scored
                    results[n_warmup] = float(mean_loss / math.log(2.0))
                else:
                    results[n_warmup] = float("nan")
    finally:
        if was_training:
            model.train()

    return results
