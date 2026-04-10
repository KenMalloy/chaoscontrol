"""Evaluation utilities for ChaosControl models.

Key concepts:
- Warming curves: measure how model performance (bpb) improves as it sees
  more context tokens before scoring. This quantifies how quickly episodic
  memory, posterior state, and other runtime buffers adapt to new data.
- State resets between segments: following the Test-Time Training (TTT)
  protocol, all runtime state (SSM hidden states, episodic buffer, posterior
  corrections, semantic tier) is cleared between evaluation segments. This
  ensures each measurement reflects the model's ability to rebuild context
  from scratch, not residual state from previous segments.
"""
from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn.functional as F

from chaoscontrol.data import batch_from_starts, maybe_autocast
from chaoscontrol.metabolic import metabolic_fork, StructuredProjections
from chaoscontrol.memory import MultiSlotOuterModel


WARMING_CURVE_STEPS = [0, 100, 500, 1000, 5000]


def compute_bpb(total_ce_nats: float, total_raw_bytes: int) -> float:
    """Compute bits-per-byte. Tokenizer-agnostic.

    Args:
        total_ce_nats: Sum of cross-entropy loss (in nats) across all predicted tokens.
        total_raw_bytes: Count of raw bytes in the evaluation text.
            This is a property of the text, independent of the model's tokenizer.

    Returns:
        Bits per byte. Lower is better.
    """
    if total_raw_bytes <= 0:
        return 0.0
    return total_ce_nats / total_raw_bytes / math.log(2.0)


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
    metabolic_mode: str = "fork",
    generation_mode: str = "noise",
    structured_proj: Any = None,
    warmup: bool = False,
    warmup_write_mode: str = "last",
    warmup_latent: bool = False,
    warmup_cold_start: bool = False,
    total_raw_bytes: int | None = None,
    tokenizer: Any = None,
    prior_bias: Any = None,
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
    total_raw_bytes_seen = 0  # raw byte count for tokenizer bpb
    vocab_size = model.vocab_size

    # Save outer model state before eval warmup so memory writes don't persist
    saved_outer_state = None
    if warmup and getattr(model, "outer_model", None) is not None:
        om = model.outer_model
        saved_outer_state = {
            "loss_ema": om.loss_ema.clone(),
            # Trigger state
            "_spike_seen": om._spike_seen,
            "_steps_since_spike": om._steps_since_spike,
            "_pre_spike_loss": om._pre_spike_loss,
        }
        # MultiSlotOuterModel fields (not present on single-slot OuterModel)
        if hasattr(om, "_slots"):
            saved_outer_state["slots"] = [s.clone() for s in om._slots]
        if hasattr(om, "_survival"):
            saved_outer_state["survival"] = list(om._survival)
        if hasattr(om, "_slot_buckets"):
            saved_outer_state["slot_buckets"] = list(om._slot_buckets)
        if hasattr(om, "_retrieval_weights"):
            saved_outer_state["_retrieval_weights"] = om._retrieval_weights
        if hasattr(om, "_compression_consequences"):
            saved_outer_state["_compression_consequences"] = list(om._compression_consequences)
        if hasattr(om, "_latent_traces"):
            saved_outer_state["latent_traces"] = [
                {"bucket_id": t["bucket_id"], "centroid_contrib": t["centroid_contrib"].clone()}
                for t in om._latent_traces
            ]
        if hasattr(om, "_compress_rng"):
            saved_outer_state["_compress_rng_state"] = om._compress_rng.getstate()
        # Single-slot OuterModel state field
        if hasattr(om, "state"):
            saved_outer_state["state"] = om.state.clone()
        if hasattr(om, "consolidation_w"):
            saved_outer_state["consolidation_w"] = om.consolidation_w.clone()
        if hasattr(om, "_last_signal_was_pain"):
            saved_outer_state["_last_signal_was_pain"] = om._last_signal_was_pain.clone()
            saved_outer_state["_last_loss"] = om._last_loss.clone()
            saved_outer_state["_last_wrote"] = om._last_wrote.clone()

        # Cold start: wipe all memory before eval loop
        if warmup_cold_start:
            if hasattr(om, "_slots"):
                om._slots = []
            if hasattr(om, "_survival"):
                om._survival = []
            if hasattr(om, "_slot_buckets"):
                om._slot_buckets = []
            if hasattr(om, "_latent_traces"):
                om._latent_traces = []
            if hasattr(om, "state"):
                om.state.zero_()

    try:
        with torch.no_grad():
            for idx in range(0, len(eval_starts), batch_size):
                batch_starts = eval_starts[idx : idx + batch_size]
                inputs, targets = batch_from_starts(tokens, batch_starts, seq_len, device)

                # Tokenizer: convert raw bytes to VQ token IDs
                raw_bytes_in_batch = int(inputs.numel())
                total_recon_loss = 0.0
                if tokenizer is not None:
                    tok_out = tokenizer(inputs)
                    token_ids = tok_out["token_ids"]  # (batch, token_seq)
                    # Reconstruction loss: information cost of VQ quantization.
                    # Must be added to token CE for correct bpb — without it,
                    # bpb only measures token prediction, not byte prediction.
                    total_recon_loss += float(tok_out["recon_loss"].item()) * raw_bytes_in_batch
                    inputs = token_ids[:, :-1]
                    targets = token_ids[:, 1:]
                    total_raw_bytes_seen += raw_bytes_in_batch

                autocast_dtype = next(model.parameters()).dtype if device.type == "cuda" else torch.float32
                with maybe_autocast(device, autocast_dtype):
                    # Standard deterministic eval
                    out = model(inputs)
                    logits = out["logits"]

                    # Warmup: write to episodic memory for future batches
                    if warmup and getattr(model, "outer_model", None) is not None:
                        batch_loss = F.cross_entropy(
                            logits.float().reshape(-1, vocab_size),
                            targets.reshape(-1),
                        ).item()

                        if warmup_write_mode == "full_sequence" and hasattr(model.outer_model, "write_sequence"):
                            # Full-sequence write: mirrors training's full_sequence path
                            running_avg = model.outer_model.loss_ema.item()
                            signal = model.outer_model.compute_consolidation_signal(batch_loss, running_avg)
                            if running_avg > 0 and signal / running_avg > 0.01:
                                model.outer_model.write_sequence(
                                    out["hidden"].detach(),
                                    bucket_id=None,
                                )
                            model.outer_model.update_survival(batch_loss)
                            model.outer_model.loss_ema = (
                                model.outer_model.ema_decay * model.outer_model.loss_ema
                                + (1 - model.outer_model.ema_decay) * batch_loss
                            )
                        else:
                            # Default: consolidation_step with last hidden
                            hidden_last = out["hidden"][:, -1, :].detach()
                            model.outer_model.consolidation_step(
                                hidden_last,
                                current_loss=batch_loss,
                                bucket_id=None,
                            )

                        # Latent reactivation on high surprise
                        if warmup_latent and hasattr(model.outer_model, "try_reactivate"):
                            running_avg = model.outer_model.loss_ema.item()
                            surprise_ratio = batch_loss / max(running_avg, 1e-6)
                            if surprise_ratio > 1.0:
                                model.outer_model.try_reactivate(
                                    bucket_id=None, surprise=surprise_ratio,
                                )

                    # Gate-aware eval (if metabolic gate is active)
                    if metabolic_gate:
                        if metabolic_mode == "mcts":
                            from chaoscontrol.metabolic import micro_mcts
                            gated_out = micro_mcts(
                                model, inputs,
                                n_rollouts=metabolic_k, horizon=8,
                                prior_bias=prior_bias,
                            )
                        elif metabolic_mode == "monte_carlo":
                            from chaoscontrol.metabolic import metabolic_monte_carlo
                            gated_out = metabolic_monte_carlo(
                                model, inputs,
                                k=metabolic_k,
                                noise_std=metabolic_noise_std,
                                generation_mode=generation_mode,
                                structured_proj=structured_proj,
                            )
                        else:
                            gated_out = metabolic_fork(
                                model, inputs,
                                k=metabolic_k,
                                noise_std=metabolic_noise_std,
                                score_mode=metabolic_score,
                                generation_mode=generation_mode,
                                structured_proj=structured_proj,
                                prior_bias=prior_bias,
                            )
                        gated_logits = gated_out["logits"]
                        gated_ce = float(
                            F.cross_entropy(
                                gated_logits.float().reshape(-1, vocab_size),
                                targets.reshape(-1), reduction="sum",
                            ).item()
                        )
                        total_loss_gated += gated_ce + total_recon_loss

                batch_ce = float(
                    F.cross_entropy(logits.float().reshape(-1, vocab_size), targets.reshape(-1), reduction="sum").item()
                )
                total_loss += batch_ce + total_recon_loss
                total_tokens += int(targets.numel())
    finally:
        if saved_outer_state is not None:
            om = model.outer_model
            om.loss_ema = saved_outer_state["loss_ema"]
            # Trigger state
            om._spike_seen = saved_outer_state["_spike_seen"]
            om._steps_since_spike = saved_outer_state["_steps_since_spike"]
            om._pre_spike_loss = saved_outer_state["_pre_spike_loss"]
            # MultiSlotOuterModel fields
            if "slots" in saved_outer_state:
                om._slots = saved_outer_state["slots"]
            if "survival" in saved_outer_state:
                om._survival = saved_outer_state["survival"]
            if "slot_buckets" in saved_outer_state:
                om._slot_buckets = saved_outer_state["slot_buckets"]
            if "_retrieval_weights" in saved_outer_state:
                om._retrieval_weights = saved_outer_state["_retrieval_weights"]
            if "_compression_consequences" in saved_outer_state:
                om._compression_consequences = saved_outer_state["_compression_consequences"]
            if "latent_traces" in saved_outer_state:
                om._latent_traces = saved_outer_state["latent_traces"]
            if "_compress_rng_state" in saved_outer_state:
                om._compress_rng.setstate(saved_outer_state["_compress_rng_state"])
            # Single-slot OuterModel state
            if "state" in saved_outer_state:
                om.state = saved_outer_state["state"]
            if "consolidation_w" in saved_outer_state:
                om.consolidation_w = saved_outer_state["consolidation_w"]
            if "_last_signal_was_pain" in saved_outer_state:
                om._last_signal_was_pain = saved_outer_state["_last_signal_was_pain"]
                om._last_loss = saved_outer_state["_last_loss"]
                om._last_wrote = saved_outer_state["_last_wrote"]
        if was_training:
            model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    # When tokenizer is active, bpb denominator is raw bytes, not VQ tokens
    if tokenizer is not None and total_raw_bytes_seen > 0:
        bpb = compute_bpb(total_loss, total_raw_bytes_seen)
    else:
        bpb = float(mean_loss / math.log(2.0))
    result = {
        "loss": float(mean_loss),
        "bpb": bpb,
        "tokens": float(total_tokens),
    }
    # When raw byte count is provided, add the proper tokenizer-agnostic bpb
    if total_raw_bytes is not None:
        result["bpb_raw"] = compute_bpb(total_loss, total_raw_bytes)
    if metabolic_gate:
        mean_loss_gated = total_loss_gated / max(total_tokens, 1)
        result["loss_gated"] = float(mean_loss_gated)
        # When tokenizer is active, bpb denominator must be raw bytes (not tokens)
        # to stay on the same scale as bpb.
        if total_raw_bytes_seen > 0:
            result["bpb_gated"] = compute_bpb(total_loss_gated, total_raw_bytes_seen)
        else:
            result["bpb_gated"] = float(mean_loss_gated / math.log(2.0))
    return result


def _reset_model_state(model: Any) -> None:
    """Reset all stateful components: SSM state, buffer, prototypes, semantic tier, posterior.

    This implements the Test-Time Training (TTT) evaluation protocol:
    between evaluation segments, all runtime state is cleared so the model
    rebuilds its beliefs from scratch. Without this reset, bpb scores would
    reflect residual state from previous segments rather than the model's
    true adaptation speed on fresh data.
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
        if hasattr(om, "_latent_traces"):
            om._latent_traces.clear()
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


def _assert_state_clean(model: Any) -> None:
    """Verify state reset was complete. Catches forgotten stateful components."""
    om = getattr(model, "outer_model", None)
    if om is not None and isinstance(om, MultiSlotOuterModel):
        assert len(om._slots) == 0, "state leak: buffer not empty after reset"
        assert len(om._survival) == 0, "state leak: survival scores not empty"
        assert len(om._slot_buckets) == 0, "state leak: bucket assignments not empty"
        if hasattr(om, "_latent_traces"):
            assert len(om._latent_traces) == 0, "state leak: latent traces not empty"
    bpm = getattr(model, "bucket_prototypes_module", None)
    if bpm is not None and hasattr(bpm, "prototypes"):
        assert bpm.prototypes.abs().sum() == 0, "state leak: prototypes not zeroed"
    st = getattr(model, "semantic_tier", None)
    if st is not None and hasattr(st, "bases"):
        assert st.bases.abs().sum() == 0, "state leak: semantic tier bases not zeroed"
    posterior = getattr(model, "posterior", None)
    if posterior is not None and hasattr(posterior, "reset"):
        if hasattr(posterior, "delta"):
            assert posterior.delta.abs().sum() == 0, "state leak: posterior delta not zeroed"
        if hasattr(posterior, "deltas"):
            assert posterior.deltas.abs().sum() == 0, "state leak: posterior deltas not zeroed"


def _assert_no_grad_leak(model: Any) -> None:
    """Verify no model parameter accumulated .grad during eval."""
    for n, p in model.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, (
            f"gradient leak: {n} has non-zero .grad after eval"
        )


def _assert_bpb_sane(bpb: float, label: str = "") -> None:
    """Catch NaN, inf, negative, or impossibly high bpb values."""
    import math
    assert math.isfinite(bpb), f"bpb is not finite: {bpb} {label}"
    assert 0 < bpb < 15, f"bpb out of sane range [0, 15]: {bpb} {label}"


def evaluate_warming_curve(
    model: Any,
    tokens: torch.Tensor,
    *,
    warmup_tokens: list[int] | None = None,
    score_tokens: int = 1024,
    segment_len: int | None = None,
    segment_starts: list[int] | None = None,
    max_segments: int = 10,
    device: torch.device | None = None,
) -> dict[int, float]:
    """Evaluate bpb warming curve following the TTT evaluation contract.

    For each warming step count N and each segment:
      1. Reset all model state (SSM, buffer, prototypes, semantic tier, posterior)
      2. Feed N warm-up tokens with memory_write_mode="append_only" (no scoring)
      3. Score the next score_tokens tokens with memory_write_mode="none"
      4. Reset again before the next segment

    Segments can be specified either via segment_starts (explicit start indices)
    or segment_len (auto-derive non-overlapping segments from token length).

    Args:
        model: ChaosStudentLM instance.
        tokens: Full token tensor (1D, long).
        warmup_tokens: List of N values (default: WARMING_CURVE_STEPS).
        score_tokens: Number of tokens to score after warming.
        segment_len: Auto-derive segments of this length. Mutually exclusive
            with segment_starts.
        segment_starts: Explicit start indices. Mutually exclusive with
            segment_len.
        device: Device to run on (default: model device).

    Returns:
        {N: mean_bpb} for each N in warmup_tokens.
    """
    if warmup_tokens is None:
        warmup_tokens = list(WARMING_CURVE_STEPS)

    if device is None:
        device = next(model.parameters()).device

    # Derive segment_starts from segment_len if not provided
    if segment_starts is None:
        if segment_len is None:
            segment_len = max(warmup_tokens) + score_tokens + 1
        total_len = int(tokens.numel())
        segment_starts = []
        pos = 0
        while pos + segment_len <= total_len:
            segment_starts.append(pos)
            pos += segment_len
        if not segment_starts:
            segment_starts = [0]

    # Cap segments to avoid O(hours) eval on large val data
    if max_segments > 0 and len(segment_starts) > max_segments:
        rng = random.Random(42)
        segment_starts = sorted(rng.sample(segment_starts, max_segments))

    was_training = model.training
    model.eval()
    vocab_size = model.vocab_size

    results: dict[int, float] = {}

    try:
        with torch.no_grad():
            for n_warmup in warmup_tokens:
                total_loss = 0.0
                total_tokens_scored = 0

                for seg_start in segment_starts:
                    _reset_model_state(model)
                    _assert_state_clean(model)

                    # Warm-up phase: feed N tokens with writes enabled, no scoring
                    if n_warmup > 0:
                        warmup_end = seg_start + n_warmup
                        chunk_size = 256
                        for chunk_start in range(seg_start, warmup_end, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, warmup_end)
                            chunk_len = chunk_end - chunk_start
                            if chunk_start + chunk_len + 1 > tokens.numel():
                                break
                            inp = tokens[chunk_start:chunk_start + chunk_len].unsqueeze(0).to(device=device, dtype=torch.long)
                            # Use append_only so the buffer fills during warmup
                            model(inp, memory_write_mode="append_only")

                    # Scoring phase: score the next score_tokens tokens (no writes)
                    score_start = seg_start + n_warmup
                    score_end = score_start + score_tokens
                    if score_end + 1 > tokens.numel():
                        continue

                    inp = tokens[score_start:score_end].unsqueeze(0).to(device=device, dtype=torch.long)
                    target = tokens[score_start + 1:score_end + 1].unsqueeze(0).to(device=device, dtype=torch.long)
                    out = model(inp, memory_write_mode="none")
                    logits = out["logits"]

                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        target.reshape(-1),
                        reduction="sum",
                    )
                    total_loss += float(loss.item())
                    total_tokens_scored += int(target.numel())

                if total_tokens_scored > 0:
                    bpb = float(total_loss / total_tokens_scored / math.log(2.0))
                    _assert_bpb_sane(bpb, f"warming_curve N={n_warmup}")
                    results[n_warmup] = bpb
                else:
                    results[n_warmup] = float("nan")
    finally:
        if was_training:
            model.train()

    return results


def causal_slot_eval(
    model,
    tokens: torch.Tensor,
    *,
    conditions: tuple[str, ...] = ("cold", "buffer_only", "slot_only", "buffer_plus_slot"),
    warmup_tokens: list[int] | None = None,
    score_tokens: int = 1024,
    window_size: int = 256,
    slot_lr: float = 1e-3,
    slot_steps: int = 24,
    segment_starts: list[int] | None = None,
    segment_len: int | None = None,
    max_segments: int = 10,
    freeze_during_scoring: bool = True,
    device: torch.device | None = None,
) -> dict[str, dict[int, float]]:
    """Evaluate buffer x Causal SLOT stacking -- the D1 2x2 factorial.

    For each condition, for each segment, for each N in warmup_tokens:
      1. Reset all model state
      2. Create fresh delta/logit_bias parameters
      3. Warmup: slide windows over segment[:N], optionally filling the buffer
         and/or optimizing delta+logit_bias via SLOT
      4. Score the next score_tokens tokens

    Model parameters stay FROZEN throughout. Only delta and logit_bias get
    gradients during SLOT optimization.

    Args:
        model: ChaosStudentLM instance.
        tokens: Full token tensor (1D, long).
        conditions: Which conditions to evaluate.
        warmup_tokens: List of N values (default: WARMING_CURVE_STEPS).
        score_tokens: Number of tokens to score after warming.
        window_size: SLOT sliding window size.
        slot_lr: Learning rate for SLOT optimizer.
        slot_steps: Number of optimization steps per window.
        segment_starts: Explicit start indices (mutually exclusive with segment_len).
        segment_len: Auto-derive segments of this length (mutually exclusive
            with segment_starts).
        freeze_during_scoring: If True, detach delta/logit_bias during scoring
            (primary metric). If False, allow continued adaptation (streaming metric).
        device: Device to run on (default: model device).

    Returns:
        {condition_name: {N: mean_bpb}} for each condition and warmup level.
    """
    if warmup_tokens is None:
        warmup_tokens = list(WARMING_CURVE_STEPS)

    if device is None:
        device = next(model.parameters()).device

    # Derive segment_starts from segment_len if not provided
    if segment_starts is None:
        if segment_len is None:
            segment_len = max(warmup_tokens) + score_tokens + 1
        total_len = int(tokens.numel())
        segment_starts = []
        pos = 0
        while pos + segment_len <= total_len:
            segment_starts.append(pos)
            pos += segment_len
        if not segment_starts:
            segment_starts = [0]

    # Cap segments to avoid O(hours) eval on large val data
    if max_segments > 0 and len(segment_starts) > max_segments:
        rng = random.Random(42)
        segment_starts = sorted(rng.sample(segment_starts, max_segments))

    was_training = model.training
    model.eval()
    # Freeze all model parameters so SLOT backward only flows into delta/logit_bias.
    # Without this, lm_head and final_norm accumulate stale .grad tensors that waste
    # VRAM and risk corrupting the next training optimizer.step().
    saved_requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    model_dim = model.dim
    vocab_size = model.vocab_size

    _CONDITION_FLAGS = {
        "cold": (False, False),
        "buffer_only": (True, False),
        "slot_only": (False, True),
        "buffer_plus_slot": (True, True),
    }

    results: dict[str, dict[int, float]] = {}

    # Reproducibility canary: run cold N=0 on first segment twice. If results
    # differ, state is leaking between eval calls.
    if len(segment_starts) > 0:
        _canary_bpbs = []
        for _trial in range(2):
            _reset_model_state(model)
            _s = segment_starts[0]
            _inp = tokens[_s:_s + score_tokens].unsqueeze(0).to(device=device, dtype=torch.long)
            _tgt = tokens[_s + 1:_s + score_tokens + 1].unsqueeze(0).to(device=device, dtype=torch.long)
            with torch.no_grad():
                _out = model(_inp, memory_write_mode="none")
                _logits = model.lm_head(model.final_norm(_out["hidden"]))
                _loss = F.cross_entropy(_logits.reshape(-1, vocab_size), _tgt.reshape(-1), reduction="sum")
            _canary_bpbs.append(float(_loss.item()))
        assert _canary_bpbs[0] == _canary_bpbs[1], (
            f"Reproducibility canary failed: cold N=0 gave different results "
            f"({_canary_bpbs[0]} vs {_canary_bpbs[1]}). State is leaking between resets."
        )

    try:
        for cond_name in conditions:
            buffer_on, slot_on = _CONDITION_FLAGS[cond_name]
            cond_results: dict[int, float] = {}

            for n_warmup in warmup_tokens:
                total_loss = 0.0
                total_tokens_scored = 0

                for seg_start in segment_starts:
                    # 1. Reset ALL model state
                    _reset_model_state(model)
                    _assert_state_clean(model)

                    # 2. Create fresh delta and logit_bias (match model dtype for bf16)
                    model_dtype = next(model.parameters()).dtype
                    delta = torch.zeros(1, 1, model_dim, device=device, dtype=model_dtype, requires_grad=True)
                    logit_bias = torch.zeros(1, 1, vocab_size, device=device, dtype=model_dtype, requires_grad=True)
                    optimizer = torch.optim.Adam([delta, logit_bias], lr=slot_lr)

                    # 3. Warmup phase -- slide windows over segment[:N]
                    # Skip warmup entirely when neither buffer nor SLOT is active:
                    # cold mode means bare artifact with no adaptation.
                    if n_warmup > 0 and (buffer_on or slot_on):
                        warmup_end = seg_start + n_warmup
                        win_start = seg_start
                        while win_start < warmup_end:
                            win_end = min(win_start + window_size, warmup_end)
                            win_len = win_end - win_start

                            # Bounds check
                            if win_end + 1 > tokens.numel():
                                break

                            window_inp = tokens[win_start:win_end].unsqueeze(0).to(device=device, dtype=torch.long)
                            window_targets = tokens[win_start + 1:win_end + 1].reshape(-1).to(device=device, dtype=torch.long)

                            # Forward through model (frozen)
                            write_mode = "append_only" if buffer_on else "none"
                            with torch.no_grad():
                                out = model(window_inp, memory_write_mode=write_mode)
                                hidden = out["hidden"].detach()

                            # SLOT optimization on this window
                            if slot_on:
                                for _step in range(slot_steps):
                                    h = hidden + delta
                                    logits = model.lm_head(model.final_norm(h)) + logit_bias
                                    loss = F.cross_entropy(
                                        logits.reshape(-1, vocab_size),
                                        window_targets,
                                    )
                                    loss.backward()
                                    optimizer.step()
                                    optimizer.zero_grad()

                            win_start = win_end

                    # 4. Scoring phase
                    score_start = seg_start + n_warmup
                    score_end = score_start + score_tokens
                    if score_end + 1 > tokens.numel():
                        continue

                    if freeze_during_scoring:
                        # Primary metric: freeze all adaptation, score in one pass.
                        # Isolates the quality of state built during warmup.
                        inp = tokens[score_start:score_end].unsqueeze(0).to(device=device, dtype=torch.long)
                        target = tokens[score_start + 1:score_end + 1].unsqueeze(0).to(device=device, dtype=torch.long)

                        with torch.no_grad():
                            out = model(inp, memory_write_mode="none")
                            hidden = out["hidden"]

                        d = delta.detach()
                        lb = logit_bias.detach()
                        logits = model.lm_head(model.final_norm(hidden + d)) + lb
                        loss = F.cross_entropy(
                            logits.reshape(-1, vocab_size),
                            target.reshape(-1),
                            reduction="sum",
                        )
                        total_loss += float(loss.detach().item())
                        total_tokens_scored += int(target.numel())
                    else:
                        # Secondary metric: fully-online score-first TTT.
                        # Competition-legal: score each window, THEN optimize delta
                        # on the scored tokens. Next window benefits from updated delta.
                        win_start = score_start
                        while win_start < score_end:
                            win_end = min(win_start + window_size, score_end)
                            if win_end + 1 > tokens.numel():
                                break

                            window_inp = tokens[win_start:win_end].unsqueeze(0).to(device=device, dtype=torch.long)
                            window_targets = tokens[win_start + 1:win_end + 1].reshape(-1).to(device=device, dtype=torch.long)

                            # Forward through model (buffer accumulates if on)
                            write_mode = "append_only" if buffer_on else "none"
                            with torch.no_grad():
                                out = model(window_inp, memory_write_mode=write_mode)
                                hidden = out["hidden"].detach()

                            # Score with current delta (from past windows only)
                            d = delta.detach()
                            lb = logit_bias.detach()
                            logits = model.lm_head(model.final_norm(hidden + d)) + lb
                            score_loss = F.cross_entropy(
                                logits.reshape(-1, vocab_size),
                                window_targets,
                                reduction="sum",
                            )
                            total_loss += float(score_loss.item())
                            total_tokens_scored += int(window_targets.numel())

                            # THEN optimize delta on this window (already scored)
                            if slot_on:
                                for _step in range(slot_steps):
                                    h = hidden + delta
                                    opt_logits = model.lm_head(model.final_norm(h)) + logit_bias
                                    opt_loss = F.cross_entropy(
                                        opt_logits.reshape(-1, vocab_size),
                                        window_targets,
                                    )
                                    opt_loss.backward()
                                    optimizer.step()
                                    optimizer.zero_grad()

                            win_start = win_end

                if total_tokens_scored > 0:
                    bpb = float(total_loss / total_tokens_scored / math.log(2.0))
                    _assert_bpb_sane(bpb, f"causal_slot {cond_name} N={n_warmup}")
                    cond_results[n_warmup] = bpb
                else:
                    cond_results[n_warmup] = float("nan")

            results[cond_name] = cond_results
    finally:
        # Restore model parameter requires_grad and training mode
        for n, p in model.named_parameters():
            if n in saved_requires_grad:
                p.requires_grad_(saved_requires_grad[n])
        _assert_no_grad_leak(model)
        if was_training:
            model.train()

    return results
