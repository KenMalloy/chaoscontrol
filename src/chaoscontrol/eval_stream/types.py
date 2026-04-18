from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DocRecord:
    doc_id: int
    tokens: list[int]
    raw_bytes: int  # needed for bpb denominator; property of text, not tokenizer


@dataclass
class ChunkRecord:
    """Per-chunk measurement record.

    NOTE (2026-04-17): Currently unused by the driver — `MetricsCollector`
    logs per-doc only. Kept in `__all__` because the plan calls for per-chunk
    logging as a future extension, and the type's field set encodes the
    Score-then-Adapt contract (loss_before_adapt pre-update, loss_after_adapt
    nullable when the `none` adapt set skips the inner loop). If per-chunk
    logging lands, wire this through instead of re-defining it ad hoc.
    """
    doc_id: int
    chunk_idx: int
    tokens: list[int]
    loss_before_adapt: float  # score-before-update loss
    loss_after_adapt: float | None  # None if no weight TTT applied


@dataclass
class RunConfig:
    # Axis 1 — what adapts
    adapt_set: str = "none"  # none, log_a, delta_proj, log_a+delta_proj, B_side, C_side, embed_rows_seen, lm_head, lora_r8, all
    # Axis 2 — what persists across doc boundaries
    persistence_mode: str = "reset"  # reset, carry_state, carry_weights, carry_both, trainable_h0, trainable_h0+carry
    # Axis 3 — Δ modulation (no-grad)
    delta_scale: float = 1.0
    log_a_shift: float = 0.0
    # Schedule
    chunk_size: int = 256  # tokens; whole_doc = -1
    steps_per_chunk: int = 1
    eval_lr: float = 0.064
    persistent_muon_moments: bool = False
    warmup_steps: int = 20  # Param Golf contract: 20 warmup + state restore pre-timer
    # Run
    seed: int = 0
    max_docs: int = 50_000
    budget_seconds: float = 600.0
    score_floor_seconds: float = 0.0
    safety_margin_seconds: float = 0.0
    checkpoint_path: str = ""
    output_path: str = ""
    summary_path: str = ""
