from dataclasses import dataclass


@dataclass
class ChaosControlConfig:
    enwik8_path: str
    device: str = "auto"
    dtype: str = "fp16"
    seed: int = 1337
    output_json: str | None = None

    vocab_size: int = 256
    model_dim: int = 128
    num_layers: int = 4
    ff_mult: int = 2
    seq_len: int = 128
    stride: int = 64
    batch_size: int = 4
    eval_batches: int = 8

    a_mode: str = "diag"  # "diag", "paired", "full"
    a_full_rank: int = 8
    a_full_gamma: float = 0.05

    rich_b_mode: str = "none"  # "none", "nn", "hub", "assembly", "hybrid"
    rich_b_bottleneck: int = 32
    rich_b_num_subnets: int = 4
    rich_b_settling_steps: int = 2

    outer_model_dim: int = 0  # 0 = disabled
    consolidation_mode: str = "symmetric"  # "symmetric", "pain_biased", "learned"
    consolidation_ema_decay: float = 0.99
    consolidation_trigger: str = "immediate"  # "immediate", "resolution", "windowed"
    consolidation_window: int = 8  # for "windowed": flush every N steps after spike
    outer_model_type: str = "single"  # "single" or "multislot"
    outer_max_slots: int = 64  # for multislot: max slots before compression
    outer_compress_ratio: int = 2  # merge N oldest slots into N//ratio

    # Metabolic gate (generation + selection fork)
    metabolic_gate: bool = False
    metabolic_k: int = 4  # number of candidate rollouts
    metabolic_threshold: float = 0.1  # surprise/running_avg threshold to trigger fork
    metabolic_threshold_mode: str = "fixed"  # "fixed" or "adaptive"
    metabolic_score: str = "memory_consistency"  # "memory_consistency", "loss_lookahead", "ensemble_agreement"
    metabolic_noise_std: float = 0.01  # perturbation magnitude for candidates

    # Wernicke layer
    wernicke_enabled: bool = False
    wernicke_k_max: int = 16
    wernicke_window: int = 8
    wernicke_router: str = "vq"  # "vq" or "moe"
    wernicke_balance_weight: float = 0.01

    crit_reg_alpha: float = 0.01
    crit_reg_beta: float = 0.001
    crit_target_coupling: float = 0.88  # target coupling strength; log(0.88) used as target_log_sv

    base_lr: float = 2e-3
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    budget_seconds: float = 60.0

    # New fields for ChaosControl repo
    model_type: str = "ssm"  # "ssm" or "transformer"
    semantic_tier_bases: int = 0  # 0 = disabled
    generation_mode: str = "noise"  # "noise" or "structured"
