from dataclasses import dataclass, field


_VALID_BUFFER_MODES = ("legacy", "append_only")
_VALID_RETRIEVAL_MODES = ("softmax_all", "bucket_mean", "bucket_recent", "bucket_topk")
_VALID_POSTERIOR_MODES = ("none", "global_delta", "bucket_delta", "residual_cache")


@dataclass
class ChaosControlConfig:
    data_path: str
    device: str = "auto"
    dtype: str = "bf16"
    seed: int = 1337
    output_json: str | None = None

    vocab_size: int = 256
    model_dim: int = 128
    num_layers: int = 4
    ff_mult: int = 2
    seq_len: int = 256
    stride: int = 128
    batch_size: int = 64
    eval_batches: int = 32

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
    metabolic_threshold_mode: str = "fixed"  # "fixed", "adaptive", or "random"
    metabolic_score: str = "memory_consistency"  # "memory_consistency", "loss_lookahead", "ensemble_agreement"
    metabolic_noise_std: float = 0.01  # perturbation magnitude for candidates
    metabolic_mode: str = "fork"  # "fork" (pick-best), "monte_carlo" (distributional), or "mcts" (tree search)

    # MCTS gate parameters (used when metabolic_mode="mcts")
    mcts_horizon: int = 8
    mcts_ucb_c: float = 1.41

    # Consolidation write mode
    consolidation_write: str = "last"  # "last" (final hidden) or "full_sequence" (trajectory-aware)

    # Latent persistence: reactivate compressed slot traces on high surprise
    latent_persistence: bool = False

    # CFR regret tracking across metabolic gate decisions
    cfr_enabled: bool = False

    # Eval warmup: replay a few training batches before scoring
    eval_warmup: bool = False
    warmup_write_mode: str = "last"  # "last" or "full_sequence"
    warmup_latent: bool = False  # try_reactivate on high surprise during warmup
    warmup_cold_start: bool = False  # True = wipe memory before eval

    # Wernicke layer
    wernicke_enabled: bool = False
    wernicke_k_max: int = 16
    wernicke_window: int = 8
    wernicke_router: str = "vq"  # "vq" or "moe"
    wernicke_balance_weight: float = 0.01
    wernicke_expert_dim: int = 0  # 0 = use model_dim (full rank); >0 = bottleneck per expert

    crit_reg_alpha: float = 0.01
    crit_reg_beta: float = 0.001
    crit_target_coupling: float = 0.88  # target coupling strength; log(0.88) used as target_log_sv

    base_lr: float = 2e-3
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    budget_seconds: float = 60.0

    # New fields for ChaosControl repo
    model_type: str = "ssm"  # "ssm", "transformer", or "mamba2"
    semantic_tier_bases: int = 0  # 0 = disabled
    semantic_tier_update_rate: float = 0.01
    generation_mode: str = "noise"  # "noise" or "structured"

    # Typed storage/consolidation (for Wernicke + memory interaction)
    typed_storage: bool = False
    typed_consolidation: bool = False
    compression_consequence: bool = False

    # Gap analysis flags
    cue_projection: bool = True  # False = use raw recurrence state as retrieval key
    dynamic_crit_per_layer: bool = False
    compression_selection: str = "survival"  # "survival" or "random" — controls slot merge ordering

    # Typed KV buffer mode ("legacy" = consolidation-based, "append_only" = per-token append)
    buffer_mode: str = "legacy"

    # Retrieval mode for typed buffer (how the model selects from stored KV entries)
    retrieval_mode: str = "softmax_all"  # "softmax_all", "bucket_mean", "bucket_recent", "bucket_topk"
    retrieval_k: int = 8  # number of neighbors for bucket_topk / bucket_recent

    # Bucket prototypes: per-bucket EMA priors decoded to model_dim
    bucket_prototypes: bool = False
    prototype_dim: int = 64
    prototype_update_rate: float = 0.1

    # Hierarchical Wernicke: number of routing levels (1 = flat, 2 = coarse+fine)
    wernicke_layers: int = 1
    wernicke_k_max_fine: int = 16  # fine-level codebook size (used when wernicke_layers >= 2)

    # Error-driven belief corrections: posterior-state options
    posterior_mode: str = "none"  # "none", "global_delta", "bucket_delta", "residual_cache"
    posterior_lr: float = 0.01
    residual_cache_k: int = 4

    # Tokenizer options
    tokenizer_type: str = "none"  # "none" or "fixed_stride"
    tokenizer_byte_dim: int = 256
    tokenizer_token_dim: int = 256
    tokenizer_codebook_size: int = 512
    tokenizer_stride: int = 4
    tokenizer_beta: float = 0.25

    # Alignment loss
    align_type: str = "none"  # "none", "contrastive", "distill", "diversity"
    align_weight: float = 0.05

    # Exp 21: optional path to a (vocab_size, model_dim) .pt tensor used to init model.embed.weight
    embed_init_path: str | None = None

    # Sleep cycle parameters
    sleep_enabled: bool = False
    sleep_stages: str = "full_cycle"  # "full_cycle", "n3_only", "rem_only", "n2_only"
    sleep_interval: int = 256  # training steps between sleep cycles
    sleep_budget: int = 128  # total sleep steps per cycle
    sleep_n2_budget: int = 64  # N2 consolidation steps
    sleep_rem_budget: int = 64  # REM dream generation steps
    sleep_n2_batches: int = 8  # batches per N2 consolidation
    sleep_rem_dreams: int = 4  # dreams per REM cycle
    sleep_rem_length: int = 128  # tokens per dream sequence
    sleep_merge_sim_threshold: float = 0.85  # similarity threshold for slot merging
    sleep_survival_floor: float = 0.1  # minimum survival score before pruning
    sleep_rem_reactivate: bool = True  # reactivate latent traces during REM
    sleep_adaptive_fatigue: bool = False  # scale sleep budget by training fatigue

    # Polyphasic sleep parameters
    polyphasic_enabled: bool = False
    polyphasic_n_partitions: int = 4
    polyphasic_k_awake: int = 3
    polyphasic_topology: str = "slot_striped"  # "slot_striped" or "bucket_owned"
    polyphasic_swap_interval: int = 256

    # Replay-eviction loop (CRCT rank-3 idle maintenance)
    replay_eviction_enabled: bool = False
    replay_eviction_mode: str = "active"
    replay_eviction_memory_streams: int = 8
    replay_eviction_threshold: float = 0.01
    replay_eviction_ema_beta: float = 0.9
    replay_eviction_min_age_steps: int = 128
    replay_eviction_max_seconds: float = 0.5
    replay_eviction_trace_path: str = ""
    replay_eviction_trace_max_rows: int = 0
    replay_eviction_probe_chunk_size: int = 16
    replay_eviction_oracle_confirm_top_k: int = 32
    replay_eviction_drift_threshold: float = 0.3
    replay_eviction_repr_drift_threshold: float = 0.2
    replay_eviction_refresh_lr: float = 0.1
    replay_eviction_refresh_margin: float = 0.001
    replay_eviction_quarantine_threshold: float = -0.01
    replay_eviction_max_quarantined: int = 8
    replay_eviction_quarantine_release_streak: int = 2
    replay_eviction_distill_peak_threshold: float = 0.04
    replay_eviction_peak_preserve_utility_threshold: float = 0.20
    replay_eviction_peak_preserve_sharpness_threshold: float = 0.20
    replay_eviction_useful_threshold: float = 0.005
    replay_eviction_min_score_count: int = 2

    def __post_init__(self) -> None:
        if self.buffer_mode not in _VALID_BUFFER_MODES:
            raise ValueError(
                f"buffer_mode must be one of {_VALID_BUFFER_MODES}, "
                f"got {self.buffer_mode!r}"
            )
        if self.retrieval_mode not in _VALID_RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval_mode must be one of {_VALID_RETRIEVAL_MODES}, "
                f"got {self.retrieval_mode!r}"
            )
        if self.posterior_mode not in _VALID_POSTERIOR_MODES:
            raise ValueError(
                f"posterior_mode must be one of {_VALID_POSTERIOR_MODES}, "
                f"got {self.posterior_mode!r}"
            )
