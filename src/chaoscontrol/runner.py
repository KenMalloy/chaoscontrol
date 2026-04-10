"""Experiment dispatcher -- loads YAML configs, builds models, runs experiments."""
from __future__ import annotations
import json
import math
import yaml
from pathlib import Path
from typing import Any

import torch

from chaoscontrol.config import ChaosControlConfig
from chaoscontrol.data import (
    resolve_device, resolve_param_dtype,
    prepare_fineweb_splits, build_lm_starts, choose_eval_starts, maybe_sync_cuda,
)
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.training import train_chaoscontrol_for_budget
from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb, evaluate_warming_curve


def load_config(path: str, *, data_path: str, budget_seconds: float | None = None) -> ChaosControlConfig:
    raw = yaml.safe_load(Path(path).read_text())
    raw["data_path"] = data_path
    if budget_seconds is not None:
        raw["budget_seconds"] = budget_seconds
    return ChaosControlConfig(**raw)


def build_model(cfg: ChaosControlConfig, device: torch.device, param_dtype: torch.dtype):
    if cfg.model_type == "transformer":
        from chaoscontrol.baselines import SimpleTransformerLM
        model = SimpleTransformerLM(
            vocab_size=cfg.vocab_size, dim=cfg.model_dim,
            num_layers=cfg.num_layers, num_heads=max(1, cfg.model_dim // 32),
        )
    elif cfg.model_type == "mamba2":
        from chaoscontrol.baselines import Mamba2LM
        model = Mamba2LM(
            vocab_size=cfg.vocab_size, dim=cfg.model_dim,
            num_layers=cfg.num_layers,
        )
    else:
        model = ChaosStudentLM(
            vocab_size=cfg.vocab_size, dim=cfg.model_dim,
            num_layers=cfg.num_layers, ff_mult=cfg.ff_mult,
            a_mode=cfg.a_mode, a_full_rank=cfg.a_full_rank,
            a_full_gamma=cfg.a_full_gamma,
            rich_b_mode=cfg.rich_b_mode, rich_b_bottleneck=cfg.rich_b_bottleneck,
            rich_b_num_subnets=cfg.rich_b_num_subnets,
            rich_b_settling_steps=cfg.rich_b_settling_steps,
            outer_model_dim=cfg.outer_model_dim,
            consolidation_mode=cfg.consolidation_mode,
            consolidation_ema_decay=cfg.consolidation_ema_decay,
            consolidation_trigger=cfg.consolidation_trigger,
            consolidation_window=cfg.consolidation_window,
            outer_model_type=cfg.outer_model_type,
            outer_max_slots=cfg.outer_max_slots,
            outer_compress_ratio=cfg.outer_compress_ratio,
            wernicke_enabled=cfg.wernicke_enabled,
            wernicke_k_max=cfg.wernicke_k_max,
            wernicke_window=cfg.wernicke_window,
            wernicke_router=cfg.wernicke_router,
            wernicke_balance_weight=cfg.wernicke_balance_weight,
            wernicke_expert_dim=cfg.wernicke_expert_dim,
            wernicke_layers=cfg.wernicke_layers,
            wernicke_k_max_fine=cfg.wernicke_k_max_fine,
            buffer_mode=cfg.buffer_mode,
            retrieval_mode=cfg.retrieval_mode,
            retrieval_k=cfg.retrieval_k,
            bucket_prototypes=cfg.bucket_prototypes,
            prototype_dim=cfg.prototype_dim,
            prototype_update_rate=cfg.prototype_update_rate,
            semantic_tier_bases=cfg.semantic_tier_bases,
            semantic_tier_update_rate=cfg.semantic_tier_update_rate,
            typed_storage=cfg.typed_storage,
            typed_consolidation=cfg.typed_consolidation,
            compression_consequence=cfg.compression_consequence,
            cue_projection=cfg.cue_projection,
            dynamic_crit_per_layer=cfg.dynamic_crit_per_layer,
            compression_selection=cfg.compression_selection,
            posterior_mode=cfg.posterior_mode,
            posterior_lr=cfg.posterior_lr,
            residual_cache_k=cfg.residual_cache_k,
        )
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(dtype=param_dtype)
    return model


def save_checkpoint(
    checkpoint_dir: Path,
    name: str,
    model: Any,
    tokenizer: Any | None,
    structured_proj: Any | None,
    cfg: ChaosControlConfig,
) -> Path:
    """Save full training checkpoint: model + tokenizer + memory state + config."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{name}.pt"
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
    }
    if tokenizer is not None:
        payload["tokenizer_state_dict"] = tokenizer.state_dict()
    if structured_proj is not None:
        payload["structured_proj"] = structured_proj
    torch.save(payload, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path} ({ckpt_path.stat().st_size / 1024:.0f} KB)")
    return ckpt_path


def load_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, Any]:
    """Load checkpoint, rebuild model + tokenizer, return all state."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ChaosControlConfig(**payload["config"])

    # Rebuild tokenizer
    tokenizer = None
    if cfg.tokenizer_type == "fixed_stride":
        from chaoscontrol.tokenizer import FixedStrideTokenizer
        tokenizer = FixedStrideTokenizer(
            byte_dim=cfg.tokenizer_byte_dim,
            token_dim=cfg.tokenizer_token_dim,
            codebook_size=cfg.tokenizer_codebook_size,
            stride=cfg.tokenizer_stride,
            beta=cfg.tokenizer_beta,
        ).to(device)
        if device.type == "cuda":
            tokenizer = tokenizer.to(dtype=param_dtype)
        cfg.vocab_size = cfg.tokenizer_codebook_size

    # Rebuild model and load weights (includes memory extra_state)
    model = build_model(cfg, device, param_dtype)
    model.load_state_dict(payload["model_state_dict"])

    if tokenizer is not None and "tokenizer_state_dict" in payload:
        tokenizer.load_state_dict(payload["tokenizer_state_dict"])

    return {
        "model": model,
        "tokenizer": tokenizer,
        "structured_proj": payload.get("structured_proj"),
        "config": cfg,
    }


def run_experiment(config_path: str, *, data_path: str, budget_seconds: float = 300,
                   checkpoint_dir: str | None = None,
                   checkpoint_name: str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path, data_path=data_path, budget_seconds=budget_seconds)
    device = resolve_device(cfg.device)
    param_dtype = resolve_param_dtype(cfg.dtype, device)

    # H100 optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_tokens, val_tokens, _test = prepare_fineweb_splits(
        cfg.data_path, device=device,
    )
    train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    eval_starts = choose_eval_starts(val_starts, batch_size=cfg.batch_size, eval_batches=cfg.eval_batches, seed=cfg.seed)

    # Create tokenizer if configured
    tokenizer = None
    if cfg.tokenizer_type == "fixed_stride":
        from chaoscontrol.tokenizer import FixedStrideTokenizer
        tokenizer = FixedStrideTokenizer(
            byte_dim=cfg.tokenizer_byte_dim,
            token_dim=cfg.tokenizer_token_dim,
            codebook_size=cfg.tokenizer_codebook_size,
            stride=cfg.tokenizer_stride,
            beta=cfg.tokenizer_beta,
        ).to(device)
        if device.type == "cuda":
            tokenizer = tokenizer.to(dtype=param_dtype)
        # Override vocab_size so the model's embedding and lm_head match codebook
        cfg.vocab_size = cfg.tokenizer_codebook_size

    model = build_model(cfg, device, param_dtype)
    model_params = sum(p.numel() for p in model.parameters())
    tok_params = sum(p.numel() for p in tokenizer.parameters()) if tokenizer else 0
    params = model_params + tok_params
    tok_info = f" | tokenizer={cfg.tokenizer_type}(stride={cfg.tokenizer_stride}, K={cfg.tokenizer_codebook_size})" if tokenizer else ""
    print(f"Model: {cfg.model_type} | dim={cfg.model_dim} | params={params:,} | {model.artifact_bytes() if hasattr(model, 'artifact_bytes') else params*2:,} bytes{tok_info}")


    train_result = train_chaoscontrol_for_budget(
        model, train_tokens=train_tokens, train_starts=train_starts,
        seq_len=cfg.seq_len, batch_size=cfg.batch_size,
        device=device, param_dtype=param_dtype,
        budget_seconds=cfg.budget_seconds,
        base_lr=cfg.base_lr, weight_decay=cfg.weight_decay,
        grad_clip_norm=cfg.grad_clip_norm, seed=cfg.seed,
        crit_reg_alpha=cfg.crit_reg_alpha, crit_reg_beta=cfg.crit_reg_beta,
        crit_target_coupling=cfg.crit_target_coupling,
        metabolic_gate=cfg.metabolic_gate, metabolic_k=cfg.metabolic_k,
        metabolic_threshold=cfg.metabolic_threshold,
        metabolic_threshold_mode=cfg.metabolic_threshold_mode,
        metabolic_score=cfg.metabolic_score,
        metabolic_noise_std=cfg.metabolic_noise_std,
        generation_mode=cfg.generation_mode,
        metabolic_mode=cfg.metabolic_mode,
        mcts_horizon=cfg.mcts_horizon,
        mcts_ucb_c=cfg.mcts_ucb_c,
        consolidation_write=cfg.consolidation_write,
        latent_persistence=cfg.latent_persistence,
        cfr_enabled=cfg.cfr_enabled,
        tokenizer=tokenizer,
        align_type=cfg.align_type,
        align_weight=cfg.align_weight,
        sleep_enabled=cfg.sleep_enabled,
        sleep_stages=cfg.sleep_stages,
        sleep_interval=cfg.sleep_interval,
        sleep_budget=cfg.sleep_budget,
        sleep_n2_budget=cfg.sleep_n2_budget,
        sleep_rem_budget=cfg.sleep_rem_budget,
        sleep_n2_batches=cfg.sleep_n2_batches,
        sleep_rem_dreams=cfg.sleep_rem_dreams,
        sleep_rem_length=cfg.sleep_rem_length,
        sleep_merge_sim_threshold=cfg.sleep_merge_sim_threshold,
        sleep_survival_floor=cfg.sleep_survival_floor,
        sleep_rem_reactivate=cfg.sleep_rem_reactivate,
        polyphasic_enabled=cfg.polyphasic_enabled,
        polyphasic_n_partitions=cfg.polyphasic_n_partitions,
        polyphasic_k_awake=cfg.polyphasic_k_awake,
        polyphasic_topology=cfg.polyphasic_topology,
        polyphasic_swap_interval=cfg.polyphasic_swap_interval,
        # Experiment 14
        buffer_mode=cfg.buffer_mode,
        wernicke_enabled=cfg.wernicke_enabled,
    )

    # Use the trained structured_proj (not a fresh random one)
    structured_proj = train_result.get("structured_proj")

    total_raw_bytes = int(val_tokens.numel())

    eval_result = evaluate_chaoscontrol_bpb(
        model, tokens=val_tokens, eval_starts=eval_starts,
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, device=device,
        metabolic_gate=cfg.metabolic_gate, metabolic_k=cfg.metabolic_k,
        metabolic_score=cfg.metabolic_score, metabolic_noise_std=cfg.metabolic_noise_std,
        metabolic_mode=cfg.metabolic_mode,
        generation_mode=cfg.generation_mode, structured_proj=structured_proj,
        warmup=cfg.eval_warmup,
        warmup_write_mode=cfg.warmup_write_mode,
        warmup_latent=cfg.warmup_latent,
        warmup_cold_start=cfg.warmup_cold_start,
        total_raw_bytes=total_raw_bytes,
        tokenizer=tokenizer,
    )

    bpb_str = f"bpb={eval_result['bpb']:.4f}"
    if "bpb_gated" in eval_result:
        bpb_str += f" bpb_gated={eval_result['bpb_gated']:.4f}"
    print(f"Result: {bpb_str} | steps={train_result['steps']} | {train_result['elapsed_s']:.1f}s")

    # Warming curve: evaluate bpb at different warm-up lengths (TTT contract)
    warming_curve: dict[int, float] = {}
    if hasattr(model, "outer_model") and model.outer_model is not None:
        warming_curve = evaluate_warming_curve(
            model, val_tokens,
            score_tokens=1024, device=device,
        )
        if warming_curve:
            curve_str = " ".join(f"{n}:{b:.3f}" for n, b in sorted(warming_curve.items()))
            print(f"Warming curve: {curve_str}")

    total_params = params + train_result.get("extra_params", 0)

    # Save checkpoint if requested
    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_name = checkpoint_name or (Path(config_path).stem + f"_seed{cfg.seed}")
        ckpt_path = save_checkpoint(
            Path(checkpoint_dir), ckpt_name, model, tokenizer, structured_proj, cfg,
        )

    return {
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
        "train": train_result,
        "eval": eval_result,
        "warming_curve": warming_curve,
        "params": total_params,
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--enwik8-path", "--data-path", required=True, dest="enwik8_path")
    p.add_argument("--budget", type=float, default=300)
    p.add_argument("--output-json", default=None)
    p.add_argument("--checkpoint-dir", default=None,
                   help="Directory for model checkpoints (reserved for future use)")
    p.add_argument("--checkpoint-name", default=None,
                   help="Checkpoint name tag (reserved for future use)")
    args = p.parse_args()

    result = run_experiment(args.config, data_path=args.data_path,
                            budget_seconds=args.budget,
                            checkpoint_dir=args.checkpoint_dir,
                            checkpoint_name=args.checkpoint_name)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved to {out}")
