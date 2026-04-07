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
    resolve_device, resolve_param_dtype, prepare_tokenized_enwik8_splits,
    build_lm_starts, choose_eval_starts, maybe_sync_cuda,
)
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.training import train_chaoscontrol_for_budget
from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb


def load_config(path: str, *, enwik8_path: str, budget_seconds: float | None = None) -> ChaosControlConfig:
    raw = yaml.safe_load(Path(path).read_text())
    raw["enwik8_path"] = enwik8_path
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
            semantic_tier_bases=cfg.semantic_tier_bases,
            typed_storage=cfg.typed_storage,
            typed_consolidation=cfg.typed_consolidation,
            compression_consequence=cfg.compression_consequence,
            cue_projection=cfg.cue_projection,
            dynamic_crit_per_layer=cfg.dynamic_crit_per_layer,
            compression_selection=cfg.compression_selection,
        )
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(dtype=param_dtype)
    return model


def run_experiment(config_path: str, *, enwik8_path: str, budget_seconds: float = 300) -> dict[str, Any]:
    cfg = load_config(config_path, enwik8_path=enwik8_path, budget_seconds=budget_seconds)
    device = resolve_device(cfg.device)
    param_dtype = resolve_param_dtype(cfg.dtype, device)

    # H100 optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_tokens, val_tokens, _test = prepare_tokenized_enwik8_splits(
        Path(cfg.enwik8_path), device=device,
    )
    train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    eval_starts = choose_eval_starts(val_starts, batch_size=cfg.batch_size, eval_batches=cfg.eval_batches, seed=cfg.seed)

    model = build_model(cfg, device, param_dtype)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.model_type} | dim={cfg.model_dim} | params={params:,} | {model.artifact_bytes() if hasattr(model, 'artifact_bytes') else params*2:,} bytes")


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
    )

    bpb_str = f"bpb={eval_result['bpb']:.4f}"
    if "bpb_gated" in eval_result:
        bpb_str += f" bpb_gated={eval_result['bpb_gated']:.4f}"
    print(f"Result: {bpb_str} | steps={train_result['steps']} | {train_result['elapsed_s']:.1f}s")

    total_params = params + train_result.get("extra_params", 0)
    return {
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
        "train": train_result,
        "eval": eval_result,
        "params": total_params,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--enwik8-path", required=True)
    p.add_argument("--budget", type=float, default=300)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    result = run_experiment(args.config, enwik8_path=args.enwik8_path, budget_seconds=args.budget)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved to {out}")
