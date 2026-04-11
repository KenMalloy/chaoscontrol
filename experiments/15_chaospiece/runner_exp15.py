#!/usr/bin/env python3
"""Single-run engine for Experiment 15 (ChaosPiece).

Handles SP8192 tokenized data and competition-correct bpb computation.
Self-contained — imports from but does not modify shared chaoscontrol modules.

Pod prerequisites:
    cd baselines/parameter_golf
    python cached_challenge_fineweb.py --variant sp8192 --train-shards 80
    # Creates: datasets/fineweb10B_sp8192/ and tokenizers/fineweb_8192_bpe.model

Usage:
    python experiments/15_chaospiece/runner_exp15.py \
        --config experiments/15_chaospiece/configs/sp_d128_L4_s1337.yaml \
        --data-path /workspace/fineweb_data/datasets/fineweb10B_sp8192 \
        --sp-model-path /workspace/fineweb_data/tokenizers/fineweb_8192_bpe.model \
        --budget 600 --output-json results/sp_d128_L4_s1337.json

    For byte-level control (no --sp-model-path, data-path points to raw byte dir):
    python experiments/15_chaospiece/runner_exp15.py \
        --config configs/bare_ssm_byte256_s1337.yaml \
        --data-path /workspace/fineweb_data/datasets/fineweb10B_byte260 \
        --budget 600 --output-json results/bare_ssm_byte256_s1337.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from chaoscontrol.data import (
    resolve_device,
    resolve_param_dtype,
    load_fineweb_tokens,
    prepare_fineweb_splits,
    build_lm_starts,
    choose_eval_starts,
    batch_from_starts,
    maybe_autocast,
    maybe_sync_cuda,
)
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.baselines import SimpleTransformerLM
from chaoscontrol.training import train_chaoscontrol_for_budget
from chaoscontrol.evaluation import compute_bpb


# ---------------------------------------------------------------------------
# SentencePiece byte LUT — replicated from competition baseline
# baselines/parameter_golf/train_gpt.py:180-204
# ---------------------------------------------------------------------------

def build_sentencepiece_luts(
    sp, vocab_size: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-token byte count lookup tables from a SentencePiece model.

    Returns:
        base_bytes_lut: (vocab_size,) int16 — raw byte length per token
            (leading ▁ stripped; its space byte is counted via has_leading_space)
        has_leading_space_lut: (vocab_size,) bool — True if piece starts with ▁
        is_boundary_token_lut: (vocab_size,) bool — True for control/unk/unused
    """
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sp_data(
    data_dir: str,
    vocab_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load pre-tokenized SP shards. Full val split used for evaluation.

    Returns (train_tokens, val_tokens, test_tokens) as CPU int16 tensors.
    test_tokens is empty for Phase A — the full val split is used so that
    SP and byte conditions are evaluated on comparable data (byte path via
    prepare_fineweb_splits also uses the full docs_val_raw.txt). Phase B
    warming curves can split a test set from train when needed.

    Clamps token IDs to [0, vocab_size-1] to handle shard header contamination.
    Binary shards have a 1024-byte header (512 uint16 values) that
    _concat_shards_mmap reads as token data. These junk values can be
    negative (int16 view of large uint16) or exceed vocab_size, crashing
    nn.Embedding. Clamping is safe: the contamination is ~0.0005% of tokens
    and affects all conditions equally.
    """
    train_tokens, val_tokens = load_fineweb_tokens(data_dir)
    # Clamp header junk to valid token range
    train_tokens = train_tokens.clamp(0, vocab_size - 1)
    val_tokens = val_tokens.clamp(0, vocab_size - 1)
    test_tokens = train_tokens[:0]  # empty tensor, same dtype
    print(f"  SP data: train={train_tokens.numel():,} val={val_tokens.numel():,} tokens")
    return train_tokens, val_tokens, test_tokens


# ---------------------------------------------------------------------------
# Evaluation with competition-correct bpb
# ---------------------------------------------------------------------------

def evaluate_bpb_sp(
    model: Any,
    *,
    tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
) -> dict[str, float]:
    """Evaluate bpb with competition-correct per-token byte counting.

    For each target token, byte count = base_bytes[tgt]
        + (has_leading_space[tgt] & ~is_boundary[prev])
    Total bpb = sum(CE_nats) / sum(byte_counts) / ln(2)
    """
    was_training = model.training
    model.eval()
    total_ce_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    vocab_size = model.vocab_size

    with torch.no_grad():
        for idx in range(0, len(eval_starts), batch_size):
            batch_starts = eval_starts[idx : idx + batch_size]
            inputs, targets = batch_from_starts(tokens, batch_starts, seq_len, device)

            autocast_dtype = next(model.parameters()).dtype if device.type == "cuda" else torch.float32
            with maybe_autocast(device, autocast_dtype):
                out = model(inputs)
                logits = out["logits"]

            # CE in nats, summed (not averaged)
            batch_ce = float(
                F.cross_entropy(
                    logits.float().reshape(-1, vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                ).item()
            )
            total_ce_nats += batch_ce
            total_tokens += int(targets.numel())

            # Per-token byte counting (competition formula)
            prev_ids = inputs.reshape(-1)
            tgt_ids = targets.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            total_bytes += int(token_bytes.to(torch.int64).sum().item())

    if was_training:
        model.train()

    bpb = compute_bpb(total_ce_nats, total_bytes)
    mean_loss = total_ce_nats / max(total_tokens, 1)
    return {
        "loss": float(mean_loss),
        "bpb": bpb,
        "tokens": float(total_tokens),
        "total_ce_nats": total_ce_nats,
        "total_scored_bytes": total_bytes,
    }


def evaluate_bpb_bytes(
    model: Any,
    *,
    tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate bpb for byte-level models. Each scored token = 1 byte."""
    was_training = model.training
    model.eval()
    total_ce_nats = 0.0
    total_tokens = 0
    vocab_size = model.vocab_size

    with torch.no_grad():
        for idx in range(0, len(eval_starts), batch_size):
            batch_starts = eval_starts[idx : idx + batch_size]
            inputs, targets = batch_from_starts(tokens, batch_starts, seq_len, device)

            autocast_dtype = next(model.parameters()).dtype if device.type == "cuda" else torch.float32
            with maybe_autocast(device, autocast_dtype):
                out = model(inputs)
                logits = out["logits"]

            batch_ce = float(
                F.cross_entropy(
                    logits.float().reshape(-1, vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                ).item()
            )
            total_ce_nats += batch_ce
            total_tokens += int(targets.numel())

    if was_training:
        model.train()

    # For raw bytes, each token = 1 byte
    bpb = compute_bpb(total_ce_nats, total_tokens)
    mean_loss = total_ce_nats / max(total_tokens, 1)
    return {
        "loss": float(mean_loss),
        "bpb": bpb,
        "tokens": float(total_tokens),
        "total_ce_nats": total_ce_nats,
        "total_scored_bytes": total_tokens,
    }


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

class _TransformerWrapper(torch.nn.Module):
    """Thin wrapper so SimpleTransformerLM accepts memory_write_mode kwarg.

    train_chaoscontrol_for_budget() unconditionally passes memory_write_mode
    to model.forward(). SimpleTransformerLM doesn't accept it. This wrapper
    absorbs the kwarg without modifying shared baselines.py.
    """

    def __init__(self, inner: SimpleTransformerLM):
        super().__init__()
        self.inner = inner
        # Duck-type attributes that training/eval code expects
        self.vocab_size = inner.vocab_size
        self.dim = inner.embed.weight.shape[1]
        self.embed = inner.embed
        self.lm_head = inner.lm_head
        self.final_norm = inner.final_norm
        self.outer_model = None
        self.wernicke = None
        self.wernicke_balance_weight = 0.0
        self.semantic_tier = None
        self.layers = inner.layers

    def forward(self, input_ids, *, return_jacobian_stats=False, memory_write_mode="none"):
        return self.inner(input_ids, return_jacobian_stats=return_jacobian_stats)

    def artifact_bytes(self):
        return self.inner.artifact_bytes()


def build_model(config: dict, device: torch.device, param_dtype: torch.dtype):
    """Build ChaosStudentLM or SimpleTransformerLM from config dict."""
    model_type = config.get("model_type", "ssm")

    if model_type == "transformer":
        inner = SimpleTransformerLM(
            vocab_size=config["vocab_size"],
            dim=config["model_dim"],
            num_layers=config["num_layers"],
            num_heads=max(1, config["model_dim"] // 32),
            ff_mult=config.get("ff_mult", 2),
        )
        model = _TransformerWrapper(inner)
    else:
        model = ChaosStudentLM(
            vocab_size=config["vocab_size"],
            dim=config["model_dim"],
            num_layers=config["num_layers"],
            ff_mult=config.get("ff_mult", 2),
            a_mode=config.get("a_mode", "diag"),
            # All bolt-ons disabled for Phase A
            outer_model_dim=0,
            wernicke_enabled=False,
        )

    model = model.to(device)
    if device.type == "cuda":
        model = model.to(dtype=param_dtype)
    return model


def match_transformer_params(
    target_params: int,
    vocab_size: int,
    ff_mult: int = 2,
) -> dict:
    """Find transformer dim/layers to approximately match a target param count.

    Strategy: try several (dim, layers) combos and pick the closest match.
    Transformer params ~ 2 * vocab * dim + layers * dim^2 * (4 + 2*ff_mult) + small norms.
    """
    best = None
    best_gap = float("inf")
    for dim in [64, 96, 128, 160, 192, 224, 256, 320, 384, 512]:
        for layers in range(2, 16):
            embed_params = 2 * vocab_size * dim  # embed + lm_head
            # Per layer: QKV (3*dim^2) + out_proj (dim^2) + FFN (2*dim^2*ff_mult) + 2 RMSNorm (2*dim)
            per_layer = dim * dim * (4 + 2 * ff_mult) + 2 * dim
            final_norm = dim  # one final RMSNorm
            total = embed_params + layers * per_layer + final_norm
            gap = abs(total - target_params)
            if gap < best_gap:
                best_gap = gap
                best = {"model_dim": dim, "num_layers": layers, "total_params": total}
    return best


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_single(
    config: dict,
    data_path: str,
    budget_seconds: float,
    sp_model_path: str | None = None,
) -> dict[str, Any]:
    """Run a single training + evaluation experiment."""
    device = resolve_device(config.get("device", "auto"))
    param_dtype = resolve_param_dtype(config.get("dtype", "bf16"), device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    is_sp = sp_model_path is not None
    vocab_size = config["vocab_size"]
    seq_len = config["seq_len"]
    stride = config.get("stride", seq_len // 2)
    batch_size = config["batch_size"]
    seed = config.get("seed", 1337)

    # -- Load data --
    if is_sp:
        train_tokens, val_tokens, test_tokens = load_sp_data(data_path, vocab_size)
        # Build byte LUT
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(sp_model_path)
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
            build_sentencepiece_luts(sp, vocab_size, device)
        )
    else:
        train_tokens, val_tokens, test_tokens = prepare_fineweb_splits(
            data_path, device=device,
        )

    train_starts = build_lm_starts(int(train_tokens.numel()), seq_len, stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), seq_len, stride)
    eval_batches = config.get("eval_batches", 32)
    eval_starts = choose_eval_starts(
        val_starts, batch_size=batch_size, eval_batches=eval_batches, seed=seed,
    )

    # -- Build model --
    model = build_model(config, device, param_dtype)
    model_params = sum(p.numel() for p in model.parameters())
    artifact_bytes = model.artifact_bytes() if hasattr(model, "artifact_bytes") else model_params * 2
    model_type = config.get("model_type", "ssm")
    print(f"Model: {model_type} | dim={config['model_dim']} | layers={config['num_layers']} | "
          f"params={model_params:,} | artifact={artifact_bytes:,} bytes ({artifact_bytes / 1e6:.1f} MB)")

    # -- Train --
    train_result = train_chaoscontrol_for_budget(
        model,
        train_tokens=train_tokens,
        train_starts=train_starts,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        param_dtype=param_dtype,
        budget_seconds=budget_seconds,
        base_lr=config.get("base_lr", 2e-3),
        weight_decay=config.get("weight_decay", 1e-2),
        grad_clip_norm=config.get("grad_clip_norm", 1.0),
        seed=seed,
        crit_reg_alpha=config.get("crit_reg_alpha", 0.01),
        crit_reg_beta=config.get("crit_reg_beta", 0.001),
        crit_target_coupling=config.get("crit_target_coupling", 0.92),
    )

    # -- Evaluate --
    if is_sp:
        eval_result = evaluate_bpb_sp(
            model,
            tokens=val_tokens,
            eval_starts=eval_starts,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
    else:
        eval_result = evaluate_bpb_bytes(
            model,
            tokens=val_tokens,
            eval_starts=eval_starts,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

    bpb_str = f"bpb={eval_result['bpb']:.4f}"
    print(f"Result: {bpb_str} | steps={train_result['steps']} | {train_result['elapsed_s']:.1f}s")

    return {
        "config": config,
        "train": train_result,
        "eval": eval_result,
        "params": model_params,
        "artifact_bytes": artifact_bytes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exp 15 single-run engine")
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--data-path", required=True, dest="data_path")
    p.add_argument("--budget", type=float, default=600)
    p.add_argument("--output-json", default=None)
    p.add_argument("--sp-model-path", default=None,
                   help="Path to SentencePiece .model file. Omit for byte-level runs.")
    args = p.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    result = run_single(
        config,
        data_path=args.data_path,
        budget_seconds=args.budget,
        sp_model_path=args.sp_model_path,
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved to {out}")
