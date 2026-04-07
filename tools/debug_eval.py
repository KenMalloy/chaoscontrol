#!/usr/bin/env python3
"""Debug: check if two different models produce identical eval outputs."""
import torch
import torch.nn.functional as F
from chaoscontrol.runner import load_config, build_model
from chaoscontrol.data import (
    resolve_device, resolve_param_dtype,
    prepare_tokenized_enwik8_splits, build_lm_starts,
    choose_eval_starts, batch_from_starts,
)

device = resolve_device("auto")
train_tokens, val_tokens, _ = prepare_tokenized_enwik8_splits(
    "/workspace/enwik8", device=device,
)

configs = [
    "experiments/01_baseline/configs/ssm_full.yaml",
    "experiments/01_baseline/configs/ssm_medium.yaml",
]

for conf in configs:
    cfg = load_config(conf, enwik8_path="/workspace/enwik8", budget_seconds=10)
    param_dtype = resolve_param_dtype(cfg.dtype, device)
    model = build_model(cfg, device, param_dtype)
    model.eval()

    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    eval_starts = choose_eval_starts(
        val_starts, batch_size=cfg.batch_size,
        eval_batches=cfg.eval_batches, seed=cfg.seed,
    )

    # First batch only
    batch_starts = eval_starts[:cfg.batch_size]
    inputs, targets = batch_from_starts(val_tokens, batch_starts, cfg.seq_len, device)

    with torch.no_grad():
        out = model(inputs)
        logits = out["logits"]
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        print(f"--- {conf} ---")
        print(f"  dim={cfg.model_dim} params={sum(p.numel() for p in model.parameters()):,}")
        print(f"  logits: mean={logits.mean().item():.6f} std={logits.std().item():.6f} shape={list(logits.shape)}")
        print(f"  loss={loss.item():.10f}")
        print(f"  logits[0,0,:5]={logits[0,0,:5].tolist()}")
        print(f"  logits[0,-1,:5]={logits[0,-1,:5].tolist()}")
