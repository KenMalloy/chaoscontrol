#!/usr/bin/env python3
"""Single-run engine for Experiment 16 Phase A.

This runner keeps the sparse-attention work self-contained inside the
experiment directory. It trains the Exp 15 style SP8192 SSM backbone, then
freezes it and evaluates a content-addressed oracle probe over recurrent
traces collected token-by-token from a chosen layer.

The current scaffold measures:
  - bare validation bpb
  - dense proxy attention entropy / effective connections
  - selector recall@k and mass capture@k
  - recent-token baseline recall@k and mass capture@k

It does not yet integrate sparse attention into the LM forward pass.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from chaoscontrol.data import (
    batch_from_starts,
    build_lm_starts,
    choose_eval_starts,
    load_fineweb_tokens,
    maybe_autocast,
    prepare_fineweb_splits,
    resolve_device,
    resolve_param_dtype,
)
from chaoscontrol.evaluation import compute_bpb
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.training import train_chaoscontrol_for_budget


def build_sentencepiece_luts(
    sp,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate the competition byte LUT logic used in Exp 15."""
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


def load_sp_data(data_dir: str, vocab_size: int = 8192) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load pre-tokenized SP shards and clamp header contamination."""
    train_tokens, val_tokens = load_fineweb_tokens(data_dir)
    train_tokens = train_tokens.clamp(0, vocab_size - 1)
    val_tokens = val_tokens.clamp(0, vocab_size - 1)
    test_tokens = train_tokens[:0]
    print(f"  SP data: train={train_tokens.numel():,} val={val_tokens.numel():,} tokens")
    return train_tokens, val_tokens, test_tokens


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
    """Competition-correct SP bpb computation."""
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
            batch_ce = float(
                F.cross_entropy(
                    logits.float().reshape(-1, vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                ).item()
            )
            total_ce_nats += batch_ce
            total_tokens += int(targets.numel())

            prev_ids = inputs.reshape(-1)
            tgt_ids = targets.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            total_bytes += int(token_bytes.to(torch.int64).sum().item())

    if was_training:
        model.train()

    return {
        "loss": float(total_ce_nats / max(total_tokens, 1)),
        "bpb": compute_bpb(total_ce_nats, total_bytes),
        "tokens": float(total_tokens),
        "total_ce_nats": total_ce_nats,
        "total_scored_bytes": total_bytes,
    }


def build_model(config: dict[str, Any], device: torch.device, param_dtype: torch.dtype) -> ChaosStudentLM:
    """Build the bare SSM backbone used for the probe."""
    model = ChaosStudentLM(
        vocab_size=config["vocab_size"],
        dim=config["model_dim"],
        num_layers=config["num_layers"],
        ff_mult=config.get("ff_mult", 2),
        a_mode=config.get("a_mode", "diag"),
        a_full_rank=config.get("a_full_rank", 8),
        a_full_gamma=config.get("a_full_gamma", 0.05),
        outer_model_dim=0,
        wernicke_enabled=False,
    )
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(dtype=param_dtype)
    return model


def _feature_from_source(
    *,
    normed: torch.Tensor,
    x_ssm: torch.Tensor,
    new_state: torch.Tensor,
    source: str,
) -> torch.Tensor:
    if source == "x":
        return x_ssm
    if source == "state":
        return new_state
    if source == "x_state":
        return torch.cat([x_ssm, new_state], dim=-1)
    if source == "norm_state":
        return torch.cat([normed, new_state], dim=-1)
    raise ValueError(f"unsupported feature source: {source}")


def attention_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-9)
    return -(probs * probs.log()).sum(dim=-1)


class ContentOracleSelector(nn.Module):
    """Small content-addressed selector used in the frozen probe."""

    def __init__(self, query_dim: int, key_dim: int, selector_dim: int = 0) -> None:
        super().__init__()
        hidden = selector_dim if selector_dim > 0 else max(query_dim, key_dim)
        self.query_proj = nn.Linear(query_dim, hidden, bias=False)
        self.key_proj = nn.Linear(key_dim, hidden, bias=False)

    def forward(
        self,
        queries: torch.Tensor,
        candidate_keys: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        q = F.normalize(self.query_proj(queries), dim=-1)
        k = F.normalize(self.key_proj(candidate_keys), dim=-1)
        scores = torch.einsum("bd,bnd->bn", q, k)
        return scores.masked_fill(~mask, -1e9)


def collect_oracle_examples(
    model: ChaosStudentLM,
    *,
    tokens: torch.Tensor,
    starts: list[int],
    seq_len: int,
    device: torch.device,
    layer_index: int,
    buffer_size: int,
    k: int,
    max_examples: int,
    query_source: str,
    write_source: str,
) -> dict[str, Any]:
    """Replay sequences token-by-token and collect oracle probe examples."""
    model.eval()
    query_rows: list[torch.Tensor] = []
    key_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    dense_topk_rows: list[torch.Tensor] = []
    entropies: list[float] = []
    effective_connections: list[float] = []
    top1_masses: list[float] = []
    recent_recall: list[float] = []
    recent_mass_capture: list[float] = []
    token_keyed_recall: list[float] = []
    token_keyed_mass_capture: list[float] = []

    with torch.no_grad():
        for start in starts:
            if len(query_rows) >= max_examples:
                break
            window = tokens[start : start + seq_len + 1]
            if window.numel() < seq_len + 1:
                continue

            states = model.init_state(1)
            buffers: list[list[torch.Tensor]] = [[] for _ in model.layers]
            buffer_token_ids: list[list[int]] = [[] for _ in model.layers]
            for pos in range(seq_len):
                token_id = window[pos].view(1, 1).to(device)
                x = model.embed(token_id).squeeze(1)

                for li, layer in enumerate(model.layers):
                    normed = layer.input_norm(x)
                    y, new_state = layer.core.step(normed, states[li], rich_b=layer.rich_b)
                    x_ssm = x + y

                    if li == layer_index:
                        query_vec = _feature_from_source(
                            normed=normed,
                            x_ssm=x_ssm,
                            new_state=new_state,
                            source=query_source,
                        ).squeeze(0)
                        write_vec = _feature_from_source(
                            normed=normed,
                            x_ssm=x_ssm,
                            new_state=new_state,
                            source=write_source,
                        ).squeeze(0)
                        buf = buffers[li]
                        if buf:
                            buf_tensor = torch.stack(buf, dim=0)
                            dense_scores = torch.matmul(buf_tensor, query_vec) / math.sqrt(query_vec.numel())
                            dense_probs = torch.softmax(dense_scores, dim=0)
                            padded_keys = torch.zeros(
                                buffer_size,
                                buf_tensor.shape[-1],
                                device="cpu",
                                dtype=torch.float32,
                            )
                            padded_target = torch.zeros(buffer_size, device="cpu", dtype=torch.float32)
                            padded_mask = torch.zeros(buffer_size, device="cpu", dtype=torch.bool)
                            valid = buf_tensor.shape[0]
                            padded_keys[:valid] = buf_tensor.to(device="cpu", dtype=torch.float32)
                            padded_target[:valid] = dense_probs.to(device="cpu", dtype=torch.float32)
                            padded_mask[:valid] = True
                            topk = dense_probs.topk(min(k, valid)).indices.to(device="cpu", dtype=torch.long)
                            dense_topk = torch.full((k,), -1, dtype=torch.long)
                            dense_topk[: topk.numel()] = topk

                            query_rows.append(query_vec.to(device="cpu", dtype=torch.float32))
                            key_rows.append(padded_keys)
                            target_rows.append(padded_target)
                            mask_rows.append(padded_mask)
                            dense_topk_rows.append(dense_topk)

                            ent = float(attention_entropy(dense_probs.unsqueeze(0)).item())
                            entropies.append(ent)
                            effective_connections.append(float(math.exp(ent)))
                            top1_masses.append(float(dense_probs.max().item()))

                            recent_idx = list(range(max(valid - k, 0), valid))
                            truth = set(int(x.item()) for x in topk)
                            hit_count = sum(int(idx in truth) for idx in recent_idx)
                            recent_recall.append(hit_count / max(len(truth), 1))
                            recent_mass_capture.append(float(dense_probs[recent_idx].sum().item()))

                            # Token-keyed baseline: select buffer positions
                            # matching the current token ID
                            current_tid = int(window[pos].item())
                            tk_ids = buffer_token_ids[li]
                            tk_matches = [i for i, tid in enumerate(tk_ids[:valid]) if tid == current_tid]
                            if len(tk_matches) > k:
                                tk_matches = tk_matches[-k:]  # most recent k
                            tk_hit = sum(int(idx in truth) for idx in tk_matches)
                            tk_truth_size = max(len(truth), 1)
                            token_keyed_recall.append(tk_hit / tk_truth_size)
                            token_keyed_mass_capture.append(
                                float(dense_probs[tk_matches].sum().item()) if tk_matches else 0.0
                            )

                            if len(query_rows) >= max_examples:
                                break

                        buf.append(write_vec.detach())
                        tk_buf = buffer_token_ids[li]
                        tk_buf.append(int(window[pos].item()))
                        if len(buf) > buffer_size:
                            buf.pop(0)
                            tk_buf.pop(0)

                    x = x_ssm + layer.ff(layer.ff_norm(x_ssm))
                    states[li] = new_state

                if len(query_rows) >= max_examples:
                    break

    if not query_rows:
        raise RuntimeError("oracle probe collected zero examples; increase eval coverage or lower seq_len")

    return {
        "queries": torch.stack(query_rows, dim=0),
        "candidate_keys": torch.stack(key_rows, dim=0),
        "target_probs": torch.stack(target_rows, dim=0),
        "mask": torch.stack(mask_rows, dim=0),
        "dense_topk": torch.stack(dense_topk_rows, dim=0),
        "recent_recall_at_k": float(sum(recent_recall) / max(len(recent_recall), 1)),
        "recent_mass_capture_at_k": float(sum(recent_mass_capture) / max(len(recent_mass_capture), 1)),
        "token_keyed_recall_at_k": float(sum(token_keyed_recall) / max(len(token_keyed_recall), 1)),
        "token_keyed_mass_capture_at_k": float(sum(token_keyed_mass_capture) / max(len(token_keyed_mass_capture), 1)),
        "full_attn_entropy": float(sum(entropies) / max(len(entropies), 1)),
        "effective_connections": float(sum(effective_connections) / max(len(effective_connections), 1)),
        "top1_mass": float(sum(top1_masses) / max(len(top1_masses), 1)),
        "examples": len(query_rows),
    }


def _batched_metrics(
    pred_scores: torch.Tensor,
    target_probs: torch.Tensor,
    mask: torch.Tensor,
    *,
    k: int,
) -> dict[str, float]:
    probs = torch.softmax(pred_scores, dim=-1)
    pred_topk = pred_scores.topk(k=min(k, pred_scores.shape[1]), dim=-1).indices
    truth_topk = target_probs.topk(k=min(k, target_probs.shape[1]), dim=-1).indices
    truth_mask = mask.gather(1, truth_topk)

    recalls: list[float] = []
    captures: list[float] = []
    for i in range(pred_topk.shape[0]):
        truth = set(int(x.item()) for x, valid in zip(truth_topk[i], truth_mask[i]) if bool(valid.item()))
        pred = [int(x.item()) for x in pred_topk[i] if bool(mask[i, x].item())]
        hit_count = sum(int(idx in truth) for idx in pred)
        recalls.append(hit_count / max(len(truth), 1))
        captures.append(float(target_probs[i, pred].sum().item()) if pred else 0.0)

    entropy = attention_entropy(probs).mean().item()
    return {
        "recall_at_k": float(sum(recalls) / max(len(recalls), 1)),
        "mass_capture_at_k": float(sum(captures) / max(len(captures), 1)),
        "oracle_entropy": float(entropy),
    }


def train_selector_probe(
    examples: dict[str, Any],
    *,
    k: int,
    selector_dim: int,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 1337,
) -> dict[str, Any]:
    """Train a small content-addressed selector to recover the dense proxy."""
    queries = examples["queries"]
    candidate_keys = examples["candidate_keys"]
    target_probs = examples["target_probs"]
    mask = examples["mask"]

    n = queries.shape[0]
    if n < 8:
        raise RuntimeError(f"selector probe needs at least 8 examples, got {n}")

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = max(1, int(0.8 * n))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < n else indices[-1:]

    selector = ContentOracleSelector(
        query_dim=queries.shape[-1],
        key_dim=candidate_keys.shape[-1],
        selector_dim=selector_dim,
    ).to(device)
    opt = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=1e-2)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for _epoch in range(epochs):
        rng.shuffle(train_idx)
        selector.train()
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start : start + batch_size]
            q = queries[idx].to(device)
            k_buf = candidate_keys[idx].to(device)
            m = mask[idx].to(device)
            tgt = target_probs[idx].to(device)
            scores = selector(q, k_buf, m)
            log_probs = F.log_softmax(scores, dim=-1)
            loss = F.kl_div(log_probs, tgt, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        selector.eval()
        with torch.no_grad():
            q = queries[val_idx].to(device)
            k_buf = candidate_keys[val_idx].to(device)
            m = mask[val_idx].to(device)
            tgt = target_probs[val_idx].to(device)
            scores = selector(q, k_buf, m)
            val_loss = float(F.kl_div(F.log_softmax(scores, dim=-1), tgt, reduction="batchmean").item())
            if val_loss < best_val:
                best_val = val_loss
                best_state = {kname: v.detach().cpu().clone() for kname, v in selector.state_dict().items()}

    if best_state is not None:
        selector.load_state_dict(best_state)
        selector.to(device)

    selector.eval()
    with torch.no_grad():
        scores = selector(
            queries.to(device),
            candidate_keys.to(device),
            mask.to(device),
        )
        metrics = _batched_metrics(scores, target_probs.to(device), mask.to(device), k=k)
    metrics["val_kl"] = best_val
    return metrics


def run_single(
    config: dict[str, Any],
    *,
    data_path: str,
    budget_seconds: float,
    sp_model_path: str,
    output_json: str | None = None,
) -> dict[str, Any]:
    """Run one Phase A oracle probe condition."""
    device = resolve_device(config.get("device", "auto"))
    param_dtype = resolve_param_dtype(config.get("dtype", "bf16"), device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    seed = int(config.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_tokens, val_tokens, _ = load_sp_data(data_path, config["vocab_size"])

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, config["vocab_size"], device
    )

    seq_len = int(config["seq_len"])
    stride = int(config.get("stride", seq_len // 2))
    batch_size = int(config["batch_size"])
    eval_batches = int(config.get("eval_batches", 32))

    train_starts = build_lm_starts(int(train_tokens.numel()), seq_len, stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), seq_len, stride)
    eval_starts = choose_eval_starts(val_starts, batch_size=batch_size, eval_batches=eval_batches, seed=seed)

    model = build_model(config, device, param_dtype)
    model_params = sum(p.numel() for p in model.parameters())
    artifact_bytes = model.artifact_bytes()
    print(
        f"Model: dim={config['model_dim']} | layers={config['num_layers']} | "
        f"params={model_params:,} | artifact={artifact_bytes:,} bytes ({artifact_bytes / 1e6:.1f} MB)"
    )

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

    bare_eval = evaluate_bpb_sp(
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

    oracle_eval_batches = int(config.get("oracle_eval_batches", min(eval_batches, 16)))
    oracle_starts = choose_eval_starts(val_starts, batch_size=1, eval_batches=oracle_eval_batches, seed=seed + 17)
    layer_index = int(config.get("oracle_layer_index", config["num_layers"] - 1))
    if layer_index < 0:
        layer_index = config["num_layers"] + layer_index

    examples = collect_oracle_examples(
        model,
        tokens=val_tokens,
        starts=oracle_starts,
        seq_len=seq_len,
        device=device,
        layer_index=layer_index,
        buffer_size=int(config["sparse_attn_buffer_size"]),
        k=int(config["sparse_attn_k"]),
        max_examples=int(config.get("oracle_max_examples", 4096)),
        query_source=config.get("oracle_query_source", "x_state"),
        write_source=config.get("oracle_write_source", "x_state"),
    )
    selector_metrics = train_selector_probe(
        examples,
        k=int(config["sparse_attn_k"]),
        selector_dim=int(config.get("sparse_attn_selector_dim", 0)),
        device=device,
        epochs=int(config.get("oracle_selector_epochs", 10)),
        batch_size=int(config.get("oracle_selector_batch_size", 128)),
        lr=float(config.get("oracle_selector_lr", 1e-3)),
        seed=seed,
    )

    result = {
        "config": config,
        "train": train_result,
        "params": model_params,
        "artifact_bytes": artifact_bytes,
        "bare_eval": bare_eval,
        "oracle_probe": {
            "examples": examples["examples"],
            "full_attn_entropy": examples["full_attn_entropy"],
            "effective_connections": examples["effective_connections"],
            "top1_mass": examples["top1_mass"],
            "recent_recall_at_k": examples["recent_recall_at_k"],
            "recent_mass_capture_at_k": examples["recent_mass_capture_at_k"],
            "token_keyed_recall_at_k": examples["token_keyed_recall_at_k"],
            "token_keyed_mass_capture_at_k": examples["token_keyed_mass_capture_at_k"],
            "selector_recall_at_k": selector_metrics["recall_at_k"],
            "selector_mass_capture_at_k": selector_metrics["mass_capture_at_k"],
            "oracle_entropy": selector_metrics["oracle_entropy"],
            "selector_val_kl": selector_metrics["val_kl"],
        },
        "implemented_metrics": [
            "bare_eval_bpb",
            "selector_recall_at_k",
            "selector_mass_capture_at_k",
            "recent_recall_at_k",
            "recent_mass_capture_at_k",
            "token_keyed_recall_at_k",
            "token_keyed_mass_capture_at_k",
            "full_attn_entropy",
            "effective_connections",
            "top1_mass",
            "oracle_entropy",
        ],
        "planned_metrics": [
            "bpb_dense",
            "bpb_oracle",
        ],
    }

    print(
        "Probe: "
        f"mass@k={result['oracle_probe']['selector_mass_capture_at_k']:.3f} | "
        f"recall@k={result['oracle_probe']['selector_recall_at_k']:.3f} | "
        f"tk_mass@k={result['oracle_probe']['token_keyed_mass_capture_at_k']:.3f} | "
        f"eff_conn={result['oracle_probe']['effective_connections']:.3f} | "
        f"bare_bpb={result['bare_eval']['bpb']:.4f}"
    )

    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str))
        print(f"Saved to {out}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 16 Phase A oracle probe")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--data-path", required=True, dest="data_path")
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    run_single(
        config,
        data_path=args.data_path,
        budget_seconds=args.budget,
        sp_model_path=args.sp_model_path,
        output_json=args.output_json,
    )
