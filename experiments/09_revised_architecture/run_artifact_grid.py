#!/usr/bin/env python3
"""Phase 3: Artifact quantization + mechanism isolation grid.

Two claims:
  1. ChaosControl is more robust to quantization than a transformer.
  2. Latent reactivation is the mechanism (not just having more parameters).

Grid: model_variant x quant_level x eval
  Variants: full_ssm, full_ssm_no_reactivation, bare_ssm, our_tfm
  Quant: bf16, int8, int6
  Eval: bpb_pretrain -> bpb_artifact -> bpb_ttt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
RESULTS = EXPERIMENT / "results_phase3"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXPERIMENT))

from chaoscontrol.artifact import serialize_artifact, load_artifact, eval_artifact
from chaoscontrol.config import ChaosControlConfig
from chaoscontrol.data import (
    resolve_device, resolve_param_dtype,
    prepare_fineweb_splits, build_lm_starts, choose_eval_starts,
)
from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb
from chaoscontrol.runner import load_checkpoint, build_model
from stats import welch_ttest, bootstrap_ci, cohens_d, sem


QUANT_LEVELS = ["bf16", "int8", "int6"]


def make_variant_configs(base_config: dict) -> list[tuple[str, dict]]:
    """Generate model variant configs from the winning stack config."""
    full = dict(base_config)

    # full_ssm: as-is
    full_ssm = dict(full)

    # full_ssm_no_reactivation: disable latent persistence only
    no_react = dict(full, latent_persistence=False)

    # bare_ssm: no memory, no Wernicke, no semantic tier
    bare = dict(full,
                outer_model_dim=0,
                wernicke_enabled=False,
                semantic_tier_bases=0,
                latent_persistence=False,
                typed_storage=False,
                typed_consolidation=False,
                compression_consequence=False)

    # our_tfm: param-matched transformer
    tfm = dict(full,
               model_type="transformer",
               outer_model_dim=0,
               wernicke_enabled=False,
               semantic_tier_bases=0,
               latent_persistence=False)

    return [
        ("full_ssm", full_ssm),
        ("full_ssm_no_reactivation", no_react),
        ("bare_ssm", bare),
        ("our_tfm", tfm),
    ]


def run_artifact_eval(
    variant_name: str,
    variant_config: dict,
    quant_level: str,
    ckpt_path: Path,
    data_path: str,
    device: torch.device,
    param_dtype: torch.dtype,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    eval_starts: list[int],
    seed: int,
) -> dict:
    """Train variant from checkpoint, serialize artifact, eval three-number metric."""
    cfg = ChaosControlConfig(**variant_config)

    # Load the pretrained checkpoint
    loaded = load_checkpoint(ckpt_path, device, param_dtype)
    pretrain_model = loaded["model"]
    pretrain_tokenizer = loaded["tokenizer"]

    # For non-full variants, we need to build a fresh model and transfer what we can
    if variant_name != "full_ssm":
        model = build_model(cfg, device, param_dtype)
        # Transfer compatible weights
        pretrain_sd = pretrain_model.state_dict()
        model_sd = model.state_dict()
        transferable = {k: v for k, v in pretrain_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(transferable, strict=False)
        ratio = len(transferable) / max(len(model_sd), 1)
        print(f"    Transferred {len(transferable)}/{len(model_sd)} params ({ratio:.0%}) from checkpoint")
        if ratio < 0.5:
            print(f"    WARNING: Low transfer ratio — {variant_name} is largely untrained. "
                  f"Pretrain bpb reflects partial weight init, not a trained baseline.")
    else:
        model = pretrain_model

    tokenizer = loaded["tokenizer"]

    # 1. bpb_pretrain: eval the bf16 model
    pretrain_eval = evaluate_chaoscontrol_bpb(
        model, tokens=val_tokens, eval_starts=eval_starts,
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, device=device,
        tokenizer=tokenizer,
    )
    bpb_pretrain = pretrain_eval["bpb"]

    if quant_level == "bf16":
        # No quantization degradation, but still measure TTT empirically
        bpb_artifact = bpb_pretrain
        # Run TTT on the unquantized model to get an empirical ttt_recovery
        # (should be ~0, but measuring it confirms the control condition)
        # Reuse the same TTT logic below by falling through with the model as-is
        art_model = model
        art_tokenizer = tokenizer

        # TTT Phase A + B (same code path as quantized case below)
        train_starts_bf16 = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.seq_len)
        ttt_starts_bf16 = choose_eval_starts(train_starts_bf16, batch_size=cfg.batch_size,
                                              eval_batches=16, seed=seed)
        from chaoscontrol.data import batch_from_starts, maybe_autocast
        import torch.nn.functional as F_bf16
        art_model.eval()
        with torch.no_grad():
            for idx in range(0, len(ttt_starts_bf16), cfg.batch_size):
                batch_starts = ttt_starts_bf16[idx:idx + cfg.batch_size]
                inputs, targets = batch_from_starts(train_tokens, batch_starts, cfg.seq_len, device)
                if art_tokenizer is not None:
                    tok_out = art_tokenizer(inputs)
                    inputs = tok_out["token_ids"][:, :-1]
                    targets = tok_out["token_ids"][:, 1:]
                autocast_dtype = next(art_model.parameters()).dtype if device.type == "cuda" else torch.float32
                with maybe_autocast(device, autocast_dtype):
                    out = art_model(inputs)
                om = getattr(art_model, "outer_model", None)
                if om is not None:
                    batch_loss = F_bf16.cross_entropy(
                        out["logits"].float().reshape(-1, art_model.vocab_size),
                        targets.reshape(-1),
                    ).item()
                    if hasattr(om, "write_sequence"):
                        running_avg = om.loss_ema.item()
                        signal = om.compute_consolidation_signal(batch_loss, running_avg)
                        if running_avg > 0 and signal / running_avg > 0.01:
                            om.write_sequence(out["hidden"].detach(), bucket_id=None)
                        om.update_survival(batch_loss)
                        om.loss_ema = om.ema_decay * om.loss_ema + (1 - om.ema_decay) * batch_loss
                    else:
                        hidden_last = out["hidden"][:, -1, :].detach()
                        om.consolidation_step(hidden_last, current_loss=batch_loss, bucket_id=None)
                    if hasattr(om, "try_reactivate"):
                        running_avg = om.loss_ema.item()
                        surprise = batch_loss / max(running_avg, 1e-6)
                        if surprise > 1.0:
                            om.try_reactivate(bucket_id=None, surprise=surprise)
        ttt_eval_bf16 = evaluate_chaoscontrol_bpb(
            art_model, tokens=val_tokens, eval_starts=eval_starts,
            batch_size=cfg.batch_size, seq_len=cfg.seq_len, device=device,
            tokenizer=art_tokenizer,
        )
        return {
            "variant": variant_name,
            "quant": quant_level,
            "seed": seed,
            "bpb_pretrain": bpb_pretrain,
            "bpb_artifact": bpb_artifact,
            "bpb_ttt": ttt_eval_bf16["bpb"],
            "quant_degradation": 0.0,
            "ttt_recovery": bpb_artifact - ttt_eval_bf16["bpb"],
            "artifact_size_bytes": 0,
        }

    # 2. Serialize artifact with requested quantization
    artifact_dir = RESULTS / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{variant_name}_{quant_level}_seed{seed}.bin"

    meta = serialize_artifact(
        model, tokenizer, cfg, artifact_path,
        lzma_preset=6,
        force_quantization=quant_level,
    )

    # 3. Load artifact and eval
    art_model, art_tokenizer, art_config = load_artifact(artifact_path, device)
    art_eval = evaluate_chaoscontrol_bpb(
        art_model, tokens=val_tokens, eval_starts=eval_starts,
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, device=device,
        tokenizer=art_tokenizer,
    )
    bpb_artifact = art_eval["bpb"]

    # 4. TTT: Phase A (forward training data with memory writes), Phase B (fresh eval)
    train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.seq_len)
    ttt_starts = choose_eval_starts(train_starts, batch_size=cfg.batch_size, eval_batches=16, seed=seed)

    # Phase A: warm memory
    from chaoscontrol.data import batch_from_starts, maybe_autocast
    import torch.nn.functional as F

    art_model.eval()
    with torch.no_grad():
        for idx in range(0, len(ttt_starts), cfg.batch_size):
            batch_starts = ttt_starts[idx:idx + cfg.batch_size]
            inputs, targets = batch_from_starts(train_tokens, batch_starts, cfg.seq_len, device)
            if art_tokenizer is not None:
                tok_out = art_tokenizer(inputs)
                inputs = tok_out["token_ids"][:, :-1]
                targets = tok_out["token_ids"][:, 1:]
            autocast_dtype = next(art_model.parameters()).dtype if device.type == "cuda" else torch.float32
            with maybe_autocast(device, autocast_dtype):
                out = art_model(inputs)
            om = getattr(art_model, "outer_model", None)
            if om is not None:
                batch_loss = F.cross_entropy(
                    out["logits"].float().reshape(-1, art_model.vocab_size),
                    targets.reshape(-1),
                ).item()
                if hasattr(om, "write_sequence"):
                    running_avg = om.loss_ema.item()
                    signal = om.compute_consolidation_signal(batch_loss, running_avg)
                    if running_avg > 0 and signal / running_avg > 0.01:
                        om.write_sequence(out["hidden"].detach(), bucket_id=None)
                    om.update_survival(batch_loss)
                    om.loss_ema = om.ema_decay * om.loss_ema + (1 - om.ema_decay) * batch_loss
                else:
                    hidden_last = out["hidden"][:, -1, :].detach()
                    om.consolidation_step(hidden_last, current_loss=batch_loss, bucket_id=None)
                if hasattr(om, "try_reactivate"):
                    running_avg = om.loss_ema.item()
                    surprise = batch_loss / max(running_avg, 1e-6)
                    if surprise > 1.0:
                        om.try_reactivate(bucket_id=None, surprise=surprise)

    # Phase B: fresh eval
    ttt_eval = evaluate_chaoscontrol_bpb(
        art_model, tokens=val_tokens, eval_starts=eval_starts,
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, device=device,
        tokenizer=art_tokenizer,
    )
    bpb_ttt = ttt_eval["bpb"]

    return {
        "variant": variant_name,
        "quant": quant_level,
        "seed": seed,
        "bpb_pretrain": bpb_pretrain,
        "bpb_artifact": bpb_artifact,
        "bpb_ttt": bpb_ttt,
        "quant_degradation": bpb_artifact - bpb_pretrain,
        "ttt_recovery": bpb_artifact - bpb_ttt,
        "artifact_size_bytes": meta["size_bytes"],
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Artifact + quantization grid")
    parser.add_argument("--checkpoint", required=True, help="Best .pt checkpoint from Phase 2")
    parser.add_argument("--data-path", required=True, help="FineWeb data dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2674, 4011])
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    param_dtype = resolve_param_dtype(args.dtype, device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load winning config from checkpoint
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_config = payload["config"]

    # Load data
    print("Loading FineWeb data...")
    train_tokens, val_tokens, _test = prepare_fineweb_splits(args.data_path, device=device)
    val_starts = build_lm_starts(int(val_tokens.numel()), 256, 128)
    eval_starts = choose_eval_starts(val_starts, batch_size=64, eval_batches=32, seed=42)

    variants = make_variant_configs(base_config)
    total = len(variants) * len(QUANT_LEVELS) * len(args.seeds)
    print(f"Grid: {len(variants)} variants x {len(QUANT_LEVELS)} quant x {len(args.seeds)} seeds = {total} runs")

    # Resume support
    results_file = RESULTS / "artifact_results.json"
    all_results = []
    done_keys = set()
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        done_keys = {(r["variant"], r["quant"], r["seed"]) for r in all_results}
        print(f"Resuming: {len(done_keys)} already done")

    completed = len(done_keys)
    for variant_name, variant_config in variants:
        for quant in QUANT_LEVELS:
            for seed in args.seeds:
                key = (variant_name, quant, seed)
                if key in done_keys:
                    continue

                completed += 1
                t0 = time.time()
                print(f"\n[{completed}/{total}] {variant_name} / {quant} / seed={seed}")

                result = run_artifact_eval(
                    variant_name, variant_config, quant,
                    ckpt_path, args.data_path, device, param_dtype,
                    train_tokens, val_tokens, eval_starts, seed,
                )
                elapsed = time.time() - t0
                print(f"  pretrain={result['bpb_pretrain']:.4f}  artifact={result['bpb_artifact']:.4f}  "
                      f"ttt={result['bpb_ttt']:.4f}  degrad={result['quant_degradation']:+.4f}  "
                      f"recovery={result['ttt_recovery']:+.4f}  ({elapsed:.1f}s)")

                all_results.append(result)
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

    # ── Analysis ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PHASE 3 RESULTS")
    print("="*70)

    # Claim 1: Quantization robustness
    print("\n  Claim 1: Quantization robustness (delta_bpb by variant x quant)")
    print(f"  {'Variant':<30} {'Quant':<8} {'delta_bpb':>10} {'SEM':>8} {'recovery':>10}")
    print(f"  {'-'*70}")

    for variant_name, _ in variants:
        for quant in QUANT_LEVELS:
            deltas = [r["quant_degradation"] for r in all_results
                      if r["variant"] == variant_name and r["quant"] == quant]
            recoveries = [r["ttt_recovery"] for r in all_results
                          if r["variant"] == variant_name and r["quant"] == quant]
            if deltas:
                mean_d = sum(deltas) / len(deltas)
                mean_r = sum(recoveries) / len(recoveries)
                print(f"  {variant_name:<30} {quant:<8} {mean_d:>+10.4f} {sem(deltas):>8.4f} {mean_r:>+10.4f}")

    # Claim 2: Reactivation is the mechanism
    print("\n  Claim 2: Reactivation is the mechanism")
    for quant in ["int8", "int6"]:
        full_deltas = [r["quant_degradation"] for r in all_results
                       if r["variant"] == "full_ssm" and r["quant"] == quant]
        no_react_deltas = [r["quant_degradation"] for r in all_results
                           if r["variant"] == "full_ssm_no_reactivation" and r["quant"] == quant]
        if full_deltas and no_react_deltas:
            t_stat, p_val = welch_ttest(full_deltas, no_react_deltas)
            d = cohens_d(full_deltas, no_react_deltas)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {quant}: full_ssm vs no_reactivation: t={t_stat:.2f}, p={p_val:.4f} {sig}, d={d:.2f}")

    print(f"\n  Results saved to: {results_file}")


if __name__ == "__main__":
    main()
