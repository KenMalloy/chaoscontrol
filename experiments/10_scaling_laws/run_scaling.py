#!/usr/bin/env python3
"""Scaling laws runner for experiment 10.

Trains parameter-matched SSM and transformer configs at 5 sizes,
logs results per-run, and generates a combined results JSON.

Conditions:
  bare_ssm   -- pure ChaosStudentLM (diag mode, no components)
  full_ssm   -- ChaosStudentLM + winning component stack from exp 09
  our_tfm    -- SimpleTransformerLM, parameter-matched to the SSM
  mamba2_ssm -- Mamba2LM baseline (requires mamba-ssm package)

Sizes: XS(64d/2L), S(128d/4L), M(256d/6L), L(384d/8L), XL(512d/10L)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXP09_RESULTS = Path(__file__).resolve().parents[1] / "09_revised_architecture" / "results"
EXP09_CONFIGS = Path(__file__).resolve().parents[1] / "09_revised_architecture" / "configs"
SEEDS = [1337, 2674, 4011]

SIZE_TABLE = {
    "XS": {"dim": 64,  "ssm_layers": 2},
    "S":  {"dim": 128, "ssm_layers": 4},
    "M":  {"dim": 256, "ssm_layers": 6},
    "L":  {"dim": 384, "ssm_layers": 8},
    "XL": {"dim": 512, "ssm_layers": 10},
}

# Competition baseline reference BPB (from their reported results).
# Used as a reference point rather than re-training, since their data
# pipeline (FineWeb shards + SentencePiece tokenizer) differs from ours.
COMP_BASELINE_BPB = 1.2244


# ── FLOPs estimation (Task 2) ───────────────────────────────────────


def estimate_flops_per_step(
    model_type: str,
    model_dim: int,
    num_layers: int,
    ff_mult: int,
    seq_len: int,
    batch_size: int,
    *,
    wernicke_enabled: bool = False,
    wernicke_k_max: int = 16,
    outer_model_dim: int = 0,
    outer_max_slots: int = 64,
    metabolic_gate: bool = False,
    metabolic_k: int = 4,
) -> int:
    """Estimate forward+backward FLOPs per training step.

    Uses the 6N approximation: total FLOPs ~ 6 * params * tokens_per_step
    for the base model, plus component-specific corrections.
    """
    tokens = batch_size * seq_len
    d = model_dim
    L = num_layers

    if model_type == "transformer":
        # Per layer: QKV proj (3*d*d) + attn output (d*d) + FFN (2*d*d*ff_mult)
        per_layer_params = 4 * d * d + 2 * d * d * ff_mult
        embed_params = 2 * 256 * d
        total_params = L * per_layer_params + embed_params
        base_flops = 6 * total_params * tokens
        # Add quadratic attention cost
        attn_flops = 6 * L * 2 * seq_len * d
        base_flops += attn_flops * batch_size
    elif model_type == "mamba2":
        # Mamba2: similar to SSM but with expand factor
        # Default expand=2: inner_dim = dim*expand
        expand = 2
        inner = d * expand
        # Per layer: in_proj, out_proj, dt_proj, plus SSM state ops
        per_layer_params = 2 * d * inner + inner * 64 + inner  # approx
        embed_params = 2 * 256 * d
        total_params = L * per_layer_params + embed_params
        base_flops = 6 * total_params * tokens
    else:
        # SSM per layer:
        #   ChaosSSMCore: in_proj + select_proj + gate_proj + out_proj + delta_proj
        #                 = 5*d*d + d (diag mode)
        #   FFN: 2*d*d*ff_mult
        per_layer_params = 5 * d * d + 2 * d * d * ff_mult
        embed_params = 2 * 256 * d
        total_params = L * per_layer_params + embed_params
        base_flops = 6 * total_params * tokens

    # Component corrections (additive)
    component_flops = 0
    if wernicke_enabled:
        component_flops += 6 * 2 * d * wernicke_k_max * tokens
    if outer_model_dim > 0:
        component_flops += 6 * 2 * d * outer_max_slots * tokens
    if metabolic_gate:
        component_flops += metabolic_k * per_layer_params * tokens

    return int(base_flops + component_flops)


# ── Parameter counting (Task 3) ─────────────────────────────────────


def count_ssm_params(dim: int, num_layers: int, ff_mult: int = 2) -> int:
    """Count params for a bare ChaosStudentLM (diag mode, no components)."""
    embed = 256 * dim
    lm_head = dim * 256
    final_norm = dim
    # Per SSM layer (diag mode):
    #   ChaosSSMCore: in_proj(d,d) + select_proj(d,d) + gate_proj(d,d) +
    #                 out_proj(d,d) + delta_proj(d,d) + log_a(d) = 5*d*d + d
    #   FeedForward: fc(d, d*ff_mult) + proj(d*ff_mult, d) = 2*d*d*ff_mult
    #   RMSNorm x2: 2*d
    per_layer = 5 * dim * dim + dim + 2 * dim * dim * ff_mult + 2 * dim
    total = embed + lm_head + final_norm + num_layers * per_layer
    return total


def count_transformer_params(dim: int, num_layers: int, ff_mult: int = 2) -> int:
    """Count params for SimpleTransformerLM."""
    embed = 256 * dim
    lm_head = dim * 256
    final_norm = dim
    # Per transformer layer:
    #   CausalSelfAttention: qkv(d, 3*d) + out_proj(d, d) = 4*d*d
    #   FeedForward: fc(d, d*ff_mult) + proj(d*ff_mult, d) = 2*d*d*ff_mult
    #   RMSNorm x2: 2*d
    per_layer = 4 * dim * dim + 2 * dim * dim * ff_mult + 2 * dim
    total = embed + lm_head + final_norm + num_layers * per_layer
    return total


def match_transformer_to_ssm(ssm_dim: int, ssm_layers: int, ff_mult: int = 2) -> dict:
    """Find transformer (num_layers, ff_mult) that matches SSM param count within 5%.

    Strategy: fix dim to match SSM dim, sweep num_layers and ff_mult.
    Returns dict with keys: num_layers, ff_mult, param_count.
    """
    target = count_ssm_params(ssm_dim, ssm_layers, ff_mult)
    best = None
    best_err = float("inf")

    for try_ff in [2, 3, 4]:
        for try_layers in range(1, 20):
            p = count_transformer_params(ssm_dim, try_layers, try_ff)
            err = abs(p - target) / target
            if err < best_err:
                best_err = err
                best = {"num_layers": try_layers, "ff_mult": try_ff, "param_count": p}
            if err < 0.05:
                return best

    # If nothing within 5%, return the closest match
    return best


# ── Experiment 09 winners (Task 4) ──────────────────────────────────


def load_exp09_winners(results_dir: str | Path | None = None) -> dict:
    """Load experiment 09 winner settings from its results and config files.

    Reads full_summary.json for winner names, then extracts the relevant
    config fields from the winning YAML files.

    Returns a merged dict of all winning settings (gate + memory + wernicke/cfr).
    """
    exp09_results = Path(results_dir) if results_dir else EXP09_RESULTS
    exp09_configs = exp09_results.parent / "configs" if results_dir else EXP09_CONFIGS

    summary_path = exp09_results / "full_summary.json"
    if not summary_path.exists():
        return {}

    summary = json.loads(summary_path.read_text())
    full_stack: dict = {}

    # L1 winner: gate settings
    if "L1" in summary:
        l1_name = summary["L1"]["winner"]
        l1_cfg_path = exp09_configs / f"{l1_name}.yaml"
        if l1_cfg_path.exists():
            l1_cfg = yaml.safe_load(l1_cfg_path.read_text())
            gate_keys = [
                "metabolic_gate", "metabolic_k", "metabolic_mode",
                "metabolic_threshold", "metabolic_threshold_mode",
                "mcts_horizon", "mcts_ucb_c",
            ]
            for k in gate_keys:
                if k in l1_cfg:
                    full_stack[k] = l1_cfg[k]
            if "metabolic_gate" not in full_stack:
                full_stack["metabolic_gate"] = False

    # L2 winner: memory settings
    if "L2" in summary:
        l2_name = summary["L2"]["winner"]
        l2_cfg_path = exp09_configs / f"{l2_name}.yaml"
        if l2_cfg_path.exists():
            l2_cfg = yaml.safe_load(l2_cfg_path.read_text())
            mem_keys = [
                "outer_model_dim", "outer_model_type", "semantic_tier_bases",
                "eval_warmup", "consolidation_write", "latent_persistence",
                "typed_consolidation",
            ]
            for k in mem_keys:
                if k in l2_cfg:
                    full_stack[k] = l2_cfg[k]

    # L3 winner: Wernicke/CFR settings
    if "L3" in summary:
        l3_name = summary["L3"]["winner"]
        l3_cfg_path = exp09_configs / f"{l3_name}.yaml"
        if l3_cfg_path.exists():
            l3_cfg = yaml.safe_load(l3_cfg_path.read_text())
            w_keys = [
                "wernicke_enabled", "wernicke_router", "wernicke_k_max",
                "cfr_enabled", "typed_storage", "compression_consequence",
            ]
            for k in w_keys:
                if k in l3_cfg:
                    full_stack[k] = l3_cfg[k]

    if full_stack:
        print(f"Loaded exp09 winners: {list(full_stack.keys())}")
    return full_stack


# ── Config generation (Task 3) ──────────────────────────────────────


def generate_configs(
    sizes: list[str],
    conditions: list[str],
    full_stack_settings: dict,
) -> dict[str, dict[str, dict]]:
    """Generate all configs for requested sizes and conditions.

    Returns {size_name: {condition_name: config_dict}}.
    Also writes each config as a YAML file to CONFIGS_DIR.
    """
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    all_configs: dict[str, dict[str, dict]] = {}

    for size_name in sizes:
        if size_name not in SIZE_TABLE:
            print(f"  [WARN] Unknown size: {size_name}, skipping")
            continue
        size_spec = SIZE_TABLE[size_name]
        dim = size_spec["dim"]
        ssm_layers = size_spec["ssm_layers"]
        configs_at_size: dict[str, dict] = {}

        for cond_name in conditions:
            if cond_name == "bare_ssm":
                cfg = {
                    "model_type": "ssm",
                    "model_dim": dim,
                    "num_layers": ssm_layers,
                    "ff_mult": 2,
                    "a_mode": "diag",
                }
            elif cond_name == "full_ssm":
                cfg = {
                    "model_type": "ssm",
                    "model_dim": dim,
                    "num_layers": ssm_layers,
                    "ff_mult": 2,
                    "a_mode": "diag",
                }
                # Overlay experiment 09 winning settings
                if full_stack_settings:
                    cfg.update(full_stack_settings)
                else:
                    print(f"  [WARN] No exp09 winners available; full_ssm at {size_name} "
                          "will be identical to bare_ssm")
                # Ensure dim/layers are for this size (override exp09 values)
                cfg["model_dim"] = dim
                cfg["num_layers"] = ssm_layers
            elif cond_name == "our_tfm":
                tfm_match = match_transformer_to_ssm(dim, ssm_layers, ff_mult=2)
                cfg = {
                    "model_type": "transformer",
                    "model_dim": dim,
                    "num_layers": tfm_match["num_layers"],
                    "ff_mult": tfm_match["ff_mult"],
                    "a_mode": "diag",  # ignored for transformer
                }
            elif cond_name == "mamba2_ssm":
                cfg = {
                    "model_type": "mamba2",
                    "model_dim": dim,
                    "num_layers": ssm_layers,
                    "ff_mult": 2,
                    "a_mode": "diag",  # ignored for mamba2
                }
            elif cond_name == "comp_tfm":
                # Competition baseline: only meaningful at XL.
                # We store a reference config but actual BPB comes from
                # the reported result, not from re-training.
                if size_name != "XL":
                    continue
                cfg = {
                    "model_type": "transformer",
                    "model_dim": 512,
                    "num_layers": 9,
                    "ff_mult": 2,
                    "a_mode": "diag",
                    "_reference_only": True,
                }
            else:
                print(f"  [WARN] Unknown condition: {cond_name}, skipping")
                continue

            configs_at_size[cond_name] = cfg

            # Write YAML file
            path = CONFIGS_DIR / f"{size_name}_{cond_name}.yaml"
            path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))

        all_configs[size_name] = configs_at_size

    return all_configs


# ── Subprocess runner (Task 5) ──────────────────────────────────────


def _launch_config(
    config_path: Path,
    data_path: str,
    budget: float,
    seed: int,
    gpu_id: int | None = None,
) -> tuple[subprocess.Popen, Path, Path, object]:
    """Launch a config run as a background process.

    Returns (proc, out_path, tmp_path, log_fh).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{config_path.stem}_seed{seed}.json"

    cfg = yaml.safe_load(config_path.read_text())
    cfg["seed"] = seed
    tmp = config_path.parent / f".tmp_{config_path.stem}_s{seed}.yaml"
    tmp.write_text(yaml.dump(cfg, default_flow_style=False))

    cmd = [
        sys.executable, "-m", "chaoscontrol.runner",
        "--config", str(tmp),
        "--data-path", data_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path = RESULTS_DIR / f"{config_path.stem}_seed{seed}.log"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, out_path, tmp, log_fh


def run_single(
    config_path: Path,
    data_path: str,
    budget: float,
    seed: int,
    *,
    gpu_id: int | None = None,
) -> dict:
    """Run a single training config via chaoscontrol.runner subprocess.

    Skips if result JSON already exists (resume support).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{config_path.stem}_seed{seed}.json"

    # Skip if already completed
    if out_path.exists():
        print(f"    [SKIP] {out_path.name} already exists")
        return json.loads(out_path.read_text())

    # Write temp YAML with seed overridden
    cfg = yaml.safe_load(config_path.read_text())
    cfg["seed"] = seed
    tmp = config_path.parent / f".tmp_{config_path.stem}_s{seed}.yaml"
    tmp.write_text(yaml.dump(cfg, default_flow_style=False))

    cmd = [
        sys.executable, "-m", "chaoscontrol.runner",
        "--config", str(tmp),
        "--data-path", data_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(cmd, check=True, env=env)
    tmp.unlink(missing_ok=True)
    return json.loads(out_path.read_text())


def run_seeds_parallel(
    config_path: Path,
    data_path: str,
    budget: float,
    seeds: list[int],
    num_gpus: int,
) -> dict[int, dict]:
    """Run all seeds for a single config, parallelizing across GPUs.

    Returns {seed: result_dict}.
    """
    results: dict[int, dict] = {}

    # Check which seeds already have results
    pending_seeds = []
    for seed in seeds:
        out_path = RESULTS_DIR / f"{config_path.stem}_seed{seed}.json"
        if out_path.exists():
            print(f"    [SKIP] {out_path.name} already exists")
            results[seed] = json.loads(out_path.read_text())
        else:
            pending_seeds.append(seed)

    if not pending_seeds:
        return results

    if num_gpus > 1 and len(pending_seeds) > 1:
        # Parallel: launch seeds concurrently across GPUs
        active: list[tuple[int, subprocess.Popen, Path, Path, object]] = []
        for i, seed in enumerate(pending_seeds):
            gpu_id = i % num_gpus
            proc, out_path, tmp, log_fh = _launch_config(
                config_path, data_path, budget, seed, gpu_id,
            )
            active.append((seed, proc, out_path, tmp, log_fh))

        for seed, proc, out_path, tmp, log_fh in active:
            proc.wait()
            log_fh.close()
            tmp.unlink(missing_ok=True)
            if proc.returncode != 0:
                print(f"    *** {config_path.stem} seed={seed} FAILED (rc={proc.returncode})")
                log_path = RESULTS_DIR / f"{config_path.stem}_seed{seed}.log"
                if log_path.exists():
                    print(f"        See {log_path}")
                continue
            results[seed] = json.loads(out_path.read_text())
    else:
        # Sequential
        for seed in pending_seeds:
            results[seed] = run_single(config_path, data_path, budget, seed)

    return results


# ── Inference benchmarks (Task 9) ───────────────────────────────────


def benchmark_inference(model, device, seq_lengths=None):
    """Benchmark inference latency and memory footprint.

    Returns {"latency_per_token_ms": float, "state_bytes": dict}.
    Only runs when CUDA is available (latency measurements on CPU are
    not meaningful for comparison).
    """
    import torch

    if seq_lengths is None:
        seq_lengths = [1024, 4096, 16384]

    model.eval()
    results = {}
    use_cuda = device.type == "cuda"

    # Latency: time 1000 forward passes on single token
    dummy = torch.randint(0, model.vocab_size, (1, 1), device=device)
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            model(dummy)
        if use_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(1000):
            model(dummy)
        if use_cuda:
            torch.cuda.synchronize()
        results["latency_per_token_ms"] = (time.perf_counter() - t0)

    # Memory footprint at different sequence lengths
    state_bytes = {}
    for sl in seq_lengths:
        if hasattr(model, "init_state") and callable(getattr(model, "init_state", None)):
            # SSM: fixed state, independent of seq_len
            state = model.init_state(1)
            state_bytes[str(sl)] = sum(
                s.nelement() * s.element_size() for s in state
                if hasattr(s, "nelement")
            )
        else:
            # Transformer: KV cache grows with seq_len
            n_layers = len(model.layers) if hasattr(model, "layers") else 4
            dim = model.embed.embedding_dim if hasattr(model, "embed") else 128
            state_bytes[str(sl)] = 2 * n_layers * dim * sl * 2  # bf16
    results["state_bytes"] = state_bytes
    return results


# ── Quantization robustness (Task 10) ───────────────────────────────


def quantize_and_eval(model, tokens, eval_starts, seq_len, device, batch_size=32):
    """Quantize model to int8 and int6, measure bpb at each level.

    Returns {"bf16_bpb": float, "int8_bpb": float, "int6_bpb": float,
             "int8_delta": float, "int6_delta": float}.
    """
    import torch
    from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb

    total_raw_bytes = int(tokens.numel())

    # bf16 baseline
    bf16_result = evaluate_chaoscontrol_bpb(
        model, tokens=tokens, eval_starts=eval_starts,
        batch_size=batch_size, seq_len=seq_len, device=device,
        total_raw_bytes=total_raw_bytes,
    )
    bf16_bpb = bf16_result["bpb"]

    # int8: quantize all 2D weight tensors
    int8_model = copy.deepcopy(model)
    with torch.no_grad():
        for _name, param in int8_model.named_parameters():
            if param.ndim == 2:
                scale = param.abs().max() / 127
                if scale > 0:
                    param.data = (param.data / scale).round().clamp(-128, 127) * scale
    int8_result = evaluate_chaoscontrol_bpb(
        int8_model, tokens=tokens, eval_starts=eval_starts,
        batch_size=batch_size, seq_len=seq_len, device=device,
        total_raw_bytes=total_raw_bytes,
    )
    del int8_model

    # int6: 6-bit range (-32..31)
    int6_model = copy.deepcopy(model)
    with torch.no_grad():
        for _name, param in int6_model.named_parameters():
            if param.ndim == 2:
                scale = param.abs().max() / 31
                if scale > 0:
                    param.data = (param.data / scale).round().clamp(-32, 31) * scale
    int6_result = evaluate_chaoscontrol_bpb(
        int6_model, tokens=tokens, eval_starts=eval_starts,
        batch_size=batch_size, seq_len=seq_len, device=device,
        total_raw_bytes=total_raw_bytes,
    )
    del int6_model

    return {
        "bf16_bpb": bf16_bpb,
        "int8_bpb": int8_result["bpb"],
        "int6_bpb": int6_result["bpb"],
        "int8_delta": int8_result["bpb"] - bf16_bpb,
        "int6_delta": int6_result["bpb"] - bf16_bpb,
    }


# ── Component metrics (Task 11) ─────────────────────────────────────


def extract_component_metrics(train_result: dict) -> dict:
    """Extract component-specific metrics from training history."""
    history = train_result.get("history", [])
    metrics: dict = {}

    # Gate fire rate
    fires = [s.get("gate_fired", False) for s in history if "gate_fired" in s]
    metrics["gate_fire_rate"] = sum(fires) / max(len(fires), 1) if fires else 0.0

    # Memory slot utilization (unique buckets / max_slots)
    buckets_used: set = set()
    for s in history:
        if "dominant_bucket" in s and s["dominant_bucket"] is not None:
            buckets_used.add(s["dominant_bucket"])
    metrics["unique_buckets"] = len(buckets_used)

    # Codebook utilization (from Wernicke bucket snapshots)
    bucket_snaps = train_result.get("bucket_snapshots", [])
    if bucket_snaps:
        last_snap = bucket_snaps[-1]
        total_entries = max(len(last_snap), 1)
        active = sum(1 for v in last_snap if v > 0)
        metrics["wernicke_codebook_utilization"] = active / total_entries

    # Spectral structure
    spectral = train_result.get("spectral_snapshots", [])
    if spectral:
        last = spectral[-1]
        metrics["spectral_max_freq"] = float(max(last)) if last else 0.0
        metrics["spectral_mean_freq"] = float(sum(last) / len(last)) if last else 0.0

    return metrics


# ── Competition baseline (Task 6) ───────────────────────────────────


def run_competition_baseline(budget: float = 600) -> dict | None:
    """Run the competition baseline train_gpt.py and extract final val_bpb.

    Returns {"bpb": float, "source": "competition_baseline", ...} or None.
    """
    baseline_dir = REPO / "baselines" / "parameter_golf"
    script = baseline_dir / "train_gpt.py"
    if not script.exists():
        print("  [WARN] Competition baseline not found, using reference BPB")
        return {
            "bpb": COMP_BASELINE_BPB,
            "source": "competition_baseline_reference",
            "hyperparams": {
                "num_layers": 9, "model_dim": 512, "num_heads": 8,
                "num_kv_heads": 4, "mlp_mult": 2, "vocab_size": 1024,
                "seq_len": 1024,
            },
        }

    result_path = RESULTS_DIR / "comp_tfm_baseline.json"
    if result_path.exists():
        print("  [SKIP] Competition baseline result already exists")
        return json.loads(result_path.read_text())

    env = dict(os.environ)
    env["MAX_WALLCLOCK_SECONDS"] = str(budget)

    print(f"  Running competition baseline (budget={budget}s)...")
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(baseline_dir),
            capture_output=True,
            text=True,
            env=env,
            timeout=int(budget * 1.5),
        )
    except subprocess.TimeoutExpired:
        print("  [ERROR] Competition baseline timed out")
        return None

    if proc.returncode != 0:
        print(f"  [ERROR] Competition baseline failed (rc={proc.returncode})")
        return None

    # Extract final BPB from stdout
    bpb_match = re.search(
        r"final_int8_zlib_roundtrip_exact.*val_bpb:([\d.]+)", proc.stdout,
    )
    if not bpb_match:
        bpb_match = re.search(r"val_bpb:([\d.]+)", proc.stdout)
    if not bpb_match:
        print("  [ERROR] Could not extract BPB from competition baseline output")
        return None

    bpb = float(bpb_match.group(1))
    result = {
        "bpb": bpb,
        "source": "competition_baseline",
        "hyperparams": {
            "num_layers": 9, "model_dim": 512, "num_heads": 8,
            "num_kv_heads": 4, "mlp_mult": 2, "vocab_size": 1024,
            "seq_len": 1024,
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"  Competition baseline: bpb={bpb:.4f}")
    return result


# ── Main (Tasks 5, 8) ───────────────────────────────────────────────


def _check_mamba2_available() -> bool:
    """Check if mamba-ssm package is installed."""
    try:
        import importlib
        importlib.import_module("mamba_ssm")
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Scaling laws runner for experiment 10",
    )
    parser.add_argument(
        "--data-path", required=True,
        help="Path to data file (e.g. enwik8)",
    )
    parser.add_argument(
        "--budget", type=float, default=600,
        help="Per-run training budget in seconds (default: 600)",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of seeds to use, 1-3 (default: 3)",
    )
    parser.add_argument(
        "--sizes", type=str, default="XS,S,M,L,XL",
        help="Comma-separated sizes to run (default: XS,S,M,L,XL)",
    )
    parser.add_argument(
        "--conditions", type=str, default="bare_ssm,full_ssm,our_tfm,mamba2_ssm",
        help="Comma-separated conditions (default: bare_ssm,full_ssm,our_tfm,mamba2_ssm)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs for seed parallelism (default: 1)",
    )
    parser.add_argument(
        "--exp09-results", type=str, default=None,
        help="Path to experiment 09 results dir (auto-detected if omitted)",
    )
    args = parser.parse_args()

    seeds = SEEDS[:args.seeds]
    sizes = [s.strip() for s in args.sizes.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    # Check mamba2 availability
    if "mamba2_ssm" in conditions and not _check_mamba2_available():
        print("  [WARN] mamba-ssm not installed; skipping mamba2_ssm condition")
        conditions = [c for c in conditions if c != "mamba2_ssm"]

    # Load experiment 09 winners for full_ssm condition
    full_stack_settings: dict = {}
    if "full_ssm" in conditions:
        full_stack_settings = load_exp09_winners(args.exp09_results)
        if not full_stack_settings:
            print("  [WARN] Experiment 09 results not found; full_ssm will "
                  "fall back to bare_ssm settings")

    # Generate all configs
    all_configs = generate_configs(sizes, conditions, full_stack_settings)

    # Print parameter summary
    print("\n" + "=" * 72)
    print("  PARAMETER SUMMARY")
    print("=" * 72)
    print(f"  {'Size':<6} {'Condition':<14} {'Params':>12} {'Artifact (bf16)':>16}")
    print(f"  {'-' * 52}")
    for size_name in sizes:
        if size_name not in all_configs:
            continue
        for cond_name in conditions:
            if cond_name not in all_configs[size_name]:
                continue
            cfg = all_configs[size_name][cond_name]
            if cfg.get("_reference_only"):
                continue
            if cfg["model_type"] == "transformer":
                p = count_transformer_params(
                    cfg["model_dim"], cfg["num_layers"], cfg.get("ff_mult", 2),
                )
            else:
                p = count_ssm_params(
                    cfg["model_dim"], cfg["num_layers"], cfg.get("ff_mult", 2),
                )
            print(f"  {size_name:<6} {cond_name:<14} {p:>12,} {p * 2:>16,} bytes")
    print()

    # Run all configs
    total_runs = sum(
        len(seeds)
        for size_name in sizes if size_name in all_configs
        for cond_name in conditions
        if cond_name in all_configs.get(size_name, {})
        and not all_configs[size_name][cond_name].get("_reference_only")
    )
    completed = 0
    all_results: dict = {}
    t_start = time.time()

    for size_name in sizes:
        if size_name not in all_configs:
            continue
        all_results[size_name] = {}

        for cond_name in conditions:
            if cond_name not in all_configs[size_name]:
                continue
            cfg = all_configs[size_name][cond_name]
            if cfg.get("_reference_only"):
                continue

            config_path = CONFIGS_DIR / f"{size_name}_{cond_name}.yaml"
            all_results[size_name][cond_name] = {}

            print(f"\n  --- {size_name}/{cond_name} ---")
            seed_results = run_seeds_parallel(
                config_path, args.data_path, args.budget,
                seeds, args.num_gpus,
            )

            for seed, result in seed_results.items():
                completed += 1
                elapsed = time.time() - t_start
                eta = (elapsed / max(completed, 1)) * (total_runs - completed)

                # Augment result with FLOPs estimate
                flops_per_step = estimate_flops_per_step(
                    model_type=cfg["model_type"],
                    model_dim=cfg["model_dim"],
                    num_layers=cfg["num_layers"],
                    ff_mult=cfg.get("ff_mult", 2),
                    seq_len=cfg.get("seq_len", 256),
                    batch_size=cfg.get("batch_size", 64),
                    wernicke_enabled=cfg.get("wernicke_enabled", False),
                    wernicke_k_max=cfg.get("wernicke_k_max", 16),
                    outer_model_dim=cfg.get("outer_model_dim", 0),
                    outer_max_slots=cfg.get("outer_max_slots", 64),
                    metabolic_gate=cfg.get("metabolic_gate", False),
                    metabolic_k=cfg.get("metabolic_k", 4),
                )
                steps = result["train"]["steps"]
                result["flops_per_step"] = flops_per_step
                result["total_flops"] = flops_per_step * steps

                # Extract component metrics
                result["component_metrics"] = extract_component_metrics(
                    result["train"],
                )

                bpb = result["eval"].get("bpb_gated", result["eval"]["bpb"])
                print(f"  [{completed}/{total_runs}] {size_name}/{cond_name} "
                      f"seed={seed} bpb={bpb:.4f} params={result['params']:,} "
                      f"steps={steps} FLOPs={result['total_flops']:.2e} "
                      f"(elapsed={elapsed:.0f}s, ETA={eta:.0f}s)")

                all_results[size_name][cond_name][seed] = result

    # Competition baseline at XL
    if "comp_tfm" in conditions and "XL" in sizes:
        print("\n" + "=" * 72)
        print("  COMPETITION BASELINE (XL reference)")
        print("=" * 72)
        comp_result = run_competition_baseline(budget=args.budget)
        if comp_result:
            all_results["comp_tfm"] = comp_result

    # Save combined results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = RESULTS_DIR / "scaling_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Combined results saved to {combined_path}")

    # Print quick summary
    print("\n" + "=" * 72)
    print("  QUICK SUMMARY")
    print("=" * 72)
    for size_name in sizes:
        if size_name not in all_results:
            continue
        for cond_name in conditions:
            if cond_name not in all_results.get(size_name, {}):
                continue
            seed_data = all_results[size_name][cond_name]
            if not isinstance(seed_data, dict) or not seed_data:
                continue
            bpbs = []
            for _seed, r in seed_data.items():
                if isinstance(r, dict) and "eval" in r:
                    ev = r["eval"]
                    bpbs.append(ev.get("bpb_gated", ev["bpb"]))
            if bpbs:
                mean_bpb = statistics.mean(bpbs)
                std_bpb = statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0
                print(f"  {size_name:<6} {cond_name:<14} "
                      f"bpb={mean_bpb:.4f} +/- {std_bpb:.4f} ({len(bpbs)} seeds)")

    print("\n  Run analyze_scaling.py to generate plots and power-law fits.")


if __name__ == "__main__":
    main()
