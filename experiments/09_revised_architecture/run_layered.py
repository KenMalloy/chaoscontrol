#!/usr/bin/env python3
"""Layered experiment runner for experiment 09 — revised architecture.

Runs layers sequentially (L0 → L0.5 → L1 → … → L6), picking winners
after each layer and injecting their settings into subsequent layers.
Results are checkpointed to disk after every single run so partial
progress survives crashes.

Layer 0:   Tokenizer architecture (bytes / BPE / fixed-stride VQ)
Layer 0.5: Codebook alignment (only if L0 winner is a learned tokenizer)
Layer 1:   Gate modes
Layer 2:   +Memory
Layer 3:   +Wernicke + Regret
Layer 3.5: Dark horses
Layer 4:   Scaling
Layer 5:   Full A-mode
Layer 6:   Inference-time adaptation depth
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

# Local statistical utilities (pure Python, no scipy dependency)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import bootstrap_ci, cohens_d, sem as compute_sem, welch_ttest

REPO = Path(__file__).resolve().parents[2]
CONFIGS = Path(__file__).resolve().parent / "configs"
RESULTS = Path(__file__).resolve().parent / "results"
SEEDS = [1337, 2674, 4011]

# ── shared defaults ──────────────────────────────────────────────────
SHARED_DEFAULTS = {
    "model_dim": 128,
    "num_layers": 4,
    "a_mode": "diag",
    "model_type": "ssm",
}


# ── helpers ──────────────────────────────────────────────────────────
def run_config(config_path: Path, data_path: str, budget: float, seed: int,
               *, gpu_id: int | None = None) -> dict:
    """Run a single config with a given seed.  Returns parsed result dict."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{config_path.stem}_seed{seed}.json"

    # Write a temporary YAML with the seed overridden
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


def _launch_config(config_path: Path, data_path: str, budget: float, seed: int,
                   gpu_id: int | None = None) -> tuple[subprocess.Popen, Path, Path]:
    """Launch a config run as a background process.  Returns (proc, out_path, tmp_path)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{config_path.stem}_seed{seed}.json"

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
    log_path = RESULTS / f"{config_path.stem}_seed{seed}.log"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, out_path, tmp, log_fh


def run_layer(config_paths: list[Path], data_path: str, budget: float,
              seeds: list[int], layer_name: str,
              *, num_gpus: int = 1) -> dict:
    """Run all configs x seeds for a layer.  Returns {config_name: {seed: result}}.

    When num_gpus > 1, seeds for each config run in parallel across GPUs.
    Configs still run sequentially (later layers depend on earlier results).
    """
    results: dict[str, dict[int, dict]] = {}
    total = len(config_paths) * len(seeds)
    done = 0

    for cfg in sorted(config_paths):
        name = cfg.stem
        results[name] = {}

        if num_gpus > 1 and len(seeds) > 1:
            # Parallel: launch all seeds concurrently, one per GPU
            print(f"  [{layer_name}] {name} — launching {len(seeds)} seeds on {num_gpus} GPUs ...")
            active: list[tuple[int, subprocess.Popen, Path, Path, object]] = []

            for i, seed in enumerate(seeds):
                gpu_id = i % num_gpus
                proc, out_path, tmp, log_fh = _launch_config(cfg, data_path, budget, seed, gpu_id)
                active.append((seed, proc, out_path, tmp, log_fh))

            # Wait for all seeds to finish
            for seed, proc, out_path, tmp, log_fh in active:
                proc.wait()
                log_fh.close()
                tmp.unlink(missing_ok=True)
                done += 1
                if proc.returncode != 0:
                    print(f"  [{layer_name}] *** {name} seed={seed} FAILED (rc={proc.returncode})")
                    log_path = RESULTS / f"{cfg.stem}_seed{seed}.log"
                    if log_path.exists():
                        print(f"      See {log_path}")
                    continue
                results[name][seed] = json.loads(out_path.read_text())
                bpb = results[name][seed]["eval"].get("bpb_gated",
                          results[name][seed]["eval"]["bpb"])
                print(f"  [{layer_name}] ({done}/{total}) {name} seed={seed} → bpb={bpb:.4f}")
        else:
            # Sequential: single GPU or single seed
            for seed in seeds:
                done += 1
                print(f"  [{layer_name}] ({done}/{total}) {name} seed={seed} ...")
                results[name][seed] = run_config(cfg, data_path, budget, seed)

    return results


def pick_winner(layer_results: dict) -> tuple[str, float, float, dict]:
    """Pick config with lowest mean bpb across seeds.

    Returns (name, mean, std, stats_info) where *stats_info* contains:
        - ``significant``: bool -- winner significantly better than runner-up
        - ``p_value``: float -- Welch t-test p-value vs runner-up
        - ``ci_95``: tuple[float, float] -- 95 % bootstrap CI for winner mean
        - ``effect_size``: float -- Cohen's d between winner and runner-up
        - ``n_seeds``: int

    Prefers ``bpb_gated`` when available (metabolic gate was active during
    eval), falling back to plain ``bpb`` otherwise.
    """
    per_config: dict[str, list[float]] = {}
    for name, seed_results in layer_results.items():
        bpbs = []
        for r in seed_results.values():
            if not isinstance(r, dict) or "eval" not in r:
                continue
            ev = r["eval"]
            bpbs.append(ev.get("bpb_gated", ev["bpb"]))
        if bpbs:
            per_config[name] = bpbs

    if not per_config:
        print("  WARNING: No configs produced results. Cannot pick a winner.")
        return ("NONE", float("inf"), 0.0, {"significant": False, "p_value": 1.0,
                "ci_95": (0.0, 0.0), "effect_size": 0.0, "n_seeds": 0})

    # Sort configs by mean bpb (ascending -- lower is better)
    ranked = sorted(per_config.items(), key=lambda kv: statistics.mean(kv[1]))
    winner_name, winner_bpbs = ranked[0]
    winner_mean = statistics.mean(winner_bpbs)
    winner_std = statistics.stdev(winner_bpbs) if len(winner_bpbs) > 1 else 0.0

    # Significance vs runner-up
    stats_info: dict = {
        "significant": False,
        "p_value": 1.0,
        "ci_95": bootstrap_ci(winner_bpbs),
        "effect_size": 0.0,
        "n_seeds": len(winner_bpbs),
    }

    if len(ranked) >= 2:
        runner_bpbs = ranked[1][1]
        if len(winner_bpbs) >= 2 and len(runner_bpbs) >= 2:
            _t, p = welch_ttest(winner_bpbs, runner_bpbs)
            d = cohens_d(winner_bpbs, runner_bpbs)
            stats_info["p_value"] = p
            stats_info["effect_size"] = d
            stats_info["significant"] = p < 0.05
            if p >= 0.05:
                print(
                    f"  WARNING: winner {winner_name} is NOT significantly "
                    f"better than runner-up {ranked[1][0]} "
                    f"(p={p:.4f}, Cohen's d={d:.2f})"
                )
        else:
            print(
                f"  WARNING: cannot test significance with n<2 seeds "
                f"(winner n={len(winner_bpbs)}, runner-up n={len(runner_bpbs)})"
            )

    return winner_name, winner_mean, winner_std, stats_info


def print_layer_summary(layer_name: str, layer_results: dict):
    """Print a ranked table with SEM, bootstrap 95% CI, and significance."""
    # Collect per-config bpbs and compute statistics
    rows = []
    per_config_bpbs: dict[str, list[float]] = {}
    for name, seed_results in layer_results.items():
        bpbs = [r["eval"].get("bpb_gated", r["eval"]["bpb"])
                for r in seed_results.values() if isinstance(r, dict) and "eval" in r]
        if not bpbs:
            print(f"  {name:<30} NO RESULTS (all seeds failed)")
            continue
        per_config_bpbs[name] = bpbs
        mean = statistics.mean(bpbs)
        se = compute_sem(bpbs)
        ci = bootstrap_ci(bpbs)
        rows.append((name, mean, se, ci, len(bpbs), bpbs))
    rows.sort(key=lambda r: r[1])

    # Compute significance of each config vs the winner (row 0)
    sig_labels: dict[str, str] = {}
    winner_bpbs = rows[0][5] if rows else []
    for name, _mean, _se, _ci, n, bpbs in rows:
        if name == rows[0][0]:
            # Winner: test vs runner-up
            if len(rows) >= 2:
                runner_bpbs = rows[1][5]
                if len(winner_bpbs) >= 2 and len(runner_bpbs) >= 2:
                    _t, p = welch_ttest(winner_bpbs, runner_bpbs)
                    sig_labels[name] = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
                else:
                    sig_labels[name] = "n<2"
            else:
                sig_labels[name] = ""
        else:
            if len(winner_bpbs) >= 2 and len(bpbs) >= 2:
                _t, p = welch_ttest(bpbs, winner_bpbs)
                sig_labels[name] = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            else:
                sig_labels[name] = "n<2"

    print(f"\n  {layer_name}")
    print(f"  {'=' * 78}")
    hdr = (
        f"  {'Config':<36} {'Sig':>4} {'Mean BPB':>10} {'SEM':>9}"
        f"  {'95% CI':>20} {'Seeds':>6}"
    )
    print(hdr)
    print(f"  {'-' * 78}")
    for name, mean, se, ci, n, _bpbs in rows:
        sig = sig_labels.get(name, "")
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(
            f"  {name:<36} {sig:>4} {mean:10.4f} +/-{se:.4f}"
            f"  {ci_str:>20} {n:6}"
        )

    # Winner line with effect-size summary
    if len(rows) >= 2 and len(rows[0][5]) >= 2 and len(rows[1][5]) >= 2:
        _t, p = welch_ttest(rows[0][5], rows[1][5])
        d = cohens_d(rows[0][5], rows[1][5])
        print(
            f"  Winner: {rows[0][0]} "
            f"(p={p:.4f} vs runner-up, Cohen's d={d:.2f})"
        )
    elif rows:
        print(f"  Winner: {rows[0][0]} (significance not testable, n<2)")
    print()


def extract_gate_settings(config_path: Path) -> dict:
    """Extract gate-related fields from a config YAML."""
    cfg = yaml.safe_load(config_path.read_text())
    gate_keys = [
        "metabolic_gate", "metabolic_k", "metabolic_mode",
        "metabolic_threshold", "metabolic_threshold_mode",
        "mcts_horizon", "mcts_ucb_c",
    ]
    settings = {k: cfg[k] for k in gate_keys if k in cfg}
    # Explicit no-gate when baseline wins
    if "metabolic_gate" not in settings:
        settings["metabolic_gate"] = False
    return settings


def write_yaml(path: Path, data: dict):
    """Write a YAML config to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


# ── layer config generators ─────────────────────────────────────────
def generate_l0_configs() -> list[Path]:
    """Generate Layer 0 tokenizer-architecture configs.

    Phase 1: bytes, BPE, fixed-stride K=512, fixed-stride K=1024,
    plus a transformer baseline on raw bytes.
    All use diag A-mode, no gate, no memory, no Wernicke.
    """
    base = {**SHARED_DEFAULTS, "metabolic_gate": False}
    configs = {
        "L0_bytes": {
            **base,
            "tokenizer_type": "none",
            "vocab_size": 256,
        },
        # L0_bpe deferred: BPE is the competition's tokenizer (sp1024), not ours.
        # Comparing BPE-on-SSM is interesting but needs plumbing beyond a config
        # flag (uint16 shard loading, external SentencePiece model, etc.).
        # The meaningful comparison is raw bytes vs. our learned VQ tokenizer
        # vs. the competition transformer (which uses BPE internally).
        "L0_fixed_k512": {
            **base,
            "tokenizer_type": "fixed_stride",
            "tokenizer_stride": 4,
            "tokenizer_codebook_size": 512,
            "tokenizer_byte_dim": 64,
            "tokenizer_token_dim": 128,
            "tokenizer_beta": 0.25,
            "vocab_size": 512,
        },
        "L0_fixed_k1024": {
            **base,
            "tokenizer_type": "fixed_stride",
            "tokenizer_stride": 4,
            "tokenizer_codebook_size": 1024,
            "tokenizer_byte_dim": 64,
            "tokenizer_token_dim": 128,
            "tokenizer_beta": 0.25,
            "vocab_size": 1024,
        },
        "L0_bytes_tfm": {
            **base,
            "tokenizer_type": "none",
            "vocab_size": 256,
            "model_type": "transformer",
        },
    }
    paths = []
    for name, cfg in configs.items():
        p = CONFIGS / f"{name}.yaml"
        write_yaml(p, cfg)
        paths.append(p)
    return paths


def generate_l05_configs(tokenizer_settings: dict) -> list[Path]:
    """Generate Layer 0.5 codebook-alignment configs.

    Only meaningful when the L0 winner is a learned tokenizer (fixed_stride).
    Inherits the winning tokenizer settings and enables Wernicke so the
    alignment loss can couple the two codebooks.
    """
    base = {
        **SHARED_DEFAULTS,
        **tokenizer_settings,
        "metabolic_gate": False,
        "wernicke_enabled": True,
        "wernicke_router": "moe",
        "wernicke_k_max": 16,
    }
    configs = {
        "L05_align_none": {**base, "align_type": "none"},
        "L05_align_contrastive": {**base, "align_type": "contrastive", "align_weight": 0.05},
        "L05_align_diversity": {**base, "align_type": "diversity", "align_weight": 0.05},
        "L05_align_distill": {**base, "align_type": "distillation", "align_weight": 0.05},
    }
    paths = []
    for name, cfg in configs.items():
        p = CONFIGS / f"{name}.yaml"
        write_yaml(p, cfg)
        paths.append(p)
    return paths


def extract_tokenizer_settings(config_path: Path) -> dict:
    """Extract tokenizer-related fields from a config YAML."""
    cfg = yaml.safe_load(config_path.read_text())
    tok_keys = [
        "tokenizer_type", "tokenizer_stride", "tokenizer_byte_dim",
        "tokenizer_token_dim", "tokenizer_codebook_size", "tokenizer_beta",
        "vocab_size",
        "align_type", "align_weight",
    ]
    return {k: cfg[k] for k in tok_keys if k in cfg}


def generate_l2_configs(gate_settings: dict, tokenizer_settings: dict | None = None) -> list[Path]:
    """Generate Layer 2 configs with the L1 winner's gate settings injected."""
    base = {**SHARED_DEFAULTS, **(tokenizer_settings or {}), **gate_settings}
    configs = {
        "L2_mem_none_cold": {
            **base,
            "eval_warmup": False,
        },
        "L2_mem_epi_cold": {
            **base,
            "outer_model_dim": 64,
            "outer_model_type": "multislot",
            "eval_warmup": False,
        },
        "L2_mem_epi_warm": {
            **base,
            "outer_model_dim": 64,
            "outer_model_type": "multislot",
            "eval_warmup": True,
        },
        "L2_mem_both_cold": {
            **base,
            "outer_model_dim": 64,
            "outer_model_type": "multislot",
            "semantic_tier_bases": 8,
            "eval_warmup": False,
        },
        "L2_mem_both_warm": {
            **base,
            "outer_model_dim": 64,
            "outer_model_type": "multislot",
            "semantic_tier_bases": 8,
            "eval_warmup": True,
        },
        "L2_mem_both_warm_fullseq": {
            **base,
            "outer_model_dim": 64,
            "outer_model_type": "multislot",
            "semantic_tier_bases": 8,
            "eval_warmup": True,
            "consolidation_write": "full_sequence",
            "latent_persistence": True,
            "typed_consolidation": True,
        },
    }
    paths = []
    for name, cfg in configs.items():
        p = CONFIGS / f"{name}.yaml"
        write_yaml(p, cfg)
        paths.append(p)
    return paths


def generate_l3_configs(gate_settings: dict, mem_settings: dict,
                        tokenizer_settings: dict | None = None) -> list[Path]:
    """Generate Layer 3 configs with L1 gate + L2 memory winners."""
    base = {**SHARED_DEFAULTS, **(tokenizer_settings or {}), **gate_settings, **mem_settings}
    configs = {
        "L3_no_wernicke": {
            **base,
        },
        "L3_wernicke": {
            **base,
            "wernicke_enabled": True,
            "wernicke_router": "moe",
            "wernicke_k_max": 16,
        },
        "L3_wernicke_cfr": {
            **base,
            "wernicke_enabled": True,
            "wernicke_router": "moe",
            "wernicke_k_max": 16,
            "cfr_enabled": True,
            "typed_storage": True,
            # CFR needs the gate to generate counterfactuals
            "metabolic_gate": True,
            "metabolic_mode": base.get("metabolic_mode", "fork"),
            "metabolic_k": base.get("metabolic_k", 4),
            "metabolic_threshold": base.get("metabolic_threshold", 0.1),
        },
        "L3_wernicke_cfr_consequence": {
            **base,
            "wernicke_enabled": True,
            "wernicke_router": "moe",
            "wernicke_k_max": 16,
            "cfr_enabled": True,
            "typed_storage": True,
            "compression_consequence": True,
            # CFR needs the gate to generate counterfactuals
            "metabolic_gate": True,
            "metabolic_mode": base.get("metabolic_mode", "fork"),
            "metabolic_k": base.get("metabolic_k", 4),
            "metabolic_threshold": base.get("metabolic_threshold", 0.1),
        },
    }
    paths = []
    for name, cfg in configs.items():
        p = CONFIGS / f"{name}.yaml"
        write_yaml(p, cfg)
        paths.append(p)
    return paths


def generate_l35_configs(gate_settings: dict, mem_settings: dict,
                         tokenizer_settings: dict | None = None) -> list[Path]:
    """Generate Layer 3.5 dark-horse configs testing cross-layer interactions."""
    best_mem = {
        "outer_model_dim": 64,
        "outer_model_type": "multislot",
        "semantic_tier_bases": 8,
        "eval_warmup": True,
        "consolidation_write": "full_sequence",
        "latent_persistence": True,
        "typed_consolidation": True,
    }
    wernicke_cfr = {
        "wernicke_enabled": True,
        "wernicke_router": "moe",
        "wernicke_k_max": 16,
        "cfr_enabled": True,
        "typed_storage": True,
    }
    configs = {
        "L35_mcts_with_memory": {
            **SHARED_DEFAULTS,
            **(tokenizer_settings or {}),
            "metabolic_gate": True,
            "metabolic_k": 4,
            "metabolic_mode": "mcts",
            "mcts_horizon": 8,
            "metabolic_threshold": 0.1,
            "metabolic_threshold_mode": "fixed",
            **best_mem,
        },
        "L35_fork_with_cfr": {
            **SHARED_DEFAULTS,
            **(tokenizer_settings or {}),
            "metabolic_gate": True,
            "metabolic_k": 4,
            "metabolic_mode": "fork",
            "metabolic_threshold": 0.1,
            "metabolic_threshold_mode": "fixed",
            **wernicke_cfr,
            **mem_settings,
        },
        "L35_full_stack_mc": {
            **SHARED_DEFAULTS,
            **(tokenizer_settings or {}),
            "metabolic_gate": True,
            "metabolic_k": 4,
            "metabolic_mode": "monte_carlo",
            "metabolic_threshold": 0.1,
            "metabolic_threshold_mode": "fixed",
            **best_mem,
            **wernicke_cfr,
        },
    }
    paths = []
    for name, cfg in configs.items():
        p = CONFIGS / f"{name}.yaml"
        write_yaml(p, cfg)
        paths.append(p)
    return paths


def generate_l4_configs(full_stack: dict,
                        tokenizer_settings: dict | None = None) -> list[Path]:
    """Generate Layer 4 scaling configs."""
    # Transformer baseline inherits the winning tokenizer so both SSM and
    # transformer see the same input representation.
    tfm_tok = {k: v for k, v in (tokenizer_settings or {}).items()
               if k.startswith("tokenizer_") or k in ("vocab_size", "align_type", "align_weight")}
    configs = {
        "L4_full_128": {**full_stack, "model_dim": 128},
        "L4_full_256": {**full_stack, "model_dim": 256},
        "L4_full_384": {**full_stack, "model_dim": 384},
        "L4_tfm_384": {
            "model_type": "transformer",
            "model_dim": 384,
            "num_layers": 4,
            "a_mode": "diag",
            **tfm_tok,
        },
    }
    paths = []
    for name, cfg in configs.items():
        p = CONFIGS / f"{name}.yaml"
        write_yaml(p, cfg)
        paths.append(p)
    return paths


def generate_l5_configs(full_stack: dict) -> list[Path]:
    """Generate Layer 5 full A-mode configs (longer budget)."""
    configs = {
        "L5_full_128": {**full_stack, "model_dim": 128, "a_mode": "full"},
        "L5_full_256": {**full_stack, "model_dim": 256, "a_mode": "full"},
        "L5_full_384": {**full_stack, "model_dim": 384, "a_mode": "full"},
    }
    paths = []
    for name, cfg in configs.items():
        p = CONFIGS / f"{name}.yaml"
        write_yaml(p, cfg)
        paths.append(p)
    return paths


def generate_l6_configs(full_stack: dict) -> list[Path]:
    """Generate Layer 6: inference-time adaptation depth.

    Tests how many memory tiers participate during eval.
    The SSM recurrence always adapts (it IS the working memory).
    The question is whether deeper tiers (episodic, semantic, latent)
    improve inference when allowed to run during the eval forward pass.
    """
    # Base: full stack but with all eval-time adaptation OFF
    base_no_adapt = {**full_stack}
    base_no_adapt["eval_warmup"] = False

    configs = {
        # WM only: standard SSM inference, no memory writes during eval
        "L6_wm_only": {
            **base_no_adapt,
        },
        # WM + episodic: surprise-gated episodic writes during eval
        "L6_wm_plus_episodic": {
            **base_no_adapt,
            "eval_warmup": True,
        },
        # WM + all tiers: cold start -- wipe memory, rebuild during eval
        "L6_wm_plus_all": {
            **base_no_adapt,
            "eval_warmup": True,
            "warmup_write_mode": "full_sequence",
            "warmup_latent": True,
            "warmup_cold_start": True,
        },
        # WM + all tiers, seeded: keep training memory, eval adds to it
        "L6_wm_plus_all_seeded": {
            **base_no_adapt,
            "eval_warmup": True,
            "warmup_write_mode": "full_sequence",
            "warmup_latent": True,
            "warmup_cold_start": False,
        },
    }
    paths = []
    for name, cfg in configs.items():
        p = CONFIGS / f"{name}.yaml"
        write_yaml(p, cfg)
        paths.append(p)
    return paths


def extract_mem_settings(config_path: Path) -> dict:
    """Extract memory-related fields from a config YAML."""
    cfg = yaml.safe_load(config_path.read_text())
    mem_keys = [
        "outer_model_dim", "outer_model_type", "semantic_tier_bases",
        "eval_warmup", "consolidation_write", "latent_persistence",
        "typed_consolidation",
    ]
    return {k: cfg[k] for k in mem_keys if k in cfg}


def extract_wernicke_settings(config_path: Path) -> dict:
    """Extract Wernicke/CFR fields from a config YAML."""
    cfg = yaml.safe_load(config_path.read_text())
    keys = [
        "wernicke_enabled", "wernicke_router", "wernicke_k_max",
        "cfr_enabled", "typed_storage", "compression_consequence",
    ]
    return {k: cfg[k] for k in keys if k in cfg}


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Layered runner for experiment 09")
    parser.add_argument("--data-path", required=True, help="Path to FineWeb data directory")
    parser.add_argument("--budget", type=float, default=150, help="Per-run budget in seconds")
    parser.add_argument("--l5-budget", type=float, default=900, help="Layer 5 budget in seconds")
    parser.add_argument("--start-layer", type=int, default=0,
                        help="Layer to start from (0=tokenizer, 1=gate, …, 6=inference)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for seed parallelism (default: 1 = sequential)")
    args = parser.parse_args()

    data_path = args.data_path
    budget = args.budget
    l5_budget = args.l5_budget
    num_gpus = args.num_gpus

    summary: dict[str, dict] = {}
    tokenizer_settings: dict = {}

    # ── Layer 0: Tokenizer Architecture (5 configs x 3 seeds) ──────
    if args.start_layer <= 0:
        print("\n" + "=" * 72)
        print("  LAYER 0: Tokenizer Architecture")
        print("=" * 72)
        l0_configs = generate_l0_configs()
        l0_results = run_layer(l0_configs, data_path, budget, SEEDS, "L0", num_gpus=num_gpus)
        print_layer_summary("Layer 0: Tokenizer Architecture", l0_results)
        l0_winner, l0_mean, l0_std, l0_stats = pick_winner(l0_results)
        print(f"  >>> L0 winner: {l0_winner} (bpb={l0_mean:.4f} +/- {l0_std:.4f})")
        summary["L0"] = {"winner": l0_winner, "mean_bpb": l0_mean, "std_bpb": l0_std}

        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / "L0_summary.json", "w") as f:
            json.dump({"results": {n: {str(s): r for s, r in sr.items()} for n, sr in l0_results.items()},
                        "winner": l0_winner}, f, indent=2, default=str)
    else:
        summary_path = RESULTS / "L0_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                l0_data = json.load(f)
            l0_winner = l0_data["winner"]
            print(f"  [L0] Loaded winner from prior run: {l0_winner}")
        else:
            # No prior L0 run — fall back to raw bytes (existing behaviour).
            # Generate the default config so downstream extraction works.
            l0_winner = "L0_bytes"
            write_yaml(CONFIGS / "L0_bytes.yaml", {
                **SHARED_DEFAULTS, "metabolic_gate": False,
                "tokenizer_type": "none", "vocab_size": 256,
            })
            print(f"  [L0] No prior results, defaulting to: {l0_winner}")

    tokenizer_settings = extract_tokenizer_settings(CONFIGS / f"{l0_winner}.yaml")

    # ── Layer 0.5: Codebook Alignment (4 configs x 3 seeds) ────────
    # Only runs when L0 winner is a learned tokenizer (alignment is
    # meaningless for raw bytes or BPE).
    l0_winner_cfg = yaml.safe_load((CONFIGS / f"{l0_winner}.yaml").read_text())
    l0_uses_learned_tokenizer = l0_winner_cfg.get("tokenizer_type", "none") not in ("none",)

    if l0_uses_learned_tokenizer and args.start_layer <= 0:
        print("\n" + "=" * 72)
        print("  LAYER 0.5: Codebook Alignment")
        print("=" * 72)
        l05_configs = generate_l05_configs(tokenizer_settings)
        l05_results = run_layer(l05_configs, data_path, budget, SEEDS, "L0.5", num_gpus=num_gpus)
        print_layer_summary("Layer 0.5: Codebook Alignment", l05_results)
        l05_winner, l05_mean, l05_std, l05_stats = pick_winner(l05_results)
        print(f"  >>> L0.5 winner: {l05_winner} (bpb={l05_mean:.4f} +/- {l05_std:.4f})")
        summary["L0.5"] = {"winner": l05_winner, "mean_bpb": l05_mean, "std_bpb": l05_std}

        with open(RESULTS / "L05_summary.json", "w") as f:
            json.dump({"results": {n: {str(s): r for s, r in sr.items()} for n, sr in l05_results.items()},
                        "winner": l05_winner}, f, indent=2, default=str)

        # Update tokenizer_settings with the alignment winner
        tokenizer_settings = extract_tokenizer_settings(CONFIGS / f"{l05_winner}.yaml")
    elif l0_uses_learned_tokenizer and args.start_layer >= 1:
        summary_path = RESULTS / "L05_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                l05_data = json.load(f)
            l05_winner = l05_data["winner"]
            print(f"  [L0.5] Loaded winner from prior run: {l05_winner}")
            tokenizer_settings = extract_tokenizer_settings(CONFIGS / f"{l05_winner}.yaml")
    else:
        print("  [L0.5] Skipped (L0 winner is not a learned tokenizer)")

    # ── Layer 1: Gate Modes (7 configs x 3 seeds) ───────────────────
    if args.start_layer <= 1:
        print("\n" + "=" * 72)
        print("  LAYER 1: Gate Modes")
        print("=" * 72)
        l1_configs = sorted(CONFIGS.glob("L1_*.yaml"))
        l1_results = run_layer(l1_configs, data_path, budget, SEEDS, "L1", num_gpus=num_gpus)
        print_layer_summary("Layer 1: Gate Modes", l1_results)
        l1_winner, l1_mean, l1_std, l1_stats = pick_winner(l1_results)
        print(f"  >>> L1 winner: {l1_winner} (bpb={l1_mean:.4f} +/- {l1_std:.4f})")
        summary["L1"] = {"winner": l1_winner, "mean_bpb": l1_mean, "std_bpb": l1_std}

        # Save layer summary
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / "L1_summary.json", "w") as f:
            json.dump({"results": {n: {str(s): r for s, r in sr.items()} for n, sr in l1_results.items()},
                        "winner": l1_winner}, f, indent=2, default=str)
    else:
        # Load prior L1 results
        with open(RESULTS / "L1_summary.json") as f:
            l1_data = json.load(f)
        l1_winner = l1_data["winner"]
        print(f"  [L1] Loaded winner from prior run: {l1_winner}")

    gate_settings = extract_gate_settings(CONFIGS / f"{l1_winner}.yaml")

    # ── Layer 2: +Memory (6 configs x 3 seeds) ──────────────────────
    if args.start_layer <= 2:
        print("\n" + "=" * 72)
        print("  LAYER 2: +Memory")
        print("=" * 72)
        l2_configs = generate_l2_configs(gate_settings, tokenizer_settings)
        l2_results = run_layer(l2_configs, data_path, budget, SEEDS, "L2", num_gpus=num_gpus)
        print_layer_summary("Layer 2: +Memory", l2_results)
        l2_winner, l2_mean, l2_std, l2_stats = pick_winner(l2_results)
        print(f"  >>> L2 winner: {l2_winner} (bpb={l2_mean:.4f} +/- {l2_std:.4f})")
        summary["L2"] = {"winner": l2_winner, "mean_bpb": l2_mean, "std_bpb": l2_std}

        with open(RESULTS / "L2_summary.json", "w") as f:
            json.dump({"results": {n: {str(s): r for s, r in sr.items()} for n, sr in l2_results.items()},
                        "winner": l2_winner}, f, indent=2, default=str)
    else:
        with open(RESULTS / "L2_summary.json") as f:
            l2_data = json.load(f)
        l2_winner = l2_data["winner"]
        print(f"  [L2] Loaded winner from prior run: {l2_winner}")

    mem_settings = extract_mem_settings(CONFIGS / f"{l2_winner}.yaml")

    # ── Layer 3: +Wernicke + Regret (4 configs x 3 seeds) ───────────
    if args.start_layer <= 3:
        print("\n" + "=" * 72)
        print("  LAYER 3: +Wernicke + Regret")
        print("=" * 72)
        l3_configs = generate_l3_configs(gate_settings, mem_settings, tokenizer_settings)
        l3_results = run_layer(l3_configs, data_path, budget, SEEDS, "L3", num_gpus=num_gpus)
        print_layer_summary("Layer 3: +Wernicke + Regret", l3_results)
        l3_winner, l3_mean, l3_std, l3_stats = pick_winner(l3_results)
        print(f"  >>> L3 winner: {l3_winner} (bpb={l3_mean:.4f} +/- {l3_std:.4f})")
        summary["L3"] = {"winner": l3_winner, "mean_bpb": l3_mean, "std_bpb": l3_std}

        with open(RESULTS / "L3_summary.json", "w") as f:
            json.dump({"results": {n: {str(s): r for s, r in sr.items()} for n, sr in l3_results.items()},
                        "winner": l3_winner}, f, indent=2, default=str)
    else:
        with open(RESULTS / "L3_summary.json") as f:
            l3_data = json.load(f)
        l3_winner = l3_data["winner"]
        print(f"  [L3] Loaded winner from prior run: {l3_winner}")

    wernicke_settings = extract_wernicke_settings(CONFIGS / f"{l3_winner}.yaml")

    # ── Layer 3.5: Dark Horses (3 configs x 3 seeds) ─────────────────
    if args.start_layer <= 3:
        print("\n" + "=" * 72)
        print("  LAYER 3.5: Dark Horses")
        print("=" * 72)
        l35_configs = generate_l35_configs(gate_settings, mem_settings, tokenizer_settings)
        l35_results = run_layer(l35_configs, data_path, budget, SEEDS, "L3.5", num_gpus=num_gpus)
        print_layer_summary("Layer 3.5: Dark Horses", l35_results)

        with open(RESULTS / "L35_summary.json", "w") as f:
            json.dump({"results": {n: {str(s): r for s, r in sr.items()} for n, sr in l35_results.items()}},
                       f, indent=2, default=str)

    # Build full stack from all winners
    full_stack = {
        **SHARED_DEFAULTS,
        **tokenizer_settings,
        **gate_settings,
        **mem_settings,
        **wernicke_settings,
    }

    # ── Layer 4: Scaling (4 configs x 3 seeds) ───────────────────────
    if args.start_layer <= 4:
        print("\n" + "=" * 72)
        print("  LAYER 4: Scaling")
        print("=" * 72)
        l4_configs = generate_l4_configs(full_stack, tokenizer_settings)
        l4_results = run_layer(l4_configs, data_path, budget, SEEDS, "L4", num_gpus=num_gpus)
        print_layer_summary("Layer 4: Scaling", l4_results)

        with open(RESULTS / "L4_summary.json", "w") as f:
            json.dump({"results": {n: {str(s): r for s, r in sr.items()} for n, sr in l4_results.items()}},
                       f, indent=2, default=str)

    # ── Layer 5: Full A-mode (3 configs x 3 seeds) ───────────────────
    if args.start_layer <= 5:
        print("\n" + "=" * 72)
        print(f"  LAYER 5: Full A-mode (budget={l5_budget}s)")
        print("=" * 72)
        l5_configs = generate_l5_configs(full_stack)
        l5_results = run_layer(l5_configs, data_path, l5_budget, SEEDS, "L5", num_gpus=num_gpus)
        print_layer_summary("Layer 5: Full A-mode", l5_results)

        with open(RESULTS / "L5_summary.json", "w") as f:
            json.dump({"results": {n: {str(s): r for s, r in sr.items()} for n, sr in l5_results.items()}},
                       f, indent=2, default=str)

    # ── Layer 6: Inference-time adaptation depth (4 configs x 3 seeds) ─
    if args.start_layer <= 6:
        print("\n" + "=" * 72)
        print("  LAYER 6: Inference-Time Adaptation Depth")
        print("=" * 72)
        l6_configs = generate_l6_configs(full_stack)
        l6_results = run_layer(l6_configs, data_path, budget, SEEDS, "L6", num_gpus=num_gpus)
        print_layer_summary("Layer 6: Inference-Time Adaptation Depth", l6_results)
        l6_winner, l6_mean, l6_std, l6_stats = pick_winner(l6_results)
        print(f"  >>> L6 winner: {l6_winner} (bpb={l6_mean:.4f} +/- {l6_std:.4f})")
        summary["L6"] = {"winner": l6_winner, "mean_bpb": l6_mean, "std_bpb": l6_std}

        with open(RESULTS / "L6_summary.json", "w") as f:
            json.dump({"results": {n: {str(s): r for s, r in sr.items()} for n, sr in l6_results.items()},
                        "winner": l6_winner}, f, indent=2, default=str)

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CROSS-LAYER PROGRESSION")
    print("=" * 72)
    for layer, info in summary.items():
        print(f"  {layer}: {info['winner']} — bpb={info['mean_bpb']:.4f} +/- {info['std_bpb']:.4f}")
    print()

    with open(RESULTS / "full_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Results saved to {RESULTS}")


if __name__ == "__main__":
    main()
