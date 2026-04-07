#!/usr/bin/env python3
"""Layered experiment runner for experiment 09 — revised architecture.

Runs 5 layers sequentially, picking winners after each layer and injecting
their settings into subsequent layers.  Results are checkpointed to disk
after every single run so partial progress survives crashes.
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
def run_config(config_path: Path, enwik8_path: str, budget: float, seed: int) -> dict:
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
        "--enwik8-path", enwik8_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)
    tmp.unlink(missing_ok=True)
    return json.loads(out_path.read_text())


def run_layer(config_paths: list[Path], enwik8_path: str, budget: float,
              seeds: list[int], layer_name: str) -> dict:
    """Run all configs x seeds for a layer.  Returns {config_name: {seed: result}}."""
    results: dict[str, dict[int, dict]] = {}
    total = len(config_paths) * len(seeds)
    done = 0
    for cfg in sorted(config_paths):
        name = cfg.stem
        results[name] = {}
        for seed in seeds:
            done += 1
            print(f"  [{layer_name}] ({done}/{total}) {name} seed={seed} ...")
            results[name][seed] = run_config(cfg, enwik8_path, budget, seed)
    return results


def pick_winner(layer_results: dict) -> tuple[str, float, float]:
    """Pick config with lowest mean bpb across seeds.  Returns (name, mean, std)."""
    stats: dict[str, tuple[float, float]] = {}
    for name, seed_results in layer_results.items():
        bpbs = [r["eval"]["bpb"] for r in seed_results.values()]
        mean = statistics.mean(bpbs)
        std = statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0
        stats[name] = (mean, std)
    winner = min(stats, key=lambda k: stats[k][0])
    return winner, stats[winner][0], stats[winner][1]


def print_layer_summary(layer_name: str, layer_results: dict):
    """Print a ranked table for the layer."""
    rows = []
    for name, seed_results in layer_results.items():
        bpbs = [r["eval"]["bpb"] for r in seed_results.values()]
        mean = statistics.mean(bpbs)
        std = statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0
        rows.append((name, mean, std, len(bpbs)))
    rows.sort(key=lambda r: r[1])
    print(f"\n{'=' * 72}")
    print(f"  {layer_name} — ranked by mean bpb")
    print(f"{'=' * 72}")
    print(f"  {'Config':<40} {'Mean BPB':>10} {'Std':>8} {'Seeds':>6}")
    print(f"  {'-' * 66}")
    for name, mean, std, n in rows:
        flag = " *" if name == rows[0][0] else ""
        print(f"  {name:<40} {mean:10.4f} {std:8.4f} {n:6}{flag}")
    # Warn if top-2 have overlapping standard errors
    if len(rows) >= 2:
        m1, s1 = rows[0][1], rows[0][2]
        m2, s2 = rows[1][1], rows[1][2]
        if m1 + s1 > m2 - s2:
            print(f"  *** WARNING: top-2 ({rows[0][0]}, {rows[1][0]}) have overlapping SE")
    print()


def extract_gate_settings(config_path: Path) -> dict:
    """Extract gate-related fields from a config YAML."""
    cfg = yaml.safe_load(config_path.read_text())
    gate_keys = [
        "metabolic_gate", "metabolic_k", "metabolic_mode",
        "metabolic_threshold", "metabolic_threshold_mode",
        "mcts_horizon", "mcts_ucb_c",
    ]
    return {k: cfg[k] for k in gate_keys if k in cfg}


def write_yaml(path: Path, data: dict):
    """Write a YAML config to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


# ── layer config generators ─────────────────────────────────────────
def generate_l2_configs(gate_settings: dict) -> list[Path]:
    """Generate Layer 2 configs with the L1 winner's gate settings injected."""
    base = {**SHARED_DEFAULTS, **gate_settings}
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


def generate_l3_configs(gate_settings: dict, mem_settings: dict) -> list[Path]:
    """Generate Layer 3 configs with L1 gate + L2 memory winners."""
    base = {**SHARED_DEFAULTS, **gate_settings, **mem_settings}
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
        },
        "L3_wernicke_cfr_consequence": {
            **base,
            "wernicke_enabled": True,
            "wernicke_router": "moe",
            "wernicke_k_max": 16,
            "cfr_enabled": True,
            "typed_storage": True,
            "compression_consequence": True,
        },
    }
    paths = []
    for name, cfg in configs.items():
        p = CONFIGS / f"{name}.yaml"
        write_yaml(p, cfg)
        paths.append(p)
    return paths


def generate_l35_configs(gate_settings: dict, mem_settings: dict) -> list[Path]:
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


def generate_l4_configs(full_stack: dict) -> list[Path]:
    """Generate Layer 4 scaling configs."""
    configs = {
        "L4_full_128": {**full_stack, "model_dim": 128},
        "L4_full_256": {**full_stack, "model_dim": 256},
        "L4_full_384": {**full_stack, "model_dim": 384},
        "L4_tfm_384": {
            "model_type": "transformer",
            "model_dim": 384,
            "num_layers": 4,
            "a_mode": "diag",
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
        # WM + all tiers: episodic + semantic consolidation + latent reactivation
        "L6_wm_plus_all": {
            **base_no_adapt,
            "eval_warmup": True,
            "consolidation_write": "full_sequence",
            "latent_persistence": True,
            "typed_consolidation": True,
        },
        # WM + all tiers, seeded from compressed LTM (not cold start)
        # Same as above but the model was trained with these features ON,
        # so episodic slots already exist from training. Eval adds to them.
        "L6_wm_plus_all_seeded": {
            **base_no_adapt,
            "eval_warmup": True,
            "consolidation_write": "full_sequence",
            "latent_persistence": True,
            "typed_consolidation": True,
            # The "seeded" part: training had memory enabled, so slots carry over.
            # This config is identical to L6_wm_plus_all but the distinction
            # matters if we add pre-eval rehydration later.
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
    parser.add_argument("--enwik8-path", required=True, help="Path to enwik8 data file")
    parser.add_argument("--budget", type=float, default=150, help="Per-run budget in seconds")
    parser.add_argument("--l5-budget", type=float, default=900, help="Layer 5 budget in seconds")
    parser.add_argument("--start-layer", type=int, default=1, help="Layer to start from (1-6)")
    args = parser.parse_args()

    enwik8_path = args.enwik8_path
    budget = args.budget
    l5_budget = args.l5_budget

    summary: dict[str, dict] = {}

    # ── Layer 1: Gate Modes (7 configs x 3 seeds) ───────────────────
    if args.start_layer <= 1:
        print("\n" + "=" * 72)
        print("  LAYER 1: Gate Modes")
        print("=" * 72)
        l1_configs = sorted(CONFIGS.glob("L1_*.yaml"))
        l1_results = run_layer(l1_configs, enwik8_path, budget, SEEDS, "L1")
        print_layer_summary("Layer 1: Gate Modes", l1_results)
        l1_winner, l1_mean, l1_std = pick_winner(l1_results)
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
        l2_configs = generate_l2_configs(gate_settings)
        l2_results = run_layer(l2_configs, enwik8_path, budget, SEEDS, "L2")
        print_layer_summary("Layer 2: +Memory", l2_results)
        l2_winner, l2_mean, l2_std = pick_winner(l2_results)
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
        l3_configs = generate_l3_configs(gate_settings, mem_settings)
        l3_results = run_layer(l3_configs, enwik8_path, budget, SEEDS, "L3")
        print_layer_summary("Layer 3: +Wernicke + Regret", l3_results)
        l3_winner, l3_mean, l3_std = pick_winner(l3_results)
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

    # ── Layer 3.5: Dark Horses (3 configs x 1 seed) ─────────────────
    if args.start_layer <= 3:
        print("\n" + "=" * 72)
        print("  LAYER 3.5: Dark Horses")
        print("=" * 72)
        l35_configs = generate_l35_configs(gate_settings, mem_settings)
        l35_results = run_layer(l35_configs, enwik8_path, budget, [SEEDS[0]], "L3.5")
        print_layer_summary("Layer 3.5: Dark Horses", l35_results)

        with open(RESULTS / "L35_summary.json", "w") as f:
            json.dump({"results": {n: {str(s): r for s, r in sr.items()} for n, sr in l35_results.items()}},
                       f, indent=2, default=str)

    # Build full stack from all winners
    full_stack = {
        **SHARED_DEFAULTS,
        **gate_settings,
        **mem_settings,
        **wernicke_settings,
    }

    # ── Layer 4: Scaling (4 configs x 1 seed) ───────────────────────
    if args.start_layer <= 4:
        print("\n" + "=" * 72)
        print("  LAYER 4: Scaling")
        print("=" * 72)
        l4_configs = generate_l4_configs(full_stack)
        l4_results = run_layer(l4_configs, enwik8_path, budget, [SEEDS[0]], "L4")
        print_layer_summary("Layer 4: Scaling", l4_results)

        with open(RESULTS / "L4_summary.json", "w") as f:
            json.dump({"results": {n: {str(s): r for s, r in sr.items()} for n, sr in l4_results.items()}},
                       f, indent=2, default=str)

    # ── Layer 5: Full A-mode (3 configs x 1 seed, 1800s budget) ─────
    if args.start_layer <= 5:
        print("\n" + "=" * 72)
        print(f"  LAYER 5: Full A-mode (budget={l5_budget}s)")
        print("=" * 72)
        l5_configs = generate_l5_configs(full_stack)
        l5_results = run_layer(l5_configs, enwik8_path, l5_budget, [SEEDS[0]], "L5")
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
        l6_results = run_layer(l6_configs, enwik8_path, budget, SEEDS, "L6")
        print_layer_summary("Layer 6: Inference-Time Adaptation Depth", l6_results)
        l6_winner, l6_mean, l6_std = pick_winner(l6_results)
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
