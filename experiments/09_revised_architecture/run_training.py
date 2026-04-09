#!/usr/bin/env python3
"""Phase 1 training runner: sequential layer ablation with no gate, no CFR.

Layers:
  L0: Tokenizer (bytes / fixed_k512 / fixed_k1024 / bytes_tfm)
  L1: Memory tier (none / epi / epi_sem) — uses L0 winner
  L2: Wernicke routing (none / vq / moe) — uses L0+L1 winners
  L3: Scaling (128d/4L, 256d/6L, 384d/8L) — full winning stack

Each config runs 3 seeds. Winner selected by Welch t-test on bpb.
All runs save full checkpoints (model + tokenizer + memory state).
Multi-GPU: seeds run in parallel on available GPUs.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# Repo root = 3 levels up from this script
REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
RESULTS = EXPERIMENT / "results_phase1"

sys.path.insert(0, str(EXPERIMENT))
from stats import welch_ttest, bootstrap_ci, cohens_d, sem

SEEDS = [1337, 2674, 4011]
BUDGET = 150  # seconds per run


# ── Config templates ────────────────────────────────────────────────


def _base_config(**overrides) -> dict:
    """SSM config with gate OFF and CFR OFF."""
    base = {
        "model_type": "ssm",
        "vocab_size": 256,
        "model_dim": 128,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 256,
        "stride": 128,
        "batch_size": 64,
        "eval_batches": 32,
        "a_mode": "diag",
        "base_lr": 2e-3,
        "weight_decay": 1e-2,
        "grad_clip_norm": 1.0,
        # Gate OFF — all budget to gradient updates
        "metabolic_gate": False,
        "cfr_enabled": False,
    }
    base.update(overrides)
    return base


def l0_configs() -> list[tuple[str, dict]]:
    """Layer 0: Tokenizer ablation."""
    return [
        ("L0_bytes", _base_config()),
        ("L0_fixed_k512", _base_config(
            tokenizer_type="fixed_stride", tokenizer_codebook_size=512,
            tokenizer_stride=4, tokenizer_byte_dim=64, tokenizer_token_dim=128,
        )),
        ("L0_fixed_k1024", _base_config(
            tokenizer_type="fixed_stride", tokenizer_codebook_size=1024,
            tokenizer_stride=4, tokenizer_byte_dim=64, tokenizer_token_dim=128,
        )),
        ("L0_bytes_tfm", _base_config(model_type="transformer")),
    ]


def l1_configs(l0_winner: dict) -> list[tuple[str, dict]]:
    """Layer 1: Memory tier ablation. Inherits L0 winner's tokenizer."""
    tok_fields = {k: v for k, v in l0_winner.items()
                  if k.startswith("tokenizer_")}
    return [
        ("L1_mem_none", _base_config(
            outer_model_dim=0, **tok_fields,
        )),
        ("L1_mem_epi", _base_config(
            outer_model_dim=64, outer_model_type="multislot",
            outer_max_slots=64, outer_compress_ratio=2,
            consolidation_write="full_sequence",
            latent_persistence=True,
            **tok_fields,
        )),
        ("L1_mem_epi_sem", _base_config(
            outer_model_dim=64, outer_model_type="multislot",
            outer_max_slots=64, outer_compress_ratio=2,
            consolidation_write="full_sequence",
            latent_persistence=True,
            semantic_tier_bases=8,
            **tok_fields,
        )),
    ]


def l2_configs(l0_winner: dict, l1_winner: dict) -> list[tuple[str, dict]]:
    """Layer 2: Wernicke routing. Inherits L0 tokenizer + L1 memory."""
    tok_fields = {k: v for k, v in l0_winner.items()
                  if k.startswith("tokenizer_")}
    mem_fields = {k: v for k, v in l1_winner.items()
                  if k.startswith(("outer_", "consolidation_", "latent_", "semantic_"))}
    return [
        ("L2_wer_none", _base_config(**tok_fields, **mem_fields)),
        ("L2_wer_vq", _base_config(
            wernicke_enabled=True, wernicke_router="vq",
            wernicke_k_max=16, wernicke_window=8,
            typed_storage=True,
            **tok_fields, **mem_fields,
        )),
        ("L2_wer_moe", _base_config(
            wernicke_enabled=True, wernicke_router="moe",
            wernicke_k_max=16, wernicke_window=8,
            typed_storage=True,
            **tok_fields, **mem_fields,
        )),
    ]


def l3_configs(l0_winner: dict, l1_winner: dict, l2_winner: dict) -> list[tuple[str, dict]]:
    """Layer 3: Scaling. Full winning stack at 3 sizes."""
    tok_fields = {k: v for k, v in l0_winner.items()
                  if k.startswith("tokenizer_")}
    mem_fields = {k: v for k, v in l1_winner.items()
                  if k.startswith(("outer_", "consolidation_", "latent_", "semantic_"))}
    wer_fields = {k: v for k, v in l2_winner.items()
                  if k.startswith(("wernicke_", "typed_"))}
    common = {**tok_fields, **mem_fields, **wer_fields}
    return [
        ("L3_dim128", _base_config(model_dim=128, num_layers=4, **common)),
        ("L3_dim256", _base_config(model_dim=256, num_layers=6, **common)),
        ("L3_dim384", _base_config(model_dim=384, num_layers=8, **common)),
    ]


# ── Execution ──────────���────────────────────────────────────────────


def _launch_config(
    config_name: str,
    config_dict: dict,
    seed: int,
    data_path: str,
    budget: float,
    gpu_id: int | None,
    checkpoint_dir: Path,
) -> tuple[subprocess.Popen, Path, Path, object]:
    """Launch a single training run as a subprocess."""
    # Write temp config YAML
    config_dict = dict(config_dict, seed=seed)
    tmp = Path(tempfile.mktemp(suffix=".yaml", prefix=f".tmp_{config_name}_s{seed}_",
                               dir=EXPERIMENT / "configs"))
    tmp.write_text(yaml.dump(config_dict, default_flow_style=False))

    out_path = RESULTS / f"{config_name}_seed{seed}.json"
    clean_name = f"{config_name}_seed{seed}"
    cmd = [
        sys.executable, "-m", "chaoscontrol.runner",
        "--config", str(tmp),
        "--data-path", data_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
        "--checkpoint-dir", str(checkpoint_dir),
        "--checkpoint-name", clean_name,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = RESULTS / f"{config_name}_seed{seed}.log"
    log_fh = open(log_path, "w")

    # Tee to both log file and RunPod web console if available
    if Path("/proc/1/fd/1").exists():
        shell_cmd = " ".join(shlex.quote(str(c)) for c in cmd)
        shell_cmd += f" 2>&1 | tee {shlex.quote(str(log_path))} /proc/1/fd/1"
        proc = subprocess.Popen(
            ["bash", "-c", shell_cmd], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, out_path, tmp, log_fh


def run_layer(
    configs: list[tuple[str, dict]],
    data_path: str,
    budget: float,
    seeds: list[int],
    layer_name: str,
    checkpoint_dir: Path,
    num_gpus: int = 1,
) -> dict[str, list[dict]]:
    """Run all configs x seeds for a layer, return {config_name: [result_per_seed]}."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}

    for config_name, config_dict in configs:
        print(f"\n{'='*60}")
        print(f"  {layer_name} :: {config_name}  ({len(seeds)} seeds)")
        print(f"{'='*60}")

        # Launch seeds in parallel across GPUs
        jobs = []
        for i, seed in enumerate(seeds):
            gpu_id = i % num_gpus if num_gpus > 1 else None
            proc, out_path, tmp, log_fh = _launch_config(
                config_name, config_dict, seed, data_path, budget,
                gpu_id, checkpoint_dir,
            )
            jobs.append((proc, out_path, tmp, log_fh, seed))

        # Wait and collect
        seed_results = []
        for proc, out_path, tmp, log_fh, seed in jobs:
            proc.wait()
            log_fh.close()
            tmp.unlink(missing_ok=True)
            if proc.returncode != 0:
                print(f"  FAILED: {config_name} seed={seed} (exit {proc.returncode})")
                continue
            if out_path.exists():
                with open(out_path) as f:
                    result = json.load(f)
                bpb = result["eval"]["bpb"]
                steps = result["train"]["steps"]
                print(f"  seed={seed}: bpb={bpb:.4f}  steps={steps}")
                seed_results.append(result)
            else:
                print(f"  MISSING: {out_path}")

        results[config_name] = seed_results

    return results


def pick_winner(results: dict[str, list[dict]]) -> tuple[str, dict]:
    """Select the best config by lowest mean bpb. Report Welch t-test vs runner-up."""
    # Extract bpb values per config
    bpb_by_config: dict[str, list[float]] = {}
    for name, seed_results in results.items():
        bpbs = [r["eval"]["bpb"] for r in seed_results]
        if bpbs:
            bpb_by_config[name] = bpbs

    if not bpb_by_config:
        raise RuntimeError("No successful runs to pick a winner from")

    # Sort by mean bpb (lower is better)
    ranked = sorted(bpb_by_config.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    winner_name, winner_bpbs = ranked[0]
    winner_mean = sum(winner_bpbs) / len(winner_bpbs)
    winner_sem = sem(winner_bpbs)
    winner_ci = bootstrap_ci(winner_bpbs)

    print(f"\n  Winner: {winner_name}")
    print(f"    mean bpb = {winner_mean:.4f} +/- {winner_sem:.4f}")
    print(f"    95% CI: [{winner_ci[0]:.4f}, {winner_ci[1]:.4f}]")

    # Statistical comparison vs runner-up
    if len(ranked) >= 2:
        runnerup_name, runnerup_bpbs = ranked[1]
        t_stat, p_val = welch_ttest(winner_bpbs, runnerup_bpbs)
        d = cohens_d(winner_bpbs, runnerup_bpbs)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"    vs {runnerup_name}: t={t_stat:.2f}, p={p_val:.4f} {sig}, d={d:.2f}")

    # Full summary table
    print(f"\n  {'Config':<25} {'mean bpb':>10} {'SEM':>8} {'steps':>8}")
    print(f"  {'-'*55}")
    for name, bpbs in ranked:
        mean_bpb = sum(bpbs) / len(bpbs)
        mean_steps = sum(
            r["train"]["steps"]
            for r in results[name]
        ) / len(results[name])
        marker = " <-- WINNER" if name == winner_name else ""
        print(f"  {name:<25} {mean_bpb:>10.4f} {sem(bpbs):>8.4f} {mean_steps:>8.0f}{marker}")

    # Return winning config dict
    winner_config = results[winner_name][0]["config"]
    return winner_name, winner_config


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Training matrix (no gate, no CFR)")
    parser.add_argument("--data-path", required=True, help="Path to FineWeb data dir with docs_raw.txt")
    parser.add_argument("--budget", type=float, default=BUDGET, help="Training budget per run (seconds)")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for seed parallelism")
    parser.add_argument("--resume-from-layer", type=int, default=0, help="Skip layers before this (0=start fresh)")
    parser.add_argument("--checkpoint-dir", default=str(EXPERIMENT / "checkpoints"),
                        help="Directory for model checkpoints")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    state_file = RESULTS / "layer_state.json"
    layer_state: dict = {}

    # Resume support: reload previous layer winners
    if state_file.exists():
        with open(state_file) as f:
            layer_state = json.load(f)
        print(f"Resumed from {state_file} (layers completed: {list(layer_state.keys())})")

    def save_state():
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(layer_state, f, indent=2, default=str)

    # ── Layer 0: Tokenizer ──
    if args.resume_from_layer <= 0:
        print("\n" + "="*70)
        print("  LAYER 0: Tokenizer ablation")
        print("="*70)
        results = run_layer(
            l0_configs(), args.data_path, args.budget,
            SEEDS, "L0", checkpoint_dir, args.num_gpus,
        )
        l0_winner_name, l0_winner_config = pick_winner(results)
        layer_state["L0"] = {"winner": l0_winner_name, "config": l0_winner_config}
        save_state()
    else:
        l0_winner_config = layer_state["L0"]["config"]
        print(f"Skipping L0 (winner: {layer_state['L0']['winner']})")

    # ── Layer 1: Memory tier ──
    if args.resume_from_layer <= 1:
        print("\n" + "="*70)
        print("  LAYER 1: Memory tier ablation")
        print("="*70)
        results = run_layer(
            l1_configs(l0_winner_config), args.data_path, args.budget,
            SEEDS, "L1", checkpoint_dir, args.num_gpus,
        )
        l1_winner_name, l1_winner_config = pick_winner(results)
        layer_state["L1"] = {"winner": l1_winner_name, "config": l1_winner_config}
        save_state()
    else:
        l1_winner_config = layer_state["L1"]["config"]
        print(f"Skipping L1 (winner: {layer_state['L1']['winner']})")

    # ── Layer 2: Wernicke ──
    if args.resume_from_layer <= 2:
        print("\n" + "="*70)
        print("  LAYER 2: Wernicke routing ablation")
        print("="*70)
        results = run_layer(
            l2_configs(l0_winner_config, l1_winner_config),
            args.data_path, args.budget,
            SEEDS, "L2", checkpoint_dir, args.num_gpus,
        )
        l2_winner_name, l2_winner_config = pick_winner(results)
        layer_state["L2"] = {"winner": l2_winner_name, "config": l2_winner_config}
        save_state()
    else:
        l2_winner_config = layer_state["L2"]["config"]
        print(f"Skipping L2 (winner: {layer_state['L2']['winner']})")

    # ── Layer 3: Scaling ──
    print("\n" + "="*70)
    print("  LAYER 3: Scaling (full winning stack)")
    print("="*70)
    results = run_layer(
        l3_configs(l0_winner_config, l1_winner_config, l2_winner_config),
        args.data_path, args.budget,
        SEEDS, "L3", checkpoint_dir, args.num_gpus,
    )
    # Don't pick a single winner for L3 — all 3 scales feed into Phase 2
    print("\n  Layer 3 complete. All checkpoints feed into Phase 2 eval ablation.")

    # Save final state
    bpb_by_scale = {}
    for name, seed_results in results.items():
        bpbs = [r["eval"]["bpb"] for r in seed_results]
        if bpbs:
            bpb_by_scale[name] = {"mean_bpb": sum(bpbs) / len(bpbs), "sem": sem(bpbs)}
    layer_state["L3"] = {"results": bpb_by_scale}
    save_state()

    # ── Summary ──
    print("\n" + "="*70)
    print("  PHASE 1 COMPLETE")
    print("="*70)
    print(f"  L0 winner: {layer_state['L0']['winner']}")
    print(f"  L1 winner: {layer_state['L1']['winner']}")
    print(f"  L2 winner: {layer_state['L2']['winner']}")
    print(f"  L3 scales: {list(bpb_by_scale.keys())}")
    print(f"\n  Checkpoints: {checkpoint_dir}")
    print(f"  Results:     {RESULTS}")
    print(f"\n  Next: run_eval_ablation.py --checkpoint-dir {checkpoint_dir} --data-path {args.data_path}")


if __name__ == "__main__":
    main()
