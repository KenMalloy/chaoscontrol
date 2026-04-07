# Experiment 10: Scaling Laws — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether the SESSM scales more parameter-efficiently than a transformer under the 16MB artifact constraint by training parameter-matched SSM and transformer configs at 5 sizes (XS through XL), fitting power-law scaling curves, and comparing against the competition baseline.

**Architecture:** The runner reads experiment 09's `full_summary.json` plus layer summary files to reconstruct the winning component stack (gate mode from L1, memory tier from L2, Wernicke/CFR from L3). It then generates `ChaosControlConfig` YAML files for `bare_ssm` and `full_ssm` at each of 5 sizes, and uses `SimpleTransformerLM` via `model_type: transformer` for `our_tfm`. The competition baseline (`train_gpt.py`) runs separately and its BPB is extracted from stdout.

**Tech Stack:** PyTorch. No new dependencies. (matplotlib for analysis plots, scipy.optimize.curve_fit for power-law fitting -- both already available in the environment.)

---

### Task 1: Create directory structure and shell entry point

**Files:** `experiments/10_scaling_laws/run.sh`

**Problem/Goal:** Provide a single shell entry point that runs the full experiment, matching the pattern used by experiment 09.

**Implementation:**

```bash
#!/usr/bin/env bash
set -euo pipefail
ENWIK8="${1:?Usage: run.sh /path/to/enwik8 [--budget SECONDS] [--seeds N] [--sizes XS,S,M,L,XL] [--conditions bare_ssm,full_ssm,our_tfm]}"
shift
BUDGET=600
SEEDS=3
SIZES="XS,S,M,L,XL"
CONDITIONS="bare_ssm,full_ssm,our_tfm"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget) BUDGET="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    --conditions) CONDITIONS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python3 "$(dirname "$0")/run_scaling.py" \
    --enwik8-path "$ENWIK8" \
    --budget "$BUDGET" \
    --seeds "$SEEDS" \
    --sizes "$SIZES" \
    --conditions "$CONDITIONS"
```

**Test:** `bash experiments/10_scaling_laws/run.sh --help` should show usage (the script will fail on the required arg, printing the usage string).

---

### Task 2: FLOPs estimation utility

**Files:** `experiments/10_scaling_laws/run_scaling.py` (this function lives at the top of the runner)

**Problem/Goal:** Estimate training FLOPs per step for SSM and transformer configs so we can produce isoFLOP comparisons. This must work from config values alone (no model instantiation needed).

**Implementation:**

```python
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

    Returns an integer FLOPs count.
    """
    tokens = batch_size * seq_len
    d = model_dim
    L = num_layers

    if model_type == "transformer":
        # Per layer: attention QKV proj (3*d*d) + attn output (d*d) + FFN (2*d*d*ff_mult)
        # Each matmul: 2*M*N*K FLOPs (fwd), doubled for bwd => 6x total
        # Attention score compute: 2*d*seq_len per token (fwd) => 6x
        per_layer_params = 4 * d * d + 2 * d * d * ff_mult
        # Embed + lm_head: 2 * vocab * d
        embed_params = 2 * 256 * d
        total_params = L * per_layer_params + embed_params
        # 6N * T approximation
        base_flops = 6 * total_params * tokens
        # Add quadratic attention cost: 6 * L * 2 * seq_len * d * tokens
        # (QK dot products + softmax @ V, both fwd and bwd)
        attn_flops = 6 * L * 2 * seq_len * d
        base_flops += attn_flops * batch_size
    else:
        # SSM per layer:
        #   ChaosSSMCore: in_proj(d,d) + select_proj(d,d) + gate_proj(d,d) + out_proj(d,d)
        #                 + delta_proj(d,d) + log_a(d) => ~5*d*d params (diag mode)
        #   FFN: 2*d*d*ff_mult
        #   RMSNorm: 2*d
        per_layer_params = 5 * d * d + 2 * d * d * ff_mult
        embed_params = 2 * 256 * d
        total_params = L * per_layer_params + embed_params
        base_flops = 6 * total_params * tokens

    # Component corrections (additive)
    component_flops = 0
    if wernicke_enabled:
        # VQ routing: 2 * d * K per token
        component_flops += 6 * 2 * d * wernicke_k_max * tokens
    if outer_model_dim > 0:
        # Memory read/write: 2 * d * max_slots per token
        component_flops += 6 * 2 * d * outer_max_slots * tokens
    if metabolic_gate:
        # Fork cost: k copies of the forward pass at gate points
        # Approximation: metabolic_k * per_layer_params * tokens (one layer re-eval)
        component_flops += metabolic_k * per_layer_params * tokens

    return int(base_flops + component_flops)
```

**Test:** Call the function with known values and verify output is reasonable. For example, a 128-dim, 4-layer transformer with ff_mult=2, seq_len=256, batch_size=64 should yield roughly `6 * ~200K * 16384 ~ 20 GFLOPs`. Print the estimate and sanity check.

---

### Task 3: Config generation (parameter-matching SSM and transformer)

**Files:** `experiments/10_scaling_laws/run_scaling.py`

**Problem/Goal:** Generate `ChaosControlConfig`-compatible YAML dicts for `bare_ssm`, `full_ssm`, and `our_tfm` at each of the 5 sizes, with transformer configs adjusted so param counts match the SSM within 5%.

**Implementation:**

The size table from the design doc:

```python
SIZE_TABLE = {
    "XS": {"dim": 64,  "ssm_layers": 2},
    "S":  {"dim": 128, "ssm_layers": 4},
    "M":  {"dim": 256, "ssm_layers": 6},
    "L":  {"dim": 384, "ssm_layers": 8},
    "XL": {"dim": 512, "ssm_layers": 10},
}
```

Parameter counting functions (no model instantiation -- pure arithmetic):

```python
def count_ssm_params(dim: int, num_layers: int, ff_mult: int = 2) -> int:
    """Count params for a bare ChaosStudentLM (diag mode, no components)."""
    # Embed + lm_head
    embed = 256 * dim
    lm_head = dim * 256
    # RMSNorm (final): dim (weight only)
    final_norm = dim
    # Per SSM layer (diag mode):
    #   ChaosSSMCore: in_proj(d,d) + select_proj(d,d) + gate_proj(d,d) + out_proj(d,d)
    #                 + delta_proj(d,d) + log_a(d) = 5*d*d + d
    #   FeedForward: fc(d, d*ff_mult) + proj(d*ff_mult, d) = 2*d*d*ff_mult
    #   RMSNorm x2: 2*d
    per_layer = 5 * dim * dim + dim + 2 * dim * dim * ff_mult + 2 * dim
    total = embed + lm_head + final_norm + num_layers * per_layer
    return total


def count_transformer_params(dim: int, num_layers: int, ff_mult: int = 2) -> int:
    """Count params for SimpleTransformerLM."""
    # Embed + lm_head
    embed = 256 * dim
    lm_head = dim * 256
    # RMSNorm (final): dim
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

    Strategy: fix dim to match SSM dim, fix ff_mult=2, sweep num_layers.
    If no exact layer count works, try ff_mult=3 with fewer layers.
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
```

Config generation using `full_stack_settings` from experiment 09:

```python
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEEDS = [1337, 2674, 4011]


def generate_configs(full_stack_settings: dict) -> dict[str, dict[str, dict]]:
    """Generate all configs for all sizes and conditions.

    Returns {size_name: {condition_name: config_dict}}.
    Also writes each config as a YAML file to CONFIGS_DIR.
    """
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    all_configs = {}

    for size_name, size_spec in SIZE_TABLE.items():
        dim = size_spec["dim"]
        ssm_layers = size_spec["ssm_layers"]
        configs_at_size = {}

        # bare_ssm: pure SSM, no components
        bare = {
            "model_type": "ssm",
            "model_dim": dim,
            "num_layers": ssm_layers,
            "ff_mult": 2,
            "a_mode": "diag",
        }
        configs_at_size["bare_ssm"] = bare

        # full_ssm: SSM + winning component stack from exp 09
        full = {**bare, **full_stack_settings}
        # Override dim/layers to this size (full_stack_settings has exp09's dim/layers)
        full["model_dim"] = dim
        full["num_layers"] = ssm_layers
        configs_at_size["full_ssm"] = full

        # our_tfm: parameter-matched transformer
        tfm_match = match_transformer_to_ssm(dim, ssm_layers, ff_mult=2)
        tfm = {
            "model_type": "transformer",
            "model_dim": dim,
            "num_layers": tfm_match["num_layers"],
            "ff_mult": tfm_match["ff_mult"],
            "a_mode": "diag",  # ignored for transformer, but ChaosControlConfig requires it
        }
        configs_at_size["our_tfm"] = tfm

        all_configs[size_name] = configs_at_size

        # Write YAML files
        for cond_name, cfg_dict in configs_at_size.items():
            path = CONFIGS_DIR / f"{size_name}_{cond_name}.yaml"
            path.write_text(yaml.dump(cfg_dict, default_flow_style=False, sort_keys=False))

    return all_configs
```

**Test:** Run `generate_configs({})` with empty full_stack_settings and verify:
1. YAML files are written for all 15 combinations (5 sizes x 3 conditions).
2. For each size, print SSM param count vs transformer param count and verify they are within 5%.
3. Spot check: at S (128d, 4L), SSM should have ~5*128^2 + 2*2*128^2 = ~147K per layer * 4 + embed ~ 300K total.

---

### Task 4: Read experiment 09 winners

**Files:** `experiments/10_scaling_laws/run_scaling.py`

**Problem/Goal:** Load the winning component configuration from experiment 09's results directory, constructing the `full_stack_settings` dict that gets injected into `full_ssm` configs at every size.

**Implementation:**

```python
EXP09_RESULTS = Path(__file__).resolve().parents[1] / "09_revised_architecture" / "results"
EXP09_CONFIGS = Path(__file__).resolve().parents[1] / "09_revised_architecture" / "configs"


def load_exp09_winners() -> dict:
    """Load experiment 09 winner settings from its results and config files.

    Reads full_summary.json for winner names, then extracts the relevant
    config fields from the winning YAML files.

    Returns a merged dict of all winning settings (gate + memory + wernicke/cfr).
    """
    summary_path = EXP09_RESULTS / "full_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Experiment 09 results not found at {summary_path}. "
            "Run experiment 09 first."
        )

    summary = json.loads(summary_path.read_text())
    full_stack = {}

    # L1 winner: gate settings
    if "L1" in summary:
        l1_name = summary["L1"]["winner"]
        l1_cfg_path = EXP09_CONFIGS / f"{l1_name}.yaml"
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
        l2_cfg_path = EXP09_CONFIGS / f"{l2_name}.yaml"
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
        l3_cfg_path = EXP09_CONFIGS / f"{l3_name}.yaml"
        if l3_cfg_path.exists():
            l3_cfg = yaml.safe_load(l3_cfg_path.read_text())
            w_keys = [
                "wernicke_enabled", "wernicke_router", "wernicke_k_max",
                "cfr_enabled", "typed_storage", "compression_consequence",
            ]
            for k in w_keys:
                if k in l3_cfg:
                    full_stack[k] = l3_cfg[k]

    print(f"Loaded exp09 winners: {list(full_stack.keys())}")
    return full_stack
```

**Test:** If experiment 09 has not been run yet, the function raises `FileNotFoundError` with a clear message. If it has been run, print the returned dict and verify it contains the expected gate/memory/wernicke keys.

---

### Task 5: Training runner (main loop)

**Files:** `experiments/10_scaling_laws/run_scaling.py`

**Problem/Goal:** The main runner that iterates over sizes and conditions, trains each config for 600s with multiple seeds, and saves per-run JSON results. Must support resuming from partial runs (skip runs whose result JSON already exists).

**Implementation:**

```python
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEEDS = [1337, 2674, 4011]


def run_single(config_path: Path, enwik8_path: str, budget: float, seed: int) -> dict:
    """Run a single training config via chaoscontrol.runner subprocess.

    Returns the parsed result dict from the output JSON.
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
        "--enwik8-path", enwik8_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)
    tmp.unlink(missing_ok=True)
    return json.loads(out_path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Scaling laws runner for experiment 10")
    parser.add_argument("--enwik8-path", required=True, help="Path to enwik8 data file")
    parser.add_argument("--budget", type=float, default=600, help="Per-run budget in seconds")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (1-3)")
    parser.add_argument("--sizes", type=str, default="XS,S,M,L,XL",
                        help="Comma-separated sizes to run")
    parser.add_argument("--conditions", type=str, default="bare_ssm,full_ssm,our_tfm",
                        help="Comma-separated conditions to run")
    args = parser.parse_args()

    seeds = SEEDS[:args.seeds]
    sizes = [s.strip() for s in args.sizes.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    # Load experiment 09 winners
    full_stack_settings = load_exp09_winners()

    # Generate all configs
    all_configs = generate_configs(full_stack_settings)

    # Print parameter summary
    print("\n" + "=" * 72)
    print("  PARAMETER SUMMARY")
    print("=" * 72)
    print(f"  {'Size':<6} {'Condition':<12} {'Params':>12} {'Artifact (bf16)':>16}")
    print(f"  {'-' * 50}")
    for size_name in sizes:
        for cond_name in conditions:
            cfg = all_configs[size_name][cond_name]
            if cfg["model_type"] == "transformer":
                p = count_transformer_params(cfg["model_dim"], cfg["num_layers"],
                                             cfg.get("ff_mult", 2))
            else:
                p = count_ssm_params(cfg["model_dim"], cfg["num_layers"],
                                     cfg.get("ff_mult", 2))
            print(f"  {size_name:<6} {cond_name:<12} {p:>12,} {p * 2:>16,} bytes")
    print()

    # Run all configs
    total_runs = len(sizes) * len(conditions) * len(seeds)
    completed = 0
    all_results = {}
    t_start = time.time()

    for size_name in sizes:
        if size_name not in all_configs:
            print(f"  [WARN] Unknown size: {size_name}, skipping")
            continue
        all_results[size_name] = {}

        for cond_name in conditions:
            if cond_name not in all_configs[size_name]:
                print(f"  [WARN] Unknown condition: {cond_name}, skipping")
                continue
            all_results[size_name][cond_name] = {}

            config_path = CONFIGS_DIR / f"{size_name}_{cond_name}.yaml"
            for seed in seeds:
                completed += 1
                elapsed = time.time() - t_start
                eta = (elapsed / max(completed - 1, 1)) * (total_runs - completed) if completed > 1 else 0
                print(f"\n  [{completed}/{total_runs}] {size_name}/{cond_name} seed={seed} "
                      f"(elapsed={elapsed:.0f}s, ETA={eta:.0f}s)")

                result = run_single(config_path, args.enwik8_path, args.budget, seed)
                all_results[size_name][cond_name][seed] = result

                # Augment result with FLOPs estimate
                cfg = all_configs[size_name][cond_name]
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

                bpb = result["eval"].get("bpb_gated", result["eval"]["bpb"])
                print(f"    bpb={bpb:.4f} params={result['params']:,} "
                      f"steps={steps} FLOPs={result['total_flops']:.2e}")

    # Save combined results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = RESULTS_DIR / "scaling_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
```

**Test:**
1. Run with `--sizes XS --conditions bare_ssm --seeds 1` to do a single quick smoke test (one run, ~600s).
2. Verify `results/XS_bare_ssm_seed1337.json` is written with `eval.bpb`, `train.steps`, `params`, `flops_per_step`, `total_flops`.
3. Verify `results/scaling_results.json` contains the nested structure.
4. Re-run the same command; verify it prints `[SKIP]` and does not re-train.

---

### Task 6: Competition baseline integration

**Files:** `experiments/10_scaling_laws/run_scaling.py` (add a `run_competition_baseline` function)

**Problem/Goal:** Run the competition's `train_gpt.py` at the XL size and extract its final `val_bpb` from stdout. This runs separately because it uses a different data pipeline (FineWeb shards + SentencePiece tokenizer) and different training code.

**Implementation:**

```python
def run_competition_baseline(budget: float = 600) -> dict | None:
    """Run the competition baseline train_gpt.py and extract final val_bpb.

    Returns {"bpb": float, "params": int, "source": "competition_baseline"} or None on failure.

    The competition script writes its BPB to stdout in the format:
        final_int8_zlib_roundtrip_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX

    Prerequisites:
    - Data shards at baselines/parameter_golf/data/datasets/fineweb10B_sp1024/
    - Tokenizer at baselines/parameter_golf/data/tokenizers/fineweb_1024_bpe.model
    - GPU available (train_gpt.py requires CUDA)
    """
    import re

    baseline_dir = REPO / "baselines" / "parameter_golf"
    script = baseline_dir / "train_gpt.py"
    if not script.exists():
        print("  [WARN] Competition baseline not found, skipping")
        return None

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
            timeout=int(budget * 1.5),  # generous timeout
        )
    except subprocess.TimeoutExpired:
        print("  [ERROR] Competition baseline timed out")
        return None

    if proc.returncode != 0:
        print(f"  [ERROR] Competition baseline failed (rc={proc.returncode})")
        print(f"  stderr: {proc.stderr[-500:]}")
        return None

    # Extract final BPB from stdout
    # Look for: final_int8_zlib_roundtrip_exact val_loss:... val_bpb:...
    bpb_match = re.search(r"final_int8_zlib_roundtrip_exact.*val_bpb:([\d.]+)", proc.stdout)
    if not bpb_match:
        # Fallback: last val_bpb line
        bpb_match = re.search(r"val_bpb:([\d.]+)", proc.stdout)

    if not bpb_match:
        print("  [ERROR] Could not extract BPB from competition baseline output")
        return None

    bpb = float(bpb_match.group(1))

    # Competition baseline params: 9 layers, dim=512, 8 heads, 4 KV heads, ff_mult=2
    # vocab=1024, tied embeddings
    result = {
        "bpb": bpb,
        "source": "competition_baseline",
        "hyperparams": {
            "num_layers": 9,
            "model_dim": 512,
            "num_heads": 8,
            "num_kv_heads": 4,
            "mlp_mult": 2,
            "vocab_size": 1024,
            "seq_len": 1024,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"  Competition baseline: bpb={bpb:.4f}")
    return result
```

Integrate into `main()` by adding after the main loop:

```python
    # Run competition baseline at XL size
    if "XL" in sizes:
        print("\n" + "=" * 72)
        print("  COMPETITION BASELINE (XL only)")
        print("=" * 72)
        comp_result = run_competition_baseline(budget=args.budget)
        if comp_result:
            all_results["comp_tfm"] = comp_result
```

**Test:** If the competition data/tokenizer is not set up, the function prints `[WARN]` and returns `None` gracefully. If it is set up, verify the BPB is extracted correctly (should be near 1.2244).

---

### Task 7: Analysis script (power-law fitting and plots)

**Files:** `experiments/10_scaling_laws/analyze_scaling.py`

**Problem/Goal:** Post-hoc analysis script that reads `scaling_results.json`, fits power-law curves to each architecture, generates publication-quality scaling plots, and outputs a summary table.

**Implementation:**

```python
#!/usr/bin/env python3
"""Analyze scaling law results from experiment 10.

Reads results/scaling_results.json, fits power-law curves, generates plots.
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Reference lines from competition
COMP_BASELINE_BPB = 1.2244
COMP_SOTA_BPB = 1.1147


def load_results() -> dict:
    """Load the combined scaling results JSON."""
    path = RESULTS_DIR / "scaling_results.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run run_scaling.py first.")
        sys.exit(1)
    return json.loads(path.read_text())


def extract_scaling_data(results: dict) -> dict[str, list[dict]]:
    """Extract (param_count, bpb, total_flops, steps) per condition.

    Returns {condition: [{"params": ..., "bpb": ..., "bpb_std": ...,
                          "total_flops": ..., "steps": ..., "size": ...}, ...]}.
    """
    data = {}
    for size_name, size_data in results.items():
        if size_name == "comp_tfm":
            continue  # Handled separately
        for cond_name, seed_data in size_data.items():
            if cond_name not in data:
                data[cond_name] = []

            bpbs = []
            params_list = []
            flops_list = []
            steps_list = []
            for seed_str, result in seed_data.items():
                ev = result["eval"]
                bpb = ev.get("bpb_gated", ev["bpb"])
                bpbs.append(bpb)
                params_list.append(result["params"])
                flops_list.append(result.get("total_flops", 0))
                steps_list.append(result["train"]["steps"])

            data[cond_name].append({
                "size": size_name,
                "params": statistics.mean(params_list),
                "bpb": statistics.mean(bpbs),
                "bpb_std": statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0,
                "total_flops": statistics.mean(flops_list),
                "steps": statistics.mean(steps_list),
            })

    # Sort each condition by param count
    for cond in data:
        data[cond].sort(key=lambda x: x["params"])

    return data


def power_law(N, A, alpha, bpb_irr):
    """Power-law model: bpb(N) = A * N^(-alpha) + bpb_irreducible."""
    return A * np.power(N, -alpha) + bpb_irr


def fit_power_law(params: list[float], bpbs: list[float]) -> dict:
    """Fit power-law to scaling data.

    Returns {"A": float, "alpha": float, "bpb_irr": float, "r_squared": float}.
    """
    params_arr = np.array(params, dtype=np.float64)
    bpbs_arr = np.array(bpbs, dtype=np.float64)

    try:
        popt, pcov = curve_fit(
            power_law, params_arr, bpbs_arr,
            p0=[100.0, 0.5, 1.0],
            bounds=([0, 0, 0], [1e6, 2.0, 5.0]),
            maxfev=10000,
        )
        A, alpha, bpb_irr = popt

        # R-squared
        residuals = bpbs_arr - power_law(params_arr, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((bpbs_arr - np.mean(bpbs_arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {"A": A, "alpha": alpha, "bpb_irr": bpb_irr, "r_squared": r_squared}
    except RuntimeError as e:
        print(f"  [WARN] Power-law fit failed: {e}")
        return {"A": 0, "alpha": 0, "bpb_irr": 0, "r_squared": 0}


def plot_scaling_curves(data: dict[str, list[dict]], fits: dict[str, dict],
                        comp_bpb: float | None = None):
    """Plot bpb vs param count for all conditions (log-log)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    colors = {"bare_ssm": "#2196F3", "full_ssm": "#4CAF50", "our_tfm": "#FF9800"}
    markers = {"bare_ssm": "o", "full_ssm": "s", "our_tfm": "^"}

    for cond, points in data.items():
        params = [p["params"] for p in points]
        bpbs = [p["bpb"] for p in points]
        stds = [p["bpb_std"] for p in points]
        color = colors.get(cond, "#999999")
        marker = markers.get(cond, "D")

        ax.errorbar(params, bpbs, yerr=stds, fmt=marker, color=color,
                     label=cond, markersize=8, capsize=4, linewidth=1.5)

        # Plot fitted curve
        if cond in fits and fits[cond]["r_squared"] > 0.5:
            fit = fits[cond]
            x_fit = np.logspace(np.log10(min(params) * 0.8),
                                np.log10(max(params) * 1.2), 100)
            y_fit = power_law(x_fit, fit["A"], fit["alpha"], fit["bpb_irr"])
            alpha_str = f"alpha={fit['alpha']:.3f}"
            ax.plot(x_fit, y_fit, "--", color=color, alpha=0.5,
                     label=f"{cond} fit ({alpha_str})")

    # Reference lines
    ax.axhline(y=COMP_BASELINE_BPB, color="red", linestyle=":", alpha=0.7,
                label=f"Competition baseline ({COMP_BASELINE_BPB})")
    ax.axhline(y=COMP_SOTA_BPB, color="darkred", linestyle=":", alpha=0.7,
                label=f"Competition SOTA ({COMP_SOTA_BPB})")

    if comp_bpb is not None:
        ax.axhline(y=comp_bpb, color="red", linestyle="-", alpha=0.5,
                    label=f"Our comp_tfm run ({comp_bpb:.4f})")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters", fontsize=12)
    ax.set_ylabel("BPB (bits per byte)", fontsize=12)
    ax.set_title("SESSM vs Transformer Scaling Laws", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scaling_curves.png", dpi=150)
    print(f"  Saved {PLOTS_DIR / 'scaling_curves.png'}")
    plt.close(fig)


def plot_isoflop(data: dict[str, list[dict]]):
    """Plot bpb vs total FLOPs for all conditions."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    colors = {"bare_ssm": "#2196F3", "full_ssm": "#4CAF50", "our_tfm": "#FF9800"}
    markers = {"bare_ssm": "o", "full_ssm": "s", "our_tfm": "^"}

    for cond, points in data.items():
        flops = [p["total_flops"] for p in points if p["total_flops"] > 0]
        bpbs = [p["bpb"] for p in points if p["total_flops"] > 0]
        if not flops:
            continue
        color = colors.get(cond, "#999999")
        marker = markers.get(cond, "D")
        ax.scatter(flops, bpbs, c=color, marker=marker, s=80, label=cond, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Total Training FLOPs", fontsize=12)
    ax.set_ylabel("BPB (bits per byte)", fontsize=12)
    ax.set_title("isoFLOP Comparison: BPB vs Compute", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "isoflop_curves.png", dpi=150)
    print(f"  Saved {PLOTS_DIR / 'isoflop_curves.png'}")
    plt.close(fig)


def plot_component_delta(data: dict[str, list[dict]]):
    """Plot component delta (bare_ssm - full_ssm) vs param count."""
    if "bare_ssm" not in data or "full_ssm" not in data:
        print("  [SKIP] Component delta plot: need both bare_ssm and full_ssm")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    bare_map = {p["size"]: p for p in data["bare_ssm"]}
    full_map = {p["size"]: p for p in data["full_ssm"]}

    sizes = []
    params = []
    deltas = []
    for size in bare_map:
        if size in full_map:
            sizes.append(size)
            params.append(bare_map[size]["params"])
            deltas.append(bare_map[size]["bpb"] - full_map[size]["bpb"])

    ax.bar(sizes, deltas, color="#4CAF50", alpha=0.7, edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel("Component Delta (bare - full, positive = components help)", fontsize=12)
    ax.set_title("Component ROI Across Scale", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate each bar
    for i, (size, delta) in enumerate(zip(sizes, deltas)):
        ax.annotate(f"{delta:+.4f}", (i, delta), textcoords="offset points",
                     xytext=(0, 10 if delta >= 0 else -15), ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "component_delta.png", dpi=150)
    print(f"  Saved {PLOTS_DIR / 'component_delta.png'}")
    plt.close(fig)


def print_summary_table(data: dict[str, list[dict]], fits: dict[str, dict]):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 90)
    print("  SCALING LAW RESULTS")
    print("=" * 90)
    print(f"  {'Size':<6} {'Condition':<12} {'Params':>10} {'BPB':>8} {'Std':>8} "
          f"{'Steps':>8} {'FLOPs':>12}")
    print(f"  {'-' * 80}")

    for cond, points in sorted(data.items()):
        for p in points:
            print(f"  {p['size']:<6} {cond:<12} {p['params']:>10,.0f} {p['bpb']:>8.4f} "
                  f"{p['bpb_std']:>8.4f} {p['steps']:>8,.0f} {p['total_flops']:>12.2e}")
        print()

    print("\n  POWER-LAW FITS: bpb(N) = A * N^(-alpha) + bpb_irr")
    print(f"  {'-' * 60}")
    print(f"  {'Condition':<12} {'A':>10} {'alpha':>10} {'bpb_irr':>10} {'R^2':>10}")
    for cond, fit in fits.items():
        print(f"  {cond:<12} {fit['A']:>10.4f} {fit['alpha']:>10.4f} "
              f"{fit['bpb_irr']:>10.4f} {fit['r_squared']:>10.4f}")

    # Kill criteria check
    print("\n  KILL CRITERIA CHECK")
    print(f"  {'-' * 60}")
    ssm_alpha = fits.get("bare_ssm", {}).get("alpha", 0)
    tfm_alpha = fits.get("our_tfm", {}).get("alpha", 0)
    if tfm_alpha > ssm_alpha + 0.1:
        print(f"  *** KILL: Transformer alpha ({tfm_alpha:.3f}) > SSM alpha ({ssm_alpha:.3f}) + 0.1")
    else:
        print(f"  OK: SSM alpha ({ssm_alpha:.3f}) vs Transformer alpha ({tfm_alpha:.3f})")

    # Check XL bare_ssm bpb
    for p in data.get("bare_ssm", []):
        if p["size"] == "XL" and p["bpb"] > 2.0:
            print(f"  *** KILL: bare_ssm at XL has bpb={p['bpb']:.4f} > 2.0")

    # Check component delta at XL
    bare_xl = next((p for p in data.get("bare_ssm", []) if p["size"] == "XL"), None)
    full_xl = next((p for p in data.get("full_ssm", []) if p["size"] == "XL"), None)
    if bare_xl and full_xl:
        delta = bare_xl["bpb"] - full_xl["bpb"]
        if delta < 0:
            print(f"  *** KILL: component_delta at XL is {delta:.4f} < 0 (full stack hurts)")
        else:
            print(f"  OK: component_delta at XL is {delta:+.4f} (full stack helps)")


def main():
    results = load_results()
    data = extract_scaling_data(results)

    # Fit power laws
    fits = {}
    for cond, points in data.items():
        if len(points) >= 3:  # Need at least 3 points for a 3-param fit
            params = [p["params"] for p in points]
            bpbs = [p["bpb"] for p in points]
            fits[cond] = fit_power_law(params, bpbs)

    # Load competition baseline if available
    comp_bpb = None
    comp_path = RESULTS_DIR / "comp_tfm_baseline.json"
    if comp_path.exists():
        comp_data = json.loads(comp_path.read_text())
        comp_bpb = comp_data.get("bpb")

    # Generate plots
    plot_scaling_curves(data, fits, comp_bpb)
    plot_isoflop(data)
    plot_component_delta(data)

    # Print summary
    print_summary_table(data, fits)

    # Save analysis summary as JSON
    analysis = {
        "fits": fits,
        "competition_baseline_bpb": comp_bpb,
        "data": {cond: points for cond, points in data.items()},
    }
    analysis_path = RESULTS_DIR / "scaling_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
```

**Test:**
1. Create a synthetic `scaling_results.json` with made-up but plausible data at 5 sizes and 3 conditions. Run `python analyze_scaling.py` and verify it produces 3 PNG plots in `plots/` and prints the summary table.
2. Verify the power-law fit produces sensible alpha values (expect 0.1-0.5 for typical language model scaling).
3. Verify kill criteria are evaluated and printed.

---

### Task 8: Wire everything together in `run_scaling.py`

**Files:** `experiments/10_scaling_laws/run_scaling.py` (final assembly)

**Problem/Goal:** Assemble all the pieces from Tasks 2-6 into a single coherent runner file.

**Implementation:**

The final `run_scaling.py` file structure:

```python
#!/usr/bin/env python3
"""Scaling laws runner for experiment 10.

Trains parameter-matched SSM and transformer configs at 5 sizes,
logs results per-run, and generates a combined results JSON.
"""
import argparse
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

# --- Paste functions from Tasks 2, 3, 4, 5, 6 here in this order: ---
# estimate_flops_per_step()    (Task 2)
# count_ssm_params()           (Task 3)
# count_transformer_params()   (Task 3)
# match_transformer_to_ssm()   (Task 3)
# generate_configs()           (Task 3)
# load_exp09_winners()         (Task 4)
# run_single()                 (Task 5)
# run_competition_baseline()   (Task 6)
# main()                       (Task 5, with Task 6 integration)
```

**Test:** `python run_scaling.py --help` prints usage with all arguments. `python run_scaling.py --enwik8-path /path/to/enwik8 --sizes XS --conditions bare_ssm --seeds 1` completes a single run and writes results.

---

### Summary: File inventory

| File | Action | Purpose |
|------|--------|---------|
| `experiments/10_scaling_laws/run.sh` | Create | Shell entry point |
| `experiments/10_scaling_laws/run_scaling.py` | Create | Training runner (Tasks 2-6, 8) |
| `experiments/10_scaling_laws/analyze_scaling.py` | Create | Post-hoc analysis (Task 7) |
| `experiments/10_scaling_laws/configs/` | Created at runtime | Generated YAML configs |
| `experiments/10_scaling_laws/results/` | Created at runtime | Per-run JSON results |
| `experiments/10_scaling_laws/plots/` | Created at runtime | PNG plots |

No existing files are modified. All 3 new files are self-contained.

---

### Task 9: Inference benchmarks (latency + memory footprint)

**Files:** `experiments/10_scaling_laws/run_scaling.py` (new function), `experiments/10_scaling_laws/analyze_scaling.py` (new plot)

**Problem/Goal:** After training each config, benchmark inference latency and memory footprint. SSM's O(d) recurrence vs transformer's O(nd) attention is a key selling point.

**Implementation:**

Add to `run_scaling.py`:

```python
def benchmark_inference(model, device, seq_lengths=[1024, 4096, 16384]):
    """Benchmark inference latency and memory footprint.
    
    Returns:
        {"latency_per_token_ms": float, "state_bytes": dict[int, int]}
    """
    model.eval()
    results = {}
    
    # Latency: time 1000 step() calls (SSM) or forward with seq_len=1 (transformer)
    if hasattr(model, "step"):
        state = model.init_state(1)
        dummy = torch.randint(0, model.vocab_size, (1, 1), device=device)
        # Warmup
        for _ in range(10):
            _, _, state = model.step(dummy, state)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1000):
            _, _, state = model.step(dummy, state)
        torch.cuda.synchronize()
        results["latency_per_token_ms"] = (time.perf_counter() - t0)
    else:
        # Transformer: forward pass on single token sequences
        dummy = torch.randint(0, model.vocab_size, (1, 1), device=device)
        for _ in range(10):
            model(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1000):
            model(dummy)
        torch.cuda.synchronize()
        results["latency_per_token_ms"] = (time.perf_counter() - t0)
    
    # Memory footprint at different sequence lengths
    state_bytes = {}
    for sl in seq_lengths:
        if hasattr(model, "init_state"):
            # SSM: fixed state, independent of seq_len
            state = model.init_state(1)
            state_bytes[sl] = sum(s.nelement() * s.element_size() for s in state)
        else:
            # Transformer: KV cache grows with seq_len
            # Estimate: 2 * num_layers * dim * seq_len * element_size (K and V)
            n_layers = len(model.layers) if hasattr(model, "layers") else 4
            dim = model.embed.embedding_dim if hasattr(model, "embed") else 128
            state_bytes[sl] = 2 * n_layers * dim * sl * 2  # bf16
    results["state_bytes"] = state_bytes
    return results
```

Call after each training run, append to results JSON.

Add to `analyze_scaling.py`:

```python
def plot_inference_efficiency(data):
    """Two-panel plot: latency per token vs size, and memory footprint vs seq_len."""
    # Panel 1: latency per token at each model size
    # Panel 2: state/KV-cache bytes at seq_len=1024,4096,16384 for SSM vs transformer at XL
```

---

### Task 10: Quantization robustness sweep

**Files:** `experiments/10_scaling_laws/run_scaling.py` (new function)

**Problem/Goal:** Measure bpb degradation under int8 and int6 quantization for each trained model. If SSM loses less bpb from quantization, that's a publishable advantage for the 16MB artifact constraint.

**Implementation:**

```python
def quantize_and_eval(model, tokens, eval_starts, seq_len, device, batch_size=32):
    """Quantize model to int8 and int6, measure bpb at each level.
    
    Returns {"bf16_bpb": float, "int8_bpb": float, "int6_bpb": float,
             "int8_delta": float, "int6_delta": float}
    """
    from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb
    
    # bf16 baseline (already have this from training)
    bf16_result = evaluate_chaoscontrol_bpb(
        model, tokens=tokens, eval_starts=eval_starts,
        batch_size=batch_size, seq_len=seq_len, device=device,
    )
    bf16_bpb = bf16_result["bpb"]
    
    # int8: quantize all linear weight tensors
    int8_model = copy.deepcopy(model)
    for name, param in int8_model.named_parameters():
        if param.ndim == 2:  # weight matrices
            scale = param.abs().max() / 127
            param.data = (param.data / scale).round().clamp(-128, 127) * scale
    int8_result = evaluate_chaoscontrol_bpb(
        int8_model, tokens=tokens, eval_starts=eval_starts,
        batch_size=batch_size, seq_len=seq_len, device=device,
    )
    
    # int6: same but 5-bit range (-32..31)
    int6_model = copy.deepcopy(model)
    for name, param in int6_model.named_parameters():
        if param.ndim == 2:
            scale = param.abs().max() / 31
            param.data = (param.data / scale).round().clamp(-32, 31) * scale
    int6_result = evaluate_chaoscontrol_bpb(
        int6_model, tokens=tokens, eval_starts=eval_starts,
        batch_size=batch_size, seq_len=seq_len, device=device,
    )
    
    return {
        "bf16_bpb": bf16_bpb,
        "int8_bpb": int8_result["bpb"],
        "int6_bpb": int6_result["bpb"],
        "int8_delta": int8_result["bpb"] - bf16_bpb,
        "int6_delta": int6_result["bpb"] - bf16_bpb,
    }
```

Add a `plot_quantization_robustness` function to `analyze_scaling.py` — grouped bar chart of quantization deltas for SSM vs transformer at each size.

---

### Task 11: Component-specific metrics logging

**Files:** `experiments/10_scaling_laws/run_scaling.py`

**Problem/Goal:** Extract and log component-specific metrics from training history for the extended analysis.

**Implementation:**

After each training run completes, extract from the training history:

```python
def extract_component_metrics(train_result: dict) -> dict:
    """Extract component-specific metrics from training history."""
    history = train_result.get("history", [])
    metrics = {}
    
    # Gate fire rate
    fires = [s.get("gate_fired", False) for s in history if "gate_fired" in s]
    metrics["gate_fire_rate"] = sum(fires) / max(len(fires), 1)
    
    # Memory slot utilization (unique buckets / max_slots)
    buckets_used = set()
    for s in history:
        if "dominant_bucket" in s and s["dominant_bucket"] is not None:
            buckets_used.add(s["dominant_bucket"])
    metrics["unique_buckets"] = len(buckets_used)
    
    # CFR regret entropy (if regret_table returned)
    regret_table = train_result.get("regret_table")
    if regret_table is not None and hasattr(regret_table, "cumulative_regret"):
        import numpy as np
        regret = regret_table.cumulative_regret
        strategy = np.exp(regret) / np.exp(regret).sum(axis=-1, keepdims=True)
        entropy = -(strategy * np.log(strategy + 1e-10)).sum(axis=-1).mean()
        metrics["cfr_entropy"] = float(entropy)
    
    # Codebook utilization (from Wernicke bucket snapshots)
    bucket_snaps = train_result.get("bucket_snapshots", [])
    if bucket_snaps:
        last_snap = bucket_snaps[-1]
        total_entries = max(len(last_snap), 1)
        active = sum(1 for v in last_snap if v > 0)
        metrics["wernicke_codebook_utilization"] = active / total_entries
    
    # Spectral structure: A-matrix eigenvalue stats from spectral snapshots
    spectral = train_result.get("spectral_snapshots", [])
    if spectral:
        last = spectral[-1]
        metrics["spectral_max_freq"] = float(max(last)) if last else 0.0
        metrics["spectral_mean_freq"] = float(sum(last) / len(last)) if last else 0.0
    
    return metrics
```

Log these alongside the per-run JSON results.

Add to `analyze_scaling.py`:
- `plot_gate_fire_rate` — fire rate vs model size, with bpb delta overlay
- `plot_codebook_utilization` — tokenizer + Wernicke utilization vs size
- `plot_spectral_evolution` — eigenvalue distribution at each size
- `plot_seed_variance` — std(bpb) per condition per size as bar chart

---

### Summary: File inventory (updated)

| File | Action | Purpose |
|------|--------|---------|
| `experiments/10_scaling_laws/run.sh` | Create | Shell entry point |
| `experiments/10_scaling_laws/run_scaling.py` | Create | Training runner (Tasks 2-6, 8-11) |
| `experiments/10_scaling_laws/analyze_scaling.py` | Create | Post-hoc analysis (Task 7 + extended plots) |
| `experiments/10_scaling_laws/configs/` | Created at runtime | Generated YAML configs |
| `experiments/10_scaling_laws/results/` | Created at runtime | Per-run JSON results |
| `experiments/10_scaling_laws/plots/` | Created at runtime | PNG plots (9 total) |

Plots produced by `analyze_scaling.py`:
1. `scaling_curves.png` — bpb vs params with power-law fits
2. `isoflop_curves.png` — bpb vs total FLOPs
3. `component_delta.png` — component ROI vs model size
4. `inference_efficiency.png` — latency per token + memory footprint
5. `quantization_robustness.png` — bpb delta from int8/int6
6. `gate_fire_rate.png` — gate utilization vs size
7. `codebook_utilization.png` — tokenizer + Wernicke codebook usage
8. `spectral_evolution.png` — A-matrix eigenvalue structure vs size
9. `seed_variance.png` — seed-to-seed bpb std

### Execution order

1. Run experiment 09 (if not already done) to produce `full_summary.json`
2. `bash experiments/10_scaling_laws/run.sh /path/to/data` — trains all configs (~8h sequential)
3. `python experiments/10_scaling_laws/analyze_scaling.py` — generates 9 plots + summary table
4. Review kill criteria output before proceeding to further experiments
