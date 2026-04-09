#!/usr/bin/env python3
"""Phase 2 eval-time ablation: test gate, CFR, memory state, and warmup on Phase 1 checkpoints.

Loads trained checkpoints from Phase 1 and evaluates under all meaningful
combinations of inference-time settings. Forward passes only -- no training.

Ablation dimensions:
  - Gate mode: none / fork_k4 / mc_k4 / mcts_k4 / mcts_k8
  - Memory state: seeded / cold / ttt
  - CFR: off / on (with warmup pass to populate regret table)
  - Warmup: none / last / full_seq_latent (skipped for cold memory)
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
RESULTS = EXPERIMENT / "results_phase2"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXPERIMENT))

from chaoscontrol.config import ChaosControlConfig
from chaoscontrol.data import (
    resolve_device, resolve_param_dtype,
    prepare_fineweb_splits, build_lm_starts, choose_eval_starts,
    batch_from_starts, maybe_autocast,
)
from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb
from chaoscontrol.runner import load_checkpoint
from chaoscontrol.regret import RegretTable
from stats import welch_ttest, bootstrap_ci, cohens_d, sem


# ── Ablation grid ───────────────────────────────────────────────────

GATE_MODES = [
    {"name": "none", "metabolic_gate": False},
    {"name": "fork_k4", "metabolic_gate": True, "metabolic_mode": "fork", "metabolic_k": 4},
    {"name": "mc_k4", "metabolic_gate": True, "metabolic_mode": "monte_carlo", "metabolic_k": 4},
    {"name": "mcts_k4", "metabolic_gate": True, "metabolic_mode": "mcts", "metabolic_k": 4},
    {"name": "mcts_k8", "metabolic_gate": True, "metabolic_mode": "mcts", "metabolic_k": 8},
]

MEMORY_STATES = ["seeded", "cold", "ttt"]

CFR_MODES = [
    {"name": "cfr_off", "enabled": False},
    {"name": "cfr_on", "enabled": True},
]

WARMUP_MODES = [
    {"name": "warmup_none", "warmup": False},
    {"name": "warmup_last", "warmup": True, "warmup_write_mode": "last", "warmup_latent": False},
    {"name": "warmup_full_seq", "warmup": True, "warmup_write_mode": "full_sequence", "warmup_latent": True},
]


def is_redundant(gate: dict, mem_state: str, cfr: dict, warmup: dict) -> bool:
    """Filter out meaningless combos."""
    # CFR without gate is a no-op
    if not gate["metabolic_gate"] and cfr["enabled"]:
        return True
    # Warmup with cold memory: cold means we wipe memory, warmup writes to it.
    # This IS meaningful (it tests reconstitution from scratch). Keep it.
    # But warmup=none + cold is the same as just cold, which we keep as the
    # baseline. warmup_last + cold and warmup_full_seq + cold are distinct.
    return False


def build_eval_grid() -> list[dict]:
    """Generate all non-redundant eval configs."""
    grid = []
    for gate, mem_state, cfr, warmup in itertools.product(
        GATE_MODES, MEMORY_STATES, CFR_MODES, WARMUP_MODES
    ):
        if is_redundant(gate, mem_state, cfr, warmup):
            continue
        grid.append({
            "gate": gate,
            "mem_state": mem_state,
            "cfr": cfr,
            "warmup": warmup,
            "label": f"{gate['name']}_{mem_state}_{cfr['name']}_{warmup['name']}",
        })
    return grid


# ── CFR warmup pass ─────────────────────────────────────────────────


def cfr_warmup_pass(
    model,
    regret_table: RegretTable,
    train_tokens: torch.Tensor,
    train_starts: list[int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
    gate_config: dict,
    n_batches: int = 16,
    seed: int = 42,
    tokenizer: object = None,
):
    """Forward a few batches with gate active to populate the regret table.

    No gradients, no optimizer. Just gate decisions + regret updates.
    """
    import random as _random
    from chaoscontrol.metabolic import metabolic_fork, micro_mcts, metabolic_monte_carlo

    rng = _random.Random(seed)
    model.eval()
    vocab_size = model.vocab_size

    param_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for _ in range(n_batches):
            batch_starts_list = [train_starts[rng.randrange(len(train_starts))]
                                 for _ in range(batch_size)]
            inputs, targets = batch_from_starts(train_tokens, batch_starts_list, seq_len, device)

            # Apply tokenizer if present (model expects VQ tokens, not raw bytes)
            if tokenizer is not None:
                tok_out = tokenizer(inputs)
                inputs = tok_out["token_ids"][:, :-1]
                targets = tok_out["token_ids"][:, 1:]

            mode = gate_config.get("metabolic_mode", "fork")
            k = gate_config.get("metabolic_k", 4)

            with maybe_autocast(device, param_dtype):
                if mode == "mcts":
                    out = micro_mcts(model, inputs, n_rollouts=k, horizon=8)
                elif mode == "monte_carlo":
                    out = metabolic_monte_carlo(model, inputs, k=k, noise_std=0.01)
                else:
                    out = metabolic_fork(model, inputs, k=k, noise_std=0.01)

            # CE for the action taken
            ce_val = F.cross_entropy(
                out["logits"].float().reshape(-1, vocab_size),
                targets.reshape(-1),
            ).item()
            actual_value = -ce_val

            # Determine bucket from Wernicke routing (use dominant bucket, not hardcoded 0)
            bucket = 0
            if "bucket_ids" in out:
                bucket = int(out["bucket_ids"].reshape(-1).mode().values.item())
            elif hasattr(model, "wernicke") and model.wernicke is not None:
                # Run a plain forward pass to get bucket routing
                with maybe_autocast(device, param_dtype):
                    plain_out = model(inputs)
                if "bucket_ids" in plain_out:
                    bucket = int(plain_out["bucket_ids"].reshape(-1).mode().values.item())

            # Counterfactual rollouts: run model.step() with alternative last tokens
            # to get actual different CE values (not copies of actual_value)
            with maybe_autocast(device, param_dtype):
                last_logits = out["logits"][:, -1, :].detach()
                k_actions = min(k, last_logits.size(-1))
                _, top_tokens = last_logits.mean(dim=0).topk(k_actions)

                # Build state up to penultimate position
                rollout_state = model.init_state(inputs.size(0))
                for t in range(inputs.size(1) - 1):
                    _, _, rollout_state = model.step(inputs[:, t:t+1], rollout_state)

                counterfactual_values = []
                for a in range(k_actions):
                    token = top_tokens[a].unsqueeze(0).expand(inputs.size(0)).unsqueeze(-1)
                    cf_state = [s.clone() for s in rollout_state]
                    cf_logits, _, _ = model.step(token, cf_state)
                    next_target = targets[:, -1]
                    cf_ce = F.cross_entropy(cf_logits, next_target).item()
                    counterfactual_values.append(-cf_ce)

            action_taken = out.get("best_idx", 0)
            if "mcts_stats" in out and "visit_counts" in out["mcts_stats"]:
                action_taken = int(out["mcts_stats"]["visit_counts"].argmax().item())

            regret_table.update(
                bucket_id=bucket % regret_table.n_buckets,
                action_taken=action_taken,
                counterfactual_values=counterfactual_values,
                actual_value=actual_value,
            )


# ── Memory state manipulation ──────────────────────────────────────


def set_memory_state(model, state: str, train_tokens=None, train_starts=None,
                     seq_len=256, batch_size=64, device=None, tokenizer=None):
    """Set the model's memory to the requested state."""
    om = getattr(model, "outer_model", None)
    if om is None:
        return  # No memory to manipulate

    if state == "seeded":
        # Keep memory as-is from checkpoint
        pass
    elif state == "cold":
        # Wipe all memory
        if hasattr(om, "_slots"):
            om._slots = []
            om._survival = []
            om._slot_buckets = []
        if hasattr(om, "_latent_traces"):
            om._latent_traces = []
        if hasattr(om, "state"):
            om.state.zero_()
    elif state == "ttt":
        # Wipe, then reconstitute via forward pass over training data
        set_memory_state(model, "cold")  # First wipe
        if train_tokens is None:
            return
        # Phase A: forward training data with memory writes
        import random as _random
        rng = _random.Random(42)
        model.eval()
        n_ttt_batches = 32  # Forward ~32 batches of training data
        with torch.no_grad():
            for _ in range(n_ttt_batches):
                starts = [train_starts[rng.randrange(len(train_starts))]
                          for _ in range(batch_size)]
                inputs, targets = batch_from_starts(train_tokens, starts, seq_len, device)
                # Apply tokenizer if present (model expects VQ token IDs, not raw bytes)
                if tokenizer is not None:
                    tok_out = tokenizer(inputs)
                    inputs = tok_out["token_ids"][:, :-1]
                    targets = tok_out["token_ids"][:, 1:]
                out = model(inputs)
                ce = F.cross_entropy(
                    out["logits"].float().reshape(-1, model.vocab_size),
                    targets.reshape(-1),
                ).item()
                hidden = out["hidden"][:, -1, :].detach()
                om.consolidation_step(hidden, current_loss=ce, bucket_id=None)


# ── Single eval run ─────────────────────────────────────────────────


def run_single_eval(
    ckpt_path: Path,
    eval_config: dict,
    data_path: str,
    device: torch.device,
    param_dtype: torch.dtype,
    train_tokens: torch.Tensor,
    train_starts: list[int],
    val_tokens: torch.Tensor,
    eval_starts: list[int],
) -> dict:
    """Load checkpoint, apply eval settings, run forward eval, return metrics."""
    from chaoscontrol.runner import load_checkpoint as _load

    loaded = _load(ckpt_path, device, param_dtype)
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    cfg = loaded["config"]
    structured_proj = loaded["structured_proj"]

    gate = eval_config["gate"]
    mem_state = eval_config["mem_state"]
    cfr = eval_config["cfr"]
    warmup_cfg = eval_config["warmup"]

    # Set memory state
    set_memory_state(
        model, mem_state,
        train_tokens=train_tokens, train_starts=train_starts,
        seq_len=cfg.seq_len, batch_size=cfg.batch_size, device=device,
        tokenizer=tokenizer,
    )

    # CFR warmup if needed
    regret_table = None
    if cfr["enabled"] and gate["metabolic_gate"]:
        regret_table = RegretTable(
            n_buckets=cfg.wernicke_k_max if cfg.wernicke_enabled else 16,
            n_actions=gate.get("metabolic_k", 4),
        )
        cfr_warmup_pass(
            model, regret_table,
            train_tokens, train_starts,
            cfg.seq_len, cfg.batch_size, device,
            gate, n_batches=16, tokenizer=tokenizer,
        )

    total_raw_bytes = int(val_tokens.numel())

    # Build CFR prior_bias from populated regret table
    # Uses bucket 0 as default (eval doesn't have per-batch bucket routing)
    cfr_bias = None
    if regret_table is not None:
        cfr_bias = regret_table.get_strategy(0).to(device)

    # Run eval
    result = evaluate_chaoscontrol_bpb(
        model,
        tokens=val_tokens,
        eval_starts=eval_starts,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        metabolic_gate=gate["metabolic_gate"],
        metabolic_k=gate.get("metabolic_k", 4),
        metabolic_score=cfg.metabolic_score,
        metabolic_noise_std=cfg.metabolic_noise_std,
        metabolic_mode=gate.get("metabolic_mode", "fork"),
        generation_mode=cfg.generation_mode,
        structured_proj=structured_proj,
        warmup=warmup_cfg["warmup"],
        warmup_write_mode=warmup_cfg.get("warmup_write_mode", "last"),
        warmup_latent=warmup_cfg.get("warmup_latent", False),
        warmup_cold_start=(mem_state == "cold"),
        total_raw_bytes=total_raw_bytes,
        tokenizer=tokenizer,
        prior_bias=cfr_bias,
    )

    return {
        "label": eval_config["label"],
        "gate": gate["name"],
        "mem_state": mem_state,
        "cfr": cfr["name"],
        "warmup": warmup_cfg["name"],
        "bpb": result["bpb"],
        "bpb_gated": result.get("bpb_gated"),
        "loss": result["loss"],
        "checkpoint": str(ckpt_path),
    }


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Eval-time ablation matrix")
    parser.add_argument("--checkpoint-dir", required=True, help="Dir with Phase 1 .pt checkpoints")
    parser.add_argument("--data-path", required=True, help="FineWeb data dir")
    parser.add_argument("--num-gpus", type=int, default=1, help="GPUs (launch shards automatically)")
    parser.add_argument("--shard-id", type=int, default=None, help="This shard's index (0-based). Internal use.")
    parser.add_argument("--num-shards", type=int, default=None, help="Total shards. Internal use.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    # Multi-GPU: launch N shards as subprocesses, each pinned to a GPU
    if args.num_gpus > 1 and args.shard_id is None:
        import shlex
        import subprocess as _sp
        procs = []
        for shard in range(args.num_gpus):
            cmd = [
                sys.executable, __file__,
                "--checkpoint-dir", args.checkpoint_dir,
                "--data-path", args.data_path,
                "--shard-id", str(shard),
                "--num-shards", str(args.num_gpus),
                "--device", args.device,
                "--dtype", args.dtype,
            ]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(shard)
            log = RESULTS / f"shard_{shard}.log"
            fh = open(log, "w")
            p = _sp.Popen(cmd, env=env, stdout=fh, stderr=_sp.STDOUT)
            procs.append((p, fh, shard))
            print(f"Launched shard {shard} on GPU {shard} (PID {p.pid})")
        # Wait for all shards
        for p, fh, shard in procs:
            p.wait()
            fh.close()
            print(f"Shard {shard} finished (exit {p.returncode})")
        # Merge shard results
        merged = []
        for shard in range(args.num_gpus):
            shard_file = RESULTS / f"eval_results_shard{shard}.json"
            if shard_file.exists():
                with open(shard_file) as f:
                    merged.extend(json.load(f))
        # Also include any prior results from the single-process run
        main_file = RESULTS / "eval_results.json"
        if main_file.exists():
            with open(main_file) as f:
                prior = json.load(f)
            # Deduplicate by (checkpoint, label)
            seen = {(r["checkpoint"], r["label"]) for r in merged}
            for r in prior:
                if (r["checkpoint"], r["label"]) not in seen:
                    merged.append(r)
        with open(main_file, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"\nMerged {len(merged)} results to {main_file}")
        return

    device = resolve_device(args.device)
    param_dtype = resolve_param_dtype(args.dtype, device)

    # Find L3 checkpoints (the scaling layer -- all scales x seeds)
    ckpt_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(ckpt_dir.glob("L3_*.pt"))
    if not checkpoints:
        # Fall back to any checkpoint
        checkpoints = sorted(ckpt_dir.glob("*.pt"))
    if not checkpoints:
        print(f"No checkpoints found in {ckpt_dir}")
        sys.exit(1)
    # Shard: only process checkpoints assigned to this shard
    if args.shard_id is not None and args.num_shards is not None:
        checkpoints = [c for i, c in enumerate(checkpoints) if i % args.num_shards == args.shard_id]
    print(f"Found {len(checkpoints)} checkpoints: {[c.stem for c in checkpoints]}")

    # Read seq_len/stride/batch_size from the first checkpoint's config
    first_payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    first_cfg = ChaosControlConfig(**first_payload["config"])
    data_seq_len = first_cfg.seq_len
    data_stride = first_cfg.stride
    data_batch_size = first_cfg.batch_size
    data_eval_batches = first_cfg.eval_batches
    print(f"Using config from {checkpoints[0].stem}: seq_len={data_seq_len}, "
          f"stride={data_stride}, batch_size={data_batch_size}")

    # Load data once
    print("Loading FineWeb data...")
    train_tokens, val_tokens, _test = prepare_fineweb_splits(args.data_path, device=device)
    train_starts = build_lm_starts(int(train_tokens.numel()), data_seq_len, data_stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), data_seq_len, data_stride)
    eval_starts = choose_eval_starts(val_starts, batch_size=data_batch_size,
                                     eval_batches=data_eval_batches, seed=42)

    # Build eval grid
    grid = build_eval_grid()
    print(f"Eval grid: {len(grid)} configs per checkpoint")
    print(f"Total evals: {len(checkpoints)} x {len(grid)} = {len(checkpoints) * len(grid)}")

    # Resume support — per-shard output files when sharded
    if args.shard_id is not None:
        results_file = RESULTS / f"eval_results_shard{args.shard_id}.json"
    else:
        results_file = RESULTS / "eval_results.json"
    all_results = []
    done_labels = set()
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        done_labels = {(r["checkpoint"], r["label"]) for r in all_results}
    # Also check main results file for evals done by prior single-process run
    main_file = RESULTS / "eval_results.json"
    if main_file.exists() and main_file != results_file:
        with open(main_file) as f:
            prior = json.load(f)
        done_labels |= {(r["checkpoint"], r["label"]) for r in prior}
    if done_labels:
        print(f"Resuming: {len(done_labels)} evals already done")

    total = len(checkpoints) * len(grid)
    completed = len(done_labels)

    for ckpt_path in checkpoints:
        for eval_config in grid:
            key = (str(ckpt_path), eval_config["label"])
            if key in done_labels:
                continue

            completed += 1
            t0 = time.time()
            print(f"[{completed}/{total}] {ckpt_path.stem} :: {eval_config['label']}", end=" ", flush=True)

            result = run_single_eval(
                ckpt_path, eval_config,
                args.data_path, device, param_dtype,
                train_tokens, train_starts, val_tokens, eval_starts,
            )
            elapsed = time.time() - t0
            bpb = result.get("bpb_gated") or result["bpb"]
            print(f"bpb={bpb:.4f} ({elapsed:.1f}s)")

            all_results.append(result)

            # Checkpoint after every eval
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # ── Analysis ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PHASE 2 COMPLETE")
    print("="*70)

    # Group by gate mode
    by_gate: dict[str, list[float]] = {}
    for r in all_results:
        gate = r["gate"]
        bpb = r.get("bpb_gated") or r["bpb"]
        by_gate.setdefault(gate, []).append(bpb)

    print(f"\n  {'Gate mode':<15} {'mean bpb':>10} {'SEM':>8} {'n':>5}")
    print(f"  {'-'*42}")
    for gate, bpbs in sorted(by_gate.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {gate:<15} {sum(bpbs)/len(bpbs):>10.4f} {sem(bpbs):>8.4f} {len(bpbs):>5}")

    # Group by memory state (use bpb_gated when available, same metric as gate table)
    by_mem: dict[str, list[float]] = {}
    for r in all_results:
        bpb = r.get("bpb_gated") or r["bpb"]
        by_mem.setdefault(r["mem_state"], []).append(bpb)

    print(f"\n  {'Memory state':<15} {'mean bpb':>10} {'SEM':>8} {'n':>5}")
    print(f"  {'-'*42}")
    for mem, bpbs in sorted(by_mem.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {mem:<15} {sum(bpbs)/len(bpbs):>10.4f} {sem(bpbs):>8.4f} {len(bpbs):>5}")

    # Best overall config
    best = min(all_results, key=lambda r: r.get("bpb_gated") or r["bpb"])
    bpb_best = best.get("bpb_gated") or best["bpb"]
    print(f"\n  Best eval config: {best['label']}")
    print(f"    bpb = {bpb_best:.4f}")
    print(f"    checkpoint = {best['checkpoint']}")
    print(f"\n  Results saved to: {results_file}")
    print(f"\n  Next: run_artifact_grid.py --checkpoint {best['checkpoint']} --data-path {args.data_path}")


if __name__ == "__main__":
    main()
