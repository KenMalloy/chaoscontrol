#!/usr/bin/env python3
"""Assemble the Phase 1 ablation matrix for the persistent-DDP launcher.

Spec source: ``docs/plans/2026-04-17-experiment-19-phase1-impl.md`` Task
1C-1 and "Track 1C — ablation matrix". The matrix is a one-at-a-time
lever-leave-out: for each precision ``P`` in the active precision set,
emit five lever combinations —

    P_stock            (all three Track 1A levers OFF)
    P_all              (all three levers ON)
    P_no_fused_clip    (fused_grad_clip OFF, other two ON)
    P_no_fused_muon    (fused_muon OFF, other two ON)
    P_no_compile       (compile_full_path OFF, other two ON)

and repeat each of those five combinations once per seed. Seeds drive
paired-t-test statistics; they never vary the lever combination inside a
condition.

The default bf16-only matrix is ``len(seeds) * 5``. With ``include_fp8=
True`` a parallel fp8_fused slice doubles it to ``len(seeds) * 10``.

Key-name contract
-----------------
The dict keys this module writes MUST match what
``experiments/19_prereqs/runner_persistent_ddp.py`` consumes:

    fused_grad_clip      — bool, read by ``run_one_seed``
    fused_muon           — bool, read by ``run_one_seed`` /
                           ``_build_optimizer_with_fused_muon``
    compile_full_path    — bool, read by ``run_one_seed``
    precision            — str,  read by ``run_one_seed`` (accepts
                           ``"bf16"``, ``"fp8"``; ``"fp8_fused"`` is
                           emitted here as the forward-compatible Phase
                           1B nickname — wiring ships in Task 1B-4)

Do not rename these keys. If the runner renames, update here and re-run
``test_key_names_match_runner_contract``.

Default base config
-------------------
Anchored at Exp 18 Test 4b's winning submission regime (see
``experiments/18_throughput_levers/run_exp18_test4b.py`` and the
canonical ``experiments/19_prereqs/run_persistent_launcher.py::
_base_config``). Callers override by passing ``base_config=<dict>`` at
the Python API or ``--base-config <path.json>`` at the CLI.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


# Anchor: Exp 18 Test 4b winner at ws=4 (LR = 2 × Test 5b's ws=2 anchor
# 0.064). Mirrors ``run_persistent_launcher._base_config``; any drift
# there must be reflected here. ``world_size`` is elided because the
# persistent-DDP launcher validates it against its own --world-size CLI
# arg, and matrix assembly is world-size-agnostic (the caller picks ws
# and the matching LR).
_DEFAULT_BASE_CONFIG: dict[str, Any] = {
    "model_type": "ssm",
    "vocab_size": 16384,
    "model_dim": 256,
    "num_layers": 4,
    "ff_mult": 2,
    "seq_len": 512,
    "stride": 256,
    "batch_size": 1024,
    "eval_batches": 16,
    "a_mode": "diag",
    "base_lr": 0.128,   # ws=4 anchor; Test 5b ws=2 winner 0.064 × 2.
    "weight_decay": 0.01,
    "grad_clip_norm": 1.0,
    "activation_checkpoint": True,
    "optimizer": "muon",
    "chunk_size": 64,
    "local_attn_window": 0,
    "local_attn_heads": 1,
    "local_attn_dim": 64,
    "device": "auto",
    "dtype": "bf16",
    "warmup_steps": 20,
}


# The three Track 1A levers. Stored as ordered tuple so the "leave-one-
# out" row order is deterministic across runs.
_LEVER_KEYS: tuple[str, str, str] = (
    "fused_grad_clip",
    "fused_muon",
    "compile_full_path",
)


# Short name hint for each "leave-one-out" condition. Matches the regex
# in tests/test_build_matrix_phase1.py::test_names_unique_and_readable —
# any change here must update that regex too.
_LEVER_LEAVE_OUT_LABEL: dict[str, str] = {
    "fused_grad_clip": "no_fused_clip",
    "fused_muon": "no_fused_muon",
    "compile_full_path": "no_compile",
}


def _lever_combos() -> list[tuple[str, dict[str, bool]]]:
    """Return the 5 canonical (label, lever-dict) pairs in stable order.

    Order:
        stock              (all levers False)
        all                (all levers True)
        no_fused_clip      (only fused_grad_clip False)
        no_fused_muon      (only fused_muon False)
        no_compile         (only compile_full_path False)
    """
    combos: list[tuple[str, dict[str, bool]]] = []
    combos.append(("stock", {k: False for k in _LEVER_KEYS}))
    combos.append(("all", {k: True for k in _LEVER_KEYS}))
    for lever in _LEVER_KEYS:
        label = _LEVER_LEAVE_OUT_LABEL[lever]
        combos.append((
            label,
            {k: (k != lever) for k in _LEVER_KEYS},
        ))
    return combos


def build_matrix_phase1(
    seeds: list[int],
    *,
    include_fp8: bool = False,
    base_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the Phase 1 ablation matrix as a flat list of entries.

    Each entry is a full launcher-entry dict: the merged base_config plus
    ``name``, ``seed``, ``precision``, and the three lever toggles. The
    output order groups by ``(precision, lever-combo)`` outer and seeds
    inner, so per-entry JSON filenames get written in a predictable
    order and partial-failure triage stays easy.

    Args:
        seeds: Distinct seeds to repeat every (precision, lever-combo)
            across. Must be non-empty AND contain no duplicates —
            duplicate seeds produce colliding ``name`` fields and the
            per-entry JSON writes in the runner would silently clobber
            each other, halving the effective paired-t sample size.
        include_fp8: If True, emit a parallel ``fp8_fused`` slice in
            addition to bf16. Phase 1B must land before these entries
            run successfully (see Task 1B-4 in the plan).
        base_config: Override for the default base config. If None,
            uses the Exp 18 Test 4b anchor defined at module top. The
            dict is deep-copied per entry so callers can mutate their
            input safely.

    Returns:
        List of entry dicts, length ``len(seeds) * 5`` (bf16-only) or
        ``len(seeds) * 10`` (with fp8).

    Raises:
        ValueError: if ``seeds`` is empty or contains duplicates.
    """
    if not seeds:
        raise ValueError("seeds must be a non-empty list")
    if len(set(seeds)) != len(seeds):
        # Duplicate seeds would produce identical name strings ->
        # per-entry JSON writes collide, paired-t silently halves.
        duplicates = sorted({s for s in seeds if seeds.count(s) > 1})
        raise ValueError(
            f"seeds must be distinct; got duplicates {duplicates}. "
            "Matrix names would collide and per-entry JSONs would "
            "overwrite each other, silently halving the paired-t "
            "sample size."
        )

    base = base_config if base_config is not None else _DEFAULT_BASE_CONFIG
    precisions: list[str] = ["bf16"]
    if include_fp8:
        precisions.append("fp8_fused")

    entries: list[dict[str, Any]] = []
    for precision in precisions:
        for combo_label, combo in _lever_combos():
            for seed in seeds:
                entry: dict[str, Any] = copy.deepcopy(base)
                entry["name"] = f"{precision}_{combo_label}_seed{seed}"
                entry["seed"] = int(seed)
                entry["precision"] = precision
                entry.update(combo)
                entries.append(entry)
    return entries


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Phase 1 ablation matrix and emit it as JSON. "
            "The output is a list of launcher-entry dicts suitable for "
            "run_persistent_launcher.py --matrix-json."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="One or more integer seeds (e.g., --seeds 1337 2674 4011 5348).",
    )
    parser.add_argument(
        "--include-fp8",
        action="store_true",
        help=(
            "Also emit the fp8_fused precision slice. Requires Phase 1B "
            "(Task 1B-4) to have landed before these entries execute."
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSON file with an override base config. "
            "If omitted, defaults to the Exp 18 Test 4b anchor."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path. If omitted, the matrix is "
            "printed to stdout."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    base_config: dict[str, Any] | None = None
    if args.base_config is not None:
        base_config = json.loads(args.base_config.read_text())
        if not isinstance(base_config, dict):
            raise ValueError(
                f"--base-config {args.base_config} must contain a JSON "
                f"object, got {type(base_config).__name__}"
            )

    matrix = build_matrix_phase1(
        seeds=list(args.seeds),
        include_fp8=args.include_fp8,
        base_config=base_config,
    )

    payload = json.dumps(matrix, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
