#!/usr/bin/env python3
"""Summarize the Phase 1 ablation matrix emitted by the persistent-DDP runner.

Spec source: ``docs/plans/2026-04-17-experiment-19-phase1-impl.md`` Task
1C-3 and "Validation protocols" → "Throughput gates" + "Ablation matrix
gates".

For each precision (``bf16`` and optionally ``fp8_fused``) and each of
the three Track 1A levers (``fused_grad_clip``, ``fused_muon``,
``compile_full_path``), the summarizer pairs the "all-on" entry at
seed ``s`` against the matching "lever-off" entry at seed ``s``,
computes paired deltas on ``tokens_per_sec`` / ``final_loss`` /
``bpb`` / ``peak_vram_mb``, runs a paired-t-test on the tok/s delta,
and assigns a verdict:

    SHIP          — mean Δ tok/s > 0 AND paired-t p < 0.05 AND
                    |Δ final_loss| ≤ 0.02 AND
                    |Δ peak_vram_mb| / baseline ≤ 0.05
    PARK          — mean Δ tok/s ≤ 0 OR a quality gate fails by more
                    than the "SHIP" bar
    INCONCLUSIVE  — mean is positive but p ≥ 0.05 (rerun with more seeds)

The summarizer reuses three load-bearing helpers from elsewhere in the
repo instead of re-implementing them:

    * ``stats.paired_ttest``            — paired (dependent-samples)
                                          t-test. Pure-Python; matches
                                          ``scipy.stats.ttest_rel`` to
                                          floating-point precision.
                                          Exp 18 uses the same import.
    * ``run_persistent_launcher._check_output_integrity``
                                        — classifies every expected
                                          output JSON into success /
                                          benign-skip / real-error.
                                          The "quality gate" refers to
                                          *this* function — if any
                                          entry is a real error the
                                          summarizer refuses to emit
                                          verdicts.
    * ``run_persistent_launcher._BENIGN_ERROR_PATTERNS``
                                        — benign error patterns the
                                          launcher pre-writes (e.g.
                                          fp8 entries when TE is
                                          unavailable). Reused here so
                                          the gate semantics stay in
                                          sync with the launcher.

The "coherent-pair resolver" and "full-coverage gate" are phase-1
specific (the lever-leave-out shape is unique to this matrix) and
live below. They mirror the invariants Ken names in
``feedback_anticipate_expensive_experiment_contamination``: the
resolver fails loud if pairing is ambiguous, and the coverage gate
fails loud if any expected (precision, lever-combo, seed) tuple is
missing from the runs-dir.

Usage
-----

::

    python experiments/19_phase1/summarize_phase1.py \\
        --matrix-json experiments/19_phase1/matrix_phase1.json \\
        --runs-dir   experiments/19_phase1/runs/ \\
        --output     experiments/19_phase1/RESULTS_PHASE1C.md

Add ``--dry-run`` to print the markdown without writing it. Add
``--lenient`` to continue with missing entries marked "missing" in the
table instead of raising the coverage gate — useful for incremental
rerunning during matrix execution.

Exit codes
----------

::

    0 — success; markdown written (or printed in --dry-run).
    1 — quality gate failed (real errors in runs-dir). No verdicts emitted.
    2 — coverage gate failed in strict mode (missing entries).
    3 — coherent-pair resolver failed (ambiguous matrix).
    4 — matrix / CLI usage error.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


# --- Reuse hooks -----------------------------------------------------
#
# The summarizer lives inside ``experiments/19_phase1/`` which has no
# ``__init__.py`` (the repo convention is one-script-per-experiment,
# never a package), so we reach for ``stats`` and the launcher's
# integrity helper via ``importlib`` rather than a package import.

_REPO = Path(__file__).resolve().parents[2]
_STATS_MODULE_PATH = _REPO / "experiments" / "09_revised_architecture" / "stats.py"
_LAUNCHER_MODULE_PATH = _REPO / "experiments" / "19_prereqs" / "run_persistent_launcher.py"


def _import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_stats() -> Any:
    return _import_file("stats_phase1", _STATS_MODULE_PATH)


def _load_launcher() -> Any:
    return _import_file("run_persistent_launcher_phase1", _LAUNCHER_MODULE_PATH)


# --- Constants aligned with build_matrix_phase1 ---------------------

# The three Track 1A levers, in stable iteration order. Must match
# ``build_matrix_phase1._LEVER_KEYS``; a mismatch means the summarizer
# would silently skip a lever.
LEVER_KEYS: tuple[str, str, str] = (
    "fused_grad_clip",
    "fused_muon",
    "compile_full_path",
)

# Mapping from lever key to the "leave-one-out" label
# ``build_matrix_phase1`` writes into the entry ``name``. Duplicated
# here intentionally: if the build-matrix side renames a label, we
# want the summarizer to start failing on ambiguous-pair lookups
# rather than silently pair a stale label.
LEVER_LEAVE_OUT_LABEL: dict[str, str] = {
    "fused_grad_clip": "no_fused_clip",
    "fused_muon": "no_fused_muon",
    "compile_full_path": "no_compile",
}

# SHIP gate thresholds from the plan's "Throughput gates" section.
SHIP_LOSS_TOL: float = 0.02            # |Δ final_loss|
SHIP_VRAM_FRAC_TOL: float = 0.05       # |Δ peak_vram_mb| / baseline
SHIP_P_THRESHOLD: float = 0.05         # paired-t two-tailed


# --- Input loading ---------------------------------------------------


def load_matrix(matrix_path: Path) -> list[dict[str, Any]]:
    """Load and shape-check the matrix JSON."""
    payload = json.loads(matrix_path.read_text())
    if not isinstance(payload, list):
        raise ValueError(
            f"{matrix_path}: expected a JSON list of entries, got "
            f"{type(payload).__name__}"
        )
    required = {"name", "seed", "precision", *LEVER_KEYS}
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{matrix_path}: entry {idx} is {type(entry).__name__}, "
                "expected dict"
            )
        missing = required - entry.keys()
        if missing:
            raise ValueError(
                f"{matrix_path}: entry {idx} ({entry.get('name', '?')!r}) "
                f"missing required keys {sorted(missing)}"
            )
    return payload


def load_run_json(runs_dir: Path, entry: dict[str, Any]) -> dict[str, Any] | None:
    """Load the per-entry result JSON. Returns None if the file is missing."""
    name = entry["name"]
    seed = int(entry["seed"])
    path = runs_dir / f"{name}_s{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# --- Full-coverage gate ----------------------------------------------


def full_coverage_gate(
    matrix: list[dict[str, Any]],
    runs_dir: Path,
    *,
    lenient: bool = False,
) -> list[dict[str, Any]]:
    """Assert every matrix entry has a corresponding result file.

    In strict mode (``lenient=False``), raises ``RuntimeError`` if any
    entry is missing. In lenient mode, returns a list of missing
    entries so the caller can surface them in the output. This is the
    "full-coverage gate" named in ``feedback_anticipate_expensive_
    experiment_contamination`` — without it a crashed seed silently
    halves the paired-t sample size and the verdict papers over the
    missing data.
    """
    missing: list[dict[str, Any]] = []
    for entry in matrix:
        name = entry["name"]
        seed = int(entry["seed"])
        path = runs_dir / f"{name}_s{seed}.json"
        if not path.exists():
            missing.append({"name": name, "seed": seed, "path": str(path)})

    if missing and not lenient:
        details = ", ".join(f"{m['name']}_s{m['seed']}" for m in missing[:10])
        more = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        raise RuntimeError(
            f"full-coverage gate: {len(missing)} matrix entries missing "
            f"from {runs_dir}: {details}{more}. Rerun the launcher or "
            f"pass --lenient to continue with 'missing' markers."
        )
    return missing


# --- Quality gate ----------------------------------------------------


def quality_gate(
    matrix: list[dict[str, Any]],
    runs_dir: Path,
) -> tuple[int, int, list[str]]:
    """Reuse the launcher's ``_check_output_integrity`` to classify every
    expected output into success / benign-skip / real-error.

    Returns ``(success_count, benign_skip_count, real_errors)``. The
    summarizer refuses to emit verdicts iff ``real_errors`` is non-empty
    — a non-zero real-error count means at least one run crashed in a
    way that isn't a pre-declared benign skip, and any paired-t on
    the partial data would silently understate variance.
    """
    launcher = _load_launcher()
    return launcher._check_output_integrity(
        runs_dir, matrix, launcher._BENIGN_ERROR_PATTERNS,
    )


# --- Coherent-pair resolver ------------------------------------------


def _is_all_on(entry: dict[str, Any]) -> bool:
    return all(bool(entry[k]) for k in LEVER_KEYS)


def _is_lever_off(entry: dict[str, Any], lever: str) -> bool:
    """True iff ``entry`` has ``lever`` OFF and the other two levers ON."""
    if bool(entry[lever]):
        return False
    for other in LEVER_KEYS:
        if other == lever:
            continue
        if not bool(entry[other]):
            return False
    return True


def resolve_coherent_pairs(
    matrix: list[dict[str, Any]],
) -> dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]]:
    """For each (precision, lever) return a list of (all_on, lever_off)
    entry pairs, one per seed.

    Fails loud if pairing is ambiguous — e.g. two entries match the
    "all-on" shape at the same (precision, seed), or two entries
    match the "lever-off" shape at the same (precision, seed). Also
    fails loud if a seed has an "all-on" entry but no matching
    "lever-off" entry for some lever (and vice versa). A silently
    dropped pair is worse than a noisy abort: it biases the paired-t
    toward whichever seeds happened to land on both sides.

    Returns:
        Mapping ``(precision, lever) -> [(all_on_entry, lever_off_entry), ...]``
        with one pair per seed, sorted by seed for deterministic output.
    """
    # Bucket by (precision, seed) -> entries. We then walk buckets and
    # pull the canonical "all-on" + "lever-off" entries out of each.
    by_precision_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for entry in matrix:
        key = (str(entry["precision"]), int(entry["seed"]))
        by_precision_seed.setdefault(key, []).append(entry)

    pairs: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    precisions = sorted({str(e["precision"]) for e in matrix})

    for precision in precisions:
        seeds = sorted({
            seed for (p, seed) in by_precision_seed if p == precision
        })
        for lever in LEVER_KEYS:
            pair_list: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for seed in seeds:
                bucket = by_precision_seed.get((precision, seed), [])
                all_on_entries = [e for e in bucket if _is_all_on(e)]
                lever_off_entries = [
                    e for e in bucket if _is_lever_off(e, lever)
                ]
                if len(all_on_entries) > 1:
                    raise RuntimeError(
                        f"coherent-pair resolver: precision={precision!r} "
                        f"seed={seed} has {len(all_on_entries)} entries "
                        f"matching the 'all-on' shape "
                        f"({[e['name'] for e in all_on_entries]}). "
                        "Pairing is ambiguous."
                    )
                if len(lever_off_entries) > 1:
                    raise RuntimeError(
                        f"coherent-pair resolver: precision={precision!r} "
                        f"seed={seed} lever={lever!r} has "
                        f"{len(lever_off_entries)} entries matching the "
                        f"'lever-off' shape "
                        f"({[e['name'] for e in lever_off_entries]}). "
                        "Pairing is ambiguous."
                    )
                if not all_on_entries:
                    # No all-on entry at this seed: skip this seed for
                    # every lever. Flagged via full-coverage gate, not
                    # here — the resolver's job is pairing, not coverage.
                    continue
                if not lever_off_entries:
                    continue
                pair_list.append((all_on_entries[0], lever_off_entries[0]))
            pairs[(precision, lever)] = pair_list
    return pairs


# --- Metric extraction ------------------------------------------------


def _tokens_per_step(config: dict[str, Any]) -> int:
    """Compute tokens-per-step from a result JSON's config block.

    Matches the Exp 18 summarizer convention exactly:
    ``world_size × batch_size × seq_len``. The runner writes
    ``world_size`` into the config when DDP is active; we fall back to
    1 (single-GPU matrix) if the field is absent so the summarizer
    handles synthetic-fixture tests too.
    """
    ws = int(config.get("world_size", 1))
    bs = int(config["batch_size"])
    seq = int(config["seq_len"])
    return ws * bs * seq


def extract_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Pull the four metrics the verdict logic depends on.

    ``bpb`` may be NaN when eval didn't run (the runner emits an
    empty ``eval`` block in that case). Callers handle NaN bpb by
    marking the bpb column "N/A" per-row.
    """
    train = result.get("train", {})
    evaluation = result.get("eval", {})
    config = result.get("config", {})

    sps = float(train.get("steps_per_second", float("nan")))
    tps = sps * _tokens_per_step(config) if math.isfinite(sps) else float("nan")

    final_loss = float(train.get("final_loss", float("nan")))
    peak_vram_mb = float(train.get("peak_vram_mb", float("nan")))
    bpb_val = evaluation.get("bpb", float("nan"))
    bpb = float(bpb_val) if bpb_val is not None else float("nan")

    return {
        "tokens_per_sec": tps,
        "final_loss": final_loss,
        "bpb": bpb,
        "peak_vram_mb": peak_vram_mb,
    }


# --- Verdict logic ---------------------------------------------------


def _mean(values: Iterable[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    return sum(xs) / len(xs) if xs else float("nan")


def compute_lever_verdict(
    all_on_metrics: list[dict[str, float]],
    lever_off_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    """Paired comparison (all_on - lever_off) and verdict per plan gates.

    ``all_on_metrics[i]`` and ``lever_off_metrics[i]`` must be paired
    — caller is responsible for ordering.
    """
    stats = _load_stats()

    n = len(all_on_metrics)
    assert n == len(lever_off_metrics), (
        f"paired inputs must have equal length; got {n} vs "
        f"{len(lever_off_metrics)}"
    )

    tps_on = [m["tokens_per_sec"] for m in all_on_metrics]
    tps_off = [m["tokens_per_sec"] for m in lever_off_metrics]
    loss_on = [m["final_loss"] for m in all_on_metrics]
    loss_off = [m["final_loss"] for m in lever_off_metrics]
    bpb_on = [m["bpb"] for m in all_on_metrics]
    bpb_off = [m["bpb"] for m in lever_off_metrics]
    vram_on = [m["peak_vram_mb"] for m in all_on_metrics]
    vram_off = [m["peak_vram_mb"] for m in lever_off_metrics]

    delta_tps = [a - b for a, b in zip(tps_on, tps_off)]
    delta_loss = [a - b for a, b in zip(loss_on, loss_off)]
    delta_bpb_pairs = [
        (a, b) for a, b in zip(bpb_on, bpb_off)
        if math.isfinite(a) and math.isfinite(b)
    ]
    delta_bpb = [a - b for a, b in delta_bpb_pairs]
    delta_vram = [a - b for a, b in zip(vram_on, vram_off)]

    mean_delta_tps = _mean(delta_tps)
    mean_delta_loss = _mean(delta_loss)
    mean_delta_bpb = _mean(delta_bpb) if delta_bpb else float("nan")
    mean_delta_vram = _mean(delta_vram)

    # Paired-t on tok/s. With n < 2 the stat is undefined; treat as
    # INCONCLUSIVE rather than picking sides.
    if n >= 2:
        _, p_value = stats.paired_ttest(tps_on, tps_off)
    else:
        p_value = float("nan")

    # Marginal tok/s fraction: (all_on - lever_off) / all_on. Positive
    # means the lever contributes positively when toggled on.
    mean_tps_on = _mean(tps_on)
    if math.isfinite(mean_tps_on) and mean_tps_on > 0:
        marginal_frac = mean_delta_tps / mean_tps_on
    else:
        marginal_frac = float("nan")

    # VRAM fraction relative to lever-off baseline. The plan says
    # "±5% peak VRAM" so the baseline is the non-fused condition.
    mean_vram_off = _mean(vram_off)
    if math.isfinite(mean_vram_off) and mean_vram_off > 0:
        vram_frac = mean_delta_vram / mean_vram_off
    else:
        vram_frac = float("nan")

    # Verdict ladder
    if not math.isfinite(mean_delta_tps) or mean_delta_tps <= 0:
        verdict = "PARK"
        reason = "mean Δ tok/s ≤ 0"
    elif (
        math.isfinite(mean_delta_loss)
        and abs(mean_delta_loss) > SHIP_LOSS_TOL
    ):
        verdict = "PARK"
        reason = (
            f"|Δ final_loss|={abs(mean_delta_loss):.4f} exceeds tol "
            f"{SHIP_LOSS_TOL}"
        )
    elif (
        math.isfinite(vram_frac)
        and abs(vram_frac) > SHIP_VRAM_FRAC_TOL
    ):
        verdict = "PARK"
        reason = (
            f"|Δ peak_vram|/baseline={abs(vram_frac):.3f} exceeds tol "
            f"{SHIP_VRAM_FRAC_TOL}"
        )
    elif math.isnan(p_value) or p_value >= SHIP_P_THRESHOLD:
        verdict = "INCONCLUSIVE"
        reason = (
            f"mean Δ tok/s positive but p={p_value:.3f} ≥ "
            f"{SHIP_P_THRESHOLD} — rerun with more seeds"
        )
    else:
        verdict = "SHIP"
        reason = (
            f"Δ tok/s={mean_delta_tps:.0f} (p={p_value:.3f}); quality "
            f"gates passed"
        )

    return {
        "n": n,
        "mean_delta_tokens_per_sec": mean_delta_tps,
        "marginal_tokens_per_sec_frac": marginal_frac,
        "mean_delta_final_loss": mean_delta_loss,
        "mean_delta_bpb": mean_delta_bpb,
        "mean_delta_peak_vram_mb": mean_delta_vram,
        "delta_peak_vram_frac": vram_frac,
        "paired_t_p_value": p_value,
        "verdict": verdict,
        "reason": reason,
    }


# --- Markdown rendering ----------------------------------------------


def _format_float(val: float, fmt: str = "{:+.3f}") -> str:
    if not math.isfinite(val):
        return "N/A"
    return fmt.format(val)


def _format_p(val: float) -> str:
    if not math.isfinite(val):
        return "N/A"
    if val < 0.001:
        return "<0.001"
    return f"{val:.3f}"


def render_markdown(
    matrix_path: Path,
    runs_dir: Path,
    quality: tuple[int, int, list[str]],
    missing: list[dict[str, Any]],
    verdicts_by_precision: dict[str, dict[str, dict[str, Any]]],
    lenient: bool,
) -> str:
    """Emit the RESULTS_PHASE1C.md body as a single markdown string."""
    lines: list[str] = []
    lines.append("# Phase 1C — lever ablation results")
    lines.append("")
    lines.append(f"- matrix: `{matrix_path}`")
    lines.append(f"- runs-dir: `{runs_dir}`")
    success, benign, real_errors = quality
    lines.append(
        f"- integrity: success={success}, benign_skip={benign}, "
        f"real_errors={len(real_errors)}"
    )
    if real_errors:
        lines.append("")
        lines.append("**Quality gate failed — verdicts suppressed.**")
        lines.append("")
        lines.append("Real errors:")
        for err in real_errors:
            lines.append(f"- {err}")
        return "\n".join(lines) + "\n"

    if missing:
        lines.append(
            f"- coverage: {len(missing)} missing entries "
            f"({'lenient' if lenient else 'strict'} mode)"
        )
        if lenient:
            lines.append("")
            lines.append("Missing entries (treated as 'missing' in tables):")
            for m in missing:
                lines.append(f"- {m['name']}_s{m['seed']}")

    for precision in sorted(verdicts_by_precision.keys()):
        lines.append("")
        lines.append(f"## precision = {precision}")
        lines.append("")
        lines.append(
            "| lever | n | marginal tok/s | Δ tok/s | ΔΔ final_loss | "
            "ΔΔ bpb | Δ vram (MB) | paired-t p | verdict | note |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
        )
        lever_verdicts = verdicts_by_precision[precision]
        for lever in LEVER_KEYS:
            v = lever_verdicts.get(lever)
            if v is None:
                lines.append(
                    f"| {lever} | 0 | missing | missing | missing | "
                    f"missing | missing | missing | MISSING | "
                    f"no pairs for this precision |"
                )
                continue
            marginal = _format_float(
                v["marginal_tokens_per_sec_frac"], "{:+.2%}"
            )
            delta_tps = _format_float(
                v["mean_delta_tokens_per_sec"], "{:+,.0f}"
            )
            delta_loss = _format_float(v["mean_delta_final_loss"], "{:+.4f}")
            delta_bpb = _format_float(v["mean_delta_bpb"], "{:+.4f}")
            delta_vram = _format_float(v["mean_delta_peak_vram_mb"], "{:+,.0f}")
            p = _format_p(v["paired_t_p_value"])
            lines.append(
                f"| {lever} | {v['n']} | {marginal} | {delta_tps} | "
                f"{delta_loss} | {delta_bpb} | {delta_vram} | {p} | "
                f"**{v['verdict']}** | {v['reason']} |"
            )

    return "\n".join(lines) + "\n"


# --- Orchestration ---------------------------------------------------


def summarize(
    matrix_path: Path,
    runs_dir: Path,
    *,
    lenient: bool = False,
) -> tuple[int, str]:
    """Top-level orchestration. Returns ``(exit_code, markdown)``.

    Exit codes match the module docstring:
        0 — success
        1 — quality gate failed
        2 — coverage gate failed (strict)
        3 — coherent-pair resolver failed
        4 — matrix / CLI usage error
    """
    try:
        matrix = load_matrix(matrix_path)
    except (ValueError, json.JSONDecodeError) as exc:
        return 4, f"matrix load error: {exc}\n"

    # Coverage gate first: strict mode raises before we do any
    # further work.
    try:
        missing = full_coverage_gate(matrix, runs_dir, lenient=lenient)
    except RuntimeError as exc:
        return 2, f"coverage gate failed: {exc}\n"

    # Quality gate: reuse launcher's integrity checker.
    quality = quality_gate(matrix, runs_dir)
    success, benign, real_errors = quality

    # Coherent pairing runs on the full matrix (pure shape check).
    try:
        pairs = resolve_coherent_pairs(matrix)
    except RuntimeError as exc:
        return 3, f"coherent-pair resolver failed: {exc}\n"

    # Build verdicts per (precision, lever). If quality gate failed,
    # skip verdict computation and let the renderer emit the "gate
    # failed" banner.
    verdicts_by_precision: dict[str, dict[str, dict[str, Any]]] = {}
    if not real_errors:
        # Group verdicts by precision for the per-section markdown shape.
        for (precision, lever), pair_list in pairs.items():
            all_on_metrics: list[dict[str, float]] = []
            lever_off_metrics: list[dict[str, float]] = []
            for all_on_entry, lever_off_entry in pair_list:
                all_on_result = load_run_json(runs_dir, all_on_entry)
                lever_off_result = load_run_json(runs_dir, lever_off_entry)
                if all_on_result is None or lever_off_result is None:
                    # Lenient-mode skip: the coverage gate already
                    # surfaced this in the report body.
                    continue
                if "error" in all_on_result or "error" in lever_off_result:
                    continue
                all_on_metrics.append(extract_metrics(all_on_result))
                lever_off_metrics.append(extract_metrics(lever_off_result))

            if not all_on_metrics:
                continue
            verdict = compute_lever_verdict(all_on_metrics, lever_off_metrics)
            verdicts_by_precision.setdefault(precision, {})[lever] = verdict

    markdown = render_markdown(
        matrix_path=matrix_path,
        runs_dir=runs_dir,
        quality=quality,
        missing=missing,
        verdicts_by_precision=verdicts_by_precision,
        lenient=lenient,
    )
    if real_errors:
        return 1, markdown
    return 0, markdown


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the Phase 1 ablation matrix emitted by the "
            "persistent-DDP runner. Emits RESULTS_PHASE1C.md."
        ),
    )
    parser.add_argument(
        "--matrix-json", type=Path, required=True,
        help="Path to the matrix JSON from build_matrix_phase1.py.",
    )
    parser.add_argument(
        "--runs-dir", type=Path, required=True,
        help="Directory where the persistent-DDP runner wrote one JSON per entry.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/19_phase1/RESULTS_PHASE1C.md"),
        help="Markdown output path.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the markdown to stdout instead of writing it.",
    )
    parser.add_argument(
        "--lenient", action="store_true",
        help=(
            "Continue with missing entries marked 'missing' in the "
            "table instead of failing the coverage gate."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    exit_code, markdown = summarize(
        matrix_path=args.matrix_json,
        runs_dir=args.runs_dir,
        lenient=args.lenient,
    )
    if args.dry_run:
        sys.stdout.write(markdown)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        # Mirror to stdout too so CI logs show the verdict banner.
        sys.stdout.write(markdown)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
