#!/usr/bin/env python3
"""Short Exp26 systems pulse and telemetry summarizer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EXP23 = REPO / "experiments" / "23_fast_path"
EXP24 = REPO / "experiments" / "24_training_time_bundle"
EXP26 = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXP23 / "configs" / "base_seq_epoch_lr0064_full_corpus.yaml"
DEFAULT_DATA_PATH = (
    REPO / "baselines" / "parameter_golf" / "datasets" / "fineweb10B_sp16384"
)
DEFAULT_SP_MODEL_16384 = (
    REPO / "baselines" / "parameter_golf" / "tokenizers" / "fineweb_16384_bpe.model"
)
DEFAULT_PROFILE_DIR = EXP26 / "validation" / "profile"

sys.path.insert(0, str(EXP23))
sys.path.insert(0, str(EXP24))
sys.path.insert(0, str(EXP26))
sys.path.insert(0, str(REPO / "src"))

from exp26 import build_validation_matrix  # noqa: E402
from fast_path import read_speed_config, write_matrix  # noqa: E402
from launch import run_matrix_entries  # noqa: E402
from run_exp26 import _prebuild_cache  # noqa: E402


def _get_path(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _health_from_result(result: dict[str, Any]) -> dict[str, Any]:
    train = result.get("train") or {}
    mechanisms = train.get("mechanisms") or {}
    crct = mechanisms.get("crct") or {}
    transport_summary = crct.get("transport_summary") or {}
    health = transport_summary.get("health") or {}
    replay = crct.get("replay_eviction") or {}
    optimizer = train.get("optimizer") or {}
    plasticity = optimizer.get("plasticity_budget") or {}
    return {
        "steps": int(train.get("steps", 0) or 0),
        "elapsed_s": float(train.get("elapsed_s", 0.0) or 0.0),
        "tokens_per_sec": float(train.get("aggregate_tokens_per_sec", 0.0) or 0.0),
        "per_gpu_tokens_per_sec": float(train.get("per_gpu_tokens_per_sec", 0.0) or 0.0),
        "final_loss": float(train.get("final_loss", 0.0) or 0.0),
        "payloads_used": int(health.get("payloads_used", 0) or 0),
        "payloads_scored": int(health.get("payloads_scored", 0) or 0),
        "payload_lag_steps_max": int(health.get("payload_lag_steps_max", 0) or 0),
        "score_seconds_max": float(health.get("score_seconds_max", 0.0) or 0.0),
        "score_stage_timing_enabled": bool(health.get("score_stage_timing_enabled", False)),
        "score_stage_samples": int(health.get("score_stage_samples", 0) or 0),
        "score_stage_encode_seconds_sum": float(
            health.get("score_stage_encode_off_seconds_sum", 0.0) or 0.0
        )
        + float(health.get("score_stage_encode_force_on_seconds_sum", 0.0) or 0.0),
        "score_stage_encode_seconds_max": float(
            health.get("score_stage_encode_off_seconds_max", 0.0) or 0.0
        )
        + float(health.get("score_stage_encode_force_on_seconds_max", 0.0) or 0.0),
        "score_stage_nll_seconds_sum": float(
            health.get("score_stage_nll_off_seconds_sum", 0.0) or 0.0
        )
        + float(health.get("score_stage_nll_mem_seconds_sum", 0.0) or 0.0),
        "score_stage_nll_seconds_max": float(
            health.get("score_stage_nll_off_seconds_max", 0.0) or 0.0
        )
        + float(health.get("score_stage_nll_mem_seconds_max", 0.0) or 0.0),
        "score_stage_plasticity_seconds_sum": float(
            health.get("score_stage_plasticity_seconds_sum", 0.0) or 0.0
        ),
        "score_stage_append_memory_seconds_sum": float(
            health.get("score_stage_append_memory_seconds_sum", 0.0) or 0.0
        ),
        "score_stage_peak_allocated_mb_max": float(
            health.get("score_stage_peak_allocated_mb_max", 0.0) or 0.0
        ),
        "weight_snapshot_published": int(health.get("weight_snapshot_published", 0) or 0),
        "weight_snapshot_applied": int(health.get("weight_snapshot_applied", 0) or 0),
        "weight_snapshot_latest_overwrites": int(health.get("weight_snapshot_latest_overwrites", 0) or 0),
        "weight_snapshot_shm_writes": int(health.get("weight_snapshot_shm_writes", 0) or 0),
        "weight_snapshot_shm_reads": int(health.get("weight_snapshot_shm_reads", 0) or 0),
        "weight_snapshot_version_lag_steps": int(health.get("weight_snapshot_version_lag_steps", 0) or 0),
        "weight_snapshot_read_seconds_sum": float(health.get("weight_snapshot_read_seconds_sum", 0.0) or 0.0),
        "weight_snapshot_read_seconds_max": float(health.get("weight_snapshot_read_seconds_max", 0.0) or 0.0),
        "weight_snapshot_read_tensor_count": int(health.get("weight_snapshot_read_tensor_count", 0) or 0),
        "weight_snapshot_read_bytes": int(health.get("weight_snapshot_read_bytes", 0) or 0),
        "request_ring_full_drops": int(_get_path(transport_summary, "coordinator.teacher_shm_request_ring_full_drops", 0) or 0),
        "result_ring_full_drops": int(_get_path(transport_summary, "memory.teacher_shm_result_ring_full_drops", 0) or 0),
        "maintenance_gpu3_starvation_reason": str(replay.get("gpu3_starvation_reason", "")),
        "maintenance_memory_streams_active": bool(replay.get("memory_streams_active", False)),
        "maintenance_jobs_pushed": int(_get_path(replay, "arm_runtime.jobs_pushed", 0) or 0),
        "maintenance_jobs_popped": int(_get_path(replay, "arm_runtime.jobs_popped", 0) or 0),
        "maintenance_probe_seconds": float(replay.get("probe_seconds", 0.0) or 0.0),
        "maintenance_probe_budget_seconds": float(replay.get("probe_budget_seconds", 0.0) or 0.0),
        "memory_rank_request_events_superseded": int(health.get("memory_rank_request_events_superseded", 0) or 0),
        "memory_rank_outer_loop_seconds_sum": float(health.get("memory_rank_outer_loop_seconds_sum", 0.0) or 0.0),
        "memory_rank_outer_loop_seconds_max": float(health.get("memory_rank_outer_loop_seconds_max", 0.0) or 0.0),
        "memory_rank_pre_pump_seconds_sum": float(health.get("memory_rank_pre_pump_seconds_sum", 0.0) or 0.0),
        "memory_rank_pre_pump_seconds_max": float(health.get("memory_rank_pre_pump_seconds_max", 0.0) or 0.0),
        "memory_rank_replay_seconds_sum": float(health.get("memory_rank_replay_seconds_sum", 0.0) or 0.0),
        "memory_rank_replay_seconds_max": float(health.get("memory_rank_replay_seconds_max", 0.0) or 0.0),
        "memory_rank_replay_ticks": int(health.get("memory_rank_replay_ticks", 0) or 0),
        "memory_rank_replay_probes_ingested": int(health.get("memory_rank_replay_probes_ingested", 0) or 0),
        "memory_rank_replay_deferred_for_packet_work": int(
            health.get("memory_rank_replay_deferred_for_packet_work", 0) or 0
        ),
        "memory_rank_replay_deferred_for_backpressure": int(
            health.get("memory_rank_replay_deferred_for_backpressure", 0) or 0
        ),
        "memory_rank_pump_loop_seconds_sum": float(health.get("memory_rank_pump_loop_seconds_sum", 0.0) or 0.0),
        "memory_rank_pump_loop_seconds_max": float(health.get("memory_rank_pump_loop_seconds_max", 0.0) or 0.0),
        "memory_rank_pump_idle_sleep_seconds_sum": float(health.get("memory_rank_pump_idle_sleep_seconds_sum", 0.0) or 0.0),
        "memory_rank_pump_idle_sleep_seconds_max": float(health.get("memory_rank_pump_idle_sleep_seconds_max", 0.0) or 0.0),
        "plasticity_packets_received": int(health.get("plasticity_packets_received", 0) or 0),
        "plasticity_lr_multiplier_max": float(plasticity.get("lr_multiplier_max", 0.0) or 0.0),
        "teacher_fail_open": int(crct.get("teacher_fail_open", 0) or 0),
    }


def summarize_profile(results_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name in {"matrix.json", "summary.json", "profile_summary.json"}:
            continue
        data = json.loads(path.read_text())
        cfg = data.get("config") or {}
        row = {
            "name": str(cfg.get("name", path.stem)),
            "arm": str(cfg.get("arm", "")),
            "path": str(path),
        }
        if "error" in data:
            row["error"] = str(data["error"])
        else:
            row.update(_health_from_result(data))
        rows.append(row)
    control_tps = next(
        (float(r.get("tokens_per_sec", 0.0)) for r in rows if r.get("arm") == "validation_fastslow_control"),
        0.0,
    )
    for row in rows:
        if control_tps > 0 and "tokens_per_sec" in row:
            row["throughput_vs_control"] = float(row["tokens_per_sec"]) / control_tps
    out = {"rows": rows}
    (results_dir / "profile_summary.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def _print_profile_summary(summary: dict[str, Any]) -> None:
    print("[exp26-profile] summary")
    for row in summary.get("rows", []):
        if "error" in row:
            print(f"  {row['arm']}: ERROR {row['error']}")
            continue
        ratio = row.get("throughput_vs_control")
        ratio_s = f" vs_control={ratio:.3f}x" if isinstance(ratio, float) else ""
        print(
            "  "
            f"{row['arm']}: steps={row['steps']} tps={row['tokens_per_sec']:.1f}"
            f"{ratio_s} payloads={row['payloads_used']}/{row['payloads_scored']} "
            f"snap={row['weight_snapshot_published']}/{row['weight_snapshot_applied']} "
            f"rings_drop={row['request_ring_full_drops']}/{row['result_ring_full_drops']} "
            f"gpu3={row['maintenance_gpu3_starvation_reason']} "
            f"plasticity={row['plasticity_packets_received']}"
        )
        if row.get("arm") == "validation_adaptive_residual_memory":
            print(
                "    "
                f"memory_loop={row.get('memory_rank_outer_loop_seconds_sum', 0.0):.3f}s "
                f"pump={row.get('memory_rank_pump_loop_seconds_sum', 0.0):.3f}s "
                f"replay={row.get('memory_rank_replay_seconds_sum', 0.0):.3f}s "
                f"replay_ticks={row.get('memory_rank_replay_ticks', 0)} "
                f"replay_defer_packet={row.get('memory_rank_replay_deferred_for_packet_work', 0)}"
            )
        if row.get("score_stage_timing_enabled"):
            stage_sum = max(
                float(row.get("score_stage_encode_seconds_sum", 0.0))
                + float(row.get("score_stage_nll_seconds_sum", 0.0))
                + float(row.get("score_stage_plasticity_seconds_sum", 0.0))
                + float(row.get("score_stage_append_memory_seconds_sum", 0.0)),
                1e-9,
            )
            print(
                "    "
                f"gpu3_stage_samples={row['score_stage_samples']} "
                f"encode_sum={row['score_stage_encode_seconds_sum']:.3f}s "
                f"nll_sum={row['score_stage_nll_seconds_sum']:.3f}s "
                f"plasticity_sum={row['score_stage_plasticity_seconds_sum']:.3f}s "
                f"append_sum={row['score_stage_append_memory_seconds_sum']:.3f}s "
                f"encode_share={row['score_stage_encode_seconds_sum'] / stage_sum:.2%} "
                f"peak_mb={row['score_stage_peak_allocated_mb_max']:.0f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--sp-model-path-16384", type=Path, default=DEFAULT_SP_MODEL_16384)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--budget", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument(
        "--arm",
        choices=("both", "control", "adaptive"),
        default="both",
        help="Run both cells, only the control, or only the ARM active cell.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--score-stage-timing",
        action="store_true",
        help="Enable rank-3 CUDA-event stage timing for short profile runs.",
    )
    args = parser.parse_args(argv)

    speed_config = read_speed_config(args.config)
    entries = build_validation_matrix(
        speed_config=speed_config,
        world_size=int(args.world_size),
        budget_seconds=float(args.budget),
        seed=int(args.seed),
    )
    if args.arm == "control":
        entries = entries[:1]
    elif args.arm == "adaptive":
        entries = entries[1:]
    if args.score_stage_timing:
        for entry in entries:
            if bool(entry.get("crct_enabled", False)):
                entry["crct_score_stage_timing_enabled"] = True

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(args.results_dir / "matrix.json", entries)
    print(
        f"[exp26-profile] entries={len(entries)} arm={args.arm} "
        f"world_size={args.world_size} budget={args.budget}s dry_run={args.dry_run}",
        flush=True,
    )
    if args.dry_run:
        for entry in entries:
            print(json.dumps(entry, indent=2, sort_keys=True, default=str))
        return 0

    _prebuild_cache(str(args.data_path))
    run_matrix_entries(
        entries=entries,
        runner_path=EXP23 / "runner_fast_path.py",
        data_path=str(args.data_path),
        sp_model_paths={16384: str(args.sp_model_path_16384)},
        results_dir=args.results_dir,
        world_size=int(args.world_size),
        limit=None,
        dry_run=False,
        skip_existing=False,
        checkpoint_dir=None,
    )
    summary = summarize_profile(args.results_dir)
    _print_profile_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
