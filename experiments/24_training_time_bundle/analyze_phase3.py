"""Phase 3 analysis helpers for the `episodic_controller_v1` matrix.

The controller matrix compares BPB across five simplex arms. Lower BPB is
better. Pairwise effects use ``delta_bpb = treatment_bpb - control_bpb``
so a negative delta means the treatment improved over its control for
that seed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


ARM_A_CONTROL = "arm_a_control"
ARM_NAMES = (
    ARM_A_CONTROL,
    "arm_b_heuristic",
    "arm_c_simplex_frozen",
    "arm_d_simplex_online",
    "arm_e_simplex_warm_online",
)
PAIRWISE_COMPARISONS = {
    "simplex_vs_heuristic": (
        "arm_d_simplex_online",
        "arm_b_heuristic",
    ),
    "warm_vs_cold": (
        "arm_e_simplex_warm_online",
        "arm_d_simplex_online",
    ),
    "online_vs_frozen": (
        "arm_d_simplex_online",
        "arm_c_simplex_frozen",
    ),
}


def analyze_controller_arms(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the six-arm controller matrix from result records.

    Records may carry ``arm`` at top level or nested under a config dict. BPB
    is read from common result fields such as ``val_bpb`` or ``bpb``. Pairwise
    comparisons are paired strictly by seed and raise ``ValueError`` on any
    missing seed instead of silently doing unpaired math.
    """
    by_arm_seed = _index_records(records)
    return {
        "summary_table": _summary_table(by_arm_seed),
        "pairwise_comparisons": _pairwise_comparisons(by_arm_seed),
    }


def _index_records(
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[int, float]]:
    by_arm_seed: dict[str, dict[int, float]] = {}
    for record in records:
        arm = _record_arm(record)
        seed = _record_seed(record)
        bpb = _record_bpb(record)
        by_arm_seed.setdefault(arm, {})
        if seed in by_arm_seed[arm]:
            raise ValueError(f"duplicate record for arm={arm!r} seed={seed!r}")
        by_arm_seed[arm][seed] = bpb
    return by_arm_seed


def _summary_table(by_arm_seed: dict[str, dict[int, float]]) -> list[dict[str, Any]]:
    control_by_seed = by_arm_seed.get(ARM_A_CONTROL, {})
    rows = []
    for arm in ARM_NAMES:
        seed_values = by_arm_seed.get(arm, {})
        values = list(seed_values.values())
        rows.append(
            {
                "arm": arm,
                "n": len(values),
                "mean_bpb": _mean(values),
                "std_bpb": _sample_std(values),
                "fraction_seeds_beating_arm_a_control": (
                    _fraction_beating_control(seed_values, control_by_seed)
                ),
            }
        )
    return rows


def _pairwise_comparisons(
    by_arm_seed: dict[str, dict[int, float]],
) -> dict[str, dict[str, Any]]:
    comparisons = {}
    for name, (treatment_arm, control_arm) in PAIRWISE_COMPARISONS.items():
        treatment = by_arm_seed.get(treatment_arm, {})
        control = by_arm_seed.get(control_arm, {})
        treatment_seeds = set(treatment)
        control_seeds = set(control)
        if treatment_seeds != control_seeds:
            missing_treatment = sorted(control_seeds - treatment_seeds)
            missing_control = sorted(treatment_seeds - control_seeds)
            raise ValueError(
                f"missing paired seeds for {name}: "
                f"treatment_arm={treatment_arm!r} missing={missing_treatment}; "
                f"control_arm={control_arm!r} missing={missing_control}"
            )

        deltas_by_seed = {
            seed: treatment[seed] - control[seed] for seed in sorted(treatment_seeds)
        }
        deltas = list(deltas_by_seed.values())
        std_delta = _sample_std(deltas)
        mean_delta = _mean(deltas)
        t_stat = _paired_t_stat(mean_delta, std_delta, len(deltas))
        comparisons[name] = {
            "name": name,
            "treatment_arm": treatment_arm,
            "control_arm": control_arm,
            "delta_bpb_by_seed": deltas_by_seed,
            "mean_delta_bpb": mean_delta,
            "std_delta_bpb": std_delta,
            "n": len(deltas),
            "t_stat": t_stat,
            "p_value": _paired_p_value(deltas, t_stat),
            "delta_convention": (
                "delta_bpb = treatment_bpb - control_bpb; negative means "
                "treatment improves because lower BPB is better"
            ),
        }
    return comparisons


def _record_arm(record: dict[str, Any]) -> str:
    for path in (
        ("arm",),
        ("config", "arm"),
        ("config", "exp24", "arm"),
        ("config", "metadata", "arm"),
        ("metadata", "arm"),
    ):
        value = _nested_get(record, path)
        if value is not None:
            return str(value)
    name = record.get("name")
    if isinstance(name, str):
        for arm in ARM_NAMES:
            if arm in name:
                return arm
    raise ValueError(f"record has no arm field: {record!r}")


def _record_seed(record: dict[str, Any]) -> int:
    for path in (
        ("seed",),
        ("config", "seed"),
        ("config", "exp24", "seed"),
        ("metadata", "seed"),
    ):
        value = _nested_get(record, path)
        if value is not None:
            return int(value)
    raise ValueError(f"record has no seed field: {record!r}")


def _record_bpb(record: dict[str, Any]) -> float:
    for path in (
        ("val_bpb",),
        ("bpb",),
        ("mean_bpb",),
        ("metrics", "val_bpb"),
        ("metrics", "bpb"),
        ("summary", "val_bpb"),
    ):
        value = _nested_get(record, path)
        if value is not None:
            return float(value)
    raise ValueError(f"record has no BPB field: {record!r}")


def _nested_get(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = record
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _fraction_beating_control(
    seed_values: dict[int, float], control_by_seed: dict[int, float]
) -> float:
    if not seed_values:
        return float("nan")
    missing_control = sorted(set(seed_values) - set(control_by_seed))
    if missing_control:
        raise ValueError(
            f"missing arm_a_control seeds for summary comparison: {missing_control}"
        )
    wins = sum(
        1 for seed, bpb in seed_values.items() if bpb < control_by_seed[seed]
    )
    return wins / len(seed_values)


def _paired_t_stat(mean_delta: float, std_delta: float, n: int) -> float | None:
    if n < 2 or math.isnan(std_delta):
        return None
    if std_delta == 0.0:
        if mean_delta == 0.0:
            return 0.0
        return math.copysign(float("inf"), mean_delta)
    return mean_delta / (std_delta / math.sqrt(n))


def _paired_p_value(deltas: list[float], t_stat: float | None) -> float | None:
    if t_stat is None or len(deltas) < 2:
        return None
    try:
        from scipy import stats  # type: ignore
    except ImportError:
        return None

    result = stats.ttest_1samp(deltas, popmean=0.0)
    return float(result.pvalue)


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        records = []
        for json_path in sorted(path.glob("*.json")):
            with json_path.open() as f:
                records.append(json.load(f))
        return records
    with path.open() as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    return [payload]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", help="JSON file or directory of JSON cells")
    args = parser.parse_args(argv)
    result = analyze_controller_arms(_load_json_records(Path(args.results_path)))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
