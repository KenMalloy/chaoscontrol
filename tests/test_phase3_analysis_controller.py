import importlib.util
import math
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
ANALYZE_PATH = (
    REPO / "experiments" / "24_training_time_bundle" / "analyze_phase3.py"
)


def _load_analyze_phase3():
    spec = importlib.util.spec_from_file_location(
        "exp24_analyze_phase3_for_tests", ANALYZE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _records():
    by_seed = {
        101: {
            "arm_a_control": 1.000,
            "arm_b_heuristic": 0.970,
            "arm_c_simplex_frozen": 0.965,
            "arm_d_simplex_online": 0.940,
            "arm_e_simplex_warm_online": 0.930,
        },
        202: {
            "arm_a_control": 1.100,
            "arm_b_heuristic": 1.040,
            "arm_c_simplex_frozen": 1.025,
            "arm_d_simplex_online": 1.000,
            "arm_e_simplex_warm_online": 0.980,
        },
        303: {
            "arm_a_control": 1.200,
            "arm_b_heuristic": 1.150,
            "arm_c_simplex_frozen": 1.130,
            "arm_d_simplex_online": 1.100,
            "arm_e_simplex_warm_online": 1.090,
        },
    }
    records = []
    for seed, arm_values in by_seed.items():
        for arm, bpb in arm_values.items():
            if arm == "arm_e_simplex_warm_online":
                # Spot-check the nested-config record path: analyzer must
                # accept arm under config["arm"], not just top-level.
                records.append(
                    {
                        "seed": seed,
                        "val_bpb": bpb,
                        "config": {"arm": arm},
                    }
                )
            else:
                records.append({"seed": seed, "arm": arm, "val_bpb": bpb})
    return records


def test_controller_phase3_analysis_summarizes_arms_and_paired_comparisons():
    mod = _load_analyze_phase3()

    result = mod.analyze_controller_arms(_records())

    comparisons = result["pairwise_comparisons"]
    assert set(comparisons) == {
        "simplex_vs_heuristic",
        "warm_vs_cold",
        "online_vs_frozen",
    }

    simplex_vs_heuristic = comparisons["simplex_vs_heuristic"]
    assert simplex_vs_heuristic["treatment_arm"] == "arm_d_simplex_online"
    assert simplex_vs_heuristic["control_arm"] == "arm_b_heuristic"
    assert simplex_vs_heuristic["delta_bpb_by_seed"] == {
        101: pytest.approx(-0.030),
        202: pytest.approx(-0.040),
        303: pytest.approx(-0.050),
    }
    assert simplex_vs_heuristic["mean_delta_bpb"] == pytest.approx(-0.040)
    assert simplex_vs_heuristic["std_delta_bpb"] == pytest.approx(0.010)
    assert simplex_vs_heuristic["n"] == 3
    assert simplex_vs_heuristic["t_stat"] == pytest.approx(
        -0.040 / (0.010 / math.sqrt(3))
    )
    assert "p_value" in simplex_vs_heuristic

    assert comparisons["warm_vs_cold"]["delta_bpb_by_seed"] == {
        101: pytest.approx(-0.010),
        202: pytest.approx(-0.020),
        303: pytest.approx(-0.010),
    }
    assert comparisons["warm_vs_cold"]["mean_delta_bpb"] == pytest.approx(
        -0.040 / 3.0
    )

    assert comparisons["online_vs_frozen"]["delta_bpb_by_seed"] == {
        101: pytest.approx(-0.025),
        202: pytest.approx(-0.025),
        303: pytest.approx(-0.030),
    }
    assert comparisons["online_vs_frozen"]["mean_delta_bpb"] == pytest.approx(
        -0.080 / 3.0
    )

    summary_by_arm = {row["arm"]: row for row in result["summary_table"]}
    assert summary_by_arm["arm_a_control"] == {
        "arm": "arm_a_control",
        "n": 3,
        "mean_bpb": pytest.approx(1.100),
        "std_bpb": pytest.approx(0.100),
        "fraction_seeds_beating_arm_a_control": pytest.approx(0.0),
    }
    assert summary_by_arm["arm_d_simplex_online"]["mean_bpb"] == pytest.approx(
        (0.940 + 1.000 + 1.100) / 3.0
    )
    assert summary_by_arm["arm_d_simplex_online"][
        "std_bpb"
    ] == pytest.approx(0.08082903768654767)
    assert summary_by_arm["arm_d_simplex_online"][
        "fraction_seeds_beating_arm_a_control"
    ] == pytest.approx(1.0)


def test_controller_phase3_analysis_rejects_missing_seed_pairing():
    mod = _load_analyze_phase3()
    records = [
        record
        for record in _records()
        if not (
            record["seed"] == 303
            and record.get("arm", record.get("config", {}).get("arm"))
            == "arm_d_simplex_online"
        )
    ]

    with pytest.raises(ValueError, match="missing paired seeds.*simplex_vs_heuristic"):
        mod.analyze_controller_arms(records)
