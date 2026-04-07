#!/usr/bin/env python3
"""Select best configs from experiments 01-06, generate configs for 07-08."""
import json
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"


def load_results(experiment_name: str) -> dict[str, dict]:
    results_dir = EXPERIMENTS / experiment_name / "results"
    results = {}
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            results[f.stem] = json.load(fh)
    return results


def best_by_bpb(results: dict[str, dict]) -> tuple[str, dict]:
    return min(results.items(), key=lambda x: x[1].get("eval", {}).get("bpb", 999))


def main():
    # Collect best from each component experiment
    component_winners = {}
    for exp in ["02_critical_dynamics", "03_state_dependent_routing",
                "04_long_term_memory", "05_typed_composition", "06_metabolic_gate"]:
        results = load_results(exp)
        if results:
            name, data = best_by_bpb(results)
            component_winners[exp] = {"name": name, "data": data}
            print(f"  {exp}: best={name} bpb={data['eval']['bpb']:.4f}")

    if not component_winners:
        print("No results found from experiments 02-06. Run them first.")
        return

    # Find overall best single component
    all_winners = [(exp, w["name"], w["data"]) for exp, w in component_winners.items()]
    best_single_exp, best_single_name, best_single_data = min(
        all_winners, key=lambda x: x[2].get("eval", {}).get("bpb", 999)
    )
    print(f"\n  Best single: {best_single_exp}/{best_single_name}")

    # Generate best_single.yaml for experiment 07
    best_single_cfg = best_single_data.get("config", {})
    best_single_cfg.pop("enwik8_path", None)
    out_dir = EXPERIMENTS / "07_full_system" / "configs"
    with open(out_dir / "best_single.yaml", "w") as f:
        yaml.dump(best_single_cfg, f, default_flow_style=False)
    print(f"  Wrote {out_dir / 'best_single.yaml'}")

    # Generate best_pair.yaml — combine the two best non-overlapping components
    if len(all_winners) >= 2:
        sorted_winners = sorted(all_winners, key=lambda x: x[2].get("eval", {}).get("bpb", 999))
        pair_cfg = dict(sorted_winners[0][2].get("config", {}))
        second_cfg = sorted_winners[1][2].get("config", {})
        # Merge non-default fields from second into first
        defaults = {"a_mode": "diag", "rich_b_mode": "none", "outer_model_dim": 0,
                    "wernicke_enabled": False, "metabolic_gate": False}
        for key, default_val in defaults.items():
            if second_cfg.get(key) != default_val:
                pair_cfg[key] = second_cfg[key]
        pair_cfg.pop("enwik8_path", None)
        with open(out_dir / "best_pair.yaml", "w") as f:
            yaml.dump(pair_cfg, f, default_flow_style=False)
        print(f"  Wrote {out_dir / 'best_pair.yaml'}")

    print("\nDone. Experiment 07 configs updated.")


if __name__ == "__main__":
    main()
