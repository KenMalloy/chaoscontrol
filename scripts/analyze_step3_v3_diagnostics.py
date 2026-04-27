"""Step 3 v3 diagnostic analysis — DuckDB queries on the new SGD-step
columns (`grad_logits_l2`, `grad_w_lh_l2`, `grad_w_lh_accum_l2`, `w_lh_l2`)
to confirm or reject the bootstrap-fix hypothesis.

The load-bearing question: does arm_f (initial_temperature=0.2) produce
larger accumulated gradients per SGD batch AND larger cumulative weight
drift than arm_d (default temperature=1.0)?

If yes → REINFORCE-on-uniform pathology confirmed at production scale,
       AND the temperature-override fix is the right intervention.
If no  → something else is going on; need to investigate.

Usage:
    .venv/bin/python scripts/analyze_step3_v3_diagnostics.py
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments/24_training_time_bundle/results"
TRACES = RESULTS / "traces"


def load_results():
    out = defaultdict(list)
    for p in sorted(RESULTS.glob("exp24_phase3_episodic_controller_v1_*.json")):
        name = p.name.replace("exp24_phase3_episodic_controller_v1_", "").replace(".json", "")
        m = re.match(r"(arm_[a-z]+_[a-z_]*?)_s(\d+)$", name)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        d = json.loads(p.read_text())
        out[arm].append({"seed": seed, "name": name, "data": d})
    return out


def per_arm_summary(arms):
    print("=== PER-ARM SUMMARY ===")
    print(f'{"arm":<32}{"n":>3}{"steps":>10}{"steps/s":>10}{"final_loss":>20}')
    rows = []
    for arm in sorted(arms):
        cells = arms[arm]
        train = [c["data"]["train"] for c in cells]
        sps = [t["steps"] / t["elapsed_s"] for t in train]
        losses = [t["final_loss"] for t in train]
        steps = [t["steps"] for t in train]
        n = len(cells)
        steps_str = f"{int(statistics.mean(steps))}"
        sps_str = (f"{statistics.mean(sps):.2f}" if n == 1
                   else f"{statistics.mean(sps):.2f} ± {statistics.stdev(sps):.2f}")
        loss_str = (f"{losses[0]:.4f}" if n == 1
                    else f"{statistics.mean(losses):.4f} ± {statistics.stdev(losses):.4f}")
        print(f'{arm:<32}{n:>3}{steps_str:>10}{sps_str:>10}{loss_str:>20}')
        rows.append({"arm": arm, "mean_loss": statistics.mean(losses),
                     "mean_steps": statistics.mean(steps)})
    return rows


def telemetry_health(arms):
    print("\n=== TELEMETRY HEALTH ===")
    print(f'{"cell":<48}{"cuda":>6}{"pushed":>9}{"drops":>7}{"errs":>5}{"sgd":>5}')
    for arm in sorted(arms):
        for c in sorted(arms[arm], key=lambda x: x["seed"]):
            d = c["data"]
            aw = d.get("train", {}).get("mechanisms", {}).get("episodic_async_writes", {})
            cuda = aw.get("cuda_stream_enabled", False)
            pushed = aw.get("pushed", 0)
            drops = aw.get("publish_drops", 0)
            errs = aw.get("drain_errors", 0)
            sgd = "-"
            if "simplex" in c["name"]:
                trace = TRACES / f'episodic_controller_v1_{c["name"]}.ndjson'
                if trace.exists():
                    max_sgd = 0
                    with trace.open() as f:
                        for line in f:
                            try:
                                r = json.loads(line)
                            except Exception:
                                continue
                            if r.get("row_type") in ("decision", "credit", "skip"):
                                max_sgd = max(max_sgd, int(r.get("sgd_steps", 0) or 0))
                    sgd = str(max_sgd)
            print(f'{c["name"]:<48}{str(cuda)[:5]:>6}{pushed:>9}{drops:>7}{errs:>5}{sgd:>5}')


def diagnostic_l2_distributions():
    print("\n=== DIAGNOSTIC L2 DISTRIBUTIONS (the bootstrap-fix question) ===")
    arm_files = defaultdict(list)
    for f in sorted(TRACES.glob("episodic_controller_v1_*.ndjson")):
        m = re.match(r"episodic_controller_v1_(arm_[a-z]+_[a-z_]*?)_s(\d+)\.ndjson$", f.name)
        if m:
            arm_files[m.group(1)].append(str(f))

    if not arm_files:
        print("  (no trace files found)")
        return

    con = duckdb.connect(":memory:")
    for arm in sorted(arm_files):
        paths_csv = ", ".join(f"'{p}'" for p in arm_files[arm])
        con.execute(
            f"CREATE OR REPLACE VIEW v_{arm} AS "
            f"SELECT * FROM read_json_auto([{paths_csv}], format='nd', "
            f"union_by_name=true)")

    # Per arm, on credit rows: distribution of the four L2 columns
    for arm in sorted(arm_files):
        print(f"\n{arm}:")
        for col in ("grad_logits_l2", "grad_w_lh_l2", "grad_w_lh_accum_l2", "w_lh_l2"):
            try:
                r = con.execute(
                    f"SELECT COUNT(*) FILTER (WHERE {col} IS NOT NULL), "
                    f"  AVG({col}), STDDEV({col}), MIN({col}), MAX({col}) "
                    f"FROM v_{arm} WHERE row_type='credit' AND {col} IS NOT NULL"
                ).fetchone()
            except Exception as exc:
                print(f"  {col:<24} ERROR: {exc}")
                continue
            n, mean, std, mn, mx = r
            if n == 0:
                print(f"  {col:<24} (no non-null rows)")
            else:
                std_v = std if std is not None else 0.0
                print(f"  {col:<24} n={n:>7}  mean={mean:.5f}  std={std_v:.5f}  "
                      f"min={mn:.5f}  max={mx:.5f}")

    # The headline arm_d vs arm_f comparison
    if "arm_d_simplex_online" in arm_files and "arm_f_simplex_sharp_online" in arm_files:
        print("\n=== HEADLINE: arm_d (default T) vs arm_f (T=0.2) ===")
        print("If arm_f's accumulated gradient + weight drift > arm_d's, "
              "the bootstrap-fix hypothesis is CONFIRMED.\n")
        for col in ("grad_w_lh_accum_l2", "w_lh_l2", "grad_logits_l2"):
            d = con.execute(
                f"SELECT AVG({col}) FROM v_arm_d_simplex_online "
                f"WHERE row_type='credit' AND {col} IS NOT NULL").fetchone()
            f = con.execute(
                f"SELECT AVG({col}) FROM v_arm_f_simplex_sharp_online "
                f"WHERE row_type='credit' AND {col} IS NOT NULL").fetchone()
            d_v = d[0] if d[0] is not None else float("nan")
            f_v = f[0] if f[0] is not None else float("nan")
            ratio = (f_v / d_v) if d_v not in (0, None, float("nan")) else float("nan")
            print(f"  {col:<24}  arm_d={d_v:.5f}  arm_f={f_v:.5f}  ratio f/d={ratio:.2f}")


def entropy_drift_per_arm():
    print("\n=== ENTROPY DRIFT BY ARM (current_entropy - entropy on credit rows) ===")
    arm_files = defaultdict(list)
    for f in sorted(TRACES.glob("episodic_controller_v1_*.ndjson")):
        m = re.match(r"episodic_controller_v1_(arm_[a-z]+_[a-z_]*?)_s(\d+)\.ndjson$", f.name)
        if m:
            arm_files[m.group(1)].append(str(f))

    con = duckdb.connect(":memory:")
    for arm in sorted(arm_files):
        paths_csv = ", ".join(f"'{p}'" for p in arm_files[arm])
        con.execute(
            f"CREATE OR REPLACE VIEW v_{arm} AS "
            f"SELECT * FROM read_json_auto([{paths_csv}], format='nd', "
            f"union_by_name=true)")
    print(f'{"arm":<32}{"n":>9}{"avg drift":>14}{"avg behv":>12}{"avg curr":>12}')
    for arm in sorted(arm_files):
        r = con.execute(
            f"SELECT COUNT(*), AVG(current_entropy - entropy), "
            f"  AVG(entropy), AVG(current_entropy) "
            f"FROM v_{arm} WHERE row_type='credit'"
        ).fetchone()
        if r[0] == 0:
            print(f'{arm:<32}{0:>9}  no credit rows')
        else:
            print(f'{arm:<32}{r[0]:>9}{r[1]:>+14.6f}{r[2]:>12.4f}{r[3]:>12.4f}')


def main():
    arms = load_results()
    if not arms:
        print("No result JSONs found in", RESULTS)
        return
    per_arm_summary(arms)
    telemetry_health(arms)
    diagnostic_l2_distributions()
    entropy_drift_per_arm()


if __name__ == "__main__":
    main()
