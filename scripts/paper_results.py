#!/usr/bin/env python3
"""CLI for the paper-results registry.

See ``paper_results/README.md`` for the schema and the rationale.

Usage:
    python scripts/paper_results.py register --experiment exp21 \\
        --condition C_ssm_random --seed 1337 --status confirmatory \\
        --config-hash sha256:abc123 \\
        --metrics '{"bpb": 1.492, "wall_clock_s": 602.5}'

    python scripts/paper_results.py verify
    python scripts/paper_results.py query --experiment exp21 \\
        --status confirmatory
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from chaoscontrol.paper_results import load, query, register, verify


def cmd_register(args: argparse.Namespace) -> None:
    metrics = json.loads(args.metrics) if args.metrics else {}
    extras = json.loads(args.extras) if args.extras else {}
    rec = register(
        experiment=args.experiment,
        condition=args.condition,
        seed=args.seed,
        status=args.status,
        metrics=metrics,
        config_hash=args.config_hash,
        artifacts=args.artifacts or [],
        extras=extras,
        registry_path=args.registry,
    )
    print(
        f"registered: {rec.experiment}/{rec.condition} seed={rec.seed} "
        f"status={rec.status} git_dirty={rec.git_dirty}"
    )


def cmd_query(args: argparse.Namespace) -> None:
    records = query(
        experiment=args.experiment,
        condition=args.condition,
        status=args.status,
        registry_path=args.registry,
    )
    for r in records:
        print(json.dumps(asdict(r), sort_keys=True))


def cmd_verify(args: argparse.Namespace) -> None:
    summary = verify(registry_path=args.registry)
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_list(args: argparse.Namespace) -> None:
    records = load(args.registry)
    print(f"{len(records)} records")
    for r in records:
        marker = "!" if r.git_dirty else " "
        print(
            f"  {marker} {r.experiment:8s} {r.condition:28s} "
            f"seed={r.seed:<6d} {r.status:12s} "
            f"bpb={r.metrics.get('bpb', float('nan')):.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ChaosControl paper-results registry")
    p.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Override registry path (default: paper_results/registry.jsonl)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="Append a run record")
    reg.add_argument("--experiment", required=True)
    reg.add_argument("--condition", required=True)
    reg.add_argument("--seed", type=int, required=True)
    reg.add_argument(
        "--status", choices=["exploratory", "confirmatory"], required=True
    )
    reg.add_argument("--metrics", help="JSON dict of metrics (e.g. '{\"bpb\": 1.5}')")
    reg.add_argument("--config-hash", required=True)
    reg.add_argument("--artifacts", nargs="*")
    reg.add_argument("--extras", help="JSON dict of extras")
    reg.set_defaults(handler=cmd_register)

    q = sub.add_parser("query", help="List matching records as JSON")
    q.add_argument("--experiment")
    q.add_argument("--condition")
    q.add_argument("--status", choices=["exploratory", "confirmatory"])
    q.set_defaults(handler=cmd_query)

    ls = sub.add_parser("list", help="Human-readable listing")
    ls.set_defaults(handler=cmd_list)

    ver = sub.add_parser("verify", help="Schema + duplicate-key check")
    ver.set_defaults(handler=cmd_verify)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
