#!/usr/bin/env python3
"""Profile the Exp24 fast-slow + Dreamworld + event_sleep arm.

This harness is intentionally synthetic: it keeps the mechanism wiring and
section boundaries from the Exp23 runner, while avoiding full FineWeb loading so
it can smoke-test locally and run quickly on one H100.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"
OUT_PATH = (
    REPO
    / "experiments"
    / "24_training_time_bundle"
    / "profile_event_sleep_arm_out.json"
)
SECTION_NAMES = (
    "encode_forward",
    "logits_and_loss",
    "backward",
    "spectral_reg",
    "predictive_aux",
    "dreamworld_replay",
    "event_sleep_gate",
    "event_sleep_decision_resolve",
    "optimizer_step",
    "fast_slow_ema",
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "exp23_runner_fast_path_profile", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyProfileLM(torch.nn.Module):
    def __init__(self, *, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.final_norm = torch.nn.LayerNorm(dim)
        self.lm_head = torch.nn.Linear(dim, vocab_size, bias=False)

    def encode(
        self,
        inputs: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        del initial_states
        hidden = self.embed(inputs.long())
        if return_final_states:
            return hidden, [hidden[:, -1].detach()]
        return hidden


class SectionTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.samples: dict[str, list[dict[str, float | None]]] = defaultdict(list)

    def time(self, name: str, fn: Callable[[], Any]) -> Any:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_event = None
            end_event = None

        t0 = time.perf_counter()
        result = fn()

        if self.device.type == "cuda":
            assert start_event is not None and end_event is not None
            end_event.record()
            torch.cuda.synchronize(self.device)
            cuda_ms = float(start_event.elapsed_time(end_event))
        else:
            cuda_ms = None
        wall_ms = (time.perf_counter() - t0) * 1000.0
        self.samples[name].append({"wall_ms": wall_ms, "cuda_ms": cuda_ms})
        return result


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _summarize_sections(
    samples: dict[str, list[dict[str, float | None]]],
) -> tuple[float, list[dict[str, Any]]]:
    wall_totals = {
        name: sum(float(sample["wall_ms"]) for sample in samples.get(name, []))
        for name in SECTION_NAMES
    }
    total_inside_budget_ms = sum(wall_totals.values())
    sections = []
    for name in SECTION_NAMES:
        wall_values = [
            float(sample["wall_ms"]) for sample in samples.get(name, [])
        ]
        cuda_values = [
            float(sample["cuda_ms"])
            for sample in samples.get(name, [])
            if sample["cuda_ms"] is not None
        ]
        sections.append(
            {
                "name": name,
                "count": len(wall_values),
                "mean_wall_ms": (
                    statistics.fmean(wall_values) if wall_values else 0.0
                ),
                "p50_wall_ms": _percentile(wall_values, 0.50),
                "p95_wall_ms": _percentile(wall_values, 0.95),
                "mean_cuda_ms": (
                    statistics.fmean(cuda_values) if cuda_values else None
                ),
                "share_wall": (
                    wall_totals[name] / total_inside_budget_ms
                    if total_inside_budget_ms > 0.0
                    else 0.0
                ),
            }
        )
    sections.sort(key=lambda item: float(item["share_wall"]), reverse=True)
    return total_inside_budget_ms, sections


def _choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    runner = _load_runner_module()
    device = _choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(args.seed)

    model = TinyProfileLM(vocab_size=args.vocab_size, dim=args.dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer.zero_grad(set_to_none=True)
    fast_slow = runner.FastSlowConsolidator.from_config(
        model,
        {
            "fast_slow_enabled": True,
            "fast_slow_interval": 32,
            "fast_slow_alpha": 0.50,
        },
    )
    dream_buffer = runner.DreamReplayBuffer(max_entries=16, max_age_steps=256)
    event_gate = runner.LossTriggeredReplayEMA(decay=0.99, warmup_steps=32)
    timer = SectionTimer(device)

    event_sleep_pending = False
    event_sleep_last_replay_step = -10**9
    event_sleep_trigger_count = 0
    event_sleep_replay_count = 0
    event_sleep_decision_count = 0
    event_sleep_queued_decision = None
    event_sleep_queued_step: int | None = None
    start = time.perf_counter()

    def resolve_event_sleep_decision() -> None:
        nonlocal event_sleep_pending
        nonlocal event_sleep_trigger_count
        nonlocal event_sleep_decision_count
        nonlocal event_sleep_queued_decision
        nonlocal event_sleep_queued_step

        if event_sleep_queued_decision is None:
            return
        decision = event_sleep_queued_decision
        decision_step = (
            0 if event_sleep_queued_step is None else event_sleep_queued_step
        )
        event_sleep_queued_decision = None
        event_sleep_queued_step = None

        event_sleep_decision_count += 1
        interval_ready = decision_step - event_sleep_last_replay_step >= 8
        buffer_ready = len(dream_buffer) >= 2
        if decision.triggered and interval_ready and buffer_ready:
            event_sleep_trigger_count += 1
            event_sleep_pending = True

    for step in range(args.steps):
        if event_sleep_queued_decision is not None:
            timer.time("event_sleep_decision_resolve", resolve_event_sleep_decision)

        if time.perf_counter() - start >= args.seconds:
            break

        tokens = torch.randint(
            0,
            args.vocab_size,
            (args.batch_size, args.seq_len + 1),
            generator=cpu_generator,
            dtype=torch.long,
        )
        tokens = tokens.to(device=device)
        inputs = tokens[:, :-1].to(torch.int32)
        targets = tokens[:, 1:].to(torch.long)

        scheduled_replay = (
            len(dream_buffer) >= 2
            and step % 8 == 0
            and args.dreamworld_weight > 0.0
        )
        event_replay = event_sleep_pending and len(dream_buffer) >= 2
        dream_entry = None
        if scheduled_replay or event_replay:
            dream_entry = dream_buffer.sample(
                generator=cpu_generator,
                current_step=step,
            )
            if event_replay:
                event_sleep_replay_count += 1
                event_sleep_pending = False
                event_sleep_last_replay_step = step

        if step % 8 == 0:
            entry = runner.capture_dream_entry(
                model,
                inputs,
                step=step,
                prefix_tokens=min(8, args.seq_len - 1),
                replay_tokens=min(4, args.seq_len - 8),
            )
            dream_buffer.add(
                step=entry.step,
                states=entry.states,
                replay_tokens=entry.replay_tokens,
            )

        def encode_forward() -> torch.Tensor:
            with runner.autocast_context(args.precision, device_type=device.type):
                return model.encode(inputs)

        hidden = timer.time("encode_forward", encode_forward)
        assert isinstance(hidden, torch.Tensor)

        def logits_and_loss() -> torch.Tensor:
            with runner.autocast_context(args.precision, device_type=device.type):
                logits = model.lm_head(model.final_norm(hidden))
                return F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    targets.reshape(-1),
                )

        loss = timer.time("logits_and_loss", logits_and_loss)
        timer.time("backward", lambda: loss.backward())

        if args.spectral_weight > 0.0:
            spectral_extra = timer.time(
                "spectral_reg",
                lambda: runner.spectral_regularization_loss(
                    model,
                    lambda_dead=args.spectral_weight,
                    lambda_sticky=args.spectral_weight,
                    min_a=0.05,
                    max_a=0.98,
                ),
            )
            if spectral_extra is not None:
                spectral_extra.backward()

        if dream_entry is not None:
            replay_weight = (
                args.event_sleep_weight
                if event_replay and args.event_sleep_weight > 0.0
                else args.dreamworld_weight
            )
            timer.time(
                "dreamworld_replay",
                lambda: runner.dreamworld_replay_backward(
                    model,
                    dream_entry,
                    replay_weight,
                    lm_head_backward_mode="single",
                    replay_batch_size=0,
                    generator=cpu_generator,
                ),
            )

        decision = timer.time(
            "event_sleep_gate",
            lambda: event_gate.update(
                loss.detach(),
                threshold=1.10,
                pressure_threshold=0.05,
                ddp_active=False,
                world_size=1,
                device=device,
            ),
        )
        if decision is not None:
            event_sleep_queued_decision = decision
            event_sleep_queued_step = step

        timer.time(
            "optimizer_step",
            lambda: (optimizer.step(), optimizer.zero_grad(set_to_none=True)),
        )
        timer.time(
            "fast_slow_ema",
            lambda: fast_slow.after_optimizer_step(model, step=step + 1),
        )

    if event_sleep_queued_decision is not None:
        timer.time("event_sleep_decision_resolve", resolve_event_sleep_decision)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - start
    total_inside_budget_ms, sections = _summarize_sections(timer.samples)
    return {
        "arm": "fast_slow_dreamworld_event_sleep",
        "device": str(device),
        "precision": args.precision,
        "steps": len(timer.samples["event_sleep_gate"]),
        "elapsed_s": elapsed_s,
        "total_inside_budget_ms": total_inside_budget_ms,
        "mean_inside_budget_ms_per_step": (
            total_inside_budget_ms / len(timer.samples["event_sleep_gate"])
            if timer.samples["event_sleep_gate"]
            else 0.0
        ),
        "tokens_per_step": int(args.batch_size * args.seq_len),
        "mechanisms": {
            "event_sleep_trigger_count": event_sleep_trigger_count,
            "event_sleep_replay_count": event_sleep_replay_count,
            "event_sleep_decision_count": event_sleep_decision_count,
            "dream_buffer_size": len(dream_buffer),
            "fast_slow_sync_count": int(fast_slow.sync_count),
        },
        "sections": sections,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dreamworld-weight", type=float, default=0.25)
    parser.add_argument("--event-sleep-weight", type=float, default=0.5)
    parser.add_argument("--spectral-weight", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
