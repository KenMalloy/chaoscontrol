#!/usr/bin/env python3
"""Profile the post-ssm_scan training step to identify the next bottleneck.

After the ssm_scan CUDA kernel landed (Phase 2), ``fwd+bwd`` of the diag
recurrence dropped from ~410 ms (torch.compile baseline) to ~2.2 ms at
B=1024 / T=512 / D=256 bf16 on 1xH100 — a 187x speedup on the scan alone.
But the scan is one kernel inside a full training step. Once the scan is
cheap, *something else* dominates. This harness measures what.

Contract
--------
* Mirrors ``runner_exp18_ssm.py``'s inner training step exactly — same
  model construction, same ``train_ssm_step`` call, same optimizer
  (Muon with ``bind_param_names``), same ``activation_checkpoint=True``.
* The env var ``CHAOSCONTROL_DIAG_SCAN_BACKEND`` must be set BEFORE
  ``python`` starts (core.py caches the backend on first resolve). The
  harness prints ``get_diag_recurrence_backend()`` up-front so a silent
  fallback to ``chunked`` is impossible to miss.
* Inputs are synthetic ``torch.randint(0, vocab_size, ...)`` — step time
  is data-independent once shapes match, no point loading the SP corpus
  for a pure-compute profile.

Submission regime: dim=256, layers=4, vocab=16384, seq=512, bs=1024
(B*T = 524,288 tokens/step) matches Test 10 ``_base()`` config.

Usage on the pod:

    source /workspace/venv/bin/activate
    cd /workspace/chaoscontrol

    # Run A: the new kernel
    CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan python \\
        experiments/19_phase1/profile_ssm_scan.py --tag ssm_scan

    # Run B: the honest baseline
    CHAOSCONTROL_DIAG_SCAN_BACKEND=chunked python \\
        experiments/19_phase1/profile_ssm_scan.py --tag chunked

Each run writes ``experiments/19_phase1/profile_traces/trace_<tag>.json``
(Chrome trace) and prints ``key_averages()`` tables to stdout.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "17_local_attn_sidecar"))

from chaoscontrol.core import get_diag_recurrence_backend, verify_diag_recurrence  # noqa: E402
from chaoscontrol.optim.muon import Muon  # noqa: E402
from chaoscontrol.train_ssm import _reject_unsupported, train_ssm_step  # noqa: E402

from runner_exp17 import build_model  # noqa: E402


# Matches experiments/18_throughput_levers/run_exp18_test10.py::_base()
# at bf16 / world_size=1. activation_checkpoint=True is load-bearing —
# one candidate bottleneck (forward-runs-twice) only exists when it's on.
SUBMISSION_CONFIG: dict[str, Any] = {
    "vocab_size": 16384,
    "model_dim": 256,
    "num_layers": 4,
    "ff_mult": 2,
    "seq_len": 512,
    "batch_size": 1024,
    "a_mode": "diag",
    "a_full_rank": 8,
    "a_full_gamma": 0.05,
    "activation_checkpoint": True,
    "local_attn_window": 0,
    "local_attn_heads": 1,
    "local_attn_dim": 64,
    "local_attn_topk": 0,
    "local_attn_topk_random": False,
    "dtype": "bf16",
    "device": "cuda",
    "chunk_size": 64,
    "base_lr": 0.064,
    "weight_decay": 0.01,
    "seed": 1337,
    "precision": "bf16",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tag",
        required=True,
        help=(
            "Label for the output trace file. Conventionally matches the "
            "backend: ``ssm_scan`` or ``chunked``."
        ),
    )
    p.add_argument(
        "--trace-dir",
        default=str(Path(__file__).parent / "profile_traces"),
        help="Directory for trace_<tag>.json. Created if missing.",
    )
    p.add_argument(
        "--force-bf16-decay",
        action="store_true",
        help=(
            "Cast decay to update.dtype before the kernel. Workaround for "
            "the ssm_scan kernel rejecting (fp32_decay, bf16_update) — the "
            "exact dtype combo _diag_terms() produces under autocast bf16 "
            "(torch.exp upcasts to fp32). Without this flag the kernel "
            "raises on step 1 in the production path. Gives a best-case "
            "ssm_scan measurement equivalent to a hypothetical kernel "
            "extension that accepts mixed dtypes."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA required; run on a GPU pod")

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(SUBMISSION_CONFIG["seed"])

    # Resolve backend NOW so the printed line reflects the cached impl.
    # If the env var was unset or the extension is missing, this is where
    # the silent fallback to chunked would happen — get_diag_recurrence_backend
    # returns {"backend": ..., "note": ...}, surfaced below.
    verify_diag_recurrence(device)
    backend_info = get_diag_recurrence_backend()
    print(
        f"[profile] tag={args.tag} "
        f"CHAOSCONTROL_DIAG_SCAN_BACKEND="
        f"{os.environ.get('CHAOSCONTROL_DIAG_SCAN_BACKEND', '<unset>')} "
        f"resolved_backend={backend_info['backend']} "
        f"note={backend_info['note']}",
        flush=True,
    )

    # --force-bf16-decay: monkey-patch ``_diag_recurrence`` to cast decay
    # to update.dtype. The ssm_scan kernel rejects (fp32, bf16) combos
    # that autocast bf16 produces naturally inside ``_diag_terms``
    # (``torch.exp(-delta * a_base)`` is on the "autocast to fp32" list,
    # but ``select * candidate`` stays bf16). This cast is cheap
    # (B*T*D bf16 allocation — ~256 MB for submission shape, 1 kernel
    # launch) and reflects what the kernel would do internally if it
    # accepted mixed dtypes. Without this the profile can't measure
    # ssm_scan at all in the real path.
    if args.force_bf16_decay:
        import chaoscontrol.core as _core

        _orig_diag = _core._diag_recurrence

        def _diag_with_dtype_bridge(decay, update):
            if decay.dtype != update.dtype:
                decay = decay.to(update.dtype)
            return _orig_diag(decay, update)

        _core._diag_recurrence = _diag_with_dtype_bridge
        print(
            "[profile] --force-bf16-decay active: casting decay->update.dtype "
            "before every _diag_recurrence call (see docstring).",
            flush=True,
        )

    model = build_model(
        SUBMISSION_CONFIG, device=device, param_dtype=torch.bfloat16
    )
    _reject_unsupported(model)
    model.train()

    # Muon must see named params — matches runner_exp18_ssm.py:143-154.
    optimizer = Muon(
        list(model.parameters()),
        lr=SUBMISSION_CONFIG["base_lr"],
        weight_decay=SUBMISSION_CONFIG["weight_decay"],
        adamw_lr=SUBMISSION_CONFIG["base_lr"],
        adamw_weight_decay=SUBMISSION_CONFIG["weight_decay"],
    )
    optimizer.bind_param_names(list(model.named_parameters()))

    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"[profile] model params={param_count:,} "
        f"dim={SUBMISSION_CONFIG['model_dim']} "
        f"layers={SUBMISSION_CONFIG['num_layers']} "
        f"vocab={SUBMISSION_CONFIG['vocab_size']} "
        f"seq={SUBMISSION_CONFIG['seq_len']} "
        f"bs={SUBMISSION_CONFIG['batch_size']} "
        f"chunk_size={SUBMISSION_CONFIG['chunk_size']} "
        f"activation_checkpoint={SUBMISSION_CONFIG['activation_checkpoint']}",
        flush=True,
    )

    # Synthetic data — step time is data-independent at fixed shape.
    # +1 on seq so inputs[:-1], targets[1:] gives (B, T) each.
    vocab = SUBMISSION_CONFIG["vocab_size"]
    seq = SUBMISSION_CONFIG["seq_len"]
    bs = SUBMISSION_CONFIG["batch_size"]
    tokens = torch.randint(
        0, vocab, (bs, seq + 1), device=device, dtype=torch.long,
    )
    inputs = tokens[:, :-1].contiguous()
    targets = tokens[:, 1:].contiguous()

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"trace_{args.tag}.json"

    def _step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = train_ssm_step(
            model=model,
            inputs=inputs,
            targets=targets,
            chunk_size=SUBMISSION_CONFIG["chunk_size"],
            ddp_active=False,
            world_size=1,
            precision=SUBMISSION_CONFIG["precision"],
            compile_full_path=False,
        )
        optimizer.step()
        # Force sync so the eventful region below gets recorded.
        _ = float(loss.detach())

    # Pre-profile warmup outside the profiler so one-time compile / caching
    # costs don't pollute the recorded windows. The profiler's own warmup
    # window does the same for steady-state measurement.
    print("[profile] pre-warmup 3 steps (outside profiler)", flush=True)
    for _ in range(3):
        _step()
    torch.cuda.synchronize()

    # Profiler schedule:
    #   wait=1  — one iter to flush any dangling state
    #   warmup=3 — discard (CUPTI ramp)
    #   active=5 — recorded
    #   repeat=1 — one cycle
    # Total iters needed: 1+3+5 = 9, call prof.step() after each.
    schedule = torch.profiler.schedule(wait=1, warmup=3, active=5, repeat=1)

    # ``on_trace_ready`` fires after each active window. Export the Chrome
    # trace via callback rather than from within the ``with`` block; the
    # latter only cleanly captures the last window and can miss kernels.
    def _on_ready(prof: torch.profiler.profile) -> None:
        prof.export_chrome_trace(str(trace_path))
        print(f"[profile] exported chrome trace to {trace_path}", flush=True)

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=_on_ready,
        record_shapes=True,
    ) as prof:
        for i in range(9):
            _step()
            prof.step()
    torch.cuda.synchronize()

    print(f"\n[profile] tag={args.tag} — top 20 by cuda_time_total")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20,
        )
    )
    print(f"\n[profile] tag={args.tag} — top 10 by self_cuda_time_total (grouped by input shape)")
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=10,
        )
    )
    print(f"\n[profile] tag={args.tag} — top 10 by self_cuda_time_total (ungrouped)")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=10,
        )
    )

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"\n[profile] tag={args.tag} peak_vram_mb={peak_mb:.1f}", flush=True)
    print(f"[profile] tag={args.tag} DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
