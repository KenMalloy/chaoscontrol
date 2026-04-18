"""Exp 20 driver. Composes DocStreamer + LegalityController + TTTRunner
+ DeltaModulator + MetricsCollector. Reads a JSON config.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

# The project isn't pip-installed in the dev venv; `tests/conftest.py` adds
# `<repo>/src` to sys.path for pytest, but this script also runs standalone
# (subprocess integration tests, pod invocations). Match conftest's bootstrap.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn.functional as F

from chaoscontrol.eval_stream.types import RunConfig
from chaoscontrol.eval_stream.doc_stream import DocStreamer
from chaoscontrol.eval_stream.legality import LegalityController
from chaoscontrol.eval_stream.persistence import StateManager, attach_trainable_h0
from chaoscontrol.eval_stream.ttt_runner import select_adapt_params, ADAPT_SET_PATTERNS
from chaoscontrol.eval_stream.delta_mod import DeltaModulator
from chaoscontrol.eval_stream.metrics import MetricsCollector
from chaoscontrol.evaluation import compute_bpb


def _ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1), reduction="sum")


def _build_model(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    from chaoscontrol.model import ChaosStudentLM
    # weights_only=False because the checkpoint payload is a dict with
    # {model, config, ...}, not a pure tensor. We trust our own checkpoints.
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = ChaosStudentLM(**cfg)
    # strict=True: any unexpected / missing key is a checkpoint mismatch we
    # want to surface, not paper over. attach_trainable_h0 runs AFTER this
    # (see run() below) so eval-only params never need a loose load.
    model.load_state_dict(blob["model"], strict=True)
    return model, cfg


def _build_optimizer(params, lr: float):
    """Single Muon optimizer. Muon handles 1D/0D params internally via its
    decoupled AdamW path (see src/chaoscontrol/optim/muon.py:145), so a
    prior "Muon for 2D + AdamW for scalars" split produced: (1) double-
    backward per chunk, (2) stale-grad bleed between optimizer steps, (3)
    catastrophic AdamW LR when Muon's LR (e.g. 0.064) was passed through
    unchanged. Collapsed to a single optimizer.
    """
    from chaoscontrol.optim.muon import Muon
    if not params:
        return []
    return [Muon(params, lr=lr)]


def _iter_chunks(tokens: list[int], chunk_size: int):
    if chunk_size < 0:  # whole_doc
        yield tokens
        return
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i + chunk_size]


def run(cfg: RunConfig, jsonl_paths: list[str], sp_model_path: str) -> None:
    # Entry assertion: Axis 1 adapting log_a is incompatible with Axis 3
    # log_a_shift (DeltaModulator reverts log_a on exit, wiping the
    # adaptation). Enforced in DeltaModulator.__enter__ as well; caught here
    # for an earlier, clearer error.
    patterns = ADAPT_SET_PATTERNS.get(cfg.adapt_set, [])
    adapts_log_a = any("log_a" in p for p in patterns) or patterns == ["*"]
    if adapts_log_a and cfg.log_a_shift != 0.0:
        raise ValueError(
            f"adapt_set={cfg.adapt_set!r} adapts log_a but log_a_shift="
            f"{cfg.log_a_shift} is nonzero; Axis 1 × Axis 3 overlap on log_a."
        )

    # Seed-parallel launches may share a node; pin this process to its local
    # GPU before any CUDA work so rank 0 doesn't collide with rank N.
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _build_model(Path(cfg.checkpoint_path))
    model.to(device)

    # attach_trainable_h0 AFTER load_state_dict so strict=True works on the
    # checkpoint (which has no h0 keys) and the newly-created h0 params live
    # on the model's device+dtype from birth (see persistence.attach_trainable_h0).
    if "trainable_h0" in cfg.persistence_mode:
        attach_trainable_h0(model)

    adapt_params = select_adapt_params(model, adapt_set=cfg.adapt_set)
    optimizers = _build_optimizer(adapt_params, cfg.eval_lr) if adapt_params else []

    streamer = DocStreamer(
        jsonl_paths=[Path(p) for p in jsonl_paths],
        sp_model_path=Path(sp_model_path),
        max_docs=cfg.max_docs,
    )
    state_mgr = StateManager(model, persistence_mode=cfg.persistence_mode)
    controller = LegalityController(model, loss_fn=_ce)
    collector = MetricsCollector(output_path=Path(cfg.output_path))

    with DeltaModulator(model, delta_scale=cfg.delta_scale,
                        log_a_shift=cfg.log_a_shift,
                        adapt_set_hint=cfg.adapt_set):
        run_start = time.monotonic()
        for doc in streamer:
            if time.monotonic() - run_start > cfg.budget_seconds:
                break
            if collector.collapsed:
                break
            state_mgr.start_doc(doc_id=doc.doc_id, batch_size=1)
            controller.mark_new_epoch()

            doc_ce_nats = 0.0
            doc_tokens = 0
            step_count = 0
            loss_before_sum = 0.0
            loss_after_sum = 0.0
            chunk_count = 0
            t0 = time.monotonic()

            for chunk_list in _iter_chunks(doc.tokens, cfg.chunk_size):
                if len(chunk_list) < 2:
                    continue
                chunk = torch.tensor(chunk_list, dtype=torch.long, device=device).unsqueeze(0)
                # Score (legality-guarded)
                loss_before = controller.score_chunk(chunk)
                loss_before_sum += loss_before
                # nats per chunk (sum): loss_before is mean-CE from controller; convert
                # But our _ce uses reduction="sum" — so loss_before IS summed nats.
                doc_ce_nats += loss_before
                doc_tokens += chunk.size(1) - 1
                chunk_count += 1

                # Adapt — single Muon optimizer; `optimizers` is at most len==1.
                if optimizers and cfg.steps_per_chunk > 0:
                    loss_after = controller.adapt_on_chunk(
                        chunk, optimizer=optimizers[0], steps=cfg.steps_per_chunk,
                    )
                    # Using `or 0.0` would drop legitimate 0.0 losses;
                    # be explicit about the None case.
                    loss_after_sum += 0.0 if loss_after is None else loss_after
                    step_count += cfg.steps_per_chunk

            wall_ms = (time.monotonic() - t0) * 1000.0
            bpb = compute_bpb(doc_ce_nats, doc.raw_bytes) if doc.raw_bytes > 0 else 0.0
            grad_norm = 0.0  # populated by controller in later task; placeholder
            state_norm = sum(float(s.norm()) for s in state_mgr.get_state()) / max(len(state_mgr.get_state()), 1)
            collector.record_doc(
                doc_id=doc.doc_id, bpb=bpb, tokens=doc_tokens,
                loss_before=loss_before_sum / max(chunk_count, 1),
                loss_after=loss_after_sum / max(chunk_count, 1) if loss_after_sum else None,
                step_count=step_count, wall_ms=wall_ms,
                grad_norm=grad_norm, state_norm=state_norm,
            )
    collector.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    raw = json.loads(Path(args.config).read_text())
    jsonl_paths = raw.pop("jsonl_paths")
    sp_model_path = raw.pop("sp_model_path")
    cfg = RunConfig(**{k: v for k, v in raw.items() if k in RunConfig.__dataclass_fields__})
    run(cfg, jsonl_paths=jsonl_paths, sp_model_path=sp_model_path)


if __name__ == "__main__":
    main()
