"""Exp 20 driver. Composes DocStreamer + LegalityController + TTTRunner
+ DeltaModulator + MetricsCollector. Reads a JSON config.
"""
from __future__ import annotations
import argparse
import hashlib
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
from chaoscontrol.eval_stream.budget import BudgetTracker, EvalDeadline
from chaoscontrol.evaluation import compute_bpb


def _ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1), reduction="sum")


def _build_model(ckpt_path: Path) -> tuple[torch.nn.Module, dict, dict]:
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
    return model, cfg, blob


def _load_episodic_cache_from_ckpt(blob: dict):
    """Construct an EpisodicCache from a checkpoint payload, or return None
    when the checkpoint has no ``episodic_cache`` key.

    Delegates to ``EpisodicCache.from_dict`` for the actual reconstruction
    so the trainer's save and the eval's load speak the same schema.
    Returns None on absence so the caller can fall back to a fresh empty
    cache (the train-no-cache + eval-cache "Arm D" path); a present-but-
    malformed payload propagates ``KeyError`` from ``from_dict`` rather
    than silently filling defaults — silent defaults are how the falsifier
    matrix's Arm B vs Arm D contrast collapses to noise.
    """
    from chaoscontrol.optim.episodic_cache import EpisodicCache
    payload = blob.get("episodic_cache")
    if payload is None:
        return None
    return EpisodicCache.from_dict(payload)


def _make_fresh_episodic_cache(cfg: RunConfig, model_dim: int):
    """Build the Arm D / train-no-cache-fallback empty cache.

    Reads shape from ``cfg`` so all four falsifier arms (especially Arm B
    vs Arm D) share an identical cache shape — diverging here would let a
    cache-shape difference look like a cache-content difference and
    silently corrupt the matrix's contrast.

    ``cfg.episodic_key_rep_dim == -1`` is a sentinel meaning "use the
    model's hidden dim" — matches the trainer's
    ``_construct_episodic_cache`` default. Negative key_rep_dim would fail
    EpisodicCache's positive-int check, which is what surfaces the
    misconfigure.
    """
    from chaoscontrol.optim.episodic_cache import EpisodicCache
    key_rep_dim = (
        int(model_dim) if cfg.episodic_key_rep_dim == -1
        else int(cfg.episodic_key_rep_dim)
    )
    return EpisodicCache(
        capacity=int(cfg.episodic_cache_capacity),
        span_length=int(cfg.episodic_span_length),
        key_rep_dim=key_rep_dim,
        grace_steps=int(cfg.episodic_grace_steps),
        fingerprint_window=int(cfg.episodic_fingerprint_window),
    )


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file's raw bytes. Used for ckpt provenance so
    a summary can pin the exact bytes that produced it. Chunked so memory
    stays bounded on multi-GB checkpoints."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk_size), b""):
            h.update(buf)
    return h.hexdigest()


def _hash_cfg(cfg: dict) -> str:
    """Deterministic sha256 of a ckpt config dict. Sorted-json so key-order
    doesn't perturb the hash."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


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
    return [Muon(params, lr=lr, fused=True)]


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
    model, ckpt_cfg, ckpt_blob = _build_model(Path(cfg.checkpoint_path))
    model.to(device)

    # attach_trainable_h0 AFTER load_state_dict so strict=True works on the
    # checkpoint (which has no h0 keys) and the newly-created h0 params live
    # on the model's device+dtype from birth (see persistence.attach_trainable_h0).
    if "trainable_h0" in cfg.persistence_mode:
        attach_trainable_h0(model)

    adapt_params = select_adapt_params(model, adapt_set=cfg.adapt_set)
    optimizers = _build_optimizer(adapt_params, cfg.eval_lr) if adapt_params else []

    # Episodic cache wiring — opt-in via cfg.episodic_cache_enabled. The
    # cache lives on CPU (per EpisodicCache contract); the controller
    # snapshots / queries it on demand and the replay path moves token
    # spans onto the model's device for forward+backward.
    episodic_cache = None
    if cfg.episodic_cache_enabled:
        # ckpt_cfg["dim"] is the trainer's hidden dim — what the trainer
        # would have used for episodic_key_rep_dim's model_dim default.
        # Reading it off the ckpt config (not a separate kwarg) ties the
        # eval-time fresh-cache shape to the same source-of-truth as
        # the trainer's _construct_episodic_cache.
        model_dim = int(ckpt_cfg["dim"])
        episodic_cache = _load_episodic_cache_from_ckpt(ckpt_blob)
        if episodic_cache is None:
            episodic_cache = _make_fresh_episodic_cache(cfg, model_dim)
            print(
                "[exp20] episodic_cache: fresh empty cache "
                f"(capacity={episodic_cache.capacity}, "
                f"span_length={episodic_cache.span_length}, "
                f"key_rep_dim={episodic_cache.key_rep_dim})",
                flush=True,
            )
        else:
            print(
                "[exp20] episodic_cache: loaded from checkpoint "
                f"(capacity={episodic_cache.capacity}, "
                f"occupied={int(episodic_cache.occupied.sum().item())})",
                flush=True,
            )

    streamer = DocStreamer(
        jsonl_paths=[Path(p) for p in jsonl_paths],
        sp_model_path=Path(sp_model_path),
        max_docs=cfg.max_docs,
    )
    state_mgr = StateManager(model, persistence_mode=cfg.persistence_mode)
    # Pass cache=None when episodic_cache is not enabled so the controller
    # path is bit-identical to the pre-cache driver. ``cache=None`` is
    # also the default kwarg, but spelling it out makes the back-compat
    # contract visible at the call site.
    #
    # Fingerprint window MUST equal the W the trainer used at write time,
    # else the rolling-hash fingerprints don't align and the cache scores
    # zero hits silently. Prefer the value carried on the loaded cache
    # (the trainer's exact W) and fall back to the cfg field for the
    # fresh-cache and no-cache paths.
    if episodic_cache is not None:
        controller_fp_window = int(episodic_cache.fingerprint_window)
    else:
        controller_fp_window = int(cfg.episodic_fingerprint_window)
    controller = LegalityController(
        model, loss_fn=_ce, cache=episodic_cache,
        fingerprint_window=controller_fp_window,
    )
    collector = MetricsCollector(output_path=Path(cfg.output_path))
    budget = BudgetTracker(
        total_budget_seconds=cfg.budget_seconds,
        score_floor_seconds=cfg.score_floor_seconds,
        safety_margin_seconds=cfg.safety_margin_seconds,
    )
    score_only_mode = not optimizers or cfg.steps_per_chunk <= 0
    docs_scored = 0
    chunks_scored = 0
    tokens_scored = 0
    adapt_steps = 0
    timed_out = False

    with DeltaModulator(model, delta_scale=cfg.delta_scale,
                        log_a_shift=cfg.log_a_shift,
                        adapt_set_hint=cfg.adapt_set):
        deadline = EvalDeadline(cfg.budget_seconds)
        for doc in streamer:
            if deadline.is_expired():
                timed_out = True
                break
            if collector.collapsed:
                break
            state_mgr.start_doc(doc_id=doc.doc_id, batch_size=1)
            controller.mark_new_epoch()
            if cfg.episodic_cache_reset_per_doc and episodic_cache is not None:
                episodic_cache.reset()
                print(
                    f"[exp20] episodic_cache: reset for doc {doc.doc_id}",
                    flush=True,
                )

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
                # Score (legality-guarded); thread the StateManager's per-doc
                # carry state in, capture final_state out. An empty state list
                # (pre-start_doc or no SSM cores) passes through as None.
                prev_state = state_mgr.get_state()
                score_t0 = time.monotonic()
                loss_before, final_states = controller.score_chunk(
                    chunk, initial_states=prev_state if prev_state else None,
                )
                budget.add_score_time(time.monotonic() - score_t0)
                if final_states:
                    state_mgr.set_state(final_states)
                loss_before_sum += loss_before
                # nats per chunk (sum): loss_before is mean-CE from controller; convert
                # But our _ce uses reduction="sum" — so loss_before IS summed nats.
                doc_ce_nats += loss_before
                doc_tokens += chunk.size(1) - 1
                chunk_count += 1
                chunks_scored += 1

                # Adapt — single Muon optimizer; `optimizers` is at most len==1.
                if optimizers and cfg.steps_per_chunk > 0 and budget.can_adapt():
                    adapt_t0 = time.monotonic()
                    loss_after = controller.adapt_on_chunk(
                        chunk,
                        optimizer=optimizers[0],
                        steps=cfg.steps_per_chunk,
                        initial_states=prev_state if prev_state else None,
                    )
                    budget.add_adapt_time(time.monotonic() - adapt_t0)
                    # Using `or 0.0` would drop legitimate 0.0 losses;
                    # be explicit about the None case.
                    loss_after_sum += 0.0 if loss_after is None else loss_after
                    step_count += cfg.steps_per_chunk
                    adapt_steps += cfg.steps_per_chunk
                # In-loop wall cap: a single adapt can take much longer than
                # ``budget.can_adapt()``'s pre-check implied (slack was 1s,
                # adapt took 50s). Check the deadline post-adapt and break
                # the chunk loop so overrun is bounded to ONE adapt's worth
                # of overshoot, not unbounded. The outer doc loop's
                # ``deadline.is_expired()`` check completes the break.
                if deadline.is_expired():
                    timed_out = True
                    break

            if timed_out:
                break

            wall_ms = (time.monotonic() - t0) * 1000.0
            bpb = compute_bpb(doc_ce_nats, doc.raw_bytes) if doc.raw_bytes > 0 else 0.0
            grad_norm = 0.0  # populated by controller in later task; placeholder
            state_norm = sum(float(s.norm()) for s in state_mgr.get_state()) / max(len(state_mgr.get_state()), 1)
            tokens_scored += doc_tokens
            docs_scored += 1
            collector.record_doc(
                doc_id=doc.doc_id, bpb=bpb, tokens=doc_tokens,
                loss_before=loss_before_sum / max(chunk_count, 1),
                loss_after=loss_after_sum / max(chunk_count, 1) if loss_after_sum else None,
                step_count=step_count, wall_ms=wall_ms,
                grad_norm=grad_norm, state_norm=state_norm,
            )
    collector.close()
    if cfg.summary_path:
        summary_path = Path(cfg.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        # Provenance pins the summary to the exact measurement artifact.
        # Computed here (not in the hot loop) because it's cheap at eval-end
        # and avoids adding a load-time side-effect to _build_model.
        ckpt_path = Path(cfg.checkpoint_path)
        ckpt_sha256 = _sha256_file(ckpt_path)
        ckpt_cfg_hash = _hash_cfg(ckpt_cfg)
        gpu_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        summary = budget.summary(
            docs_scored=docs_scored,
            chunks_scored=chunks_scored,
            tokens_scored=tokens_scored,
            adapt_steps=adapt_steps,
            timed_out=timed_out,
            collapsed=collector.collapsed,
            score_only_mode=score_only_mode,
            elapsed_seconds=deadline.elapsed(),
            ckpt_sha256=ckpt_sha256,
            ckpt_cfg_hash=ckpt_cfg_hash,
            stream_seed=cfg.seed,
            gpu_name=gpu_name,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            chunk_size=cfg.chunk_size,
            max_docs=cfg.max_docs,
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


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
