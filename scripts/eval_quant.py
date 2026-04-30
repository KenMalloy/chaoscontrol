"""Measure bpb penalty of GPTQ int6 + LZMA roundtrip on an SSM checkpoint.

Takes a checkpoint produced by ``runner_exp21.py --output-ckpt`` (format:
``{"model": state_dict, "config": CareStudentLM kwargs}``) and:

  1. Loads the bf16 model via the ``run_exp20_eval.py::_build_model`` pattern.
  2. Eval pass 1 — scores the FineWeb stream in bf16 to get ``bpb_bf16``.
  3. Calibration — AR-self-generated token sequences via
     ``ar_self_generated_calibration`` (Issue #1017 compliant; no val data).
  4. Quantization — ``GPTQQuantizer.calibrate`` + ``quantize_state_dict``.
  5. Pack — ``pack_int6_lzma`` → serialized artifact bytes.
  6. Unpack + dequantize — ``unpack_int6_lzma`` → ``dequantize_state_dict``.
  7. Eval pass 2 — same FineWeb stream (same seed, fresh iterator), this time
     on the dequantized model, to get ``bpb_int6``.
  8. Writes a JSON comparison (``bpb_bf16``, ``bpb_int6``, ``delta_bpb``,
     ``artifact_bytes``, ``artifact_margin_mb``, and pass-level metrics).

The int6-lzma blob is also written to ``<output_json>.int6.ptz.lzma`` for
offline inspection.

Usage (see ``--help`` for the full arg list)::

    python scripts/eval_quant.py \\
        --ckpt path/to/model.pt \\
        --eval-jsonl path/to/fineweb_eval.jsonl \\
        --sp-model-path path/to/sp.model \\
        --output-json path/to/result.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

# The project is not always pip-installed in dev venvs; `tests/conftest.py`
# and `scripts/run_exp20_eval.py` both bootstrap `src/` onto sys.path for
# standalone runs. Match that pattern so this script works the same way in
# the dev tree and on the training pod.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn.functional as F


def _pick_device(device_arg: str) -> torch.device:
    """Resolve ``--device`` — ``"auto"`` prefers CUDA if available."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_checkpoint(
    ckpt_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load ``{"model": state_dict, "config": kwargs}`` into a fresh model.

    Mirrors ``scripts/run_exp20_eval.py::_build_model`` one-for-one — same
    ``weights_only=False`` (the payload is a dict, not a pure tensor),
    same ``strict=True`` (any key mismatch is a checkpoint bug, not a
    silent miss), same return shape. Kept verbatim so this script and
    ``run_exp20_eval.py`` measure the exact same artifact the downstream
    consumer loads.
    """
    from chaoscontrol.model import CareStudentLM

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: dict[str, Any] = blob["config"]
    model = CareStudentLM(**cfg)
    model.load_state_dict(blob["model"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def _iter_chunks(tokens: list[int], chunk_size: int):
    """Yield fixed-size token chunks (or the whole doc if chunk_size < 0).

    Matches the chunk layout in ``run_exp20_eval.py::_iter_chunks`` so our
    bpb measurement doesn't drift from how the downstream eval harness
    splits docs.
    """
    if chunk_size < 0:
        yield tokens
        return
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i : i + chunk_size]


def _eval_bpb(
    model: torch.nn.Module,
    *,
    label: str,
    jsonl_paths: list[Path],
    sp_model_path: Path,
    budget_seconds: float,
    chunk_size: int,
    device: torch.device,
    max_docs: int = 50_000,
) -> dict[str, float]:
    """Score the FineWeb stream, return bpb + bookkeeping.

    Per-doc: SSM state resets at the start of each doc (``initial_states=None``)
    and threads through chunks within the same doc via ``final_states``.
    The inner loop mirrors the canonical scoring pattern in
    ``src/chaoscontrol/eval_stream/legality.py::score_chunk`` (lines 77-79)
    — full chunk into forward, CE on ``logits[:, :-1]`` against
    ``chunk[:, 1:]``, carry ``final_states`` from the full chunk. This
    keeps ``bpb_bf16`` comparable to what a ``run_exp20_eval.py`` run
    would report on the same checkpoint + stream. LegalityController-free
    because this script only measures artifact quality (the TTT +
    leak-detection machinery is overkill for a score-only pipeline).

    ``label`` identifies the pass (``"bf16"`` or ``"int6"``) in the error
    message if the stream yields zero scorable docs/bytes — otherwise a
    mis-pointed ``--eval-jsonl`` or too-small ``--budget-seconds`` silently
    returns ``bpb=0.0`` and the pipeline happily reports a green delta.
    Also raises if the accumulated bpb is non-finite — a dequantized int6
    weight NaN'ing during pass 2 would otherwise land silently in the
    output JSON.

    Returns::

        {"bpb", "docs_scored", "tokens_scored", "wall_seconds",
         "ce_nats", "raw_bytes"}
    """
    from chaoscontrol.eval_stream.doc_stream import DocStreamer
    from chaoscontrol.evaluation import compute_bpb

    streamer = DocStreamer(
        jsonl_paths=jsonl_paths,
        sp_model_path=sp_model_path,
        max_docs=max_docs,
    )

    total_ce_nats = 0.0
    total_raw_bytes = 0
    total_tokens = 0
    docs_scored = 0
    t_start = time.monotonic()

    with torch.no_grad():
        for doc in streamer:
            if time.monotonic() - t_start > budget_seconds:
                break
            # Reset recurrence at doc boundary. initial_states=None gets the
            # zero-init branch in CareStudentLM.forward (see model.py line
            # 1128 — when None is passed, the layers are called with
            # initial_state=None, which maps to zero-init in CareSSMCore).
            prev_final_states: list[torch.Tensor] | None = None
            for chunk_list in _iter_chunks(doc.tokens, chunk_size):
                # Need at least 2 tokens to form one (input, target) pair —
                # matches the guard in legality.py::score_chunk (line 59-63).
                if len(chunk_list) < 2:
                    continue
                chunk = torch.tensor(
                    chunk_list, dtype=torch.long, device=device
                ).unsqueeze(0)
                # Mirror canonical scoring in
                # src/chaoscontrol/eval_stream/legality.py::score_chunk
                # (lines 77-79): full chunk into forward so SSM state
                # advances T tokens, then slice logits to drop the
                # final-position logit that has no next-token target.
                out = model(chunk, initial_states=prev_final_states)
                logits = out["logits"][:, :-1]
                targets = chunk[:, 1:]
                # reduction="sum" so we accumulate total nats across the
                # corpus; compute_bpb divides by raw byte count at the end.
                # No .float() upcast — matches legality.py:79 exactly
                # so bpb_bf16 here tracks what the downstream eval harness sees.
                ce_sum = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="sum",
                )
                total_ce_nats += float(ce_sum.item())
                total_tokens += int(targets.numel())
                # Thread SSM state across chunks within the doc — full-chunk
                # final_states, matching legality.py:80-84.
                prev_final_states = out.get("final_states")
            total_raw_bytes += int(doc.raw_bytes)
            docs_scored += 1

    wall_seconds = time.monotonic() - t_start
    # Fail loud if the stream yielded nothing scorable — a mis-typed
    # ``--eval-jsonl`` or a too-small ``--budget-seconds`` otherwise sneaks
    # through as ``bpb=0.0`` and the calibration + roundtrip still runs,
    # producing a green-looking result JSON with no error signal.
    if docs_scored == 0 or total_raw_bytes == 0:
        raise RuntimeError(
            f"_eval_bpb pass '{label}' scored zero docs/bytes "
            f"(docs_scored={docs_scored}, raw_bytes={total_raw_bytes}) "
            f"over jsonl_paths={[str(p) for p in jsonl_paths]}; "
            "check --eval-jsonl and --budget-seconds."
        )
    bpb = float(compute_bpb(total_ce_nats, total_raw_bytes))
    # Fail loud on non-finite bpb. Pass 2 can hit this if dequantized int6
    # weights produce NaN logits (bad Hessian → Cholesky underflow → NaN
    # inverse → NaN out); pass 1 can hit it if the bf16 checkpoint itself
    # is corrupted. Either way we want a stack trace, not a silent
    # bpb=NaN written to JSON that downstream tooling happily reads.
    # Mirrors the ``runner_exp21.py`` finiteness guard on training-side
    # bpb (runner_exp21.py:444-452).
    if not math.isfinite(bpb):
        raise RuntimeError(
            f"_eval_bpb pass '{label}' produced non-finite bpb={bpb} "
            f"(ce_nats={total_ce_nats}, raw_bytes={total_raw_bytes}). "
            "For pass 'int6' this usually means a dequantized weight "
            "NaN'd during inference; for pass 'bf16' it usually means "
            "the loaded checkpoint is corrupt."
        )
    return {
        "bpb": bpb,
        "docs_scored": int(docs_scored),
        "tokens_scored": int(total_tokens),
        "wall_seconds": float(wall_seconds),
        "ce_nats": float(total_ce_nats),
        "raw_bytes": int(total_raw_bytes),
    }


def _make_logit_fn(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Adapt ``CareStudentLM`` to the ``logit_fn`` contract.

    ``ar_self_generated_calibration`` calls ``logit_fn(tokens) -> (B, T, V)``
    logits; ``CareStudentLM.forward`` returns a ``dict`` whose ``"logits"``
    entry has exactly that shape. The ``initial_states=None`` is implicit
    in the bare call signature — AR sampling doesn't persist state across
    calibration sequences, which matches what ``collect_hessians`` expects
    when it forwards each sequence independently.
    """

    def logit_fn(tokens: torch.Tensor) -> torch.Tensor:
        return model(tokens)["logits"]

    return logit_fn


def _calibrate_and_quantize(
    model: torch.nn.Module,
    *,
    device: torch.device,
    num_seqs: int,
    seq_len: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    """Run AR calibration, build Hessians, quantize the state dict.

    The two-step ``.calibrate()`` + ``.quantize_state_dict()`` flow matches
    the API contract verified in ``tests/test_quantization_gptq.py`` (line
    170-180). Issue #1017 compliance comes from using
    ``ar_self_generated_calibration`` — the model rolls its own tokens
    from a random seed, so no val data enters the Hessian.

    The returned tuple is fed directly into ``pack_int6_lzma``.
    """
    from chaoscontrol.quantization import (
        GPTQQuantizer,
        ar_self_generated_calibration,
    )

    logit_fn = _make_logit_fn(model)
    # vocab_size must come from the model, not the default (1024). The real
    # SSM uses SP8192 / SP16384 — a mismatched vocab would silently roll
    # tokens the model can't embed, producing a garbage Hessian.
    calib_tokens = ar_self_generated_calibration(
        logit_fn,
        num_seqs=num_seqs,
        seq_len=seq_len,
        vocab_size=model.vocab_size,
        device=device,
        seed=seed,
    )

    quantizer = GPTQQuantizer()
    # collect_hessians will run each calibration sequence through the model
    # with forward hooks on every nn.Linear. Default forward_fn=None lets it
    # call model(seq) directly — CareStudentLM's forward accepts
    # (batch, seq) int64 ids and returns a dict; the hooks only care about
    # the Linear inputs, not the model's return value.
    quantizer.calibrate(model, calib_tokens, device=device)
    # min_numel=65536 (default) filters out small 1D tensors (biases, norms)
    # so only large 2D weights go through the int6 path. Keep the default —
    # it's what the SOTA transformer record used.
    result, meta = quantizer.quantize_state_dict(model.state_dict())
    return result, meta


def _roundtrip_to_model(
    blob: bytes,
    cfg: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Rebuild a fresh ``CareStudentLM`` from an int6-lzma blob.

    ``unpack_int6_lzma`` decompresses to ``(quantized, meta)``, then
    ``dequantize_state_dict`` reconstructs a float state dict. The
    meta map iterates every original key (passthrough + int6 alike) so
    the reconstructed dict is 1:1 with the original — ``strict=True`` on
    load is the load-bearing assertion that packaging+dequantization
    didn't drop or add any keys.
    """
    from chaoscontrol.model import CareStudentLM
    from chaoscontrol.quantization import (
        dequantize_state_dict,
        unpack_int6_lzma,
    )

    quantized, meta = unpack_int6_lzma(blob)
    deq_state = dequantize_state_dict(quantized, meta)
    model = CareStudentLM(**cfg)
    model.load_state_dict(deq_state, strict=True)
    model.to(device)
    model.eval()
    return model


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure bpb penalty of GPTQ int6 + LZMA roundtrip on an SSM "
            "checkpoint produced by runner_exp21.py --output-ckpt."
        ),
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="Checkpoint path written by runner_exp21.py --output-ckpt.",
    )
    parser.add_argument(
        "--eval-jsonl",
        required=True,
        type=Path,
        nargs="+",
        help="One or more FineWeb eval JSONL paths.",
    )
    parser.add_argument(
        "--sp-model-path",
        required=True,
        type=Path,
        help="SentencePiece tokenizer .model file.",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=600.0,
        help="Time budget per eval pass (default: 600s).",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Where to write the result JSON.",
    )
    parser.add_argument(
        "--calibration-seqs",
        type=int,
        default=64,
        help="Number of AR-self-generated calibration sequences (default: 64).",
    )
    parser.add_argument(
        "--calibration-seq-len",
        type=int,
        default=2048,
        help="Length of each calibration sequence in tokens (default: 2048).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for both eval stream determinism and AR calibration.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='"auto" (default), "cuda", or "cpu".',
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help=(
            "Chunk size for per-doc CE accumulation (default: 256 — matches "
            "RunConfig.chunk_size). Use -1 for whole-doc."
        ),
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=50_000,
        help="Maximum docs to pull from DocStreamer (default: 50000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Seed both CPU and (maybe) CUDA so the two eval passes see the same
    # sampling decisions anywhere randomness sneaks in. DocStreamer itself
    # is deterministic (no RNG), but being defensive here is cheap.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _pick_device(args.device)

    # --- Step 1: Load bf16 model ---
    bf16_model, cfg = _load_checkpoint(args.ckpt, device)

    # --- Step 2: Eval pass 1 — bf16 ---
    bf16_metrics = _eval_bpb(
        bf16_model,
        label="bf16",
        jsonl_paths=list(args.eval_jsonl),
        sp_model_path=args.sp_model_path,
        budget_seconds=args.budget_seconds,
        chunk_size=args.chunk_size,
        device=device,
        max_docs=args.max_docs,
    )

    # --- Steps 3 + 4: Calibrate + quantize ---
    quantized, meta = _calibrate_and_quantize(
        bf16_model,
        device=device,
        num_seqs=args.calibration_seqs,
        seq_len=args.calibration_seq_len,
        seed=args.seed,
    )

    # --- Step 5: Pack ---
    from chaoscontrol.quantization import pack_int6_lzma

    blob = pack_int6_lzma(quantized, meta)
    artifact_bytes = len(blob)

    # Write the blob next to the output JSON for offline inspection. The
    # side file name keeps the `.int6.ptz.lzma` convention used elsewhere
    # in the repo so downstream tooling can recognize it.
    blob_path = args.output_json.with_suffix(args.output_json.suffix + ".int6.ptz.lzma")
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(blob)

    # Free the bf16 model before constructing the dequantized one — on CUDA
    # this avoids holding both models in VRAM simultaneously.
    del bf16_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Step 6: Unpack + dequantize into a fresh model ---
    int6_model = _roundtrip_to_model(blob, cfg, device)

    # --- Step 7: Eval pass 2 — int6 ---
    # DocStreamer's __iter__ restarts doc_id=0 every call, so building a
    # fresh streamer with the same JSONL paths reproduces the SAME doc
    # order as pass 1. This is the SAME eval stream (same docs, same
    # order) that pass 1 saw — documented explicitly here because it's
    # easy to miss that a new DocStreamer instance is safer than trying
    # to "reset" a mid-iteration one (which the class doesn't support;
    # see doc_stream.py docstring).
    # Cap pass 2 at pass 1's doc count so delta_bpb compares the SAME slice of
    # the eval stream. Pass 2 budget auto-scales to ``max(budget, 2*pass1_wall)``
    # so a slower int6 inference path doesn't trigger the docs-mismatch
    # RuntimeError below and waste a pod run. The user's --budget-seconds
    # is still honored as a floor — auto-scale only raises it, never lowers.
    pass2_budget = max(args.budget_seconds, 2.0 * bf16_metrics["wall_seconds"])
    int6_metrics = _eval_bpb(
        int6_model,
        label="int6",
        jsonl_paths=list(args.eval_jsonl),
        sp_model_path=args.sp_model_path,
        budget_seconds=pass2_budget,
        chunk_size=args.chunk_size,
        device=device,
        max_docs=min(args.max_docs, bf16_metrics["docs_scored"]),
    )
    if int6_metrics["docs_scored"] != bf16_metrics["docs_scored"]:
        raise RuntimeError(
            "eval pass docs_scored mismatch: "
            f"bf16={bf16_metrics['docs_scored']} int6={int6_metrics['docs_scored']}. "
            "Pass 2 truncated before matching pass 1 despite auto-scaled "
            f"budget of {pass2_budget:.1f}s — int6 inference is >2x slower "
            "than bf16. Raise --budget-seconds and retry."
        )
    delta_bpb = int6_metrics["bpb"] - bf16_metrics["bpb"]
    # bpb1 and bpb2 are each finite per _eval_bpb's guard, but float
    # subtraction of two finite numbers is always finite — assert anyway
    # to make any future regression (e.g. a sign-bug introducing inf)
    # a loud stack trace rather than a silent JSON write.
    if not math.isfinite(delta_bpb):
        raise RuntimeError(
            f"delta_bpb is non-finite: {delta_bpb} "
            f"(bpb_bf16={bf16_metrics['bpb']}, bpb_int6={int6_metrics['bpb']})"
        )

    # --- Step 8: Write result JSON ---
    # Parameter Golf's 16 MB budget is decimal (10^6 bytes), not binary
    # (MiB, 2^20). An artifact at 16,000,000 bytes is AT the budget —
    # margin should be 0.0. Using a MiB divisor would falsely report
    # +0.74 positive margin at that boundary.
    artifact_margin_mb = 16.0 - (artifact_bytes / 1e6)

    result = {
        "ckpt_path": str(args.ckpt),
        "eval_jsonl": [str(p) for p in args.eval_jsonl],
        "sp_model_path": str(args.sp_model_path),
        "bpb_bf16": bf16_metrics["bpb"],
        "bpb_int6": int6_metrics["bpb"],
        "delta_bpb": delta_bpb,
        "artifact_bytes": artifact_bytes,
        "artifact_margin_mb": artifact_margin_mb,
        "bf16_docs_scored": bf16_metrics["docs_scored"],
        "bf16_tokens_scored": bf16_metrics["tokens_scored"],
        "bf16_wall_seconds": bf16_metrics["wall_seconds"],
        "int6_docs_scored": int6_metrics["docs_scored"],
        "int6_tokens_scored": int6_metrics["tokens_scored"],
        "int6_wall_seconds": int6_metrics["wall_seconds"],
        "calibration_seqs": args.calibration_seqs,
        "calibration_seq_len": args.calibration_seq_len,
        "seed": args.seed,
    }

    # Atomic write via .tmp + rename so a killed process never leaves a
    # partial JSON on disk that downstream tooling would happily parse.
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    tmp_path.write_text(json.dumps(result, indent=2))
    tmp_path.rename(args.output_json)

    print(
        f"[eval_quant] bpb_bf16={result['bpb_bf16']:.4f} "
        f"bpb_int6={result['bpb_int6']:.4f} "
        f"delta={result['delta_bpb']:+.4f} "
        f"artifact={artifact_bytes/1e6:.3f} MB "
        f"margin={artifact_margin_mb:+.3f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
