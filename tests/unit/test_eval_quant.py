"""End-to-end unit tests for ``scripts/eval_quant.py``.

The script under test runs the bf16 → int6-LZMA quantization pipeline against
a Param-Golf-style FineWeb stream and writes a 17-field result JSON. These
tests exercise the FULL pipeline on a tiny model + tiny stream — load
checkpoint, eval pass 1 in bf16, AR-self-generated calibration, GPTQ int6
quantization, LZMA pack, unpack, dequantize, eval pass 2, JSON write — and
assert the output contract end-to-end. No code under test is stubbed.

Test fixture sizing:

  - vocab_size=64, dim=16, num_layers=1 — proven legal in
    test_runner_exp21_output_ckpt.py.
  - SP model trained inline via SentencePieceTrainer on a ~5KB seeded
    corpus (vocab 64, BPE).
  - 4-doc eval JSONL.
  - calibration: 4 sequences × 32 tokens.
  - chunk_size=32.
  - budget-seconds=30 (safety cap; the pipeline finishes in ~6s on CPU).

`min_numel` monkey-patch — REQUIRED, documented in the script's
``_calibrate_and_quantize``. The default ``min_numel=65536`` filters out
EVERY Linear weight in a dim=16 model (largest is ``lm_head`` at
16×64=1024 elements, three orders of magnitude under the cutoff). To
exercise the int6 GPTQ path on a tiny model that runs in CPU-bound
seconds, we wrap ``GPTQQuantizer.quantize_state_dict`` to default
``min_numel=1``. ``scripts/eval_quant.py`` does not currently expose this
threshold via CLI; the wrap is the smallest hack that lets the test prove
the int6 lane runs without scaling the model into a many-second budget.
The non-monkey-patched default behavior is already covered by the
tests in ``tests/test_quantization_gptq.py`` against a toy MLP.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import sentencepiece as spm
import torch

REPO = Path(__file__).resolve().parents[2]
# scripts/eval_quant.py uses runtime sys.path bootstrap; mirror it here so
# the import-and-call path resolves chaoscontrol the same way the script
# does at __main__ time.
sys.path.insert(0, str(REPO / "src"))


# === Required output JSON fields (exact spec from Task 3) ===
REQUIRED_JSON_FIELDS: set[str] = {
    "ckpt_path",
    "eval_jsonl",
    "sp_model_path",
    "bpb_bf16",
    "bpb_int6",
    "delta_bpb",
    "artifact_bytes",
    "artifact_margin_mb",
    "bf16_docs_scored",
    "bf16_tokens_scored",
    "bf16_wall_seconds",
    "int6_docs_scored",
    "int6_tokens_scored",
    "int6_wall_seconds",
    "calibration_seqs",
    "calibration_seq_len",
    "seed",
}
# Fields whose runtime value must be both finite and numeric (no NaN, no
# inf, no None). String / list fields skip this gate.
NUMERIC_FIELDS: set[str] = {
    "bpb_bf16",
    "bpb_int6",
    "delta_bpb",
    "artifact_bytes",
    "artifact_margin_mb",
    "bf16_docs_scored",
    "bf16_tokens_scored",
    "bf16_wall_seconds",
    "int6_docs_scored",
    "int6_tokens_scored",
    "int6_wall_seconds",
    "calibration_seqs",
    "calibration_seq_len",
    "seed",
}


def _train_tiny_sp_model(out_dir: Path) -> Path:
    """Train a vocab-64 BPE SentencePiece model on a fixed mini-corpus.

    Returns the path to the saved ``.model`` file. Repeats a small set of
    English sentences to give the BPE trainer enough material to hit
    vocab_size=64 without complaining about hard_vocab_limit. Bos/eos/pad
    are disabled so the produced ids are dense in [0, 64) — consistent
    with how chaoscontrol's SP shards are built (append_eos=False).
    """
    corpus = "\n".join(
        [
            "the quick brown fox jumps over the lazy dog",
            "a stitch in time saves nine",
            "to be or not to be that is the question",
            "all that glitters is not gold",
            "the rain in spain stays mainly in the plain",
            "now is the time for all good men to come to the aid",
        ]
        * 20
    )
    corpus_path = out_dir / "corpus.txt"
    corpus_path.write_text(corpus)
    model_prefix = out_dir / "tiny"
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=64,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
    )
    return out_dir / "tiny.model"


def _tiny_chaos_config() -> dict:
    """ChaosStudentLM kwargs that build in <100ms on CPU.

    Mirrors the tiny-config used in ``test_runner_exp21_output_ckpt.py``
    so this checkpoint reads back via the same loader contract that
    ``scripts/eval_quant.py::_load_checkpoint`` uses. wernicke / outer /
    posterior modules are off so the forward path is exactly
    embed → ssm-block → lm_head, the smallest call graph that still
    contains nn.Linear layers for GPTQ to bite into.
    """
    return dict(
        vocab_size=64,
        dim=16,
        num_layers=1,
        ff_mult=2,
        a_mode="diag",
        a_full_rank=8,
        a_full_gamma=0.05,
        outer_model_dim=0,
        wernicke_enabled=False,
        local_attn_window=0,
    )


def _write_tiny_checkpoint(ckpt_path: Path, *, seed: int = 0) -> dict:
    """Build a randomly-initialized ChaosStudentLM and save in the
    Task-1 ``--output-ckpt`` format.

    No training is needed — random weights through the eval + calibration
    + quantize + dequantize + eval pipeline test the contract, not model
    quality.
    """
    from chaoscontrol.model import ChaosStudentLM

    cfg = _tiny_chaos_config()
    torch.manual_seed(seed)
    model = ChaosStudentLM(**cfg)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model": state, "config": cfg}, ckpt_path)
    return cfg


def _write_tiny_eval_jsonl(jsonl_path: Path) -> None:
    """4-doc FineWeb-shape JSONL — schema matches DocStreamer's
    ``obj.get("text", "")`` + raw-bytes accounting in
    ``src/chaoscontrol/eval_stream/doc_stream.py``.

    Each doc is ~60-80 chars so the chunk_size=32 path takes 2-3 chunks
    per doc and exercises the inter-chunk SSM-state-threading branch.
    """
    docs = [
        {"text": "the quick brown fox jumps over the lazy dog and goes home"},
        {"text": "a stitch in time saves nine threads and patches the cloth"},
        {"text": "to be or not to be that is the question for the prince"},
        {
            "text": "all that glitters is not gold even when it shines"
            " like the sun"
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(d) for d in docs))


def _import_eval_quant():
    """Load ``scripts/eval_quant.py`` as a module.

    The script lives outside any package, so import via spec/loader.
    Cached at module level so multiple tests share the same module object.
    """
    if "scripts.eval_quant" in sys.modules:
        return sys.modules["scripts.eval_quant"]
    import importlib.util

    script_path = REPO / "scripts" / "eval_quant.py"
    spec = importlib.util.spec_from_file_location(
        "scripts.eval_quant", script_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts.eval_quant"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def eval_quant_module():
    """Import scripts/eval_quant.py once per test (module is cached by name)."""
    return _import_eval_quant()


@pytest.fixture
def tiny_pipeline_paths(tmp_path):
    """Build all tiny inputs the eval_quant pipeline needs in tmp_path.

    Returns a dict of paths so individual tests can mutate one input
    (e.g. swap eval_jsonl for an empty file in the C1 test) without
    rebuilding the others.
    """
    sp_model_path = _train_tiny_sp_model(tmp_path)
    ckpt_path = tmp_path / "tiny.ckpt"
    _write_tiny_checkpoint(ckpt_path)
    eval_jsonl = tmp_path / "eval.jsonl"
    _write_tiny_eval_jsonl(eval_jsonl)
    output_json = tmp_path / "result.json"
    return {
        "sp_model_path": sp_model_path,
        "ckpt_path": ckpt_path,
        "eval_jsonl": eval_jsonl,
        "output_json": output_json,
    }


def _make_argv(paths: dict, **overrides) -> list[str]:
    """Build a CLI argv list from a tiny_pipeline_paths fixture dict."""
    argv = [
        "--ckpt",
        str(paths["ckpt_path"]),
        "--eval-jsonl",
        str(paths["eval_jsonl"]),
        "--sp-model-path",
        str(paths["sp_model_path"]),
        "--output-json",
        str(paths["output_json"]),
        "--budget-seconds",
        str(overrides.get("budget_seconds", 30)),
        "--calibration-seqs",
        str(overrides.get("calibration_seqs", 4)),
        "--calibration-seq-len",
        str(overrides.get("calibration_seq_len", 32)),
        "--chunk-size",
        str(overrides.get("chunk_size", 32)),
        "--device",
        "cpu",
        "--max-docs",
        str(overrides.get("max_docs", 4)),
        "--seed",
        str(overrides.get("seed", 0)),
    ]
    return argv


def _patched_quantize_state_dict(monkeypatch):
    """Force every nn.Linear through the int6 path on the tiny model.

    See the module-level docstring on why this monkey-patch is necessary
    for any tiny-fixture coverage of the int6 lane.
    """
    from chaoscontrol.quantization import GPTQQuantizer

    orig = GPTQQuantizer.quantize_state_dict

    def wrapped(self, state_dict, **kwargs):
        kwargs.setdefault("min_numel", 1)
        return orig(self, state_dict, **kwargs)

    monkeypatch.setattr(GPTQQuantizer, "quantize_state_dict", wrapped)


# =====================================================================
# Test 1 — happy path round-trip
# =====================================================================
def test_pipeline_produces_complete_result_json(
    tiny_pipeline_paths, eval_quant_module, monkeypatch
):
    """End-to-end: load → bf16 eval → calibrate → GPTQ → pack → unpack
    → dequantize → int6 eval → JSON write succeeds and the JSON has all
    17 required fields, each finite and non-null.

    Also asserts:
      - process exit 0 (main() returns 0)
      - artifact_bytes < 16 MB (Param Golf decimal budget)
      - bf16_docs_scored == int6_docs_scored (paired-pass invariant
        from the C1/I1 fix; tiny model can't fail this absent a bug)
      - side file ``<output_json>.int6.ptz.lzma`` exists and unpacks
        via ``unpack_int6_lzma`` into the (quantized, meta) tuple shape
        the dequantizer expects
    """
    from chaoscontrol.quantization import unpack_int6_lzma

    _patched_quantize_state_dict(monkeypatch)
    argv = _make_argv(tiny_pipeline_paths)
    rc = eval_quant_module.main(argv)
    assert rc == 0, f"main returned non-zero exit code: {rc}"

    output_json = tiny_pipeline_paths["output_json"]
    assert output_json.exists(), "output JSON was not written"
    result = json.loads(output_json.read_text())

    # All 17 fields present
    actual_fields = set(result.keys())
    missing = REQUIRED_JSON_FIELDS - actual_fields
    extra = actual_fields - REQUIRED_JSON_FIELDS
    assert not missing, f"output JSON missing required fields: {missing}"
    assert not extra, (
        f"output JSON has unexpected fields: {extra} — schema drift?"
    )

    # Numeric fields must be finite + non-null
    for k in NUMERIC_FIELDS:
        v = result[k]
        assert v is not None, f"field {k!r} is None"
        assert isinstance(v, (int, float)), (
            f"field {k!r} has non-numeric type: {type(v).__name__}"
        )
        assert math.isfinite(float(v)), f"field {k!r} is non-finite: {v}"

    # String / list fields must be present and non-empty
    assert isinstance(result["ckpt_path"], str) and result["ckpt_path"]
    assert isinstance(result["sp_model_path"], str) and result["sp_model_path"]
    assert (
        isinstance(result["eval_jsonl"], list) and len(result["eval_jsonl"]) >= 1
    )

    # Paired-pass invariant — bf16 and int6 must score the SAME slice of
    # the eval stream. The script raises if they don't, but verify here
    # in case a future refactor weakens the assertion.
    assert result["bf16_docs_scored"] == result["int6_docs_scored"], (
        f"paired-pass docs mismatch: "
        f"bf16={result['bf16_docs_scored']} int6={result['int6_docs_scored']}"
    )
    assert result["bf16_docs_scored"] > 0, (
        "bf16 pass scored zero docs (eval JSONL fixture failed)"
    )

    # Param Golf decimal-MB budget — artifact must fit under 16e6 bytes.
    # On the tiny model this is hilariously slack (~5KB) but we assert
    # the comparison so the contract is locked.
    assert result["artifact_bytes"] < 16_000_000, (
        f"artifact exceeds Param Golf 16 MB budget: "
        f"{result['artifact_bytes']} bytes"
    )

    # Side file is written next to the JSON and is loadable.
    blob_path = output_json.with_suffix(output_json.suffix + ".int6.ptz.lzma")
    assert blob_path.exists(), (
        f"side file {blob_path} was not written"
    )
    assert blob_path.stat().st_size == result["artifact_bytes"], (
        f"side file size {blob_path.stat().st_size} != "
        f"artifact_bytes {result['artifact_bytes']}"
    )
    quantized, meta = unpack_int6_lzma(blob_path.read_bytes())
    assert isinstance(quantized, dict) and quantized, (
        "unpacked quantized dict is empty"
    )
    assert isinstance(meta, dict) and meta, "unpacked meta dict is empty"


# =====================================================================
# Test 2 — empty-JSONL fail-loud regression (C1 fix)
# =====================================================================
def test_empty_eval_jsonl_raises_with_pass_label(
    tiny_pipeline_paths, eval_quant_module, monkeypatch
):
    """Empty eval JSONL must raise RuntimeError that names ``pass 'bf16'``.

    Pre-fix (C1), an empty stream silently produced ``bpb=0.0`` and the
    rest of the pipeline ran to completion with a green-looking JSON.
    The fix raises ``RuntimeError("_eval_bpb pass 'bf16' scored zero
    docs/bytes ...")``. We confirm the exception fires AND identifies
    the bf16 pass — both pieces are needed because ``int6`` would also
    raise zero-docs and we'd miss bf16-side regressions otherwise.
    """
    _patched_quantize_state_dict(monkeypatch)
    # Overwrite the eval file with whitespace only — DocStreamer skips
    # blank lines + JSONDecodeError + missing-text, so this is the
    # "stream yielded nothing" path the C1 fix protects against.
    tiny_pipeline_paths["eval_jsonl"].write_text("\n\n   \n")
    argv = _make_argv(tiny_pipeline_paths)

    with pytest.raises(RuntimeError) as exc_info:
        eval_quant_module.main(argv)

    msg = str(exc_info.value)
    assert "pass 'bf16'" in msg, (
        f"RuntimeError did not name the bf16 pass — message was: {msg!r}"
    )
    # Also confirm it's the empty-stream path and not some other failure
    # that happened to trip RuntimeError.
    assert "scored zero" in msg, (
        f"RuntimeError is not the zero-docs guard — message was: {msg!r}"
    )


# =====================================================================
# Test 3 — decimal-MB regression (I2 fix)
# =====================================================================
def test_artifact_margin_uses_decimal_mb(
    tiny_pipeline_paths, eval_quant_module, monkeypatch
):
    """``artifact_margin_mb`` must use decimal MB (10^6), not MiB (2^20).

    Pre-fix (I2), a 16,000,000-byte artifact would report margin = +0.74
    MB instead of 0.0 because the divisor was 2^20 = 1,048,576 bytes/MiB.
    Param Golf's 16 MB budget is a decimal cap — reporting binary MiB
    would silently grant a 5% headroom phantom that does not exist on
    the leaderboard.

    Asserting at the 16M-byte boundary is impossible on a 5KB tiny
    artifact, but the arithmetic identity holds at every byte count:
    ``margin_mb = 16.0 - artifact_bytes / 1e6`` is unambiguously the
    decimal formula. We assert that identity to within float epsilon.
    """
    _patched_quantize_state_dict(monkeypatch)
    argv = _make_argv(tiny_pipeline_paths)
    rc = eval_quant_module.main(argv)
    assert rc == 0

    result = json.loads(tiny_pipeline_paths["output_json"].read_text())
    expected = 16.0 - (result["artifact_bytes"] / 1e6)
    actual = result["artifact_margin_mb"]
    # MiB divisor (1024**2 = 1_048_576) would give a different number;
    # at ~5KB artifact the difference is small in absolute terms but
    # large enough to fail this 1e-9 tolerance.
    assert abs(actual - expected) < 1e-9, (
        f"artifact_margin_mb = {actual} != decimal expected {expected} "
        f"(artifact_bytes={result['artifact_bytes']})"
    )


# =====================================================================
# Test 4 — static check: no .float() upcast in _eval_bpb (C2 fix)
# =====================================================================
def test_eval_bpb_does_not_upcast_logits():
    """Static-grep ``scripts/eval_quant.py`` for ``logits.float()``.

    Pre-fix (C2), ``_eval_bpb`` upcast logits to fp32 before CE,
    which made the bpb measurement diverge from the canonical
    ``run_exp20_eval`` path (which uses bf16 logits). The fix removes
    the upcast. This is a static guard — the comment in
    ``_eval_bpb`` (line 161-162) pins the contract; this test catches
    a future revert.

    We grep the entire script to keep the check brittle-to-the-bad-state.
    There's no legitimate ``logits.float()`` anywhere in this script —
    if one shows up it's almost certainly the upcast slipping back in.
    """
    src = (REPO / "scripts" / "eval_quant.py").read_text()
    assert "logits.float()" not in src, (
        "logits.float() upcast found in scripts/eval_quant.py — this "
        "diverges the bpb measurement from run_exp20_eval. Remove the "
        "upcast and use bf16 CE so artifact bpb tracks downstream."
    )
