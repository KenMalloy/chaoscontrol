import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.eval_stream.doc_stream import DocStreamer
from chaoscontrol.eval_stream.legality import LegalityController


def test_legality_controller_matches_naive_forward(tmp_path):
    torch.manual_seed(0)
    # CPU-pin the test: bf16/tf32 on GPU exceeds the 1e-4 tolerance; the
    # bit-exact guarantee is about the harness composition, not the dtype path.
    device = torch.device("cpu")
    m = ChaosStudentLM(
        vocab_size=32, dim=16, num_layers=2, block_type="ssm", a_mode="diag",
    ).to(device)
    m.eval()  # must be in eval for deterministic comparison under block_type="attention" too
    tokens = torch.randint(1, 32, (1, 64), device=device)

    # Naive forward-only — call .eval() first so the comparison is invariant
    # under blocks that include dropout (attention variant).
    with torch.no_grad():
        out = m(tokens)
        logits = out["logits"] if isinstance(out, dict) else out
        naive = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            tokens[:, 1:].reshape(-1), reduction="sum",
        ).item()

    controller = LegalityController(
        m, loss_fn=lambda lg, tg: F.cross_entropy(
            lg.reshape(-1, lg.size(-1)).float(), tg.reshape(-1), reduction="sum"
        ),
    )
    harness_score, _ = controller.score_chunk(tokens)
    assert abs(harness_score - naive) < 1e-4


def test_chunked_carry_state_differs_from_reset(tmp_path):
    """Canary for the Task 3.5 state-plumbing invariant.

    Failure mode this guards against: ``persistence_mode=carry_state`` is a
    silent no-op (state threaded from chunk N's final_states isn't actually
    consumed by chunk N+1's forward).

    We test this directly by running the driver twice at matched
    ``chunk_size`` and comparing ``carry_state`` vs ``reset``. If state
    doesn't thread, both modes zero the recurrence at every chunk start and
    produce identical bpb. A nontrivial difference proves state is being
    consumed.

    DEVIATION FROM PLAN: the original canary asserted ``|bpb_whole -
    bpb_carry| < 1e-3``. That tolerance is unreachable with non-overlapping
    chunks because chunking drops the boundary teacher-forcing targets
    (e.g. 71→69 targets on a 72-token doc at chunk_size=32 → ~2.9% CE gap,
    empirically verified). Driver is NOT modified; if future phases want
    strict whole-doc equivalence they need a sliding/overlapping chunker,
    which is out of scope for the harness phase.
    """
    import json
    import subprocess
    import sys

    import sentencepiece as spm

    # Tiny SP model
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["alpha beta gamma", "delta epsilon zeta"] * 50))
    sp_prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(sp_prefix),
        vocab_size=64, character_coverage=1.0, model_type="bpe",
    )

    # Two docs, each long enough to produce multiple chunks. Two docs are
    # required because ``reset`` and ``carry_state`` only differ at doc
    # boundaries — within a doc they're the same (state carries across
    # chunks; the persistence mode governs what happens at doc_id+=1).
    jsonl = tmp_path / "docs.jsonl"
    with jsonl.open("w") as fh:
        text1 = " ".join(["alpha beta gamma delta epsilon zeta"] * 12)
        text2 = " ".join(["zeta epsilon delta gamma beta alpha"] * 12)
        fh.write(json.dumps({"text": text1}) + "\n")
        fh.write(json.dumps({"text": text2}) + "\n")

    # Tiny checkpoint
    m = ChaosStudentLM(vocab_size=64, dim=16, num_layers=2, block_type="ssm", a_mode="diag")
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(),
                "config": {"vocab_size": 64, "dim": 16, "num_layers": 2,
                           "block_type": "ssm", "a_mode": "diag"}}, ckpt)

    def _run(chunk_size: int, persistence: str, out_name: str) -> float:
        cfg_path = tmp_path / f"cfg_{out_name}.json"
        out_path = tmp_path / f"{out_name}.jsonl"
        cfg_path.write_text(json.dumps({
            "adapt_set": "none", "persistence_mode": persistence,
            "chunk_size": chunk_size, "steps_per_chunk": 0,
            "max_docs": 2, "seed": 0,
            "jsonl_paths": [str(jsonl)],
            "sp_model_path": f"{sp_prefix}.model",
            "checkpoint_path": str(ckpt),
            "output_path": str(out_path),
        }))
        result = subprocess.run(
            [sys.executable, "scripts/run_exp20_eval.py", "--config", str(cfg_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        # Return second doc's bpb — that's where reset vs carry_state diverges,
        # since both modes start doc 0 from zeros.
        lines = out_path.read_text().strip().splitlines()
        rec = json.loads(lines[1])
        return rec["bpb"]

    bpb_reset = _run(chunk_size=32, persistence="reset", out_name="reset")
    bpb_carry = _run(chunk_size=32, persistence="carry_state", out_name="carry")
    # A silent no-op would make these identical. Any nontrivial difference
    # proves the threaded state is being consumed across the doc boundary.
    assert abs(bpb_reset - bpb_carry) > 1e-6, (bpb_reset, bpb_carry)
