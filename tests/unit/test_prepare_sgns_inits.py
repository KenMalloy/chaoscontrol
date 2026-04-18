"""End-to-end test for scripts/prepare_sgns_inits.py.

Covers the full pipeline: synthetic SGNS tensor → meanstd + fullcov + shuffled
init artifacts with matching shapes and the expected distributional properties.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def test_prepare_sgns_inits_produces_three_artifacts(tmp_path):
    """CLI end-to-end: synthetic SGNS tensor in → three init tensors out."""
    # Synthesize an anisotropic SGNS-like tensor — non-unit row norms,
    # non-zero mean — so the moment-matching transforms have something
    # to actually change.
    torch.manual_seed(0)
    vocab, dim = 64, 16
    sgns = torch.randn(vocab, dim) * 3.0 + 1.5
    sgns_path = tmp_path / "sgns_v64_d16.pt"
    torch.save(sgns, sgns_path)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "prepare_sgns_inits.py"),
            "--sgns", str(sgns_path),
            "--out-dir", str(out_dir),
            "--reference-seed", "0",
            "--shuffled-seed", "42",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"

    meanstd = torch.load(out_dir / "sgns_init_meanstd.pt", map_location="cpu")
    fullcov = torch.load(out_dir / "sgns_init_fullcov.pt", map_location="cpu")
    shuffled = torch.load(out_dir / "sgns_init_shuffled.pt", map_location="cpu")

    assert meanstd.shape == (vocab, dim)
    assert fullcov.shape == (vocab, dim)
    assert shuffled.shape == (vocab, dim)

    # meanstd preserves per-row direction (cosine) of the input SGNS tensor.
    cos = torch.nn.functional.cosine_similarity(sgns, meanstd, dim=-1)
    assert torch.all(cos > 0.999), (
        f"meanstd should preserve per-row direction; min cos={cos.min().item()}"
    )

    # shuffled is a row-permutation of meanstd — same multiset of rows.
    ms_sorted = meanstd.norm(dim=-1).sort().values
    sh_sorted = shuffled.norm(dim=-1).sort().values
    torch.testing.assert_close(ms_sorted, sh_sorted)

    # shuffled destroys ID→vector mapping — at least some rows differ.
    row_match = (shuffled == meanstd).all(dim=-1)
    assert not row_match.all(), "shuffled must permute at least some rows"


def test_prepare_sgns_inits_rejects_bad_shape(tmp_path):
    """1D tensor input should fail the shape check with a clear message."""
    bad = torch.randn(100)
    path = tmp_path / "bad.pt"
    torch.save(bad, path)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "prepare_sgns_inits.py"),
            "--sgns", str(path),
            "--out-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode != 0
    assert "shape (V, D)" in result.stderr or "shape (V, D)" in result.stdout
