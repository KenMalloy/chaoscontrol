"""End-to-end test for scripts/prepare_sgns_inits.py.

Covers the full pipeline: synthetic SGNS tensor → meanstd + fullcov + shuffled
init artifacts with matching shapes and the expected distributional properties.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def _load_prepare_module():
    path = REPO / "scripts" / "prepare_sgns_inits.py"
    spec = importlib.util.spec_from_file_location("prepare_sgns_inits_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_sgns_inits_produces_core_and_mechanism_artifacts(tmp_path):
    """CLI end-to-end: synthetic SGNS tensor in, dissection init tensors out."""
    # Synthesize an anisotropic SGNS-like tensor — non-unit row norms,
    # non-zero mean — so the moment-matching transforms have something
    # to actually change.
    torch.manual_seed(0)
    vocab, dim = 64, 16
    sgns = torch.randn(vocab, dim) * 3.0 + 1.5
    sgns_path = tmp_path / "sgns_v64_d16.pt"
    torch.save(sgns, sgns_path)
    token_counts = torch.arange(vocab, 0, -1, dtype=torch.float32)
    token_counts_path = tmp_path / "token_counts.pt"
    torch.save(token_counts, token_counts_path)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "prepare_sgns_inits.py"),
            "--sgns", str(sgns_path),
            "--out-dir", str(out_dir),
            "--reference-seed", "0",
            "--shuffled-seed", "42",
            "--norm-seed", "11",
            "--norm-shuffled-seed", "12",
            "--fullcov-shuffled-seed", "13",
            "--random-fullcov-seed", "14",
            "--token-counts", str(token_counts_path),
            "--freq-buckets", "4",
            "--freq-shuffled-seed", "15",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"

    meanstd = torch.load(out_dir / "sgns_init_meanstd.pt", map_location="cpu")
    fullcov = torch.load(out_dir / "sgns_init_fullcov.pt", map_location="cpu")
    shuffled = torch.load(out_dir / "sgns_init_shuffled.pt", map_location="cpu")
    zero = torch.load(out_dir / "sgns_init_zero.pt", map_location="cpu")
    norm_only = torch.load(out_dir / "sgns_init_norm_only.pt", map_location="cpu")
    norm_only_shuffled = torch.load(
        out_dir / "sgns_init_norm_only_shuffled.pt", map_location="cpu"
    )
    fullcov_shuffled = torch.load(
        out_dir / "sgns_init_fullcov_shuffled.pt", map_location="cpu"
    )
    random_fullcov = torch.load(
        out_dir / "sgns_init_random_fullcov.pt", map_location="cpu"
    )
    freq_bucket_shuffle = torch.load(
        out_dir / "sgns_init_freq_bucket_shuffle.pt", map_location="cpu"
    )

    assert meanstd.shape == (vocab, dim)
    assert fullcov.shape == (vocab, dim)
    assert shuffled.shape == (vocab, dim)
    assert zero.shape == (vocab, dim)
    assert norm_only.shape == (vocab, dim)
    assert norm_only_shuffled.shape == (vocab, dim)
    assert fullcov_shuffled.shape == (vocab, dim)
    assert random_fullcov.shape == (vocab, dim)
    assert freq_bucket_shuffle.shape == (vocab, dim)

    assert torch.count_nonzero(zero) == 0

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

    # norm_only carries only the per-token norm channel from meanstd.
    torch.testing.assert_close(norm_only.norm(dim=-1), meanstd.norm(dim=-1))
    norm_cos = torch.nn.functional.cosine_similarity(meanstd, norm_only, dim=-1)
    assert norm_cos.abs().mean() < 0.5

    # norm_only_shuffled keeps the same norm multiset, but not the token ID
    # to norm assignment.
    torch.testing.assert_close(
        norm_only_shuffled.norm(dim=-1).sort().values,
        meanstd.norm(dim=-1).sort().values,
    )
    assert not torch.allclose(
        norm_only_shuffled.norm(dim=-1),
        meanstd.norm(dim=-1),
    )

    # fullcov_shuffled is a row permutation of the stronger full-cov control.
    torch.testing.assert_close(
        fullcov_shuffled.norm(dim=-1).sort().values,
        fullcov.norm(dim=-1).sort().values,
    )

    # random_fullcov is synthetic: same shape/moments target, not a row shuffle.
    assert not torch.allclose(random_fullcov, fullcov)
    assert not torch.allclose(
        random_fullcov.norm(dim=-1).sort().values,
        fullcov.norm(dim=-1).sort().values,
    )

    # Frequency-bucket shuffle preserves row multisets inside each frequency
    # rank bucket.
    order = torch.argsort(token_counts, descending=True, stable=True)
    bucket_ids = torch.empty(vocab, dtype=torch.long)
    bucket_ids[order] = torch.div(
        torch.arange(vocab) * 4, vocab, rounding_mode="floor"
    ).clamp(max=3)
    for bucket in bucket_ids.unique():
        mask = bucket_ids == bucket
        torch.testing.assert_close(
            freq_bucket_shuffle[mask].norm(dim=-1).sort().values,
            meanstd[mask].norm(dim=-1).sort().values,
        )


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


def test_prepare_sgns_inits_data_dir_counts_default_is_capped(tmp_path, monkeypatch):
    """The data-dir counting path must not default to scanning all shards."""
    module = _load_prepare_module()
    vocab, dim = 16, 4
    sgns_path = tmp_path / "sgns.pt"
    torch.save(torch.randn(vocab, dim), sgns_path)

    captured: dict[str, int | None] = {}

    def fake_count_data_dir_tokens(data_dir, vocab_size, max_tokens):
        captured["max_tokens"] = max_tokens
        assert data_dir == tmp_path / "data"
        assert vocab_size == vocab
        return torch.arange(vocab, 0, -1, dtype=torch.float32)

    monkeypatch.setattr(module, "_count_data_dir_tokens", fake_count_data_dir_tokens)

    out_dir = tmp_path / "out"
    rc = module.main([
        "--sgns", str(sgns_path),
        "--out-dir", str(out_dir),
        "--data-dir-for-counts", str(tmp_path / "data"),
    ])

    assert rc == 0
    assert captured["max_tokens"] == 50_000_000
    assert (out_dir / "sgns_init_freq_bucket_shuffle.pt").is_file()


def test_count_data_dir_tokens_rejects_large_oor_fraction(tmp_path, monkeypatch):
    """Wrong tokenizer/data paths should fail instead of poisoning freq buckets."""
    module = _load_prepare_module()

    def fake_load_fineweb_tokens(_data_dir):
        train = torch.tensor([0, 1, 2, 999, 1000, -3], dtype=torch.long)
        val = torch.tensor([], dtype=torch.long)
        return train, val

    import chaoscontrol.data

    monkeypatch.setattr(chaoscontrol.data, "load_fineweb_tokens", fake_load_fineweb_tokens)

    try:
        module._count_data_dir_tokens(tmp_path, vocab_size=8, max_tokens=6)
    except ValueError as exc:
        assert "tokenizer/data mismatch" in str(exc)
    else:
        raise AssertionError("expected OOR-heavy token stream to fail")


def test_prepare_sgns_inits_writes_class_bucket_artifact(tmp_path, monkeypatch):
    """The default Exp21b class-bucket condition needs a generated artifact."""
    module = _load_prepare_module()
    vocab, dim = 8, 4
    sgns_path = tmp_path / "sgns.pt"
    torch.save(torch.randn(vocab, dim), sgns_path)

    monkeypatch.setattr(
        module,
        "_sentencepiece_pieces",
        lambda path, vocab_size: [
            "<unk>",
            "\u2581hello",
            "world",
            "123",
            ".",
            "9th",
            "\u2581",
            "hello-world",
        ],
    )

    out_dir = tmp_path / "out"
    rc = module.main([
        "--sgns", str(sgns_path),
        "--out-dir", str(out_dir),
        "--sp-model", str(tmp_path / "fake.model"),
    ])

    assert rc == 0
    class_bucket = torch.load(
        out_dir / "sgns_init_class_bucket_shuffle.pt", map_location="cpu"
    )
    assert class_bucket.shape == (vocab, dim)
