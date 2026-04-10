#!/usr/bin/env python3
"""Tests for Experiment 15 (ChaosPiece) runner.

Tests correctness of:
1. bpb formula on synthetic data
2. Byte LUT construction (if sentencepiece available)
3. Model building for both SSM and transformer
4. match_transformer_params utility
5. Dry run: tiny training + eval produces expected output structure
"""
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chaoscontrol.evaluation import compute_bpb
from runner_exp15 import (
    build_model,
    evaluate_bpb_bytes,
    evaluate_bpb_sp,
    match_transformer_params,
)


# ---------------------------------------------------------------------------
# bpb formula
# ---------------------------------------------------------------------------

class TestBpbFormula:
    def test_known_values(self):
        """bpb = total_ce_nats / total_bytes / ln(2).
        If CE = ln(2) nats per byte over 100 bytes, bpb should be exactly 1.0."""
        total_ce = math.log(2.0) * 100
        total_bytes = 100
        assert abs(compute_bpb(total_ce, total_bytes) - 1.0) < 1e-10

    def test_zero_bytes_returns_zero(self):
        assert compute_bpb(10.0, 0) == 0.0

    def test_higher_ce_means_higher_bpb(self):
        bpb_low = compute_bpb(100.0, 1000)
        bpb_high = compute_bpb(200.0, 1000)
        assert bpb_high > bpb_low

    def test_more_bytes_means_lower_bpb(self):
        bpb_few = compute_bpb(100.0, 100)
        bpb_many = compute_bpb(100.0, 1000)
        assert bpb_many < bpb_few


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

class TestModelBuilding:
    def test_ssm_vocab_8192(self):
        config = {
            "model_type": "ssm",
            "vocab_size": 8192,
            "model_dim": 64,
            "num_layers": 2,
            "ff_mult": 2,
        }
        model = build_model(config, torch.device("cpu"), torch.float32)
        assert model.vocab_size == 8192
        assert model.embed.weight.shape == (8192, 64)

    def test_transformer_vocab_8192(self):
        config = {
            "model_type": "transformer",
            "vocab_size": 8192,
            "model_dim": 64,
            "num_layers": 2,
            "ff_mult": 2,
        }
        model = build_model(config, torch.device("cpu"), torch.float32)
        assert model.vocab_size == 8192
        assert model.embed.weight.shape == (8192, 64)

    def test_ssm_forward_shape(self):
        config = {
            "model_type": "ssm",
            "vocab_size": 8192,
            "model_dim": 64,
            "num_layers": 2,
        }
        model = build_model(config, torch.device("cpu"), torch.float32)
        x = torch.randint(0, 8192, (2, 16))
        out = model(x)
        assert out["logits"].shape == (2, 16, 8192)

    def test_transformer_forward_shape(self):
        config = {
            "model_type": "transformer",
            "vocab_size": 8192,
            "model_dim": 64,
            "num_layers": 2,
        }
        model = build_model(config, torch.device("cpu"), torch.float32)
        x = torch.randint(0, 8192, (2, 16))
        out = model(x)
        assert out["logits"].shape == (2, 16, 8192)

    def test_ssm_no_bolt_ons(self):
        """Phase A SSMs should have no Wernicke, no outer model."""
        config = {
            "model_type": "ssm",
            "vocab_size": 8192,
            "model_dim": 64,
            "num_layers": 2,
        }
        model = build_model(config, torch.device("cpu"), torch.float32)
        assert model.outer_model is None
        assert model.wernicke is None


# ---------------------------------------------------------------------------
# Param matching
# ---------------------------------------------------------------------------

class TestParamMatching:
    def test_reasonable_match(self):
        """match_transformer_params should get within 20% of target."""
        target = 3_000_000
        result = match_transformer_params(target, vocab_size=8192)
        assert result is not None
        gap = abs(result["total_params"] - target) / target
        assert gap < 0.20, f"Gap too large: {gap:.1%}"

    def test_small_target(self):
        result = match_transformer_params(500_000, vocab_size=8192)
        assert result is not None
        assert result["num_layers"] >= 2

    def test_large_target(self):
        result = match_transformer_params(8_000_000, vocab_size=8192)
        assert result is not None
        assert result["total_params"] > 0


# ---------------------------------------------------------------------------
# Byte-level eval
# ---------------------------------------------------------------------------

class TestByteLevelEval:
    def test_eval_returns_expected_keys(self):
        """Byte-level eval should return bpb, loss, tokens, total_scored_bytes."""
        config = {
            "model_type": "ssm",
            "vocab_size": 256,
            "model_dim": 32,
            "num_layers": 1,
        }
        model = build_model(config, torch.device("cpu"), torch.float32)
        # Fake token tensor
        tokens = torch.randint(0, 256, (2000,))
        eval_starts = [0, 100, 200]
        result = evaluate_bpb_bytes(
            model,
            tokens=tokens,
            eval_starts=eval_starts,
            batch_size=2,
            seq_len=64,
            device=torch.device("cpu"),
        )
        assert "bpb" in result
        assert "loss" in result
        assert "tokens" in result
        assert "total_scored_bytes" in result
        assert result["bpb"] > 0
        # For byte-level, scored_bytes should equal scored tokens
        assert result["total_scored_bytes"] == result["tokens"]

    def test_bpb_in_sane_range(self):
        """Untrained model on bytes should have bpb around 8 (log2(256))."""
        config = {
            "model_type": "ssm",
            "vocab_size": 256,
            "model_dim": 32,
            "num_layers": 1,
        }
        model = build_model(config, torch.device("cpu"), torch.float32)
        tokens = torch.randint(0, 256, (5000,))
        eval_starts = list(range(0, 4000, 100))
        result = evaluate_bpb_bytes(
            model,
            tokens=tokens,
            eval_starts=eval_starts,
            batch_size=4,
            seq_len=64,
            device=torch.device("cpu"),
        )
        # Untrained: should be near log2(256) = 8.0, but not exactly
        assert 4.0 < result["bpb"] < 12.0


# ---------------------------------------------------------------------------
# SP8192 eval with synthetic LUT
# ---------------------------------------------------------------------------

class TestSPEval:
    def test_evaluate_bpb_sp_with_synthetic_lut(self):
        """Test evaluate_bpb_sp using a hand-crafted byte LUT.

        Setup: vocab=16, 3 tokens with known byte counts.
        Token 0: boundary (0 bytes), Token 1: 3 bytes, Token 2: 5 bytes.
        None have leading spaces for simplicity.
        Expected: total_scored_bytes = sum of byte counts for all target tokens.
        """
        vocab = 16
        config = {"model_type": "ssm", "vocab_size": vocab, "model_dim": 32, "num_layers": 1}
        model = build_model(config, torch.device("cpu"), torch.float32)

        # Build synthetic LUT
        base_bytes = torch.zeros(vocab, dtype=torch.int16)
        base_bytes[1] = 3
        base_bytes[2] = 5
        base_bytes[3] = 2
        base_bytes[4] = 4
        # Fill remaining with 1 byte each
        for i in range(5, vocab):
            base_bytes[i] = 1
        has_leading_space = torch.zeros(vocab, dtype=torch.bool)
        is_boundary = torch.zeros(vocab, dtype=torch.bool)
        is_boundary[0] = True  # Token 0 is boundary

        # Create token sequence using only tokens 1-5
        tokens = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5] * 20, dtype=torch.long)
        eval_starts = [0, 10, 20]
        seq_len = 8

        result = evaluate_bpb_sp(
            model,
            tokens=tokens,
            eval_starts=eval_starts,
            batch_size=2,
            seq_len=seq_len,
            device=torch.device("cpu"),
            base_bytes_lut=base_bytes,
            has_leading_space_lut=has_leading_space,
            is_boundary_token_lut=is_boundary,
        )

        assert "bpb" in result
        assert "total_scored_bytes" in result
        assert result["bpb"] > 0
        assert result["total_scored_bytes"] > 0
        # Verify scored bytes is reasonable: each batch has seq_len target tokens
        # with byte counts from our LUT (not all 1)
        avg_bytes_per_token = result["total_scored_bytes"] / result["tokens"]
        assert avg_bytes_per_token > 1.0, "Byte LUT should produce > 1 byte/token on average"

    def test_sp_eval_leading_space_adjustment(self):
        """Verify leading space adds 1 byte when previous token is not boundary."""
        vocab = 8
        config = {"model_type": "ssm", "vocab_size": vocab, "model_dim": 32, "num_layers": 1}
        model = build_model(config, torch.device("cpu"), torch.float32)

        base_bytes = torch.ones(vocab, dtype=torch.int16) * 2  # each token = 2 base bytes
        has_leading_space = torch.zeros(vocab, dtype=torch.bool)
        has_leading_space[3] = True  # Token 3 has leading space
        is_boundary = torch.zeros(vocab, dtype=torch.bool)
        is_boundary[0] = True  # Token 0 is boundary

        # Sequence: [1, 3, 1, 3, ...] — token 3 follows non-boundary token 1
        # So token 3 should get +1 byte from leading space
        tokens = torch.tensor([1, 3, 1, 3, 1, 3, 1, 3, 1, 3] * 10, dtype=torch.long)
        eval_starts = [0]
        seq_len = 8

        result = evaluate_bpb_sp(
            model,
            tokens=tokens,
            eval_starts=eval_starts,
            batch_size=1,
            seq_len=seq_len,
            device=torch.device("cpu"),
            base_bytes_lut=base_bytes,
            has_leading_space_lut=has_leading_space,
            is_boundary_token_lut=is_boundary,
        )

        # 8 target tokens: tokens at positions 1-8 of the sequence
        # Sequence is [1,3,1,3,1,3,1,3,1,...], targets are positions 1-8
        # Token 3 follows non-boundary 1: gets 2+1=3 bytes
        # Token 1 has no leading space: gets 2 bytes
        # Pattern: 3,1,3,1,3,1,3,1 -> alternating 3 and 2 bytes -> avg 2.5
        scored_tokens = result["tokens"]
        avg_bytes = result["total_scored_bytes"] / scored_tokens
        assert avg_bytes > 2.0, f"Leading space should increase avg bytes above 2.0, got {avg_bytes}"


# ---------------------------------------------------------------------------
# Byte LUT (only if sentencepiece installed)
# ---------------------------------------------------------------------------

sp_available = False
try:
    import sentencepiece
    sp_available = True
except ImportError:
    pass


@pytest.mark.skipif(not sp_available, reason="sentencepiece not installed")
class TestByteLUT:
    def test_lut_shapes(self):
        """LUT tensors should have correct shapes."""
        from runner_exp15 import build_sentencepiece_luts
        import sentencepiece as spm

        # Try to find a model file in the repo
        candidates = list(REPO.glob("baselines/parameter_golf/tokenizers/*.model"))
        if not candidates:
            pytest.skip("No SentencePiece model file found in repo")
        sp = spm.SentencePieceProcessor()
        sp.Load(str(candidates[0]))
        vocab = sp.vocab_size()

        base_bytes, has_space, is_boundary = build_sentencepiece_luts(
            sp, vocab, torch.device("cpu"),
        )
        assert base_bytes.shape[0] >= vocab
        assert has_space.shape[0] >= vocab
        assert is_boundary.shape[0] >= vocab
        # Control tokens should be boundary tokens
        assert is_boundary[0].item() is True  # <unk> or <s>
        # Most regular tokens should have > 0 bytes
        assert (base_bytes > 0).sum() > vocab * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
