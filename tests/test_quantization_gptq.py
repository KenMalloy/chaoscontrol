"""Tests for the GPTQ int6 + LZMA packaging pipeline."""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from chaoscontrol.quantization import (
    GPTQQuantizer,
    ar_self_generated_calibration,
    pack_int6_lzma,
    quantize_int6_gptq,
    quantize_int6_percentile,
    unpack_int6_lzma,
)


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()


def _synthetic_hessian(cols: int, seed: int = 0) -> torch.Tensor:
    """Build a well-conditioned positive-definite Hessian from random data."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(256, cols, generator=g)
    return (x.T @ x) / x.shape[0] + 0.01 * torch.eye(cols)


class _TinyLogitModel(nn.Module):
    """Minimal embedding + linear → logits stub for AR calibration tests."""

    def __init__(self, vocab: int = 32, dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(tokens))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(tokens)


class GPTQRoundTripTests(unittest.TestCase):
    def test_round_trip_small_linear_is_close(self) -> None:
        """GPTQ on a 64x64 Linear should yield roughly-equal weights."""
        torch.manual_seed(1337)
        layer = nn.Linear(64, 64, bias=False)
        W = layer.weight.data.clone()
        H = _synthetic_hessian(cols=64, seed=1337)

        q, s = quantize_int6_gptq(W, hessian=H)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(s.dtype, torch.float16)
        self.assertTrue(
            bool(((q >= -31) & (q <= 31)).all().item()),
            "int6 codes must live in [-31, 31]",
        )

        recon = q.float() * s.float().unsqueeze(-1)
        # GPTQ degrades by design; assert the recon is close enough that
        # the *relative* Frobenius error is much smaller than the raw
        # clip-scale noise floor. Empirically for a well-conditioned H
        # this lands < 0.05 for a 64x64 random layer.
        rel_err = (W - recon).norm() / W.norm().clamp_min(1e-8)
        self.assertLess(
            rel_err.item(), 0.15,
            f"GPTQ relative Frobenius error too high: {rel_err.item():.4f}",
        )

    def test_percentile_fallback_for_1d_tensor(self) -> None:
        """1D tensors must still round-trip through the percentile path."""
        torch.manual_seed(7)
        t = torch.randn(512)
        q, s = quantize_int6_percentile(t)
        recon = q.float() * float(s.item())
        self.assertLess((t - recon).abs().max().item(), 0.2)


class GPTQCompressionTests(unittest.TestCase):
    def test_int6_lzma_bytes_smaller_than_fp32(self) -> None:
        """int6 + LZMA of a matrix must beat the raw fp32 serialization."""
        torch.manual_seed(42)
        W = torch.randn(256, 256)
        H = _synthetic_hessian(cols=256, seed=42)
        q, s = quantize_int6_gptq(W, hessian=H)

        quantized: dict[str, torch.Tensor] = {"W.q": q, "W.scale": s}
        meta = {"W": {"type": "int6", "dtype": "torch.float32", "shape": [256, 256]}}

        blob = pack_int6_lzma(quantized, meta)
        fp32_bytes = _tensor_bytes(W)
        self.assertLess(
            len(blob), fp32_bytes,
            f"expected int6+lzma < fp32 ({len(blob)} vs {fp32_bytes})",
        )
        # Sanity: we expect at least a 3x reduction for a random 256x256
        # matrix. int6 alone is 8/32=0.25x, LZMA squeezes further.
        self.assertLess(
            len(blob), fp32_bytes // 3,
            f"expected > 3x compression, got {fp32_bytes / max(len(blob), 1):.2f}x",
        )


class ARCalibrationTests(unittest.TestCase):
    def _run_calibration(self, seed: int) -> list[torch.Tensor]:
        torch.manual_seed(0)  # model init stays fixed across calls
        model = _TinyLogitModel(vocab=32, dim=16)
        return ar_self_generated_calibration(
            model.forward_logits,
            num_seqs=4, seq_len=16, vocab_size=32,
            temperature=0.8, batch_size=2, device="cpu", seed=seed,
        )

    def test_fixed_seed_is_deterministic(self) -> None:
        """Identical seed + identical model must yield identical tokens."""
        run_a = self._run_calibration(seed=123)
        run_b = self._run_calibration(seed=123)
        self.assertEqual(len(run_a), len(run_b))
        for a, b in zip(run_a, run_b):
            self.assertTrue(
                torch.equal(a, b),
                f"AR calibration diverged for seed 123: {a} vs {b}",
            )

    def test_different_seeds_differ(self) -> None:
        """Paranoia check — different seeds must not accidentally collide."""
        run_a = self._run_calibration(seed=1)
        run_b = self._run_calibration(seed=2)
        any_diff = any(not torch.equal(a, b) for a, b in zip(run_a, run_b))
        self.assertTrue(any_diff, "different seeds produced identical tokens")


class _TinyLogitMLP(nn.Module):
    """An embed + 2-layer MLP + logit head. Usable as a logit_fn target."""

    def __init__(self, vocab: int = 32, in_dim: int = 96, hidden: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, in_dim)
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, in_dim, bias=False)
        self.head = nn.Linear(in_dim, vocab, bias=False)
        torch.nn.init.normal_(self.embed.weight, std=0.5)

    def forward_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.fc2(torch.tanh(self.fc1(x)))
        return self.head(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(tokens)


class ToyMLPIntegrationTests(unittest.TestCase):
    def test_quantized_mlp_forward_within_tolerance(self) -> None:
        """End-to-end: AR calibrate → quantize → pack → unpack → dequantize."""
        torch.manual_seed(2026)
        vocab = 32
        in_dim, hidden = 96, 128
        model = _TinyLogitMLP(vocab=vocab, in_dim=in_dim, hidden=hidden)
        model.eval()

        calib_tokens = ar_self_generated_calibration(
            model.forward_logits,
            num_seqs=4, seq_len=32, vocab_size=vocab,
            temperature=1.0, batch_size=2, device="cpu", seed=99,
        )

        quantizer = GPTQQuantizer()
        quantizer.calibrate(model, calib_tokens)
        self.assertIn("fc1.weight", quantizer.hessians)
        self.assertIn("fc2.weight", quantizer.hessians)
        self.assertIn("head.weight", quantizer.hessians)

        # Force all Linear layers through the int6 path by lowering the
        # numel cutoff — a 96x128 layer has 12288 elements.
        result, meta = quantizer.quantize_state_dict(
            model.state_dict(), min_numel=1024,
        )
        self.assertEqual(meta["fc1.weight"]["type"], "int6")
        self.assertEqual(meta["fc2.weight"]["type"], "int6")

        blob = pack_int6_lzma(result, meta)
        self.assertGreater(len(blob), 0)

        packed, packed_meta = unpack_int6_lzma(blob)
        deq = quantizer.dequantize_state_dict(packed, packed_meta)

        quantized_model = _TinyLogitMLP(vocab=vocab, in_dim=in_dim, hidden=hidden)
        quantized_model.load_state_dict(deq)
        quantized_model.eval()

        probe = torch.randint(0, vocab, (8, 32), dtype=torch.int64)
        with torch.no_grad():
            baseline = model.forward_logits(probe)
            noisy = quantized_model.forward_logits(probe)
        rel_err = (baseline - noisy).norm() / baseline.norm().clamp_min(1e-8)
        # int6 on a tiny random MLP is coarse — the Exp 19 design note
        # says ~10% is the ballpark on real ChaosControl weights; random
        # toy weights land higher, so we allow 35% here while still
        # catching any regression that blows the pipeline apart.
        self.assertLess(
            rel_err.item(), 0.35,
            f"quantized MLP forward diverged too much: {rel_err.item():.4f}",
        )


if __name__ == "__main__":
    unittest.main()
