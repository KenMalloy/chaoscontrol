"""Tests for ChaosAttentionBlock — Exp 19 Phase 2 scientific control.

ChaosAttentionBlock is a causal multi-head self-attention sibling to
ChaosSSMBlock. It exists so Exp 19 Phase 2 can run the same training stack
with attention substituted for SSM and measure per-token learning efficiency.

These tests cover the block contract:
  - Shape parity with ChaosSSMBlock (forward and return_jacobian_stats).
  - Causal masking: outputs at position t must not depend on inputs at
    positions > t.
  - Backward pass produces non-zero gradients on every learnable tensor.
  - Shape interchangeability with ChaosSSMBlock inside ChaosStudentLM via
    the `block_type="attention"` constructor flag.
  - No Flash Attention imports — SDPA only.
  - RoPE position sensitivity: outputs depend on absolute position (the
    block is NOT permutation-equivariant), parameter count is unchanged
    vs the pre-RoPE version, and the rotation math matches the RoFormer
    reference formula to within float32 precision.
"""
from __future__ import annotations

import importlib
import sys
import unittest

import torch
import torch.nn.functional as F

from chaoscontrol.model import (
    ChaosAttentionBlock,
    ChaosSSMBlock,
    ChaosStudentLM,
)


class TestChaosAttentionBlockForwardShape(unittest.TestCase):
    def test_forward_shape_matches_input(self) -> None:
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        x = torch.randn(3, 11, 32)
        out = block(x)
        assert out.shape == (3, 11, 32), f"shape mismatch: {out.shape}"

    def test_forward_dtype_preserved(self) -> None:
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        x = torch.randn(2, 8, 32)
        out = block(x)
        assert out.dtype == x.dtype

    def test_return_jacobian_stats_shape_contract(self) -> None:
        """Match ChaosSSMBlock's return contract when return_jacobian_stats=True."""
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        x = torch.randn(2, 8, 32)
        result = block(x, return_jacobian_stats=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        out, stats = result
        assert out.shape == (2, 8, 32)
        assert isinstance(stats, dict)
        assert "lambda_max" in stats
        assert "sv_log_var" in stats

    def test_init_signature_shape_matches_ssm_block(self) -> None:
        """ChaosAttentionBlock(dim, ff_mult=...) mirrors ChaosSSMBlock(dim, ff_mult=...)."""
        dim = 32
        attn = ChaosAttentionBlock(dim, ff_mult=2, num_heads=4)
        ssm = ChaosSSMBlock(dim, ff_mult=2, a_mode="diag", rich_b_mode="none")
        x = torch.randn(2, 8, dim)
        attn_out = attn(x)
        ssm_out = ssm(x)
        assert attn_out.shape == ssm_out.shape

    def test_dim_not_divisible_by_heads_raises(self) -> None:
        with self.assertRaises(ValueError):
            ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=5)


class TestChaosAttentionBlockCausality(unittest.TestCase):
    """A causal block's output at position t must be independent of inputs at
    positions > t. We test this empirically: mutate inputs at positions > t
    and check that outputs at positions <= t are unchanged.
    """

    def _make_block(self) -> ChaosAttentionBlock:
        torch.manual_seed(0)
        block = ChaosAttentionBlock(dim=16, ff_mult=2, num_heads=4)
        block.eval()  # disable dropout for deterministic comparison
        return block

    def test_future_tokens_do_not_affect_past(self) -> None:
        block = self._make_block()
        seq_len = 8
        x = torch.randn(2, seq_len, 16)
        with torch.no_grad():
            out_ref = block(x)

        # For every position t in [0, seq_len - 2], zero out positions > t
        # and confirm that outputs at positions [0..t] are byte-identical to
        # the reference run. Everything in the (future) region we perturbed is
        # of course expected to differ.
        for t in range(seq_len - 1):
            x_mut = x.clone()
            x_mut[:, t + 1 :, :] = 0.0
            with torch.no_grad():
                out_mut = block(x_mut)
            past = out_mut[:, : t + 1, :]
            past_ref = out_ref[:, : t + 1, :]
            # Allow tight tolerance for any floating-point reassociation SDPA
            # might do, but the causal invariant is strict; use a very small
            # atol to catch real causality bugs.
            assert torch.allclose(past, past_ref, atol=1e-6, rtol=1e-5), (
                f"causality violated at t={t}: "
                f"max diff = {(past - past_ref).abs().max().item():.3e}"
            )

    def test_random_future_noise_does_not_affect_past(self) -> None:
        """Stronger variant: replace future tokens with fresh random noise.
        This probes attention leakage more aggressively than zeroing.
        """
        block = self._make_block()
        seq_len = 6
        x = torch.randn(2, seq_len, 16)
        with torch.no_grad():
            out_ref = block(x)

        for t in range(seq_len - 1):
            x_mut = x.clone()
            x_mut[:, t + 1 :, :] = torch.randn_like(x_mut[:, t + 1 :, :])
            with torch.no_grad():
                out_mut = block(x_mut)
            past = out_mut[:, : t + 1, :]
            past_ref = out_ref[:, : t + 1, :]
            assert torch.allclose(past, past_ref, atol=1e-6, rtol=1e-5), (
                f"causality violated at t={t} (random perturbation): "
                f"max diff = {(past - past_ref).abs().max().item():.3e}"
            )


class TestChaosAttentionBlockBackward(unittest.TestCase):
    def test_backward_produces_nonzero_grads_on_every_learnable(self) -> None:
        torch.manual_seed(0)
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = block(x)
        loss = out.pow(2).mean()
        loss.backward()

        # Inputs get gradient.
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

        # Every named parameter of the block has non-zero gradient.
        offenders: list[str] = []
        for name, p in block.named_parameters():
            if p.grad is None:
                offenders.append(f"{name}: grad is None")
                continue
            if p.grad.abs().sum().item() == 0.0:
                offenders.append(f"{name}: grad is zero")
        assert not offenders, "parameters with missing/zero grad: " + ", ".join(offenders)

    def test_qkv_and_out_proj_receive_nonzero_grads_explicitly(self) -> None:
        """Sanity check on the named projections called out in the task spec."""
        torch.manual_seed(0)
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        x = torch.randn(2, 8, 32)
        loss = block(x).pow(2).mean()
        loss.backward()
        assert block.qkv_proj.weight.grad is not None
        assert block.qkv_proj.weight.grad.abs().sum().item() > 0
        assert block.out_proj.weight.grad is not None
        assert block.out_proj.weight.grad.abs().sum().item() > 0
        assert block.ff.fc.weight.grad is not None
        assert block.ff.fc.weight.grad.abs().sum().item() > 0
        assert block.ff.proj.weight.grad is not None
        assert block.ff.proj.weight.grad.abs().sum().item() > 0


class TestChaosStudentLMWithAttentionBlock(unittest.TestCase):
    """Interchangeability: build ChaosStudentLM with ChaosAttentionBlock layers
    and verify the forward contract is identical to the SSM path.
    """

    def test_student_lm_attention_forward_shape_matches_ssm(self) -> None:
        vocab_size, dim, num_layers = 64, 32, 2
        ssm_model = ChaosStudentLM(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            ff_mult=2,
            block_type="ssm",
            a_mode="diag",
            rich_b_mode="none",
        )
        attn_model = ChaosStudentLM(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            ff_mult=2,
            block_type="attention",
            attention_num_heads=4,
        )
        ids = torch.randint(0, vocab_size, (2, 16))
        ssm_out = ssm_model(ids)
        attn_out = attn_model(ids)
        assert ssm_out["logits"].shape == attn_out["logits"].shape
        assert attn_out["logits"].shape == (2, 16, vocab_size)
        assert attn_out["hidden"].shape == (2, 16, dim)

    def test_student_lm_attention_uses_attention_block(self) -> None:
        model = ChaosStudentLM(
            vocab_size=64,
            dim=32,
            num_layers=2,
            ff_mult=2,
            block_type="attention",
            attention_num_heads=4,
        )
        for layer in model.layers:
            assert isinstance(layer, ChaosAttentionBlock)
        assert model.block_type == "attention"

    def test_student_lm_attention_gradients_flow(self) -> None:
        model = ChaosStudentLM(
            vocab_size=64,
            dim=32,
            num_layers=2,
            ff_mult=2,
            block_type="attention",
            attention_num_heads=4,
        )
        ids = torch.randint(0, 64, (2, 16))
        out = model(ids)
        loss = F.cross_entropy(out["logits"].reshape(-1, 64), ids.reshape(-1))
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_unknown_block_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            ChaosStudentLM(
                vocab_size=64, dim=32, num_layers=2, ff_mult=2,
                block_type="lstm",
            )

    def test_student_lm_attention_step_shape(self) -> None:
        """ChaosStudentLM.step() must work when block_type='attention'.

        The attention block's step() is degenerate (single token, no history),
        but it exists so ChaosStudentLM.step() / dream_step() iteration code
        does not crash. This test pins the shape contract so future refactors
        of the ``isinstance(layer, ChaosSSMHybridBlock)`` branching or the
        attention block's step() signature are caught automatically.
        """
        model = ChaosStudentLM(
            vocab_size=64,
            dim=32,
            num_layers=2,
            ff_mult=2,
            block_type="attention",
            attention_num_heads=4,
        )
        model.eval()
        states = model.init_state(batch_size=2)
        ids = torch.randint(0, 64, (2, 1))
        logits, hidden, new_states = model.step(ids, states)
        assert logits.shape == (2, 64)
        assert hidden.shape == (2, 32)
        assert len(new_states) == 2
        assert all(s.shape == (2, 32) for s in new_states)
        # ChaosAttentionBlock.step returns zero new_state (no cross-token
        # state by design — see the docstring on ChaosAttentionBlock.step).
        assert all(s.abs().sum().item() == 0.0 for s in new_states)

    def test_student_lm_attention_causality(self) -> None:
        """Full LM is causal when ChaosAttentionBlock is used.

        Perturb a single future input embedding (by replacing the token id at
        position t+1) and check that logits at positions <= t are unchanged.
        This ties the block-level causality test to the end-to-end model path.
        """
        torch.manual_seed(0)
        model = ChaosStudentLM(
            vocab_size=64,
            dim=32,
            num_layers=2,
            ff_mult=2,
            block_type="attention",
            attention_num_heads=4,
        )
        model.eval()
        ids = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            out_ref = model(ids)["logits"]

        for t in range(6):
            ids_mut = ids.clone()
            # Replace the token at position t+1 with a different id.
            ids_mut[0, t + 1] = (ids[0, t + 1].item() + 1) % 64
            with torch.no_grad():
                out_mut = model(ids_mut)["logits"]
            past_ref = out_ref[:, : t + 1, :]
            past_mut = out_mut[:, : t + 1, :]
            assert torch.allclose(past_mut, past_ref, atol=1e-5, rtol=1e-5), (
                f"end-to-end causality violated at t={t}: "
                f"max diff = {(past_mut - past_ref).abs().max().item():.3e}"
            )


class TestChaosAttentionBlockPositionSensitivity(unittest.TestCase):
    """RoPE tests for ChaosAttentionBlock (Codex review 2026-04-13, finding #1).

    Before RoPE was added, ChaosAttentionBlock ran SDPA on raw token embeddings
    with no positional signal. Attention is permutation-equivariant under the
    causal mask: reordering tokens produces an identical output *at positions
    where the full set of attended keys is the same*. In particular, at the
    LAST position of a sequence the causal mask lets the token attend to all
    keys, so permuting earlier tokens leaves the last-position output
    unchanged in the no-position-signal regime. That is the fairness problem
    these tests guard against — the SSM recurrence at the same position IS
    sensitive to order, so the Exp 19 Phase 2 comparison would be biased
    against the SSM if attention lacked any position signal.

    RoPE rotates Q and K by a per-position angle so the dot product
    q_i · k_j depends on (i - j), making the block sensitive to absolute
    position while remaining parameter-free.
    """

    @staticmethod
    def _expected_param_count(dim: int, ff_mult: int) -> int:
        """Formula: 2*dim (RMSNorms) + (2*ff_mult + 4)*dim^2 (FF + QKV + out_proj)."""
        return 2 * dim + (2 * ff_mult + 4) * dim * dim

    def test_parameter_count_matches_pre_rope_formula(self) -> None:
        """RoPE must add ZERO learnable parameters.

        The buffer registration is ``persistent=False`` so cos/sin tables do
        not show up in ``parameters()``. If a future refactor accidentally
        promotes them to nn.Parameter (or inserts any learned projection in
        the RoPE path), this test will catch it.
        """
        for dim, ff_mult, num_heads in [(32, 2, 4), (64, 2, 8), (128, 4, 8)]:
            with self.subTest(dim=dim, ff_mult=ff_mult, num_heads=num_heads):
                block = ChaosAttentionBlock(
                    dim=dim, ff_mult=ff_mult, num_heads=num_heads
                )
                actual = sum(p.numel() for p in block.parameters())
                expected = self._expected_param_count(dim, ff_mult)
                assert actual == expected, (
                    f"param count mismatch: expected {expected}, got {actual}. "
                    f"RoPE must be parameter-free."
                )

    def test_rope_buffers_are_not_persistent(self) -> None:
        """cos/sin tables must NOT appear in state_dict (non-persistent buffers).

        Checkpoint size is user-visible; a persistent RoPE buffer would bloat
        every saved model and propagate a fixed max-seq-len into checkpoints.
        """
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        # Trigger cache build so the buffers are non-empty.
        _ = block(torch.randn(1, 8, 32))
        state = block.state_dict()
        assert "_rope_cos" not in state
        assert "_rope_sin" not in state

    def test_permuting_earlier_tokens_changes_last_position_output(self) -> None:
        """At the last position, the causal mask lets the token attend to ALL
        keys, so without any position signal the softmax over a permuted key
        set would yield an identical output (attention is a set operation over
        keys/values when Q is fixed). With RoPE, the dot product q_i · k_j
        depends on (i - j), so reordering earlier keys must change the output.

        The task prompt suggests comparing [A,B,C] vs [B,A,C] at the differing
        positions. Those positions have different tokens so they trivially
        differ. The DISCRIMINATING assertion is at position 2 (token C in
        both sequences, same query, same key/value SET, different key
        ORDERING) — that output is identical without RoPE and different
        with RoPE.
        """
        torch.manual_seed(0)
        block = ChaosAttentionBlock(dim=16, ff_mult=2, num_heads=4)
        block.eval()

        # Build token embeddings A, B, C deterministically.
        a = torch.randn(1, 16)
        b = torch.randn(1, 16)
        c = torch.randn(1, 16)

        # Sequence 1: [A, B, C]. Sequence 2: [B, A, C].
        seq1 = torch.stack([a, b, c], dim=1)  # (1, 3, 16)
        seq2 = torch.stack([b, a, c], dim=1)

        with torch.no_grad():
            out1 = block(seq1)
            out2 = block(seq2)

        # Position 2 holds the same token (C) in both, with the same SET of
        # prior keys {A, B} but different ORDER. With RoPE the outputs must
        # differ; without RoPE they would be identical to within float noise.
        diff = (out1[:, 2, :] - out2[:, 2, :]).abs().max().item()
        assert diff > 1e-4, (
            f"position 2 output matches across permuted prefixes "
            f"(max diff {diff:.2e}) — RoPE is not rotating queries or keys. "
            f"Attention remains permutation-equivariant."
        )

    def test_same_token_at_different_positions_produces_different_outputs(
        self,
    ) -> None:
        """Translation sensitivity: the same query token embedding placed at
        absolute positions t and t+k must produce different hidden states
        (after the FF residual). RoPE is the only thing in the block that
        depends on absolute position, so if this test passes, the position
        signal is live all the way through to the block output.

        Construction: fix a context prefix that is IDENTICAL in both
        sequences, then place the same target token at position t=2 in one
        sequence and at position t=7 in the other. The shared prefix covers
        positions [0..1] in both, so we specifically check the OUTPUTS at
        t=2 vs t=7, not at the prefix positions.
        """
        torch.manual_seed(1)
        block = ChaosAttentionBlock(dim=16, ff_mult=2, num_heads=4)
        block.eval()

        # Shared prefix (two tokens) and the target token.
        prefix = torch.randn(1, 2, 16)  # positions [0, 1]
        target = torch.randn(1, 1, 16)
        # Filler tokens to pad out sequence 2 so the target lands at t=7.
        filler = torch.randn(1, 5, 16)  # positions [2..6] in seq2

        # Sequence 1: [prefix, TARGET, filler] — target at t=2, length 8.
        seq1 = torch.cat([prefix, target, filler], dim=1)
        # Sequence 2: [prefix, filler, TARGET] — target at t=7, length 8.
        seq2 = torch.cat([prefix, filler, target], dim=1)

        with torch.no_grad():
            out1 = block(seq1)
            out2 = block(seq2)

        # Target at t=2 in seq1, t=7 in seq2. Different positions -> different
        # RoPE rotation applied to Q at that position -> different attention
        # output -> different block output.
        out1_target = out1[:, 2, :]
        out2_target = out2[:, 7, :]
        diff = (out1_target - out2_target).abs().max().item()
        assert diff > 1e-4, (
            f"same token at different positions produced identical outputs "
            f"(max diff {diff:.2e}) — RoPE is not translating the position."
        )

    def test_rope_numerical_parity_with_roformer_reference(self) -> None:
        """Assert the block's RoPE matches the RoFormer paper formula exactly.

        Reference (inline; same math as lucidrains/rotary-embedding-torch and
        the original Su et al. 2021 paper, interleaved-pair form):

            x_rot[..., 0::2] = x[..., 0::2] * cos_pair - x[..., 1::2] * sin_pair
            x_rot[..., 1::2] = x[..., 1::2] * cos_pair + x[..., 0::2] * sin_pair

        where cos_pair[t, i] = cos(t / base^(2i/head_dim)) for i in
        [0, head_dim/2), and similarly for sin_pair.

        We build a deterministic Q tensor, call ``block.apply_rope`` through
        the staticmethod, and compare against the reference computed inline
        at float32. Max abs diff must be below 1e-6.
        """
        torch.manual_seed(2)
        dim = 32
        num_heads = 4
        head_dim = dim // num_heads  # 8
        seq_len = 12
        base = 10000.0

        block = ChaosAttentionBlock(dim=dim, ff_mult=2, num_heads=num_heads)
        # Build the cache at the target seq_len and float32 so we can read
        # the cos/sin tables directly.
        block._ensure_rope_cache(seq_len, torch.device("cpu"))

        # Deterministic (1, num_heads, seq_len, head_dim) tensor.
        x = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float32)

        # --- Inline RoFormer reference -------------------------------------
        half = head_dim // 2
        i = torch.arange(half, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (2.0 * i / head_dim))  # (half,)
        positions = torch.arange(seq_len, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]  # (seq_len, half)
        cos_pair_ref = torch.cos(angles)  # (seq_len, half)
        sin_pair_ref = torch.sin(angles)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even_ref = x_even * cos_pair_ref - x_odd * sin_pair_ref
        rot_odd_ref = x_odd * cos_pair_ref + x_even * sin_pair_ref
        x_rot_ref = torch.empty_like(x)
        x_rot_ref[..., 0::2] = rot_even_ref
        x_rot_ref[..., 1::2] = rot_odd_ref
        # -------------------------------------------------------------------

        # Compare to the block's apply_rope using the cached cos/sin tables.
        cos = block._rope_cos[:seq_len]
        sin = block._rope_sin[:seq_len]
        x_rot_block = ChaosAttentionBlock.apply_rope(x, cos, sin)

        max_diff = (x_rot_block - x_rot_ref).abs().max().item()
        assert max_diff < 1e-6, (
            f"RoPE math does not match RoFormer reference: max diff {max_diff:.3e}"
        )

    def test_apply_rope_is_identity_at_position_zero(self) -> None:
        """At t=0, cos=1 and sin=0 for every frequency, so RoPE is identity.

        This property is what lets ChaosAttentionBlock.step() remain correct
        without a special single-token branch: the step hits _attn with
        seq_len=1, the cache gives cos=[1,...,1] and sin=[0,...,0], and
        apply_rope passes Q and K through unchanged.
        """
        torch.manual_seed(3)
        dim = 16
        num_heads = 4
        head_dim = dim // num_heads
        block = ChaosAttentionBlock(dim=dim, ff_mult=2, num_heads=num_heads)
        block._ensure_rope_cache(1, torch.device("cpu"))
        cos = block._rope_cos[:1]
        sin = block._rope_sin[:1]

        x = torch.randn(2, num_heads, 1, head_dim)
        x_rot = ChaosAttentionBlock.apply_rope(x, cos, sin)
        assert torch.allclose(x_rot, x, atol=1e-7), (
            "RoPE is not identity at position 0"
        )

    def test_head_dim_must_be_even_for_rope(self) -> None:
        """Explicit constructor check: odd head_dim must raise."""
        # dim=30, num_heads=5 -> head_dim=6 is even (valid).
        _ = ChaosAttentionBlock(dim=30, ff_mult=2, num_heads=5)
        # dim=30, num_heads=3 -> head_dim=10 is even (valid).
        _ = ChaosAttentionBlock(dim=30, ff_mult=2, num_heads=3)
        # dim=30, num_heads=6 -> head_dim=5 is odd -> must raise.
        with self.assertRaises(ValueError):
            ChaosAttentionBlock(dim=30, ff_mult=2, num_heads=6)

    def test_backward_with_rope_produces_nonzero_grads(self) -> None:
        """Forward + loss + backward end-to-end with RoPE active.

        Backward-pass smoke test: checks that RoPE's ``torch.empty_like``
        re-interleave path does not block gradient flow, and that every
        learnable tensor (Q/K/V fused qkv_proj, out_proj, FF fc, FF proj,
        both RMSNorm weights) receives non-zero, non-NaN gradients.
        """
        torch.manual_seed(4)
        block = ChaosAttentionBlock(dim=32, ff_mult=2, num_heads=4)
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = block(x)
        loss = out.pow(2).mean()
        loss.backward()

        # Input gradient non-zero and non-NaN.
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0
        assert torch.isfinite(x.grad).all()

        # Every learnable parameter gets a live gradient.
        offenders: list[str] = []
        for name, p in block.named_parameters():
            if p.grad is None:
                offenders.append(f"{name}: grad is None")
                continue
            if not torch.isfinite(p.grad).all():
                offenders.append(f"{name}: grad has NaN/inf")
                continue
            if p.grad.abs().sum().item() == 0.0:
                offenders.append(f"{name}: grad is zero")
        assert not offenders, (
            "parameters with missing/zero/NaN grad (with RoPE): "
            + ", ".join(offenders)
        )

        # Explicit check on the named projections the task asks about.
        assert block.qkv_proj.weight.grad.abs().sum().item() > 0
        assert block.out_proj.weight.grad.abs().sum().item() > 0
        assert block.ff.fc.weight.grad.abs().sum().item() > 0
        assert block.ff.proj.weight.grad.abs().sum().item() > 0


class TestNoFlashAttentionImport(unittest.TestCase):
    """Task constraint: ChaosAttentionBlock must not depend on Flash Attention.
    We verify by asserting no flash_attn module is loaded after importing
    chaoscontrol.model.
    """

    def test_no_flash_attn_in_sys_modules_after_import(self) -> None:
        # Force a clean import of the model module.
        if "chaoscontrol.model" in sys.modules:
            del sys.modules["chaoscontrol.model"]
        importlib.import_module("chaoscontrol.model")
        flash_modules = [
            name for name in sys.modules if name.startswith("flash_attn")
        ]
        assert flash_modules == [], (
            f"ChaosAttentionBlock transitively imported flash_attn modules: "
            f"{flash_modules}"
        )


if __name__ == "__main__":
    unittest.main()
