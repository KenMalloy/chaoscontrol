"""Forward/backward parity tests for ChaosStudentLM activation checkpointing.

All tests run on CPU in float32 so bf16 quantization noise does not enter
the parity checks. The checkpoint code path must reproduce the non-checkpoint
path to ~float32 noise on logits and gradients; a larger drift would mean
the checkpoint wrapper is recomputing something different than the original
forward saw.
"""
from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from chaoscontrol.model import ChaosStudentLM


def _make_model(*, seed: int, activation_checkpoint: bool | None) -> ChaosStudentLM:
    torch.manual_seed(seed)
    kwargs = dict(
        vocab_size=64,
        dim=16,
        num_layers=3,
        ff_mult=2,
        a_mode="diag",
        rich_b_mode="none",
        outer_model_dim=0,
    )
    if activation_checkpoint is not None:
        kwargs["activation_checkpoint"] = activation_checkpoint
    return ChaosStudentLM(**kwargs)


def _make_hybrid_model(*, seed: int, activation_checkpoint: bool) -> ChaosStudentLM:
    torch.manual_seed(seed)
    return ChaosStudentLM(
        vocab_size=64,
        dim=16,
        num_layers=3,
        ff_mult=2,
        a_mode="diag",
        rich_b_mode="none",
        outer_model_dim=0,
        local_attn_window=4,
        local_attn_heads=2,
        local_attn_dim=16,
        activation_checkpoint=activation_checkpoint,
    )


def _clone_with_flag(ref: ChaosStudentLM, activation_checkpoint: bool) -> ChaosStudentLM:
    clone = copy.deepcopy(ref)
    clone.activation_checkpoint = activation_checkpoint
    return clone


class TestActivationCheckpointForwardParity(unittest.TestCase):
    def test_forward_parity_train_mode_bs4_seq32_fp32(self) -> None:
        """Train mode + grad enabled: the alt model actually invokes
        ``torch.utils.checkpoint.checkpoint`` (use_ckpt=True branch).
        """
        ref = _make_model(seed=0, activation_checkpoint=False)
        alt = _clone_with_flag(ref, activation_checkpoint=True)

        input_ids = torch.randint(0, 64, (4, 32))

        ref.train()
        alt.train()
        out_ref = ref(input_ids)["logits"]
        out_alt = alt(input_ids)["logits"]

        self.assertTrue(out_alt.requires_grad)
        diff = (out_ref - out_alt).abs().max().item()
        self.assertLess(diff, 1e-6, msg=f"train-mode logits drift: {diff}")

    def test_forward_parity_eval_mode_skip_path(self) -> None:
        """Eval + no_grad: the checkpoint skip path must not drift. This
        is a separate guard from the train-mode parity test because the
        skip condition (``torch.is_grad_enabled() and x.requires_grad``)
        is evaluated per-forward and a bug that flipped it the wrong way
        would show up as eval-time drift before it showed up as a train
        bug.
        """
        ref = _make_model(seed=0, activation_checkpoint=False)
        alt = _clone_with_flag(ref, activation_checkpoint=True)

        input_ids = torch.randint(0, 64, (4, 32))

        ref.eval()
        alt.eval()
        with torch.no_grad():
            out_ref = ref(input_ids)["logits"]
            out_alt = alt(input_ids)["logits"]

        diff = (out_ref - out_alt).abs().max().item()
        self.assertLess(diff, 1e-6, msg=f"eval-mode logits drift: {diff}")

    def test_forward_parity_with_jacobian_stats(self) -> None:
        ref = _make_model(seed=1, activation_checkpoint=False)
        alt = _clone_with_flag(ref, activation_checkpoint=True)

        input_ids = torch.randint(0, 64, (2, 16))

        ref.train()
        alt.train()
        out_ref = ref(input_ids, return_jacobian_stats=True)
        out_alt = alt(input_ids, return_jacobian_stats=True)

        self.assertIn("jacobian_stats", out_ref)
        self.assertIn("jacobian_stats", out_alt)
        self.assertTrue(out_alt["logits"].requires_grad)
        logits_diff = (out_ref["logits"] - out_alt["logits"]).abs().max().item()
        self.assertLess(logits_diff, 1e-5, msg=f"logits drift with stats: {logits_diff}")

    def test_forward_parity_hybrid_block(self) -> None:
        """Train-mode parity check on the ChaosSSMHybridBlock path. Uses the
        checkpoint branch (use_ckpt=True) actively because the hybrid block
        has more internal operations than the pure SSM block and would be
        the first place a checkpoint-recompute determinism drift shows up.
        """
        torch.manual_seed(2)
        ref = _make_hybrid_model(seed=2, activation_checkpoint=False)
        alt = _clone_with_flag(ref, activation_checkpoint=True)

        input_ids = torch.randint(0, 64, (4, 16))

        ref.train()
        alt.train()
        out_ref = ref(input_ids)["logits"]
        out_alt = alt(input_ids)["logits"]

        self.assertTrue(out_alt.requires_grad)
        diff = (out_ref - out_alt).abs().max().item()
        self.assertLess(diff, 1e-5, msg=f"hybrid logits drift: {diff}")


class TestActivationCheckpointGradientParity(unittest.TestCase):
    def test_gradients_match_bs4_seq32_fp32(self) -> None:
        ref = _make_model(seed=3, activation_checkpoint=False)
        alt = _clone_with_flag(ref, activation_checkpoint=True)

        input_ids = torch.randint(0, 64, (4, 32))

        ref.train()
        alt.train()

        loss_ref = ref(input_ids)["logits"].sum()
        loss_ref.backward()

        loss_alt = alt(input_ids)["logits"].sum()
        loss_alt.backward()

        max_diff = 0.0
        for (name_r, p_ref), (name_a, p_alt) in zip(
            ref.named_parameters(), alt.named_parameters(), strict=True
        ):
            self.assertEqual(name_r, name_a)
            self.assertIsNotNone(p_ref.grad, msg=f"ref grad None for {name_r}")
            self.assertIsNotNone(p_alt.grad, msg=f"alt grad None for {name_a}")
            diff = (p_ref.grad - p_alt.grad).abs().max().item()
            if diff > max_diff:
                max_diff = diff

        self.assertLess(max_diff, 1e-5, msg=f"max grad drift: {max_diff}")


class TestActivationCheckpointTrainingParity(unittest.TestCase):
    def _run_fixed_steps(self, model: ChaosStudentLM, steps: int) -> list[float]:
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(9999)
        losses: list[float] = []
        for _ in range(steps):
            ids = torch.randint(0, 64, (2, 16), generator=gen)
            targets = torch.randint(0, 64, (2, 16), generator=gen)
            opt.zero_grad(set_to_none=True)
            logits = model(ids)["logits"]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 64), targets.reshape(-1)
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    def test_five_step_loss_trajectory(self) -> None:
        ref = _make_model(seed=7, activation_checkpoint=False)
        alt = _clone_with_flag(ref, activation_checkpoint=True)

        ref.train()
        alt.train()

        ref_losses = self._run_fixed_steps(ref, steps=5)
        alt_losses = self._run_fixed_steps(alt, steps=5)

        for i, (r, a) in enumerate(zip(ref_losses, alt_losses, strict=True)):
            self.assertLess(
                abs(r - a),
                1e-4,
                msg=f"step {i}: ref={r:.6f} alt={a:.6f} diff={abs(r - a):.2e}",
            )


class TestActivationCheckpointBackwardsCompat(unittest.TestCase):
    def test_default_off_matches_no_kwarg(self) -> None:
        """Building without the kwarg must equal building with ``False``."""
        torch.manual_seed(42)
        no_kwarg = ChaosStudentLM(
            vocab_size=64, dim=16, num_layers=3, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
        )
        torch.manual_seed(42)
        explicit_off = ChaosStudentLM(
            vocab_size=64, dim=16, num_layers=3, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
            activation_checkpoint=False,
        )

        self.assertFalse(no_kwarg.activation_checkpoint)
        self.assertFalse(explicit_off.activation_checkpoint)

        input_ids = torch.randint(0, 64, (2, 16))
        no_kwarg.eval()
        explicit_off.eval()
        with torch.no_grad():
            out_a = no_kwarg(input_ids)["logits"]
            out_b = explicit_off(input_ids)["logits"]

        self.assertTrue(torch.equal(out_a, out_b))


if __name__ == "__main__":
    unittest.main()
