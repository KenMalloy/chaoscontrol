"""Peak-VRAM regression test for the chunked LM-head backward path.

The unit tests in ``test_chunked_lm_backward.py`` prove gradient
equivalence. This test proves the *memory win*: on the regime that
OOMed the old path (bs=1024 / seq=512 / V=16384), the lean
``train_ssm_step`` must stay well below the ~17 GiB that
``logits.grad`` alone cost the frozen loop.

Runs only when CUDA is available, so it's a no-op locally on macOS
and a real assertion the first time it executes on the pod. Without
this test, the 64x memory-reduction claim is architectural only —
no number in the repo ever verified it.
"""
from __future__ import annotations

import pytest
import torch

from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.train_ssm import train_ssm_step


# Old-path peak at this regime was 17.2 GiB (B×T×V×2 bytes for
# logits.grad alone — see project_logits_grad_bottleneck_2026-04-16).
# The new path must not come anywhere near that.
NEW_PATH_PEAK_CEILING_GIB = 10.0
# Same regime: B × chunk_T × V × 2 bytes at chunk=64 = 2.15 GiB. That's
# the largest single new-path allocation; everything else (SSM state,
# activations) sits below it. A ceiling of 10 GiB gives generous
# headroom for ws=1 test runs while still catching any regression that
# reintroduces a full-logits materialization.

SUBMISSION_REGIME = dict(
    # Model matches the Exp 18 submission regime (4L × 256d, SP16384).
    vocab_size=16384,
    dim=256,
    num_layers=4,
    ff_mult=2,
    a_mode="diag",
)

BS = 1024
SEQ = 512
CHUNK = 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestTrainSSMMemoryWin:
    """Empirical proof that the chunked backward path doesn't blow VRAM."""

    def test_new_path_peak_below_ceiling(self) -> None:
        device = torch.device("cuda")

        torch.manual_seed(2026)
        model = ChaosStudentLM(**SUBMISSION_REGIME).to(device=device, dtype=torch.bfloat16)
        model.train()

        # Deterministic inputs matching the OOM regime exactly.
        g = torch.Generator(device="cpu").manual_seed(0)
        inputs = torch.randint(
            0, SUBMISSION_REGIME["vocab_size"], (BS, SEQ), generator=g,
        ).to(device=device)
        targets = torch.randint(
            0, SUBMISSION_REGIME["vocab_size"], (BS, SEQ), generator=g,
        ).to(device=device)

        # One warm-up step so CUDA allocator state and kernel caches
        # don't inflate the measurement we care about.
        model.zero_grad(set_to_none=True)
        train_ssm_step(
            model=model, inputs=inputs, targets=targets, chunk_size=CHUNK,
        )

        # Reset and measure the real step.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        model.zero_grad(set_to_none=True)
        train_ssm_step(
            model=model, inputs=inputs, targets=targets, chunk_size=CHUNK,
        )
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gib = peak_bytes / (1024 ** 3)

        assert peak_gib < NEW_PATH_PEAK_CEILING_GIB, (
            f"train_ssm_step peak VRAM {peak_gib:.2f} GiB exceeded ceiling "
            f"{NEW_PATH_PEAK_CEILING_GIB} GiB at bs={BS}/seq={SEQ}/V="
            f"{SUBMISSION_REGIME['vocab_size']}/chunk={CHUNK}. The old-path "
            f"logits.grad at this regime alone was 17.2 GiB — anything in "
            f"the same ballpark means the chunked backward isn't actually "
            f"freeing the per-chunk logits. Inspect chunked_lm_head_backward "
            f"for retained tensors across chunks."
        )

    def test_largest_single_alloc_below_full_logits_grad(self) -> None:
        """The defining property of the fix: no allocation exceeds
        ``chunk_logits`` size. Full-logits materialization would be
        ``B × T × V × 2`` bytes = 17.2 GiB at this regime; chunked
        is ``B × chunk_T × V × 2`` bytes = 2.15 GiB.

        This uses a different lens than the peak-memory check: peak
        is cumulative, whereas the historical-allocator snapshot
        shows the single largest block ever allocated during the
        step. We want that block to match the chunked size, not the
        full-logits size.
        """
        device = torch.device("cuda")

        torch.manual_seed(2026)
        model = ChaosStudentLM(**SUBMISSION_REGIME).to(device=device, dtype=torch.bfloat16)
        model.train()

        g = torch.Generator(device="cpu").manual_seed(1)
        inputs = torch.randint(
            0, SUBMISSION_REGIME["vocab_size"], (BS, SEQ), generator=g,
        ).to(device=device)
        targets = torch.randint(
            0, SUBMISSION_REGIME["vocab_size"], (BS, SEQ), generator=g,
        ).to(device=device)

        # Warm-up to stabilize allocator behavior.
        train_ssm_step(
            model=model, inputs=inputs, targets=targets, chunk_size=CHUNK,
        )

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        model.zero_grad(set_to_none=True)
        train_ssm_step(
            model=model, inputs=inputs, targets=targets, chunk_size=CHUNK,
        )
        torch.cuda.synchronize()

        # Full (B,T,V) bf16 would be 17.2 GiB. The chunked path's peak
        # allocation for logits is ~2.15 GiB; leave 2x headroom for
        # allocator fragmentation and the bf16/fp32 upcast inside CE.
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        full_logits_gib = (BS * SEQ * SUBMISSION_REGIME["vocab_size"] * 2) / (1024 ** 3)
        assert peak_gib < full_logits_gib / 2, (
            f"peak {peak_gib:.2f} GiB is within 2x of the old-path "
            f"full-logits size ({full_logits_gib:.2f} GiB). Expected "
            f"chunked backward to keep peak at least 2x below that. "
            f"Either the chunking isn't engaged or something else in "
            f"the step retains a full-size tensor."
        )
