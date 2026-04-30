"""Peak-VRAM regression test for the chunked LM-head backward path.

The unit tests in ``test_chunked_lm_backward.py`` prove gradient
equivalence. This test proves the *user-facing memory win*: at
bs=1024 / seq=512 / V=16384 / 4L × 256d bf16, the frozen path OOMs
on an 80 GiB H100 (measured 2026-04-16: peak reached 78.13 GiB
before OutOfMemoryError). The chunked-backward path fits with
meaningful headroom.

The 50 GiB ceiling below is not tight — encoder activations at this
regime are ~37 GiB and they dominate total peak. The ceiling's job
is to catch a structural regression: any change that reintroduces
full-logits materialization would add ~17 GiB (full (B,T,V) bf16 =
17.2 GiB) on top of the ~37 GiB encoder base, pushing total above
50 GiB and failing loudly. That's the failure mode worth guarding.

Runs only when CUDA is available, so it's a no-op locally on macOS
and a real assertion every time it executes on the pod.
"""
from __future__ import annotations

import pytest
import torch

from chaoscontrol.model import CareStudentLM
from chaoscontrol.train_ssm import train_ssm_step


# Empirical values from the 2026-04-16 pod run:
#   Old path: OOM at peak 78.13 GiB (>80 GiB H100 capacity)
#   New path: peak 36.78 GiB at bs=1024, 18.43 GiB at bs=512 (linear scaling)
# A 50 GiB ceiling sits ~13 GiB above the observed new-path peak —
# enough headroom for allocator fragmentation drift, tight enough to
# catch a logits.grad regression (+17.2 GiB would push past 50 GiB).
NEW_PATH_PEAK_CEILING_GIB = 50.0

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
    """Empirical proof the chunked path fits where the frozen path OOMs."""

    def test_fits_where_old_path_ooms(self) -> None:
        device = torch.device("cuda")

        torch.manual_seed(2026)
        model = CareStudentLM(**SUBMISSION_REGIME).to(device=device, dtype=torch.bfloat16)
        model.train()

        # Deterministic inputs matching the OOM regime exactly.
        g = torch.Generator(device="cpu").manual_seed(0)
        inputs = torch.randint(
            0, SUBMISSION_REGIME["vocab_size"], (BS, SEQ), generator=g,
        ).to(device=device)
        targets = torch.randint(
            0, SUBMISSION_REGIME["vocab_size"], (BS, SEQ), generator=g,
        ).to(device=device)

        # Warm-up step so CUDA allocator state and kernel caches don't
        # inflate the measurement we care about.
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
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        assert peak_gib < NEW_PATH_PEAK_CEILING_GIB, (
            f"train_ssm_step peak VRAM {peak_gib:.2f} GiB exceeded "
            f"{NEW_PATH_PEAK_CEILING_GIB} GiB ceiling at bs={BS}/seq={SEQ}/"
            f"V={SUBMISSION_REGIME['vocab_size']}/chunk={CHUNK}. At this "
            f"regime encoder activations are ~37 GiB, so the chunked "
            f"backward's logits contribution should be ~1-5 GiB — total "
            f"stays around ~37-42 GiB under the fix. Anything above 50 GiB "
            f"means the chunked backward is no longer suppressing the "
            f"full (B,T,V) logits.grad and the fix has regressed. "
            f"Check chunked_lm_head_backward for tensors retained across "
            f"chunks, or for a caller bypassing the detach boundary."
        )
