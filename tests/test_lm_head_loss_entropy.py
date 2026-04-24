"""Smoke tests for the per-token-entropy LM-head forward API.

These checks only verify that the Python symbol exists and has the expected
signature. The build + numerical equivalence check against a reference
`H[softmax(logits)]` runs on the CUDA pod in Stage D.4; this file is the
macOS-side guard that catches import/API regressions without CUDA.
"""
from __future__ import annotations

import inspect


def test_fused_lm_head_forward_with_ce_entropy_exists_and_has_expected_signature():
    from chaoscontrol.kernels._lm_head_loss import (
        fused_lm_head_forward_with_ce_entropy,
    )

    sig = inspect.signature(fused_lm_head_forward_with_ce_entropy)
    params = sig.parameters
    assert "x" in params
    assert "weight" in params
    assert "target" in params
    assert "tile_size" in params
    # Return type annotation should be a 4-tuple.
    assert (
        "tuple" in str(sig.return_annotation).lower()
        or sig.return_annotation.__class__.__name__ == "_GenericAlias"
    )
