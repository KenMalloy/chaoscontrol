"""LZMA artifact packaging for int6-quantized state dicts.

Mirrors the final packaging path in the local SOTA transformer record
(baselines/parameter_golf/sota/train_gpt.py, around line 2042):

    torch.save({"w": weights, "m": meta}, buf)
    blob = lzma.compress(buf.getvalue(), preset=9)

LZMA preset 9 is the slowest / strongest setting; the int6 code alphabet
is narrow enough that the extra compression headroom is worth the wall
time. The container dict ``{"w": ..., "m": ...}`` keeps the quantized
result alongside the meta map emitted by
``GPTQQuantizer.quantize_state_dict`` so that ``unpack_int6_lzma`` can
feed it straight back into ``GPTQQuantizer.dequantize_state_dict``.
"""
from __future__ import annotations

import io
import lzma
from typing import Any

import torch


LZMA_PRESET = 9


def pack_int6_lzma(
    quantized: dict[str, torch.Tensor],
    meta: dict[str, dict],
    *,
    preset: int = LZMA_PRESET,
) -> bytes:
    """Serialize + LZMA-compress a quantized state dict.

    ``quantized`` is the ``result`` returned by
    ``GPTQQuantizer.quantize_state_dict`` — a flat dict where int6 entries
    live under ``<name>.q`` / ``<name>.scale`` and passthroughs live under
    ``<name>``. ``meta`` is the companion map keyed by the original tensor
    name that says how to dequantize each key.

    Returns raw bytes ready to be written to disk (e.g.
    ``final_model.int6.ptz``) or embedded in a larger submission blob.
    """
    buf = io.BytesIO()
    torch.save({"w": quantized, "m": meta}, buf)
    return lzma.compress(buf.getvalue(), preset=preset)


def unpack_int6_lzma(
    blob: bytes,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    """Inverse of ``pack_int6_lzma``.

    Decompresses the blob and returns ``(quantized, meta)`` ready to pass
    to ``GPTQQuantizer.dequantize_state_dict``. ``map_location`` is
    forwarded to ``torch.load``.
    """
    raw = lzma.decompress(blob)
    payload: dict[str, Any] = torch.load(
        io.BytesIO(raw), map_location=map_location, weights_only=False,
    )
    if not isinstance(payload, dict) or "w" not in payload or "m" not in payload:
        raise ValueError("blob is not a valid int6-lzma container")
    return payload["w"], payload["m"]
