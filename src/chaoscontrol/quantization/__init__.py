"""GPTQ int6 quantization + LZMA packaging lifted from the SOTA transformer.

Public API:

    GPTQQuantizer                 -- calibrate + quantize + dequantize a model
    ar_self_generated_calibration -- rule-compliant AR calibration sampler
    collect_hessians              -- X^T X accumulator over nn.Linear layers
    quantize_int6_gptq            -- Hessian-aware per-tensor GPTQ
    quantize_int6_percentile      -- percentile-search fallback
    pack_int6_lzma                -- serialize + LZMA-compress to bytes
    unpack_int6_lzma              -- inverse of pack_int6_lzma
"""
from chaoscontrol.quantization.gptq import (
    DEFAULT_BLOCK_SIZE,
    INT6_CLIP_RANGE,
    PERCENTILE_GRID,
    GPTQQuantizer,
    ar_self_generated_calibration,
    collect_hessians,
    dequantize_state_dict,
    quantize_int6_gptq,
    quantize_int6_percentile,
)
from chaoscontrol.quantization.packaging import (
    LZMA_PRESET,
    pack_int6_lzma,
    unpack_int6_lzma,
)

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "GPTQQuantizer",
    "INT6_CLIP_RANGE",
    "LZMA_PRESET",
    "PERCENTILE_GRID",
    "ar_self_generated_calibration",
    "collect_hessians",
    "dequantize_state_dict",
    "pack_int6_lzma",
    "quantize_int6_gptq",
    "quantize_int6_percentile",
    "unpack_int6_lzma",
]
