"""Dump Phase D4 PyTorch controller pretrain weights to the C++ CSWG format.

Binary layout, little-endian:

    magic:      4 bytes, ``b"CSWG"``
    version:    uint32, currently 1
    n_layers:   uint32
    d_global:   uint32
    d_slot:     uint32, currently the policy-head slot count
    feature_dim:uint32, event-feature width consumed by ``in_proj``
    dtype:      uint32, currently 1 = fp16
    tensors:    raw fp16 payloads in ``TENSOR_ORDER``

The payload stores fp16 values so the C++ side can load a compact,
pickle-free bootstrap artifact. The current C++ reference loader widens
those values back to fp32 tensors for validation and reference execution.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any

import torch


MAGIC = b"CSWG"
VERSION = 1
DTYPE_FP16 = 1
HEADER_STRUCT = struct.Struct("<4sIIIIII")

TENSOR_ORDER = (
    "trunk.in_proj.weight",
    "trunk.in_proj.bias",
    "trunk.decay",
    "trunk.w_in",
    "trunk.w_out",
    "trunk.bias",
    "policy_head.weight",
    "policy_head.bias",
    "value_head.weight",
    "value_head.bias",
)


def dump_checkpoint_to_cpp(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    dtype: str = "fp16",
) -> dict[str, Any]:
    """Write a D4 pretrain checkpoint to a CSWG C++ binary."""

    if dtype != "fp16":
        raise ValueError(f"only dtype='fp16' is supported; got {dtype!r}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("weights", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a state dict or a 'weights' dict")

    tensors = _validate_and_collect_state(state)
    n_layers, d_global, feature_dim, d_slot = _infer_dimensions(tensors)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    max_param_drift = 0.0
    total_payload_bytes = 0
    with out.open("wb") as f:
        f.write(
            HEADER_STRUCT.pack(
                MAGIC,
                VERSION,
                n_layers,
                d_global,
                d_slot,
                feature_dim,
                DTYPE_FP16,
            )
        )
        for name in TENSOR_ORDER:
            original = tensors[name]
            half = original.to(torch.float16).contiguous()
            widened = half.to(torch.float32)
            max_param_drift = max(
                max_param_drift,
                float((original - widened).abs().max().item()),
            )
            payload = half.numpy().tobytes(order="C")
            f.write(payload)
            total_payload_bytes += len(payload)

    return {
        "path": str(out),
        "magic": MAGIC.decode("ascii"),
        "version": VERSION,
        "n_layers": n_layers,
        "d_global": d_global,
        "d_slot": d_slot,
        "feature_dim": feature_dim,
        "dtype": dtype,
        "dtype_code": DTYPE_FP16,
        "tensor_order": TENSOR_ORDER,
        "payload_bytes": total_payload_bytes,
        "fp32_vs_fp16_max_abs_drift": max_param_drift,
    }


def _validate_and_collect_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    missing = [name for name in TENSOR_ORDER if name not in state]
    if missing:
        raise KeyError(f"checkpoint missing controller tensor(s): {missing}")
    out: dict[str, torch.Tensor] = {}
    for name in TENSOR_ORDER:
        value = torch.as_tensor(state[name], dtype=torch.float32, device="cpu")
        out[name] = value.contiguous()
    return out


def _infer_dimensions(tensors: dict[str, torch.Tensor]) -> tuple[int, int, int, int]:
    in_proj_weight = tensors["trunk.in_proj.weight"]
    in_proj_bias = tensors["trunk.in_proj.bias"]
    decay = tensors["trunk.decay"]
    w_in = tensors["trunk.w_in"]
    w_out = tensors["trunk.w_out"]
    trunk_bias = tensors["trunk.bias"]
    policy_weight = tensors["policy_head.weight"]
    policy_bias = tensors["policy_head.bias"]
    value_weight = tensors["value_head.weight"]
    value_bias = tensors["value_head.bias"]

    if in_proj_weight.dim() != 2:
        raise ValueError("trunk.in_proj.weight must have shape [D_global, F]")
    d_global = int(in_proj_weight.shape[0])
    feature_dim = int(in_proj_weight.shape[1])
    if tuple(in_proj_bias.shape) != (d_global,):
        raise ValueError("trunk.in_proj.bias must have shape [D_global]")
    if decay.dim() != 2 or int(decay.shape[1]) != d_global:
        raise ValueError("trunk.decay must have shape [n_layers, D_global]")
    n_layers = int(decay.shape[0])
    expected_layer_mat = (n_layers, d_global, d_global)
    if tuple(w_in.shape) != expected_layer_mat:
        raise ValueError("trunk.w_in must have shape [n_layers, D_global, D_global]")
    if tuple(w_out.shape) != expected_layer_mat:
        raise ValueError("trunk.w_out must have shape [n_layers, D_global, D_global]")
    if tuple(trunk_bias.shape) != (n_layers, d_global):
        raise ValueError("trunk.bias must have shape [n_layers, D_global]")
    if policy_weight.dim() != 2 or int(policy_weight.shape[1]) != d_global:
        raise ValueError("policy_head.weight must have shape [D_slot, D_global]")
    d_slot = int(policy_weight.shape[0])
    if tuple(policy_bias.shape) != (d_slot,):
        raise ValueError("policy_head.bias must have shape [D_slot]")
    if tuple(value_weight.shape) != (1, d_global):
        raise ValueError("value_head.weight must have shape [1, D_global]")
    if tuple(value_bias.shape) != (1,):
        raise ValueError("value_head.bias must have shape [1]")
    return n_layers, d_global, feature_dim, d_slot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = dump_checkpoint_to_cpp(args.checkpoint, args.output)
    print(
        "wrote {path} n_layers={n_layers} d_global={d_global} "
        "d_slot={d_slot} feature_dim={feature_dim} dtype={dtype} "
        "payload_bytes={payload_bytes} fp32_vs_fp16_max_abs_drift={drift:.8g}".format(
            drift=manifest["fp32_vs_fp16_max_abs_drift"],
            **manifest,
        )
    )


if __name__ == "__main__":
    main()
