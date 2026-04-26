"""Dump a trained ``SimplexPolicy`` to the CSWG v3 binary format.

CSWG v3 is self-describing: a small fixed header points at a JSON manifest,
and the manifest names every tensor with shape, dtype, offset, and byte
count. The C++/runner side no longer needs a duplicated fixed tensor order
to know whether an artifact contains the base simplex policy only or the
optional residual HxH branch.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


MAGIC = b"CSWG"
VERSION = 3
DTYPE_FP16 = 1
DTYPE_NAME = {DTYPE_FP16: "fp16"}
HEADER_STRUCT = struct.Struct("<4sIIII")
HEADER_SIZE = HEADER_STRUCT.size

TENSOR_NAMES: tuple[str, ...] = (
    "W_vp",
    "b_vp",
    "W_lh",
    "b_lh",
    "W_sb",
    "alpha",
    "temperature",
    "bucket_embed",
    "lambda_hxh",
    "W_q",
    "W_k",
    "W_v",
    "W_o",
    "W_e",
)


def _state_tensor(state: dict[str, Any], name: str) -> torch.Tensor:
    if name not in state:
        raise KeyError(
            f"checkpoint missing tensor {name!r}; have {sorted(state.keys())}"
        )
    return torch.as_tensor(
        state[name], dtype=torch.float32, device="cpu"
    ).contiguous()


def _dims_from_cfg(cfg: Any) -> dict[str, int]:
    return {
        "n_vertices": int(cfg.n_vertices),
        "k_v": int(cfg.k_v),
        "k_e": int(cfg.k_e),
        "k_s": int(cfg.k_s),
        "h": int(cfg.h),
        "n_heads": int(getattr(cfg, "n_heads", 0)),
        "n_buckets": int(cfg.n_buckets),
        "bucket_embed_dim": int(cfg.bucket_embed_dim),
    }


def dump_model_to_cswg_v3(model: Any, output_path: str | Path) -> dict[str, Any]:
    """Write a self-describing fp16 CSWG v3 artifact and return its manifest."""
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise ValueError("model must have a .cfg attribute (PretrainConfig)")

    dims = _dims_from_cfg(cfg)
    state = model.state_dict()
    payload_parts: list[bytes] = []
    tensor_entries: list[dict[str, Any]] = []
    payload_offset = 0
    max_drift = 0.0
    for name in TENSOR_NAMES:
        tensor = _state_tensor(state, name)
        half = tensor.to(torch.float16).contiguous()
        widened = half.to(torch.float32)
        if tensor.numel() > 0:
            max_drift = max(
                max_drift,
                float((tensor - widened).abs().max().item()),
            )
        raw = half.cpu().numpy().tobytes(order="C")
        tensor_entries.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": "fp16",
            "offset": payload_offset,
            "nbytes": len(raw),
        })
        payload_parts.append(raw)
        payload_offset += len(raw)

    manifest = {
        "magic": MAGIC.decode("ascii"),
        "version": VERSION,
        "dtype_code": DTYPE_FP16,
        "dtype": DTYPE_NAME[DTYPE_FP16],
        "dims": dims,
        "tensors": tensor_entries,
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(
            HEADER_STRUCT.pack(
                MAGIC,
                VERSION,
                DTYPE_FP16,
                len(manifest_bytes),
                0,
            )
        )
        f.write(manifest_bytes)
        for part in payload_parts:
            f.write(part)

    manifest.update({
        "path": str(out),
        "header_bytes": HEADER_SIZE,
        "manifest_bytes": len(manifest_bytes),
        "payload_bytes": payload_offset,
        "file_bytes": HEADER_SIZE + len(manifest_bytes) + payload_offset,
        "fp32_vs_fp16_max_abs_drift": max_drift,
    })
    return manifest


def load_cswg_v3(path: str | Path) -> dict[str, Any]:
    """Parse a CSWG v3 binary into ``{"header", "manifest", "tensors"}``."""
    raw = Path(path).read_bytes()
    if len(raw) < HEADER_SIZE:
        raise ValueError(
            f"CSWG file too small: {len(raw)} bytes, need {HEADER_SIZE}"
        )
    magic, version, dtype_code, manifest_nbytes, reserved = (
        HEADER_STRUCT.unpack(raw[:HEADER_SIZE])
    )
    if magic != MAGIC:
        raise ValueError(f"bad CSWG magic: expected {MAGIC!r}, got {magic!r}")
    if version != VERSION:
        raise ValueError(
            f"unsupported CSWG version: expected {VERSION}, got {version}"
        )
    if dtype_code != DTYPE_FP16:
        raise ValueError(
            f"unsupported dtype code: expected {DTYPE_FP16} (fp16), got {dtype_code}"
        )
    manifest_end = HEADER_SIZE + int(manifest_nbytes)
    if manifest_end > len(raw):
        raise ValueError(
            f"CSWG manifest truncated: header says {manifest_nbytes} bytes, "
            f"file has {len(raw) - HEADER_SIZE}"
        )
    manifest = json.loads(raw[HEADER_SIZE:manifest_end].decode("utf-8"))
    if int(manifest.get("version", -1)) != VERSION:
        raise ValueError("CSWG manifest version does not match header")
    payload = raw[manifest_end:]
    tensors: dict[str, torch.Tensor] = {}
    for entry in manifest["tensors"]:
        name = str(entry["name"])
        shape = tuple(int(x) for x in entry["shape"])
        offset = int(entry["offset"])
        nbytes = int(entry["nbytes"])
        end = offset + nbytes
        if end > len(payload):
            raise ValueError(
                f"CSWG payload truncated reading {name!r}: "
                f"need bytes [{offset}, {end}), payload has {len(payload)}"
            )
        flat = np.frombuffer(payload[offset:end], dtype=np.float16).copy()
        tensor = torch.from_numpy(flat.astype(np.float32))
        tensors[name] = tensor.reshape(shape).contiguous()

    expected_end = max(
        (int(e["offset"]) + int(e["nbytes"]) for e in manifest["tensors"]),
        default=0,
    )
    if expected_end != len(payload):
        raise ValueError(
            f"CSWG file has trailing payload bytes: manifest ends at "
            f"{expected_end}, payload_size={len(payload)}"
        )

    header = {
        "magic": magic.decode("ascii"),
        "version": int(version),
        "dtype_code": int(dtype_code),
        "manifest_bytes": int(manifest_nbytes),
        "reserved": int(reserved),
    }
    return {"header": header, "manifest": manifest, "tensors": tensors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to a torch.save() of {'config': cfg.__dict__, "
        "'weights': state_dict}.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.checkpoint is None:
        raise SystemExit(
            "this CLI requires --checkpoint pointing at a torch.save state_dict; "
            "the typical entry point is pretrain_controller.py main()."
        )

    import importlib.util

    pretrain_path = Path(__file__).resolve().parent / "pretrain_controller.py"
    spec = importlib.util.spec_from_file_location("pretrain_controller", pretrain_path)
    assert spec is not None and spec.loader is not None
    pretrain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pretrain)

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = pretrain.PretrainConfig(**blob["config"])
    model = pretrain.SimplexPolicy(cfg)
    model.load_state_dict(blob["weights"])
    manifest = dump_model_to_cswg_v3(model, args.output)
    dims = manifest["dims"]
    print(
        "wrote {path} version={version} dims=(N={n_vertices}, K_v={k_v}, "
        "K_e={k_e}, K_s={k_s}, H={h}, heads={n_heads}) file_bytes={file_bytes} "
        "drift={fp32_vs_fp16_max_abs_drift:.6g}".format(
            path=manifest["path"],
            version=manifest["version"],
            file_bytes=manifest["file_bytes"],
            fp32_vs_fp16_max_abs_drift=manifest["fp32_vs_fp16_max_abs_drift"],
            **dims,
        )
    )


if __name__ == "__main__":
    main()
