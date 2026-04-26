"""Dump a trained ``SimplexPolicy`` to the CSWG v2 binary format.

CSWG v2 layout, little-endian. 32-byte header followed by an fp16
tensor payload in fixed order:

    offset  size  field
    0       4     magic           = b"CSWG"
    4       4     version         = 2
    8       4     n_vertices      = 16  (N)
    12      4     k_v             = 16
    16      4     k_e             = 1
    20      4     k_s             = 4
    24      4     h               = 32
    28      2     dtype           = 1 (fp16)
    30      2     reserved        = 0

    32 ..   payload (fp16 little-endian, in TENSOR_ORDER)

Total payload (V1): W_vp(512) + b_vp(32) + W_lh(32) + b_lh(1)
+ W_sb(4) + alpha(1) + temperature(1) + bucket_embed(64)
= 647 fp16 = 1294 bytes. File total = 32 + 1294 = 1326 bytes.

The C++ loader for v2 lives in S2; this module only writes the binary
and exposes the parser used by S4 round-trip tests.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


MAGIC = b"CSWG"
VERSION = 2
DTYPE_FP16 = 1
# Header: 4-byte magic + 6 uint32 + 2 uint16 = 32 bytes.
HEADER_STRUCT = struct.Struct("<4sIIIIIIHH")
HEADER_SIZE = HEADER_STRUCT.size
assert HEADER_SIZE == 32, f"header struct must be 32 bytes, got {HEADER_SIZE}"

# Tensor order is part of the on-wire contract: changing it bumps the
# CSWG version. Each entry is (state_dict_key, expected_shape_fn).
# expected_shape_fn takes the dimension dict (see _dim_dict) and returns
# the tuple shape; this lets the parser validate against the header
# without hard-coding scalars vs vectors.
TENSOR_ORDER: tuple[tuple[str, str], ...] = (
    ("W_vp", "(K_v, H)"),
    ("b_vp", "(H,)"),
    ("W_lh", "(H,)"),
    ("b_lh", "()"),
    ("W_sb", "(K_s,)"),
    ("alpha", "()"),
    ("temperature", "()"),
    ("bucket_embed", "(N_buckets, BucketEmbedDim)"),
)
# n_buckets and bucket_embed_dim are baked into V1 (8 x 8). They are
# not in the CSWG v2 header so the loader must agree with this module.
N_BUCKETS = 8
BUCKET_EMBED_DIM = 8


def _expected_shape(name: str, dims: dict[str, int]) -> tuple[int, ...]:
    K_v = dims["k_v"]
    K_s = dims["k_s"]
    H = dims["h"]
    return {
        "W_vp": (K_v, H),
        "b_vp": (H,),
        "W_lh": (H,),
        "b_lh": (),
        "W_sb": (K_s,),
        "alpha": (),
        "temperature": (),
        "bucket_embed": (N_BUCKETS, BUCKET_EMBED_DIM),
    }[name]


def _state_tensor(state: dict[str, Any], name: str) -> torch.Tensor:
    if name not in state:
        raise KeyError(
            f"checkpoint missing tensor {name!r}; have {sorted(state.keys())}"
        )
    return torch.as_tensor(state[name], dtype=torch.float32, device="cpu").contiguous()


def dump_model_to_cswg_v2(model: Any, output_path: str | Path) -> dict[str, Any]:
    """Write the simplex policy weights to a CSWG v2 binary.

    ``model`` must be a ``SimplexPolicy`` (or any object exposing
    ``cfg`` with the architectural dims and a ``state_dict()`` whose
    keys match ``TENSOR_ORDER``). Returns a manifest dict.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise ValueError("model must have a .cfg attribute (PretrainConfig)")
    dims = {
        "n_vertices": int(cfg.n_vertices),
        "k_v": int(cfg.k_v),
        "k_e": int(cfg.k_e),
        "k_s": int(cfg.k_s),
        "h": int(cfg.h),
    }
    if (cfg.n_buckets, cfg.bucket_embed_dim) != (N_BUCKETS, BUCKET_EMBED_DIM):
        raise ValueError(
            "CSWG v2 bakes bucket_embed at (8, 8); "
            f"got ({cfg.n_buckets}, {cfg.bucket_embed_dim})"
        )

    state = model.state_dict()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = 0
    max_drift = 0.0
    with out.open("wb") as f:
        f.write(
            HEADER_STRUCT.pack(
                MAGIC,
                VERSION,
                dims["n_vertices"],
                dims["k_v"],
                dims["k_e"],
                dims["k_s"],
                dims["h"],
                DTYPE_FP16,
                0,           # reserved
            )
        )
        for name, _ in TENSOR_ORDER:
            tensor = _state_tensor(state, name)
            expected = _expected_shape(name, dims)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"tensor {name!r} shape {tuple(tensor.shape)} != expected {expected}"
                )
            half = tensor.to(torch.float16).contiguous()
            widened = half.to(torch.float32)
            max_drift = max(max_drift, float((tensor - widened).abs().max().item()))
            buf = half.cpu().numpy().tobytes(order="C")
            f.write(buf)
            payload_bytes += len(buf)

    return {
        "path": str(out),
        "magic": MAGIC.decode("ascii"),
        "version": VERSION,
        "n_vertices": dims["n_vertices"],
        "k_v": dims["k_v"],
        "k_e": dims["k_e"],
        "k_s": dims["k_s"],
        "h": dims["h"],
        "dtype_code": DTYPE_FP16,
        "tensor_order": TENSOR_ORDER,
        "header_bytes": HEADER_SIZE,
        "payload_bytes": payload_bytes,
        "file_bytes": HEADER_SIZE + payload_bytes,
        "fp32_vs_fp16_max_abs_drift": max_drift,
    }


def load_cswg_v2(path: str | Path) -> dict[str, Any]:
    """Parse a CSWG v2 binary. Returns ``{"header": dict, "tensors": dict}``.

    Raises ``ValueError`` with a clear message on bad magic or wrong
    version (in particular, v1 payloads are rejected here -- the C++
    side has its own check).
    """
    raw = Path(path).read_bytes()
    if len(raw) < HEADER_SIZE:
        raise ValueError(
            f"CSWG file too small: {len(raw)} bytes, need at least {HEADER_SIZE}"
        )
    (
        magic,
        version,
        n_vertices,
        k_v,
        k_e,
        k_s,
        h,
        dtype_code,
        reserved,
    ) = HEADER_STRUCT.unpack(raw[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(
            f"bad CSWG magic: expected {MAGIC!r}, got {magic!r}"
        )
    if version != VERSION:
        raise ValueError(
            f"unsupported CSWG version: expected {VERSION}, got {version} "
            f"(v1 layout is rejected; rebuild with this dumper)"
        )
    if dtype_code != DTYPE_FP16:
        raise ValueError(
            f"unsupported dtype code: expected {DTYPE_FP16} (fp16), got {dtype_code}"
        )
    dims = {
        "n_vertices": int(n_vertices),
        "k_v": int(k_v),
        "k_e": int(k_e),
        "k_s": int(k_s),
        "h": int(h),
        "dtype_code": int(dtype_code),
        "reserved": int(reserved),
    }

    cursor = HEADER_SIZE
    tensors: dict[str, torch.Tensor] = {}
    for name, _ in TENSOR_ORDER:
        shape = _expected_shape(name, dims)
        n_elems = 1
        for d in shape:
            n_elems *= d
        n_bytes = n_elems * 2  # fp16
        end = cursor + n_bytes
        if end > len(raw):
            raise ValueError(
                f"CSWG payload truncated reading {name!r}: "
                f"need {n_bytes} bytes at offset {cursor}, file ends at {len(raw)}"
            )
        flat = np.frombuffer(raw[cursor:end], dtype=np.float16).copy()
        cursor = end
        if shape == ():
            tensors[name] = torch.from_numpy(flat.astype(np.float32))[0].clone()
        else:
            tensors[name] = (
                torch.from_numpy(flat.astype(np.float32)).reshape(shape).contiguous()
            )

    if cursor != len(raw):
        raise ValueError(
            f"CSWG file has trailing bytes: cursor={cursor}, file_size={len(raw)}"
        )

    return {"header": dims, "tensors": tensors}


# Back-compat shim: the top-level pretrain script imports this name.
def dump_checkpoint_to_cpp(*args, **kwargs):
    raise RuntimeError(
        "dump_checkpoint_to_cpp is the v1 entry point. The simplex "
        "controller uses CSWG v2; call dump_model_to_cswg_v2 instead."
    )


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

    # Local import; importing pretrain_controller here would create a
    # cycle when pretrain_controller imports the dumper.
    import importlib.util
    pretrain_path = Path(__file__).resolve().parent / "pretrain_controller.py"
    spec = importlib.util.spec_from_file_location("pretrain_controller", pretrain_path)
    assert spec is not None and spec.loader is not None
    pretrain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pretrain)

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = blob["config"]
    cfg = pretrain.PretrainConfig(**cfg_dict)
    model = pretrain.SimplexPolicy(cfg)
    model.load_state_dict(blob["weights"])
    manifest = dump_model_to_cswg_v2(model, args.output)
    print(
        "wrote {path} version={version} dims=(N={n_vertices}, K_v={k_v}, "
        "K_e={k_e}, K_s={k_s}, H={h}) file_bytes={file_bytes} "
        "drift={fp32_vs_fp16_max_abs_drift:.6g}".format(**manifest)
    )


if __name__ == "__main__":
    main()
