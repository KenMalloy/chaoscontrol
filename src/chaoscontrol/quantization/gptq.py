"""GPTQ int6 quantizer with autoregressive self-generated calibration.

Ported from the local SOTA transformer record
(baselines/parameter_golf/sota/train_gpt.py). Covers AR self-generated
calibration, Hessian accumulation via forward hooks on nn.Linear,
Cholesky-based error-compensated quantization, and a percentile-search
fallback. "int6" means a value range of [-31, 31] stored inside torch.int8;
there is no packed 6-bit byte layout — the narrow alphabet is what lets
the downstream LZMA stage (quantization.packaging) shrink the artifact.
"""
from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn


INT6_CLIP_RANGE = 31
DEFAULT_BLOCK_SIZE = 128
PERCENTILE_GRID = (0.9990, 0.9995, 0.9999, 0.99999, 1.0)

LogitFn = Callable[[torch.Tensor], torch.Tensor]


def ar_self_generated_calibration(
    logit_fn: LogitFn,
    *,
    num_seqs: int = 64,
    seq_len: int = 2048,
    vocab_size: int = 1024,
    temperature: float = 0.8,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> list[torch.Tensor]:
    """Sample calibration sequences autoregressively from the model itself.

    Parameter Golf Issue #1017 forbids touching validation data for
    calibration. The SOTA record sidesteps this by seeding with uniformly
    random first tokens and letting the model roll its own continuations —
    the Hessian is then estimated on the model's output distribution at the
    given ``temperature``, which carries no validation-set information.

    ``logit_fn`` must accept ``(batch, t)`` int64 ids and return
    ``(batch, t, vocab)`` logits. Returns a list of ``num_seqs`` tensors
    shaped ``(1, seq_len)`` int64.
    """
    if num_seqs <= 0 or seq_len <= 0:
        raise ValueError("num_seqs and seq_len must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    device = torch.device(device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_start in range(0, num_seqs, batch_size):
            bs = min(batch_size, num_seqs - batch_start)
            tokens = torch.randint(
                0, vocab_size, (bs, 1),
                device=device, generator=rng, dtype=torch.int64,
            )
            for _ in range(seq_len - 1):
                logits = logit_fn(tokens)
                next_logit = logits[:, -1, :]
                probs = torch.softmax(next_logit.float() / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens = torch.cat([tokens, next_tok.to(tokens.dtype)], dim=1)
            for i in range(bs):
                all_tokens.append(tokens[i : i + 1].detach().clone())
    return all_tokens


def collect_hessians(
    model: nn.Module,
    token_seqs: Iterable[torch.Tensor],
    *,
    device: torch.device | str = "cpu",
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Accumulate ``H = X^T X`` per ``nn.Linear`` layer from token sequences.

    Hooks every ``nn.Linear``, runs each sequence through ``model``, and
    returns a dict keyed by ``{module_name}.weight``. Hessians are averaged
    and damped by 1% of the mean diagonal — the standard GPTQ Cholesky
    stabilizer. ``forward_fn`` defaults to ``model(sequence)``; pass a
    lambda if a different entry point is needed.
    """
    device = torch.device(device)
    hessians: dict[str, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        param_name = name + ".weight"
        cols = module.weight.shape[1]
        hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device="cpu")

        def make_hook(pname: str):
            def hook_fn(_module, inputs, _output):
                x = inputs[0].detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                hessians[pname] += (x.T @ x).cpu()
            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(param_name)))

    try:
        model.eval()
        with torch.inference_mode():
            num_seqs = 0
            for seq in token_seqs:
                seq = seq.to(device)
                if forward_fn is None:
                    model(seq)
                else:
                    forward_fn(model, seq)
                num_seqs += 1
    finally:
        for h in hooks:
            h.remove()

    if num_seqs == 0:
        raise ValueError("token_seqs was empty; cannot build Hessian")
    for name in list(hessians.keys()):
        H = hessians[name]
        H /= num_seqs
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    return hessians


def quantize_int6_percentile(
    tensor: torch.Tensor, clip_range: int = INT6_CLIP_RANGE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Percentile-search int6 fallback for 1D params / Hessian-less cases."""
    t32 = tensor.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float("inf")
        for pct in PERCENTILE_GRID:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(
                torch.round(t32 / s.float()[:, None]), -clip_range, clip_range,
            ).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(
        amax / clip_range if amax > 0 else 1.0, dtype=torch.float16,
    )
    q = torch.clamp(
        torch.round(t32 / scale.float()), -clip_range, clip_range,
    ).to(torch.int8)
    return q, scale


def quantize_int6_gptq(
    weight: torch.Tensor,
    hessian: torch.Tensor | None = None,
    clip_range: int = INT6_CLIP_RANGE,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hessian-aware int6 quantization with Cholesky error compensation.

    For each column in a permutation of ``weight`` (largest Hessian diagonal
    first), quantize to int6, compute the residual, and propagate a
    correction across the remaining columns using the upper-triangular
    Cholesky factor of ``H^-1``. The 5-point percentile search runs the full
    block loop per clip and keeps the best-MSE result. Falls back to
    ``quantize_int6_percentile`` when ``weight`` is 1D or ``hessian`` is
    None. Returns ``(int8 code, float16 per-row scale)``.
    """
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return quantize_int6_percentile(t32, clip_range)

    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    idx = torch.arange(cols)
    H[idx, idx] += damp

    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]

    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    best_q: torch.Tensor | None = None
    best_scale: torch.Tensor | None = None
    best_err = float("inf")

    for pct in PERCENTILE_GRID:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()

        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()

        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int8)
            Err1 = torch.zeros(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = torch.clamp(
                    torch.round(w / sf), -clip_range, clip_range,
                ).to(torch.int8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse

    assert best_q is not None and best_scale is not None
    return best_q[:, inv_perm], best_scale


def _dtype_from_str(dtype_str: str | None) -> torch.dtype | None:
    if not dtype_str:
        return None
    return getattr(torch, dtype_str.removeprefix("torch."), None)


class GPTQQuantizer:
    """Apply GPTQ int6 quantization to every ``nn.Linear`` weight in a model.

    Holds calibration artifacts so callers can inspect or reuse them. The
    quantized dict produced by ``quantize_state_dict`` is the exact shape
    that ``chaoscontrol.quantization.packaging.pack_int6_lzma`` expects.
    """

    def __init__(
        self,
        clip_range: int = INT6_CLIP_RANGE,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        self.clip_range = clip_range
        self.block_size = block_size
        self.hessians: dict[str, torch.Tensor] = {}

    def calibrate(
        self,
        model: nn.Module,
        token_seqs: Iterable[torch.Tensor],
        *,
        device: torch.device | str = "cpu",
        forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        self.hessians = collect_hessians(
            model, token_seqs, device=device, forward_fn=forward_fn,
        )
        return self.hessians

    def quantize_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        int6_param_names: set[str] | None = None,
        min_numel: int = 65536,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
        """Quantize eligible float tensors to int6; pass through the rest.

        ``int6_param_names=None`` means "every float tensor above
        ``min_numel``". Small tensors keep fp16 because the per-row scale
        overhead isn't worth it. Returns ``(result, meta)`` — ``result``
        has ``<name>`` for passthroughs or ``<name>.q`` + ``<name>.scale``
        for int6 entries.
        """
        result: dict[str, torch.Tensor] = {}
        meta: dict[str, dict] = {}
        for name, tensor in state_dict.items():
            t = tensor.detach().cpu().contiguous()
            if not t.is_floating_point():
                result[name] = t
                meta[name] = {"type": "passthrough", "dtype": str(t.dtype)}
                continue
            eligible = t.numel() > min_numel and (
                int6_param_names is None or name in int6_param_names
            )
            if not eligible:
                result[name] = t.to(torch.float16)
                meta[name] = {"type": "passthrough_fp16", "dtype": str(tensor.dtype)}
                continue
            H = self.hessians.get(name)
            if H is not None and t.ndim == 2:
                q, s = quantize_int6_gptq(
                    t, hessian=H,
                    clip_range=self.clip_range, block_size=self.block_size,
                )
            else:
                q, s = quantize_int6_percentile(t, clip_range=self.clip_range)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {
                "type": "int6", "dtype": str(tensor.dtype), "shape": list(t.shape),
            }
        return result, meta

    @staticmethod
    def dequantize_state_dict(
        result: dict[str, torch.Tensor],
        meta: dict[str, dict],
    ) -> dict[str, torch.Tensor]:
        """Inverse of ``quantize_state_dict``. Returns a float state dict.

        Stateless — usable without instantiating a ``GPTQQuantizer`` via
        the module-level ``dequantize_state_dict`` alias.
        """
        return dequantize_state_dict(result, meta)


def dequantize_state_dict(
    result: dict[str, torch.Tensor],
    meta: dict[str, dict],
) -> dict[str, torch.Tensor]:
    """Reconstruct a float state dict from an int6-quantized payload.

    Matches the layout produced by ``GPTQQuantizer.quantize_state_dict``
    (and therefore ``unpack_int6_lzma``). Passthroughs are returned in
    their original dtype when recoverable from ``meta``.
    """
    out: dict[str, torch.Tensor] = {}
    for name, info in meta.items():
        kind = info["type"]
        orig_dtype = _dtype_from_str(info.get("dtype"))
        if kind == "passthrough":
            out[name] = result[name]
            continue
        if kind == "passthrough_fp16":
            t = result[name]
            out[name] = t.to(orig_dtype) if orig_dtype is not None else t
            continue
        if kind != "int6":
            raise ValueError(f"unknown meta type {kind!r} for key {name!r}")
        q = result[name + ".q"]
        s = result[name + ".scale"]
        if s.ndim > 0:
            view_shape = (q.shape[0],) + (1,) * (q.ndim - 1)
            deq = q.float() * s.float().view(view_shape)
        else:
            deq = q.float() * float(s.item())
        out[name] = deq.to(orig_dtype) if orig_dtype is not None else deq
    return out
