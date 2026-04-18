"""Modded-NanoGPT-style lean transformer variant for Exp 21 ablations.

Kept separate from ``baselines.SimpleTransformerLM`` to avoid perturbing
existing callers (which duck-type the latter into ``ChaosStudentLM``'s
``outer_model``/``wernicke``/``semantic_tier`` slots). This module implements
the 4-cell ablation's A/B cell: d_model=256, n_head=4, n_layer=8, ff_mult=4,
RoPE + RMSNorm + ReLU^2 + SDPA (Flash-Attn backend) + QK-norm; no auxiliary
embeddings; untied embed/LM-head.

RoPE math is inlined to keep the variant self-contained — importing
``ChaosAttentionBlock.apply_rope`` would pull ``model.py``'s full SSM import
chain for a five-line helper. The rotation math here is byte-identical to
``src/chaoscontrol/model.py:354`` (interleaved-pair RoFormer form), and the
cos/sin cache build mirrors ``model.py:339-352``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chaoscontrol.core import RMSNorm


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """ReLU^2 activation: (max(0, x))^2. Used inside the MLP."""
    return F.relu(x).square()


def _build_rope_cache(
    head_dim: int, seq_len: int, base: float = 10_000.0
) -> torch.Tensor:
    """Build combined (cos, sin) RoPE cache with shape (2, seq_len, head_dim).

    Math is identical to ``model.py:_ensure_rope_cache`` (lines 339-352):
    inverse-frequency table over the first half_dim, positions [0, seq_len),
    outer-product into angles, then ``repeat_interleave`` by 2 so that each
    (2i, 2i+1) channel pair shares its cos/sin. The first axis stacks
    cos/sin so we register a single buffer.
    """
    half = head_dim // 2
    freqs = torch.arange(0, half, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (2.0 * freqs / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.einsum("t,i->ti", positions, inv_freq)
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
    return torch.stack([cos, sin], dim=0)  # (2, seq_len, head_dim)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotary embedding. Byte-identical to model.py:354.

    ``x`` shape (..., seq, head_dim); ``cos``/``sin`` shape (seq, head_dim)
    with each frequency repeated twice along the last axis.
    """
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_pair = cos[..., 0::2]
    sin_pair = sin[..., 0::2]
    rot_even = x_even * cos_pair - x_odd * sin_pair
    rot_odd = x_odd * cos_pair + x_even * sin_pair
    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


class NanoGPTLeanBlock(nn.Module):
    """Pre-RMSNorm attention + MLP block with RoPE and QK-norm.

    Structure (modded-NanoGPT lean):
        x = x + out_proj( SDPA( qk_norm(rope(q)), qk_norm(rope(k)), v ) )
        x = x + mlp_out( relu^2( mlp_in( rmsnorm(x) ) ) )
    """

    def __init__(self, d_model: int, n_head: int, ffn_mult: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.attn_norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        # QK-norm: per-head RMSNorm over head_dim, applied post-RoPE. Two
        # separate norms (distinct learnable scale for Q vs K) matches
        # modded-NanoGPT and most recent QK-norm variants.
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp_norm = RMSNorm(d_model)
        ffn_dim = ffn_mult * d_model
        self.mlp_in = nn.Linear(d_model, ffn_dim, bias=False)
        self.mlp_out = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        # rope_cache: (2, T, head_dim) — caller slices to current seq length.
        cos = rope_cache[0]
        sin = rope_cache[1]

        # --- Attention ---
        h = self.attn_norm(x)
        B, T, D = h.shape
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, T, 3D) -> per-head: (B, H, T, Dh). We transpose BEFORE RoPE so
        # the ``(T, head_dim)`` cos/sin broadcast cleanly against the
        # ``(B, H, T, head_dim)`` q/k — same call shape as model.py:400-412.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # RoPE on Q and K only (not V). Cache broadcasts over batch/head axes.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # QK-norm over head_dim, post-RoPE (modded-NanoGPT ordering).
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(attn_out)

        # --- MLP ---
        m = self.mlp_norm(x)
        m = self.mlp_in(m)
        m = relu_squared(m)
        x = x + self.mlp_out(m)
        return x


class NanoGPTLeanLM(nn.Module):
    """Lean modded-NanoGPT transformer for Exp 21 ablations.

    Defaults (design spec): d_model=256, n_head=4, n_layer=8, ffn_mult=4,
    max_seq_len=2048. Untied embed/LM-head. Returns the same ``{"logits",
    "hidden"}`` dict as ``SimpleTransformerLM`` and exposes the same
    duck-typed hooks so the shared train/eval path accepts it unchanged.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_head: int = 4,
        n_layer: int = 8,
        ffn_mult: int = 4,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [NanoGPTLeanBlock(d_model, n_head, ffn_mult) for _ in range(n_layer)]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Non-persistent RoPE cache buffer: moves with .to(device) but isn't
        # checkpointed (it's pure function of head_dim/max_seq_len/base).
        head_dim = d_model // n_head
        self.register_buffer(
            "rope_cache",
            _build_rope_cache(head_dim, max_seq_len),
            persistent=False,
        )

        # Duck-typing compatibility with ChaosStudentLM for shared train/eval
        # loop (matches SimpleTransformerLM's hooks).
        self.outer_model = None
        self.wernicke = None
        self.wernicke_balance_weight = 0.0
        self.semantic_tier = None

    def artifact_bytes(self) -> int:
        return int(sum(p.numel() for p in self.parameters()) * 2)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the pre-final_norm hidden state (B, T, D).

        Contract for ``train_ssm_for_budget``: the chunked LM-head backward
        runs ``model.encode()`` once, detaches, then loops chunked
        forward/backward through ``final_norm + lm_head``. Must produce
        the identical tensor to ``forward()["hidden"]`` or the chunked
        path and the frozen forward path diverge.
        """
        T = input_ids.shape[1]
        if T > self.rope_cache.shape[1]:
            head_dim = self.rope_cache.shape[-1]
            new_cache = _build_rope_cache(head_dim, T).to(self.rope_cache.device)
            self.rope_cache = new_cache
        rope = self.rope_cache[:, :T]
        h = self.embed(input_ids)
        for block in self.blocks:
            h = block(h, rope)
        return h

    def forward(self, input_ids: torch.Tensor, *, return_jacobian_stats: bool = False):
        hidden = self.encode(input_ids)
        h = self.final_norm(hidden)
        logits = self.lm_head(h)
        out = {"logits": logits, "hidden": hidden}
        if return_jacobian_stats:
            # Transformer has no SSM Jacobian stats; return zeros to match
            # the SimpleTransformerLM contract.
            out["jacobian_stats"] = {
                "lambda_max": torch.tensor(0.0),
                "sv_log_var": torch.tensor(0.0),
            }
        return out


def build_nanogpt_lean(
    vocab_size: int = 8192,
    d_model: int = 256,
    n_head: int = 4,
    n_layer: int = 8,
    ffn_mult: int = 4,
    max_seq_len: int = 2048,
) -> NanoGPTLeanLM:
    """Factory with design-spec defaults (~10.49M params at V=8192)."""
    return NanoGPTLeanLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        n_layer=n_layer,
        ffn_mult=ffn_mult,
        max_seq_len=max_seq_len,
    )
