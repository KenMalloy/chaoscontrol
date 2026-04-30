"""CareSSMBlock, CareSSMHybridBlock, AttentionControlBlock and CareStudentLM — full model assembly.

AttentionControlBlock is a scientific control for Exp 19 Phase 2: it provides a
causal-attention sibling to CareSSMBlock sharing the same block interface so
an apples-to-apples comparison can be run inside a single CareStudentLM via
the ``block_type`` constructor flag. It is not a submission candidate and
does not participate in the hybrid SSM+local-attention path used by
CareSSMHybridBlock — the two are independent choices.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from chaoscontrol.core import RMSNorm, FeedForward, CareSSMCore
from chaoscontrol.local_attn import LocalAttention, RollingKVCache
from chaoscontrol.routing import RichBNN, DistributedB
from chaoscontrol.memory import OuterModel, MultiSlotOuterModel, SemanticTier, BucketPrototypes
from chaoscontrol.posterior import GlobalDelta, BucketDelta, ResidualCache
from chaoscontrol.wernicke import WernickeLayer, HierarchicalWernicke


class CareSSMBlock(nn.Module):
    """Single block: input_norm -> CareSSMCore (with optional rich_b) -> residual -> ff_norm -> FeedForward -> residual."""

    def __init__(
        self,
        dim: int,
        ff_mult: int = 2,
        *,
        a_mode: str = "diag",
        a_full_rank: int = 8,
        a_full_gamma: float = 0.05,
        ssm_delta_rank: int = 0,
        rich_b_mode: str = "none",
        rich_b_bottleneck: int = 32,
        rich_b_num_subnets: int = 4,
        rich_b_settling_steps: int = 2,
    ) -> None:
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.ff_norm = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_mult)
        self.core = CareSSMCore(
            dim,
            a_mode=a_mode,
            a_full_rank=a_full_rank,
            a_full_gamma=a_full_gamma,
            delta_rank=ssm_delta_rank,
        )

        if rich_b_mode == "none":
            self.rich_b: nn.Module | None = None
        elif rich_b_mode == "nn":
            self.rich_b = RichBNN(dim, bottleneck=rich_b_bottleneck)
        elif rich_b_mode in ("hub", "assembly", "hybrid"):
            self.rich_b = DistributedB(
                dim,
                num_subnets=rich_b_num_subnets,
                bottleneck=rich_b_bottleneck,
                topology=rich_b_mode,
                settling_steps=rich_b_settling_steps,
            )
        else:
            raise ValueError(f"unsupported rich_b_mode: {rich_b_mode}")

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-token step through the block.

        Args:
            x: (batch, dim) — single token
            state: (batch, dim) — previous recurrence state for this block's core

        Returns:
            (output, new_state) — output is (batch, dim), new_state is (batch, dim)
        """
        normed = self.input_norm(x)
        y, new_state = self.core.step(normed, state, rich_b=self.rich_b)
        x = x + y
        x = x + self.ff(self.ff_norm(x))
        return x, new_state

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_jacobian_stats: bool = False,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Sequence forward. See CareSSMCore.forward for state-threading semantics.

        The block's residual + FF chain is unchanged regardless of the state
        kwargs — only the recurrence is seeded, and ``final_state`` is plumbed
        out so CareStudentLM can assemble a per-block list.

        Returns:
            x                                     (both False)
            (x, stats)                            (stats=True)
            (x, final_state)                      (final_state=True)
            (x, stats, final_state)               (both True)
        """
        normed = self.input_norm(x)
        result = self.core(
            normed,
            rich_b=self.rich_b,
            return_jacobian_stats=return_jacobian_stats,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )
        # Unpack per the core's documented return tuple shape.
        if return_jacobian_stats and return_final_state:
            y, stats, final_state = result
        elif return_jacobian_stats:
            y, stats = result
            final_state = None
        elif return_final_state:
            y, final_state = result
            stats = None
        else:
            y = result
            stats = None
            final_state = None
        x = x + y
        x = x + self.ff(self.ff_norm(x))
        if return_jacobian_stats and return_final_state:
            return x, stats, final_state
        if return_jacobian_stats:
            return x, stats
        if return_final_state:
            return x, final_state
        return x


class CareSSMHybridBlock(nn.Module):
    """SSM block with local attention sidecar.

    Structure: input_norm -> SSM -> local_attn (gated) -> ff_norm -> FF -> residual.
    The attention sidecar queries a rolling KV cache of recent positions.
    Gate initialized near-zero so the block starts as a pure SSM block.
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 2,
        *,
        a_mode: str = "diag",
        a_full_rank: int = 8,
        a_full_gamma: float = 0.05,
        ssm_delta_rank: int = 0,
        local_attn_window: int = 64,
        local_attn_heads: int = 1,
        local_attn_dim: int = 64,
        local_attn_topk: int = 0,
        local_attn_topk_random: bool = False,
    ) -> None:
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.ff_norm = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_mult)
        self.core = CareSSMCore(
            dim, a_mode=a_mode, a_full_rank=a_full_rank,
            a_full_gamma=a_full_gamma, delta_rank=ssm_delta_rank,
        )
        self.rich_b = None  # compatibility with CareSSMBlock

        # Attention sidecar (local window, top-k selective, or top-k random control)
        self.local_attn_window = local_attn_window
        self.local_attn_dim = local_attn_dim
        self.local_attn_topk = local_attn_topk
        self.local_attn_topk_random = local_attn_topk_random
        self.local_attn = LocalAttention(dim, local_attn_dim, local_attn_heads)
        self.k_proj = nn.Linear(dim, local_attn_dim, bias=False)
        self.v_proj = nn.Linear(dim, local_attn_dim, bias=False)
        self.gate_proj = nn.Linear(dim, 1, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(-4.0))

    def _init_kv_cache(self) -> RollingKVCache:
        return RollingKVCache(self.local_attn_window, self.local_attn_dim)

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        *,
        kv_cache: RollingKVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.input_norm(x)
        y, new_state = self.core.step(normed, state)
        x_ssm_base = x + y
        x_ssm = x_ssm_base

        if kv_cache is not None:
            keys, values, mask = kv_cache.last(self.local_attn_window)
            if mask.any():
                attn_out = self.local_attn(x_ssm, keys, values, mask)
                gate = torch.sigmoid(self.gate_proj(x_ssm) + self.gate_bias)
                x_ssm = x_ssm + gate * attn_out
            # Write K/V from clean trunk output AFTER the read so retrieval
            # remains causal (current token cannot attend to itself), and so
            # the cache contains pure SSM features rather than mixed attention
            # outputs.
            kv_cache.write(self.k_proj(x_ssm_base), self.v_proj(x_ssm_base))

        x_out = x_ssm + self.ff(self.ff_norm(x_ssm))
        return x_out, new_state

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_jacobian_stats: bool = False,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Parallel sequence-level forward.

        SSM runs the compiled diag scan over the full sequence, then local
        attention is computed as a parallel causal sliding-window operation.
        No sequential Python loop — matches pure CareSSMBlock throughput
        plus a small attention matmul overhead.

        The step() method with RollingKVCache is still available for
        autoregressive inference.

        ``initial_state`` and ``return_final_state`` thread through to
        ``CareSSMCore.forward`` identically to ``CareSSMBlock``. The
        local-attention sidecar is stateless across forward calls here
        (KV cache lives only in the step() path via RollingKVCache), so
        the final_state is the SSM core's final state only.
        """
        # 1. SSM: parallel over full sequence (compiled diag scan)
        normed = self.input_norm(x)
        if return_final_state:
            ssm_out, final_state = self.core.forward(
                normed,
                initial_state=initial_state,
                return_final_state=True,
            )
        else:
            ssm_out = self.core.forward(normed, initial_state=initial_state)
            final_state = None
        x_ssm = x + ssm_out  # residual

        # 2. Attention sidecar: local window, top-k selective, or top-k random
        K = self.k_proj(x_ssm)  # (batch, seq, attn_dim)
        V = self.v_proj(x_ssm)  # (batch, seq, attn_dim)
        attn_out = self.local_attn.forward_sequence(
            x_ssm, K, V, window=self.local_attn_window,
            topk=self.local_attn_topk,
            topk_random=self.local_attn_topk_random,
        )
        gate = torch.sigmoid(self.gate_proj(x_ssm) + self.gate_bias)
        x_ssm = x_ssm + gate * attn_out

        # 3. FF: parallel
        y = x_ssm + self.ff(self.ff_norm(x_ssm))

        stats = (
            {"lambda_max": torch.tensor(0.0), "sv_log_var": torch.tensor(0.0)}
            if return_jacobian_stats else None
        )
        if return_jacobian_stats and return_final_state:
            return y, stats, final_state
        if return_jacobian_stats:
            return y, stats
        if return_final_state:
            return y, final_state
        return y


class AttentionControlBlock(nn.Module):
    """Causal multi-head self-attention sibling to CareSSMBlock.

    Scientific control for Exp 19 Phase 2's "SSM vs Attention under equal
    infrastructure" comparison. Not a submission candidate and NOT related to
    CareSSMHybridBlock's local-attention sidecar — this is pure attention,
    used when the whole model runs as a transformer for the controlled
    comparison against the pure-SSM path.

    Matches CareSSMBlock's block interface exactly so the two can be swapped
    in CareStudentLM via the ``block_type`` constructor flag.

    Architecture (mirrors CareSSMBlock's pre-norm + two-residual shape):
      - input_norm (RMSNorm) -> Q/K/V projection -> RoPE on Q/K -> causal SDPA
        -> out_proj -> residual
      - ff_norm (RMSNorm) -> FeedForward -> residual

    Attention uses torch.nn.functional.scaled_dot_product_attention with
    is_causal=True. No Flash Attention dependency; PyTorch's SDPA has native
    fast paths on H100 (including FlashAttention v2 under the hood when the
    shapes/dtypes are supported).

    Positional signal: Rotary Position Embeddings (RoPE, Su et al., RoFormer,
    2021) applied to Q and K only (not V). Parameter-free — cos/sin tables
    are derived from position indices and a base constant (10000) and stored
    as non-persistent buffers so checkpoints are unaffected. This matches the
    way CareSSMBlock's SSM recurrence gets position "for free" from the
    state-transition dynamics: no learned positional parameters enter the
    comparison on either side. Rationale: without any position signal,
    self-attention is permutation-equivariant (modulo the causal mask),
    whereas the SSM recurrence is intrinsically positional. Codex review
    2026-04-13 flagged this as a High-severity fairness issue for the
    Exp 19 Phase 2 control.

    Parameter footprint at dim=256, ff_mult=2, num_heads=8:
      - QKV projection (fused): 3 * dim^2 = 196,608
      - out_proj:              dim^2     = 65,536
      - FF (fc + proj):        2 * dim * (dim * ff_mult) = 262,144
      - RMSNorm weights:       2 * dim   = 512
      - total:                              ~524,800 parameters

    RoPE contributes ZERO learnable parameters; the cos/sin buffers are
    registered with ``persistent=False`` and do not show up in
    ``sum(p.numel() for p in block.parameters())`` or in the state_dict.

    This is ~1 * dim^2 FEWER params than CareSSMBlock's core (which has
    in/select/gate/delta/out = 5 projections). Exp 19 Phase 2 compares
    per-token learning efficiency at matched block count, not matched
    per-block parameter count — the difference is documented honestly in
    the review brief.
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 2,
        *,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        head_dim = dim // num_heads
        # RoPE pairs adjacent channels (2i, 2i+1) into a complex rotation, so
        # head_dim must be even. Without this assertion, odd head_dim would
        # silently drop the last channel and produce subtly wrong rotations.
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be even for RoPE; "
                f"got dim={dim}, num_heads={num_heads}"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dropout = attn_dropout
        self.rope_base = rope_base

        self.input_norm = RMSNorm(dim)
        self.ff_norm = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_mult)

        # Fused QKV projection matches the CareSSMCore convention of bias=False
        # and keeps the per-block parameter footprint compact.
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # RoPE cos/sin cache. Non-persistent buffers: they are not saved in
        # the state_dict, do not bloat checkpoints, and are lazily rebuilt on
        # the first forward pass (or any later forward with a longer sequence
        # than the cache covers). Stored in float32 for precision; cast to
        # the input dtype at application time (same pattern as RMSNorm).
        # Initial shape is (0, head_dim) — zero-length sentinel so the first
        # forward triggers a real build. We cannot know the target device at
        # __init__ time; buffers will move with .to(device) alongside the
        # module, and the cache-extend path handles device/dtype transitions.
        self.register_buffer(
            "_rope_cos",
            torch.empty(0, head_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_rope_sin",
            torch.empty(0, head_dim, dtype=torch.float32),
            persistent=False,
        )

    def _ensure_rope_cache(self, seq_len: int, device: torch.device) -> None:
        """Lazily (re)build the RoPE cos/sin cache when needed.

        Rebuilds if the cache is shorter than ``seq_len`` or on a different
        device. Stored in float32; dtype conversion happens at application
        time. Cheap to rebuild — O(seq_len * head_dim).
        """
        cache_len = self._rope_cos.shape[0]
        same_device = self._rope_cos.device == device
        if cache_len >= seq_len and same_device:
            return
        # Rotation frequencies: 1 / base^(2i / head_dim) for i in [0, head_dim/2).
        # Each pair (2i, 2i+1) shares the same frequency.
        half = self.head_dim // 2
        freqs = torch.arange(0, half, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.rope_base ** (2.0 * freqs / self.head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        # angles[t, i] = t * inv_freq[i], shape (seq_len, half)
        angles = torch.einsum("t,i->ti", positions, inv_freq)
        # Expand to (seq_len, head_dim) by repeating each freq for the pair.
        # cos/sin are the same for both channels of a (2i, 2i+1) pair; the
        # interleaved application math in ``apply_rope`` reads only the even
        # indices of these tensors per pair (see below).
        cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
        self._rope_cos = cos
        self._rope_sin = sin

    @staticmethod
    def apply_rope(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply interleaved rotary embedding to the last dim of ``x``.

        Math (interleaved pairs; RoFormer paper form):
            x_rot[..., 0::2] = x[..., 0::2] * cos_pair - x[..., 1::2] * sin_pair
            x_rot[..., 1::2] = x[..., 1::2] * cos_pair + x[..., 0::2] * sin_pair
        where cos_pair[..., i] = cos(theta_i) for i in [0, head_dim/2).

        ``cos`` and ``sin`` are shape (seq_len, head_dim) with each frequency
        repeated twice along the last axis (cos = [c0, c0, c1, c1, ...]), so
        the even/odd slices pick the same cos_pair/sin_pair for the
        corresponding channel of each rotation pair.

        ``x`` has shape (..., seq_len, head_dim); the cos/sin broadcast over
        the leading axes. Result is the same dtype as ``x``.

        This is defined as a staticmethod so tests can exercise the rotation
        math in isolation without instantiating the block, and so the
        numerical-parity test has a single source of truth it can call.
        """
        # Cast cos/sin to input dtype for the multiplication. The underlying
        # buffers are float32; we match x's dtype here so bf16 forward passes
        # stay in bf16 for the actual multiply.
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos_pair = cos[..., 0::2]
        sin_pair = sin[..., 0::2]
        rot_even = x_even * cos_pair - x_odd * sin_pair
        rot_odd = x_odd * cos_pair + x_even * sin_pair
        # Re-interleave into the original shape.
        out = torch.empty_like(x)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd
        return out

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        """Run pre-norm causal self-attention over the sequence dim."""
        batch, seq, dim = x.shape
        qkv = self.qkv_proj(x)  # (batch, seq, 3*dim)
        q, k, v = qkv.chunk(3, dim=-1)
        # (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        # Apply RoPE to Q and K (not V) after the QKV reshape and before SDPA.
        # V carries no positional rotation — this is the standard RoFormer
        # formulation. step() hits this path with seq=1, where cos=1 and
        # sin=0 for every frequency at position 0, so the rotation collapses
        # to the identity and no step-specific branch is needed.
        self._ensure_rope_cache(seq, x.device)
        cos = self._rope_cos[:seq]  # (seq, head_dim)
        sin = self._rope_sin[:seq]
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)
        # SDPA with is_causal=True installs the upper-triangular mask internally.
        # Dropout is only applied in training mode.
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout_p
        )
        # (batch, num_heads, seq, head_dim) -> (batch, seq, dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.out_proj(attn_out)

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-token step through the block.

        This interface exists so AttentionControlBlock is drop-in compatible with
        CareSSMBlock in CareStudentLM.step() / dream_step(). Because single-
        token attention has no history (we intentionally do NOT maintain a KV
        cache — the task forbids hybrid SSM+attention paths here), this
        collapses to a pass through V + out_proj (the causal mask makes the
        softmax over a single position a no-op: q_0 attends to k_0 only,
        weighted 1.0).

        The returned "new_state" is a zero tensor of the same shape as the
        incoming state, preserving the shape contract that lets CareStudentLM
        iterate blocks uniformly. Attention has no cross-token state; this
        method is NOT valid for incremental decoding — it exists only so
        CareStudentLM.step() / dream_step() do not crash when
        block_type="attention".

        Args:
            x: (batch, dim) — single token
            state: (batch, dim) — ignored, carried for interface parity

        Returns:
            (output, new_state) — output is (batch, dim), new_state is zeros
            of the same shape as the incoming state.
        """
        normed = self.input_norm(x)
        # Promote (batch, dim) -> (batch, 1, dim), attend, drop the seq dim.
        y_seq = self._attn(normed.unsqueeze(1)).squeeze(1)
        x = x + y_seq
        x = x + self.ff(self.ff_norm(x))
        new_state = torch.zeros_like(state)
        return x, new_state

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_jacobian_stats: bool = False,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Sequence forward pass. Shape: (batch, seq, dim) -> (batch, seq, dim).

        return_jacobian_stats is accepted for interface parity with
        CareSSMBlock. Attention has no per-step Jacobian in the SSM sense, so
        we return dummy zero stats matching the SSM diag-mode contract (see
        CareSSMCore.forward around the ``return_jacobian_stats`` branch).

        ``initial_state`` / ``return_final_state`` are accepted for block
        interface parity with CareSSMBlock. Attention has no cross-token
        recurrent state in this block (no KV cache threaded across forward
        calls — that lives in step()), so initial_state is ignored and
        final_state is returned as zeros of shape (batch, dim) to match the
        SSM contract. This mirrors the choice made in ``step()`` at line
        458; it preserves uniform iteration in CareStudentLM without
        pretending attention has an SSM-shaped recurrent state.
        """
        normed = self.input_norm(x)
        y = self._attn(normed)
        x = x + y
        x = x + self.ff(self.ff_norm(x))
        stats = None
        if return_jacobian_stats:
            stats = {
                "lambda_max": torch.tensor(0.0, device=x.device),
                "sv_log_var": torch.tensor(0.0, device=x.device),
            }
        final_state = None
        if return_final_state:
            batch = x.shape[0]
            final_state = torch.zeros(batch, self.dim, device=x.device, dtype=x.dtype)
        if return_jacobian_stats and return_final_state:
            return x, stats, final_state
        if return_jacobian_stats:
            return x, stats
        if return_final_state:
            return x, final_state
        return x


class CareStudentLM(nn.Module):
    """CARE trunk language model wiring the diagonal SSM path plus optional tiers."""

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        num_layers: int,
        ff_mult: int = 2,
        block_type: str = "ssm",
        attention_num_heads: int = 8,
        attention_dropout: float = 0.0,
        a_mode: str = "diag",
        a_full_rank: int = 8,
        a_full_gamma: float = 0.05,
        ssm_delta_rank: int = 0,
        rich_b_mode: str = "none",
        rich_b_bottleneck: int = 32,
        rich_b_num_subnets: int = 4,
        rich_b_settling_steps: int = 2,
        outer_model_dim: int = 0,
        consolidation_mode: str = "symmetric",
        consolidation_ema_decay: float = 0.99,
        consolidation_trigger: str = "immediate",
        consolidation_window: int = 8,
        outer_model_type: str = "single",
        outer_max_slots: int = 64,
        outer_compress_ratio: int = 2,
        wernicke_enabled: bool = False,
        wernicke_k_max: int = 16,
        wernicke_window: int = 8,
        wernicke_router: str = "vq",
        wernicke_balance_weight: float = 0.01,
        wernicke_expert_dim: int = 0,
        wernicke_layers: int = 1,
        wernicke_k_max_fine: int = 16,
        buffer_mode: str = "legacy",
        retrieval_mode: str = "softmax_all",
        retrieval_k: int = 8,
        bucket_prototypes: bool = False,
        prototype_dim: int = 64,
        prototype_update_rate: float = 0.1,
        semantic_tier_bases: int = 0,
        semantic_tier_update_rate: float = 0.01,
        typed_storage: bool = False,
        typed_consolidation: bool = False,
        compression_consequence: bool = False,
        cue_projection: bool = True,
        dynamic_crit_per_layer: bool = False,
        compression_selection: str = "survival",
        posterior_mode: str = "none",
        posterior_lr: float = 0.01,
        residual_cache_k: int = 4,
        local_attn_window: int = 0,
        local_attn_heads: int = 1,
        local_attn_dim: int = 64,
        local_attn_topk: int = 0,
        local_attn_topk_random: bool = False,
        depth_recurrence_shared_layers: list[int] | None = None,
        depth_recurrence_count: int = 1,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.activation_checkpoint = bool(activation_checkpoint)
        self.local_attn_window = local_attn_window
        self.embed = nn.Embedding(vocab_size, dim)

        # Typed KV buffer config
        self.buffer_mode = buffer_mode
        self.retrieval_mode = retrieval_mode
        self.retrieval_k = retrieval_k

        # Wernicke layer: typed compositional preprocessing
        self.wernicke: WernickeLayer | HierarchicalWernicke | None = None
        self.wernicke_balance_weight = wernicke_balance_weight
        if wernicke_enabled:
            if wernicke_layers >= 2:
                self.wernicke = HierarchicalWernicke(
                    dim=dim,
                    k_coarse=wernicke_k_max,
                    k_fine=wernicke_k_max_fine,
                    window=wernicke_window,
                    router_type=wernicke_router,
                    balance_weight=wernicke_balance_weight,
                    expert_dim=wernicke_expert_dim if wernicke_expert_dim > 0 else None,
                )
            else:
                self.wernicke = WernickeLayer(
                    dim,
                    k_max=wernicke_k_max,
                    window=wernicke_window,
                    router_type=wernicke_router,
                    balance_weight=wernicke_balance_weight,
                    expert_dim=wernicke_expert_dim if wernicke_expert_dim > 0 else None,
                )

        self.block_type = block_type
        if block_type == "attention":
            # Pure-attention scientific control for Exp 19 Phase 2. All layers
            # are AttentionControlBlock; local_attn_window / hybrid-SSM kwargs
            # are ignored in this path because the comparison is "full SSM
            # stack vs full attention stack", not "SSM with sidecar vs
            # attention with sidecar". If the comparison is ever widened, the
            # CareSSMHybridBlock sidecar is still the right mechanism for
            # mixed configurations and should not be conflated with this flag.
            self.layers = nn.ModuleList([
                AttentionControlBlock(
                    dim,
                    ff_mult,
                    num_heads=attention_num_heads,
                    attn_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ])
        elif block_type == "ssm":
            ssm_block_kwargs = dict(
                a_mode=a_mode,
                a_full_rank=a_full_rank,
                a_full_gamma=a_full_gamma,
                ssm_delta_rank=ssm_delta_rank,
                rich_b_mode=rich_b_mode,
                rich_b_bottleneck=rich_b_bottleneck,
                rich_b_num_subnets=rich_b_num_subnets,
                rich_b_settling_steps=rich_b_settling_steps,
            )
            if local_attn_window > 0:
                ssm_layers = [
                    CareSSMBlock(dim, ff_mult, **ssm_block_kwargs)
                    for _ in range(num_layers - 1)
                ]
                hybrid_layer = CareSSMHybridBlock(
                    dim, ff_mult,
                    a_mode=a_mode,
                    a_full_rank=a_full_rank,
                    a_full_gamma=a_full_gamma,
                    ssm_delta_rank=ssm_delta_rank,
                    local_attn_window=local_attn_window,
                    local_attn_heads=local_attn_heads,
                    local_attn_dim=local_attn_dim,
                    local_attn_topk=local_attn_topk,
                    local_attn_topk_random=local_attn_topk_random,
                )
                self.layers = nn.ModuleList(ssm_layers + [hybrid_layer])
            else:
                self.layers = nn.ModuleList([
                    CareSSMBlock(dim, ff_mult, **ssm_block_kwargs)
                    for _ in range(num_layers)
                ])
        else:
            raise ValueError(
                f"unsupported block_type: {block_type!r} (expected 'ssm' or 'attention')"
            )

        # Weight-tied depth recurrence: indices of physical layers that form
        # the "shared group" are replayed `depth_recurrence_count` times in
        # the forward loop. Because the same nn.Module object is re-used, no
        # new parameters are introduced — autograd accumulates gradients on
        # the shared weights across passes. At count=1 (default) the virtual
        # sequence equals list(range(num_layers)), i.e. bit-identical to the
        # non-recurrent path. Runs after the block_type switch so it works
        # uniformly for both "ssm" and "attention" configurations — the
        # virtual layer mechanism is block-type-agnostic.
        shared = list(depth_recurrence_shared_layers or [])
        self.depth_recurrence_shared_layers: list[int] = shared
        self.depth_recurrence_count: int = int(depth_recurrence_count)
        if self.depth_recurrence_count < 1:
            raise ValueError(
                f"depth_recurrence_count must be >= 1, got {self.depth_recurrence_count}"
            )
        if shared:
            if any(not (0 <= i < len(self.layers)) for i in shared):
                raise ValueError(
                    f"depth_recurrence_shared_layers indices out of range "
                    f"[0, {len(self.layers)}): {shared}"
                )
            if sorted(shared) != shared:
                raise ValueError(
                    f"depth_recurrence_shared_layers must be sorted ascending: {shared}"
                )
            if any(shared[i] + 1 != shared[i + 1] for i in range(len(shared) - 1)):
                raise ValueError(
                    f"depth_recurrence_shared_layers must be contiguous: {shared}"
                )
            a, b = shared[0], shared[-1]
            prefix = list(range(0, a))
            suffix = list(range(b + 1, len(self.layers)))
            self._virtual_layer_indices: list[int] = (
                prefix + shared * self.depth_recurrence_count + suffix
            )
        else:
            self._virtual_layer_indices = list(range(len(self.layers)))

        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.outer_model_type = outer_model_type
        self.outer_model: OuterModel | MultiSlotOuterModel | None = None
        if outer_model_dim > 0:
            common_kw = dict(
                consolidation_mode=consolidation_mode,
                ema_decay=consolidation_ema_decay,
                trigger=consolidation_trigger,
                trigger_window=consolidation_window,
            )
            if outer_model_type == "multislot":
                self.outer_model = MultiSlotOuterModel(
                    dim, outer_dim=outer_model_dim,
                    max_slots=outer_max_slots,
                    compress_ratio=outer_compress_ratio,
                    compression_selection=compression_selection,
                    **common_kw,
                )
            else:
                self.outer_model = OuterModel(
                    dim, outer_dim=outer_model_dim, **common_kw,
                )

        # Semantic tier: always-on background bias from episodic experience
        self.semantic_tier: SemanticTier | None = None
        if semantic_tier_bases > 0:
            self.semantic_tier = SemanticTier(dim, num_bases=semantic_tier_bases, update_rate=semantic_tier_update_rate)

        # Bucket prototypes: per-bucket semantic priors
        self.bucket_prototypes_module: BucketPrototypes | None = None
        if bucket_prototypes and outer_model_dim > 0:
            # Determine total buckets from Wernicke config
            if wernicke_layers >= 2:
                total_k = wernicke_k_max * wernicke_k_max_fine
            else:
                total_k = wernicke_k_max
            self.bucket_prototypes_module = BucketPrototypes(
                k_max=total_k,
                prototype_dim=prototype_dim,
                model_dim=dim,
                update_rate=prototype_update_rate,
            )

        # Error-driven posterior module: stores belief corrections from prediction error
        self.posterior: GlobalDelta | BucketDelta | ResidualCache | None = None
        self.posterior_mode = posterior_mode
        total_buckets = (
            self.wernicke.total_buckets
            if hasattr(self.wernicke, "total_buckets")
            else (self.wernicke.k_max if self.wernicke is not None else wernicke_k_max)
        )
        if posterior_mode == "global_delta":
            self.posterior = GlobalDelta(dim, lr=posterior_lr)
        elif posterior_mode == "bucket_delta":
            self.posterior = BucketDelta(
                k_max=total_buckets, model_dim=dim, lr=posterior_lr,
            )
        elif posterior_mode == "residual_cache":
            self.posterior = ResidualCache(
                model_dim=dim, k=residual_cache_k,
            )

        # Gap-analysis flags
        self.typed_storage = typed_storage
        self.typed_consolidation = typed_consolidation
        self.compression_consequence = compression_consequence
        self.cue_projection = cue_projection
        self.dynamic_crit_per_layer = dynamic_crit_per_layer

    def init_state(self, batch_size: int) -> list[torch.Tensor]:
        """Initialize recurrence states for all layers."""
        device = self.embed.weight.device
        dtype = self.embed.weight.dtype
        return [torch.zeros(batch_size, self.dim, device=device, dtype=dtype) for _ in range(len(self.layers))]

    def init_kv_caches(self) -> list[RollingKVCache | None]:
        """Initialize KV caches for hybrid blocks. Returns one entry per layer (None for pure SSM blocks)."""
        return [
            layer._init_kv_cache() if isinstance(layer, CareSSMHybridBlock) else None
            for layer in self.layers
        ]

    def step(
        self,
        token_ids: torch.Tensor,
        states: list[torch.Tensor],
        kv_caches: list[RollingKVCache | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Single-token forward step through the SSM recurrence only.

        Intentionally simpler than forward() — skips Wernicke typed composition,
        episodic/semantic memory reads, and outer model interactions. This is the
        "world model" for MCTS planning: analogous to System 2 deliberation
        operating on a compressed internal model rather than full perception.

        For components that require the full forward path (Wernicke routing,
        memory cue-dependent retrieval), use forward() on complete sequences.

        Args:
            token_ids: (batch, 1) — single token ids
            states: list of (batch, dim) per layer
            kv_caches: from init_kv_caches(), required when local_attn_window > 0.
                Mutated in place (rolling buffer accumulates K/V entries).

        Returns:
            (logits, hidden, new_states)
            logits: (batch, vocab)
            hidden: (batch, dim)
            new_states: list of (batch, dim) per layer
        """
        # Guard: depth recurrence is only wired into forward(), not step().
        # step() iterates physical layers once regardless of the virtual
        # layer schedule, so calling step() with count > 1 would silently
        # run a different (shallower) model than forward() did on the same
        # weights. Exp 19 Phase 1 trains full sequences via forward() — if
        # a future caller wants recurrent single-token rollout (MCTS
        # planning, autoregressive generation), they need to either update
        # this method to iterate `self._virtual_layer_indices` with
        # per-virtual-step state management, or explicitly accept that the
        # rollout runs a different architecture than training did.
        if self.depth_recurrence_count > 1:
            raise NotImplementedError(
                "CareStudentLM.step() does not implement depth recurrence. "
                f"depth_recurrence_count={self.depth_recurrence_count} is set, "
                "but step() iterates physical layers once — calling it would "
                "silently run a shallower model than forward(). Use forward() "
                "for full-sequence inference, or update step() to handle "
                "virtual-layer rollout with per-virtual-step state management."
            )

        x = self.embed(token_ids).squeeze(1)  # (batch, dim)

        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        new_states = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, CareSSMHybridBlock):
                if kv_caches[i] is None:
                    raise RuntimeError(
                        f"Layer {i} is a hybrid block but no KV cache provided. "
                        "Call init_kv_caches() and pass the result to step()."
                    )
                x, new_s = layer.step(x, states[i], kv_cache=kv_caches[i])
            else:
                x, new_s = layer.step(x, states[i])
            new_states.append(new_s)

        hidden = x
        x = self.final_norm(x.unsqueeze(1)).squeeze(1)  # RMSNorm expects (..., dim)
        logits = self.lm_head(x)

        return logits, hidden, new_states

    def dream_step(
        self,
        token_ids: torch.Tensor,
        states: list[torch.Tensor],
        kv_caches: list[RollingKVCache | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Full-tier single-token forward step for REM dream generation.

        Like step() but includes ALL tiers (Wernicke, memory, semantic) that
        step() intentionally skips. During REM dream generation the model needs
        to experience the full forward path so that dream sequences reflect the
        complete perception pipeline.

        Args:
            token_ids: (batch, 1) — single token ids
            states: list of (batch, dim) per layer
            kv_caches: from init_kv_caches(), required when local_attn_window > 0.
                Mutated in place (rolling buffer accumulates K/V entries).

        Returns:
            (logits, hidden, new_states)
            logits: (batch, vocab)
            hidden: (batch, dim)
            new_states: list of (batch, dim) per layer
        """
        # Guard: depth recurrence is only wired into forward(), not dream_step().
        # See step()'s guard for the full rationale — same failure mode, same
        # fix story. REM dream rollout with count > 1 would silently run a
        # shallower model than forward() trained.
        if self.depth_recurrence_count > 1:
            raise NotImplementedError(
                "CareStudentLM.dream_step() does not implement depth recurrence. "
                f"depth_recurrence_count={self.depth_recurrence_count} is set, "
                "but dream_step() iterates physical layers once — calling it would "
                "silently run a shallower model than forward(). Use forward() "
                "for full-sequence REM replay, or update dream_step() to handle "
                "virtual-layer rollout with per-virtual-step state management."
            )

        x = self.embed(token_ids).squeeze(1)  # (batch, dim)

        # Wernicke (NOT skipped, unlike step)
        if self.wernicke is not None:
            x_seq = x.unsqueeze(1)  # Wernicke expects (batch, seq, dim)
            x_seq, bucket_ids, _ = self.wernicke(x_seq)
            x = x_seq.squeeze(1)

        # Memory read (NOT skipped)
        if self.outer_model is not None:
            batch_size = x.size(0)
            if hasattr(self.outer_model, "_slots"):
                outer_read = self.outer_model.read(batch_size, cue=x)
            else:
                outer_read = self.outer_model.read(batch_size)
            x = x + outer_read

        # Semantic tier (NOT skipped)
        if self.semantic_tier is not None:
            x = x + self.semantic_tier.read(x.size(0))

        # SSM recurrence — use CareSSMBlock.step()
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        new_states = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, CareSSMHybridBlock):
                if kv_caches[i] is None:
                    raise RuntimeError(
                        f"Layer {i} is a hybrid block but no KV cache provided. "
                        "Call init_kv_caches() and pass the result to dream_step()."
                    )
                x, new_s = layer.step(x, states[i], kv_cache=kv_caches[i])
            else:
                x, new_s = layer.step(x, states[i])
            new_states.append(new_s)

        hidden = x
        logits = self.lm_head(self.final_norm(x.unsqueeze(1)).squeeze(1))
        return logits, hidden, new_states

    def artifact_bytes(self) -> int:
        return int(sum(p.numel() for p in self.parameters()) * 2)

    @torch.no_grad()
    def append_memory_from_hidden(
        self,
        hidden: torch.Tensor,
        *,
        bucket_ids: torch.Tensor | None = None,
        score: torch.Tensor | None = None,
        max_tokens: int | None = None,
        event_ids: torch.Tensor | None = None,
    ) -> bool:
        """Append sequence hidden states into the append-only outer memory.

        ``forward(memory_write_mode='append_only')`` already owns this write
        shape, but the throughput runner deliberately calls ``encode()`` plus
        a fused LM-head backward instead of materializing full logits through
        ``forward()``. CRCT needs the same cache lifecycle on that fast path,
        so this helper factors the write side out without changing the default
        side-effect-free ``encode()`` contract.

        Returns ``True`` when a write happened. A ``False`` return means the
        model is not configured with a multislot append-only outer memory, so
        callers can fail loudly instead of training against a zero-utility
        teacher by accident.
        """
        if (
            self.buffer_mode != "append_only"
            or self.outer_model is None
            or not isinstance(self.outer_model, MultiSlotOuterModel)
        ):
            return False
        batch, seq, dim = hidden.shape
        h_flat = hidden.detach().reshape(-1, dim)
        event_source = None
        if event_ids is not None:
            event_source = event_ids.detach().reshape(-1).to(
                device=hidden.device,
                dtype=torch.long,
            )
        max_tokens_i = 0 if max_tokens is None else int(max_tokens)
        if max_tokens_i > 0 and h_flat.shape[0] > max_tokens_i:
            if score is not None:
                flat_score = score.detach().reshape(-1).to(
                    device=hidden.device,
                    dtype=torch.float32,
                )
                selected = torch.topk(
                    flat_score,
                    k=max_tokens_i,
                    largest=True,
                    sorted=False,
                ).indices
            else:
                selected = torch.linspace(
                    0,
                    h_flat.shape[0] - 1,
                    steps=max_tokens_i,
                    device=hidden.device,
                ).long()
            h_flat = h_flat.index_select(0, selected)
        else:
            selected = None
        event_flat = None
        if event_source is not None:
            if event_source.numel() == batch * seq:
                event_flat = (
                    event_source.index_select(0, selected)
                    if selected is not None
                    else event_source
                )
            elif event_source.numel() == h_flat.shape[0]:
                event_flat = event_source
            else:
                raise ValueError(
                    "event_ids must either cover the full hidden shape "
                    f"({batch * seq}) or the selected write count "
                    f"({h_flat.shape[0]}), got {event_source.numel()}"
                )
        encoded_flat = torch.tanh(
            self.outer_model.encoder(
                h_flat.to(dtype=self.outer_model.encoder.weight.dtype)
            )
        )
        if bucket_ids is None:
            bids_flat = torch.zeros(
                h_flat.shape[0],
                dtype=torch.long,
                device=hidden.device,
            )
        else:
            bids_flat = bucket_ids.detach().reshape(-1).to(
                device=hidden.device,
                dtype=torch.long,
            )
            if selected is not None:
                bids_flat = bids_flat.index_select(0, selected)
        if event_flat is None:
            self.outer_model.append_kv_batch(encoded_flat, bids_flat)
        else:
            self.outer_model._append_kv_batch_committed(
                encoded_flat,
                bids_flat,
                event_ids=event_flat,
        )
        return True

    def encode_paired_for_score(
        self,
        input_ids: torch.Tensor,
        *,
        cache_read_cutoff: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return ``(h_off, h_mem, memory_meta)`` for rank-side scoring.

        GPU3/GPU7 exact scoring asks the same question every request:
        ``memory_off`` vs ``force_on`` for the same token batch and cache
        cutoff. Running embed/Wernicke once and stacking the off/on streams
        through the SSM loop preserves the oracle contract while avoiding two
        separate recurrent launches. Train ranks never call this method.
        """
        x_base = self.embed(input_ids)
        bucket_ids = None
        if self.wernicke is not None:
            x_base, bucket_ids, _balance_loss = self.wernicke(x_base)

        memory_gate: torch.Tensor | None = None
        memory_residual: torch.Tensor | None = None

        def _expanded_gate(reference: torch.Tensor) -> torch.Tensor:
            nonlocal memory_gate
            if memory_gate is None:
                memory_gate = torch.ones(
                    reference.shape[:2],
                    device=reference.device,
                    dtype=reference.dtype,
                )
            return memory_gate.unsqueeze(-1).to(dtype=reference.dtype)

        def _add_force_bias(reference: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            nonlocal memory_residual
            gate = _expanded_gate(reference)
            bias = bias.to(device=reference.device, dtype=reference.dtype)
            memory_residual = bias if memory_residual is None else memory_residual + bias
            return reference + bias * gate

        if (
            isinstance(self.outer_model, MultiSlotOuterModel)
            and self.cue_projection
        ):
            self._last_outer_cue = x_base.detach().mean(dim=1)
        else:
            self._last_outer_cue = None

        x_mem = x_base
        if self.outer_model is not None and self.buffer_mode == "append_only":
            batch = x_base.size(0)
            model_dim = x_base.size(2)
            if isinstance(self.outer_model, MultiSlotOuterModel) and self.outer_model._slots:
                if bucket_ids is not None:
                    per_sample_buckets = bucket_ids.mode(dim=1).values
                else:
                    per_sample_buckets = torch.zeros(
                        batch,
                        dtype=torch.long,
                        device=x_base.device,
                    )
                outer_read = torch.zeros(
                    batch,
                    1,
                    model_dim,
                    device=x_base.device,
                    dtype=x_base.dtype,
                )
                for b_id in per_sample_buckets.unique():
                    mask = per_sample_buckets == b_id
                    cue = None
                    if self.retrieval_mode in ("bucket_topk", "softmax_all"):
                        cue = self.outer_model.cue_proj(
                            x_base[mask]
                            .detach()
                            .mean(dim=1)
                            .to(dtype=self.outer_model.cue_proj.weight.dtype)
                        )
                    read = self.outer_model.read_bucket(
                        int(mask.sum()),
                        bucket_id=int(b_id.item()),
                        mode=self.retrieval_mode,
                        k=self.retrieval_k,
                        cue=cue,
                        read_cutoff=cache_read_cutoff,
                    )
                    outer_read[mask] = read.unsqueeze(1).to(dtype=x_base.dtype)
                x_mem = _add_force_bias(x_mem, outer_read)

            if self.bucket_prototypes_module is not None and bucket_ids is not None:
                per_sample_buckets = bucket_ids.mode(dim=1).values
                proto_bias = torch.zeros(
                    batch,
                    1,
                    model_dim,
                    device=x_base.device,
                    dtype=x_base.dtype,
                )
                for b_id in per_sample_buckets.unique():
                    mask = per_sample_buckets == b_id
                    proto = self.bucket_prototypes_module.read(
                        int(mask.sum()),
                        int(b_id.item()),
                    )
                    proto_bias[mask] = proto.unsqueeze(1).to(dtype=x_base.dtype)
                x_mem = _add_force_bias(x_mem, proto_bias)
        elif self.outer_model is not None:
            if isinstance(self.outer_model, MultiSlotOuterModel):
                cue = x_base.detach().mean(dim=1) if self.cue_projection else None
                outer_read = self.outer_model.read(
                    x_base.size(0),
                    cue=cue,
                    read_cutoff=cache_read_cutoff,
                )
            else:
                outer_read = self.outer_model.read(x_base.size(0))
            x_mem = _add_force_bias(x_mem, outer_read.unsqueeze(1))

        if self.semantic_tier is not None:
            semantic_bias = self.semantic_tier.read(x_base.size(0))
            x_mem = _add_force_bias(x_mem, semantic_bias.unsqueeze(1))

        x_pair = torch.cat([x_base, x_mem], dim=0)
        for layer_idx in self._virtual_layer_indices:
            layer = self.layers[layer_idx]
            x_pair = layer(x_pair, return_jacobian_stats=False)
        h_off, h_mem = x_pair.chunk(2, dim=0)

        if self.posterior is not None:
            if isinstance(self.posterior, GlobalDelta):
                posterior_bias = self.posterior.read(h_mem.size(0))
                h_mem = _add_force_bias(h_mem, posterior_bias.unsqueeze(1))
            elif isinstance(self.posterior, BucketDelta) and bucket_ids is not None:
                per_sample_buckets = bucket_ids.mode(dim=1).values
                posterior_bias = torch.cat([
                    self.posterior.read(
                        bucket_id=int(per_sample_buckets[i].item()),
                        batch_size=1,
                    )
                    for i in range(h_mem.size(0))
                ], dim=0)
                h_mem = _add_force_bias(h_mem, posterior_bias.unsqueeze(1))
            elif isinstance(self.posterior, ResidualCache):
                query = h_mem.detach().mean(dim=1)
                posterior_bias = self.posterior.read(query)
                h_mem = _add_force_bias(h_mem, posterior_bias.unsqueeze(1))

        memory_meta = {
            "memory_mode": "force_on",
            "cache_read_cutoff": cache_read_cutoff,
            "memory_gate": memory_gate,
            "memory_residual": memory_residual,
        }
        return h_off, h_mem, memory_meta

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        memory_mode: str = "packet",
        cache_read_cutoff: int | None = None,
        episodic_residual: torch.Tensor | None = None,
        episodic_gate: torch.Tensor | None = None,
        memory_slot_mask: torch.Tensor | None = None,
        memory_slot_override_index: int | None = None,
        memory_slot_override_values: torch.Tensor | None = None,
        return_memory_meta: bool = False,
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | dict[str, Any]:
        """Run every pre-LM-head computation and return the hidden state.

        Mirrors the encoder portion of ``forward()`` — embed → wernicke →
        outer-model read → semantic tier → layers → posterior — and
        returns the hidden ``(batch, seq, dim)`` tensor at the point
        where ``forward()`` names ``hidden = x`` (currently model.py
        around line 1051).

        Introduced 2026-04-16 so ``train_ssm`` can run the encoder
        once and then loop chunked forward/backward through the
        LM head without materializing the full ``(B, T, V)`` logits
        gradient. ``forward()`` is frozen for reproducibility of prior
        experiments; any config that ``forward()`` supports should
        produce an identical hidden state here.

        Unlike ``forward()`` this method does not return Jacobian stats,
        balance-loss, or bucket-ids. The bare-SSM path — the only path
        ``train_ssm`` supports — produces none of those. Configs that
        need them must continue to use ``forward()`` + ``training.py``.

        ``memory_mode`` controls the memory read/injection point for CRCT:
        ``"off"`` disables memory reads; ``"force_on"`` uses the historical
        full-strength read for GPU3 oracle/counterfactual scoring; ``"packet"``
        consumes a precomputed episodic residual packet before recurrence and
        performs no local sidecar reads. The default is ``"packet"`` with a
        zero-residual no-op, matching the train-rank trunk contract: semantic
        gist/cue construction happens locally, targeted episodic retrieval
        happens off-path, and the trunk only receives the prepared packet.

        ``memory_slot_mask`` optionally provides a per-sample physical-slot
        mask for rank-3 maintenance probes. It is ignored when
        ``memory_mode='off'`` and lets the sidecar oracle evaluate
        ``[baseline, no-sidecar, hide-slot-i]`` variants as one expanded
        batch without changing train-rank trunk execution.

        ``memory_slot_override_index``/``memory_slot_override_values`` let
        rank-3 maintenance score refresh candidates in one expanded GPU3
        batch.  The override is sparse and per sample: retrieval treats the
        selected physical slot as if it contained the supplied value, without
        mutating the SlotTable or exposing the candidate to train ranks.

        Side effects: none. Unlike ``forward()`` this path never
        performs memory writes (those live in ``forward()`` after
        ``lm_head`` and only fire when ``memory_write_mode='append_only'``,
        which is unused by the bare-SSM submission regime).
        """
        if memory_mode not in {"off", "force_on", "packet"}:
            raise ValueError(
                "memory_mode must be one of 'off', 'force_on', or 'packet', "
                f"got {memory_mode!r}"
            )
        if memory_mode != "packet" and (
            episodic_residual is not None or episodic_gate is not None
        ):
            raise ValueError(
                "episodic_residual/episodic_gate are only valid with "
                "memory_mode='packet'"
            )
        if memory_mode == "packet" and (
            (episodic_residual is None) != (episodic_gate is None)
        ):
            raise ValueError(
                "memory_mode='packet' requires both episodic_residual and "
                "episodic_gate, or neither for a zero-residual no-op"
            )
        if initial_states is not None:
            if len(initial_states) != len(self.layers):
                raise ValueError(
                    f"initial_states length {len(initial_states)} does not match "
                    f"num_layers {len(self.layers)}"
                )
        x = self.embed(input_ids)
        memory_gate: torch.Tensor | None = None
        memory_residual: torch.Tensor | None = None

        def _expanded_gate(reference: torch.Tensor) -> torch.Tensor | None:
            nonlocal memory_gate
            if memory_mode == "off":
                return None
            if memory_gate is None:
                memory_gate = torch.ones(
                    reference.shape[:2],
                    device=reference.device,
                    dtype=reference.dtype,
                )
            return memory_gate.unsqueeze(-1).to(dtype=reference.dtype)

        def _add_memory_bias(reference: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            nonlocal memory_residual
            gate = _expanded_gate(reference)
            if gate is None:
                return reference
            bias = bias.to(device=reference.device, dtype=reference.dtype)
            if memory_mode == "force_on":
                memory_residual = (
                    bias
                    if memory_residual is None
                    else memory_residual + bias
                )
            return reference + bias * gate

        def _packet_gate(reference: torch.Tensor) -> torch.Tensor:
            assert episodic_gate is not None
            gate = episodic_gate.to(device=reference.device, dtype=reference.dtype)
            if gate.dim() == 1:
                if gate.shape[0] != reference.shape[0]:
                    raise ValueError(
                        f"episodic_gate shape {tuple(gate.shape)} cannot gate "
                        f"hidden shape {tuple(reference.shape[:2])}"
                    )
                gate = gate[:, None].expand(reference.shape[0], reference.shape[1])
            elif gate.dim() == 2:
                if gate.shape[0] != reference.shape[0]:
                    raise ValueError(
                        f"episodic_gate shape {tuple(gate.shape)} cannot gate "
                        f"hidden shape {tuple(reference.shape[:2])}"
                    )
                if gate.shape[1] == 1 and reference.shape[1] != 1:
                    gate = gate.expand(reference.shape[0], reference.shape[1])
                elif gate.shape != reference.shape[:2]:
                    raise ValueError(
                        f"episodic_gate shape {tuple(gate.shape)} cannot gate "
                        f"hidden shape {tuple(reference.shape[:2])}"
                    )
            else:
                raise ValueError(
                    "episodic_gate must have shape (B,), (B, 1), or (B, T)"
                )
            return gate

        def _packet_residual(reference: torch.Tensor) -> torch.Tensor:
            assert episodic_residual is not None
            residual = episodic_residual.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            if residual.dim() == 2:
                if residual.shape != (reference.shape[0], reference.shape[2]):
                    raise ValueError(
                        f"episodic_residual shape {tuple(residual.shape)} "
                        f"cannot feed hidden shape {tuple(reference.shape)}"
                    )
                residual = residual.unsqueeze(1)
            elif residual.dim() == 3:
                if residual.shape[0] != reference.shape[0] or residual.shape[2] != reference.shape[2]:
                    raise ValueError(
                        f"episodic_residual shape {tuple(residual.shape)} "
                        f"cannot feed hidden shape {tuple(reference.shape)}"
                    )
                if residual.shape[1] != 1:
                    raise ValueError(
                        "episodic_residual packets must be compact with shape "
                        f"(B, D) or (B, 1, D); got {tuple(residual.shape)}"
                    )
            else:
                raise ValueError(
                    "episodic_residual must have compact shape (B, D) or "
                    "(B, 1, D)"
                )
            return residual

        # Wernicke layer: compose bytes into typed units before SSM recurrence
        bucket_ids = None
        if self.wernicke is not None:
            x, bucket_ids, _balance_loss = self.wernicke(x)

        if memory_mode == "packet" and episodic_residual is not None:
            packet_gate = _packet_gate(x)
            packet_residual = _packet_residual(x)
            x = x.addcmul_(packet_residual, packet_gate.unsqueeze(-1))
            memory_gate = packet_gate
            memory_residual = packet_residual

        if (
            memory_mode not in {"off", "packet"}
            and isinstance(self.outer_model, MultiSlotOuterModel)
            and self.cue_projection
        ):
            # Rank-3 sidecar maintenance needs the same pre-SSM cue used by
            # real memory reads.  Stashing this tensor makes the cheap
            # saliency map falsifiable against the full oracle path without
            # forcing train ranks to observe any sidecar state.
            self._last_outer_cue = x.detach().mean(dim=1)
        else:
            self._last_outer_cue = None

        # Buffer read path (mirrors forward() body 1-for-1 for the configs
        # that wire an outer_model). For bare-SSM both branches are skipped
        # because ``self.outer_model`` is None.
        if (
            memory_mode not in {"off", "packet"}
            and self.outer_model is not None
            and self.buffer_mode == "append_only"
        ):
            batch = x.size(0)
            model_dim = x.size(2)
            if isinstance(self.outer_model, MultiSlotOuterModel) and self.outer_model._slots:
                if bucket_ids is not None:
                    per_sample_buckets = bucket_ids.mode(dim=1).values
                else:
                    per_sample_buckets = torch.zeros(batch, dtype=torch.long, device=x.device)
                outer_read = torch.zeros(batch, 1, model_dim, device=x.device, dtype=x.dtype)
                for b_id in per_sample_buckets.unique():
                    mask = per_sample_buckets == b_id
                    cue = None
                    if self.retrieval_mode in ("bucket_topk", "softmax_all"):
                        cue = self.outer_model.cue_proj(
                            x[mask].detach().mean(dim=1).to(dtype=self.outer_model.cue_proj.weight.dtype)
                        )
                    read = self.outer_model.read_bucket(
                        int(mask.sum()), bucket_id=int(b_id.item()),
                        mode=self.retrieval_mode, k=self.retrieval_k, cue=cue,
                        read_cutoff=cache_read_cutoff,
                        slot_mask=memory_slot_mask[mask] if memory_slot_mask is not None else None,
                        slot_override_index=memory_slot_override_index,
                        slot_override_values=(
                            memory_slot_override_values[mask]
                            if memory_slot_override_values is not None
                            else None
                        ),
                    )
                    outer_read[mask] = read.unsqueeze(1).to(dtype=x.dtype)
                x = _add_memory_bias(x, outer_read)

            if self.bucket_prototypes_module is not None and bucket_ids is not None:
                per_sample_buckets = bucket_ids.mode(dim=1).values
                proto_bias = torch.zeros(batch, 1, model_dim, device=x.device, dtype=x.dtype)
                for b_id in per_sample_buckets.unique():
                    mask = per_sample_buckets == b_id
                    proto = self.bucket_prototypes_module.read(int(mask.sum()), int(b_id.item()))
                    proto_bias[mask] = proto.unsqueeze(1).to(dtype=x.dtype)
                x = _add_memory_bias(x, proto_bias)

        elif memory_mode not in {"off", "packet"} and self.outer_model is not None:
            if isinstance(self.outer_model, MultiSlotOuterModel):
                cue = x.detach().mean(dim=1) if self.cue_projection else None
                outer_read = self.outer_model.read(
                    x.size(0),
                    cue=cue,
                    read_cutoff=cache_read_cutoff,
                    slot_mask=memory_slot_mask,
                    slot_override_index=memory_slot_override_index,
                    slot_override_values=memory_slot_override_values,
                )
            else:
                outer_read = self.outer_model.read(x.size(0))
            x = _add_memory_bias(x, outer_read.unsqueeze(1))

        if memory_mode not in {"off", "packet"} and self.semantic_tier is not None:
            semantic_bias = self.semantic_tier.read(x.size(0))
            x = _add_memory_bias(x, semantic_bias.unsqueeze(1))

        # Virtual-layer loop — weight-tied depth recurrence when configured;
        # otherwise list(range(num_layers)). Matches forward() exactly.
        use_ckpt = self.activation_checkpoint and torch.is_grad_enabled() and x.requires_grad
        final_states: list[torch.Tensor | None] | None = None
        if return_final_states:
            final_states = [None] * len(self.layers)
        for layer_idx in self._virtual_layer_indices:
            layer = self.layers[layer_idx]
            init_state = initial_states[layer_idx] if initial_states is not None else None
            if use_ckpt:
                if return_final_states:
                    x, fstate = _checkpoint(
                        layer,
                        x,
                        return_jacobian_stats=False,
                        initial_state=init_state,
                        return_final_state=True,
                        use_reentrant=False,
                    )
                    final_states[layer_idx] = fstate
                else:
                    x = _checkpoint(
                        layer,
                        x,
                        return_jacobian_stats=False,
                        initial_state=init_state,
                        use_reentrant=False,
                    )
            else:
                if return_final_states:
                    x, fstate = layer(
                        x,
                        return_jacobian_stats=False,
                        initial_state=init_state,
                        return_final_state=True,
                    )
                    final_states[layer_idx] = fstate
                else:
                    x = layer(x, return_jacobian_stats=False, initial_state=init_state)

        # Posterior correction bias — bare-SSM path has self.posterior is None
        # so the block is skipped.
        if memory_mode not in {"off", "packet"} and self.posterior is not None:
            if isinstance(self.posterior, GlobalDelta):
                posterior_bias = self.posterior.read(x.size(0))
                x = _add_memory_bias(x, posterior_bias.unsqueeze(1))
            elif isinstance(self.posterior, BucketDelta) and bucket_ids is not None:
                per_sample_buckets = bucket_ids.mode(dim=1).values
                posterior_bias = torch.cat([
                    self.posterior.read(bucket_id=int(per_sample_buckets[i].item()), batch_size=1)
                    for i in range(x.size(0))
                ], dim=0)
                x = _add_memory_bias(x, posterior_bias.unsqueeze(1))
            elif isinstance(self.posterior, ResidualCache):
                query = x.detach().mean(dim=1)
                posterior_bias = self.posterior.read(query)
                x = _add_memory_bias(x, posterior_bias.unsqueeze(1))

        if return_final_states:
            assert final_states is not None
            assert all(s is not None for s in final_states), (
                "final_states has unvisited slots — virtual layer indices did not "
                "cover every physical layer"
            )
            if return_memory_meta:
                out: dict[str, Any] = {"hidden": x, "final_states": final_states}
                out["memory_meta"] = {
                    "memory_mode": memory_mode,
                    "cache_read_cutoff": cache_read_cutoff,
                    "memory_gate": memory_gate,
                    "memory_residual": memory_residual,
                }
                return out
            return x, final_states
        if return_memory_meta:
            out = {"hidden": x}
            out["memory_meta"] = {
                "memory_mode": memory_mode,
                "cache_read_cutoff": cache_read_cutoff,
                "memory_gate": memory_gate,
                "memory_residual": memory_residual,
            }
            return out
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_jacobian_stats: bool = False,
        memory_write_mode: str = "none",
        initial_states: list[torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
            input_ids: (batch, seq) token IDs.
            return_jacobian_stats: if True, compute and return Jacobian stats.
            memory_write_mode: "none" (default, side-effect free),
                "append_only" (append per-token KV pairs to buffer after forward).
                The caller (training loop or eval) decides when writes happen.
            initial_states: optional list of per-block initial recurrent states.
                When provided, length must equal ``num_layers``; each tensor is
                ``(batch, dim)``. ``None`` preserves the historical zero-init
                behavior and produces bit-identical logits to the pre-Task-3.5
                forward. The returned dict always includes ``final_states``:
                one ``(batch, dim)`` tensor per physical block, captured after
                the last virtual-layer step touches that block.
        """
        if initial_states is not None:
            if len(initial_states) != len(self.layers):
                raise ValueError(
                    f"initial_states length {len(initial_states)} does not match "
                    f"num_layers {len(self.layers)}"
                )
        x = self.embed(input_ids)

        # Wernicke layer: compose bytes into typed units before SSM recurrence
        balance_loss = None
        bucket_ids = None
        if self.wernicke is not None:
            x, bucket_ids, balance_loss = self.wernicke(x)

        # Buffer read path
        if self.outer_model is not None and self.buffer_mode == "append_only":
            # Typed buffer path: per-sample within-bucket retrieval.
            # Each sample reads from its own dominant bucket (mode over seq dim),
            # not one global bucket for the whole batch.
            # NOTE: This is a known simplification — writes are per-token (each
            # token goes to its Wernicke bucket) but reads are per-sample (one
            # dominant bucket per sequence). Per-token reads would require
            # per-position retrieval, which has the same O(batch*seq) cost we
            # avoid in writes. The dominant-bucket approximation is reasonable
            # when most tokens in a sequence share a primary type.
            batch = x.size(0)
            model_dim = x.size(2)
            if isinstance(self.outer_model, MultiSlotOuterModel) and self.outer_model._slots:
                if bucket_ids is not None:
                    per_sample_buckets = bucket_ids.mode(dim=1).values  # (batch,)
                else:
                    per_sample_buckets = torch.zeros(batch, dtype=torch.long, device=x.device)
                outer_read = torch.zeros(batch, 1, model_dim, device=x.device, dtype=x.dtype)
                for b_id in per_sample_buckets.unique():
                    mask = per_sample_buckets == b_id
                    cue = None
                    if self.retrieval_mode in ("bucket_topk", "softmax_all"):
                        cue = self.outer_model.cue_proj(
                            x[mask].detach().mean(dim=1).to(dtype=self.outer_model.cue_proj.weight.dtype)
                        )
                    read = self.outer_model.read_bucket(
                        int(mask.sum()), bucket_id=int(b_id.item()),
                        mode=self.retrieval_mode, k=self.retrieval_k, cue=cue,
                    )
                    outer_read[mask] = read.unsqueeze(1).to(dtype=x.dtype)
                x = x + outer_read

            # Bucket prototypes: per-sample prior bias
            if self.bucket_prototypes_module is not None and bucket_ids is not None:
                per_sample_buckets = bucket_ids.mode(dim=1).values  # (batch,)
                proto_bias = torch.zeros(batch, 1, model_dim, device=x.device, dtype=x.dtype)
                for b_id in per_sample_buckets.unique():
                    mask = per_sample_buckets == b_id
                    proto = self.bucket_prototypes_module.read(int(mask.sum()), int(b_id.item()))
                    proto_bias[mask] = proto.unsqueeze(1).to(dtype=x.dtype)
                x = x + proto_bias

        elif self.outer_model is not None:
            # Legacy path: unchanged
            if isinstance(self.outer_model, MultiSlotOuterModel):
                cue = x.detach().mean(dim=1) if self.cue_projection else None
                outer_read = self.outer_model.read(x.size(0), cue=cue)
            else:
                outer_read = self.outer_model.read(x.size(0))
            x = x + outer_read.unsqueeze(1).to(dtype=x.dtype)  # broadcast across seq dim

        if self.semantic_tier is not None:
            semantic_bias = self.semantic_tier.read(x.size(0))
            x = x + semantic_bias.unsqueeze(1).to(dtype=x.dtype)

        # Iterate over virtual layer indices so weight-tied depth recurrence
        # replays the shared group N times against the same physical modules.
        # When recurrence is disabled this sequence is list(range(num_layers))
        # and the behavior is bit-identical to the non-recurrent path.
        # NOTE: at count > 1, all_stats and per_layer_jacobian_stats contain
        # one entry per virtual step (not per physical layer).
        #
        # Activation checkpointing wraps each virtual-layer call, not each
        # physical layer. Under depth recurrence this means the recomputed
        # backward pass re-runs the shared block once per virtual step, which
        # is what we want — the autograd graph of the original forward has the
        # same structure. Gradient accumulation on the weight-tied shared
        # parameters happens via the usual autograd mechanism; checkpointing
        # is orthogonal. Skipped entirely when ``x`` has no grad (eval /
        # torch.no_grad) because checkpointing a no-grad path is a no-op at
        # best and a warning-emitting behavior drift at worst.
        use_ckpt = self.activation_checkpoint and torch.is_grad_enabled() and x.requires_grad
        all_stats: list[dict] = []
        # Per-physical-block final state, indexed by physical layer index.
        # Under depth recurrence (_virtual_layer_indices repeats indices),
        # the last virtual hit to a given physical block overwrites the slot,
        # matching the "state at end of sequence for this physical block"
        # semantics the Task 3.5 contract exposes.
        final_states: list[torch.Tensor | None] = [None] * len(self.layers)
        for layer_idx in self._virtual_layer_indices:
            layer = self.layers[layer_idx]
            init_state = initial_states[layer_idx] if initial_states is not None else None
            if use_ckpt:
                result = _checkpoint(
                    layer,
                    x,
                    return_jacobian_stats=return_jacobian_stats,
                    initial_state=init_state,
                    return_final_state=True,
                    use_reentrant=False,
                )
            else:
                result = layer(
                    x,
                    return_jacobian_stats=return_jacobian_stats,
                    initial_state=init_state,
                    return_final_state=True,
                )
            if return_jacobian_stats:
                x, stats, fstate = result
                all_stats.append(stats)
            else:
                x, fstate = result
            final_states[layer_idx] = fstate

        # Error-driven posterior read — add correction bias after SSM recurrence,
        # before LM head. The posterior update happens AFTER loss computation
        # in the training loop (same pattern as buffer writes).
        if self.posterior is not None:
            if isinstance(self.posterior, GlobalDelta):
                posterior_bias = self.posterior.read(x.size(0))
                x = x + posterior_bias.unsqueeze(1)
            elif isinstance(self.posterior, BucketDelta) and bucket_ids is not None:
                per_sample_buckets = bucket_ids.mode(dim=1).values
                posterior_bias = torch.cat([
                    self.posterior.read(bucket_id=int(per_sample_buckets[i].item()), batch_size=1)
                    for i in range(x.size(0))
                ], dim=0)
                x = x + posterior_bias.unsqueeze(1)
            elif isinstance(self.posterior, ResidualCache):
                # Use mean-pooled hidden as query context
                query = x.detach().mean(dim=1)  # (batch, dim)
                posterior_bias = self.posterior.read(query)
                x = x + posterior_bias.unsqueeze(1)

        hidden = x
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Memory writes are deferred to the caller (training loop / eval) so
        # forward() stays side-effect free by default.
        # Each token is written to its own Wernicke bucket, not collapsed
        # to one dominant bucket per batch, preserving per-token type fidelity.
        if (
            memory_write_mode == "append_only"
            and self.buffer_mode == "append_only"
            and self.outer_model is not None
            and isinstance(self.outer_model, MultiSlotOuterModel)
        ):
            wrote = self.append_memory_from_hidden(hidden, bucket_ids=bucket_ids)
            if (
                wrote
                and self.bucket_prototypes_module is not None
                and bucket_ids is not None
            ):
                batch, seq, dim = hidden.shape
                h_flat = hidden.detach().reshape(-1, dim)
                encoded_flat = torch.tanh(
                    self.outer_model.encoder(
                        h_flat.to(dtype=self.outer_model.encoder.weight.dtype)
                    )
                )
                bids_flat = bucket_ids.detach().reshape(-1)
                # Update bucket prototypes (encoded_flat is in outer_dim, which must
                # match prototype_dim — both default to 64 but assert to catch misconfig)
                assert encoded_flat.shape[-1] == self.bucket_prototypes_module.prototype_dim, (
                    f"outer_dim ({encoded_flat.shape[-1]}) != prototype_dim "
                    f"({self.bucket_prototypes_module.prototype_dim})"
                )
                self.bucket_prototypes_module.update_batch(bids_flat, encoded_flat)

        out: dict[str, Any] = {"logits": logits, "hidden": hidden}
        # Every physical block gets visited by at least one virtual-layer
        # iteration in standard configs; None slots would indicate a bug in
        # _virtual_layer_indices rather than a supported state. Assert here
        # so Task 3.5's contract (list[Tensor] of length num_layers) is
        # guaranteed to downstream callers.
        assert all(s is not None for s in final_states), (
            "final_states has unvisited slots — virtual layer indices did not "
            "cover every physical layer"
        )
        out["final_states"] = final_states
        if balance_loss is not None:
            out["balance_loss"] = balance_loss
        if bucket_ids is not None:
            out["bucket_ids"] = bucket_ids
        if return_jacobian_stats:
            # Average stats across layers
            merged: dict[str, torch.Tensor] = {}
            for key in all_stats[0]:
                merged[key] = torch.stack([s[key] for s in all_stats]).mean()
            out["jacobian_stats"] = merged
            if self.dynamic_crit_per_layer:
                out["per_layer_jacobian_stats"] = all_stats
        return out
