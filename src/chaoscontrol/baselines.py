"""Baseline models for comparison."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from chaoscontrol.core import RMSNorm, FeedForward


class SimpleTransformerLM(nn.Module):
    """Minimal causal transformer for baseline comparison."""

    def __init__(self, vocab_size=256, dim=128, num_layers=4, num_heads=4, ff_mult=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_mult) for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Duck-typing compatibility with ChaosStudentLM for shared train/eval
        self.outer_model = None
        self.wernicke = None
        self.wernicke_balance_weight = 0.0
        self.semantic_tier = None

    def artifact_bytes(self):
        return int(sum(p.numel() for p in self.parameters()) * 2)

    def forward(self, input_ids, *, return_jacobian_stats=False):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        hidden = x
        x = self.final_norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits, "hidden": hidden}
        if return_jacobian_stats:
            # Transformer has no Jacobian stats — return zeros
            out["jacobian_stats"] = {
                "lambda_max": torch.tensor(0.0),
                "sv_log_var": torch.tensor(0.0),
            }
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=2):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ff_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.ff = FeedForward(dim, ff_mult)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)
