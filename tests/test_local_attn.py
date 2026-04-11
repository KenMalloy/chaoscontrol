"""Tests for LocalAttention and RollingKVCache."""
import torch
from chaoscontrol.local_attn import LocalAttention, RollingKVCache


def test_rolling_kv_cache_write_and_read():
    cache = RollingKVCache(window=4, dim=8)
    for i in range(6):
        k = torch.full((2, 8), float(i))
        v = torch.full((2, 8), float(i) * 10)
        cache.write(k, v)
    keys, values, mask = cache.last(4)
    assert keys.shape == (2, 4, 8)
    assert values.shape == (2, 4, 8)
    assert mask.shape == (2, 4)
    # After 6 writes into window=4, oldest entries are positions 2-5
    assert keys[0, -1, 0].item() == 5.0


def test_rolling_kv_cache_partial_fill():
    cache = RollingKVCache(window=8, dim=4)
    cache.write(torch.ones(1, 4), torch.ones(1, 4))
    cache.write(torch.ones(1, 4) * 2, torch.ones(1, 4) * 2)
    keys, values, mask = cache.last(8)
    assert mask[0, :2].all()
    assert not mask[0, 2:].any()


def test_local_attention_output_shape():
    attn = LocalAttention(model_dim=32, attn_dim=8, num_heads=1)
    query = torch.randn(2, 32)
    keys = torch.randn(2, 16, 8)
    values = torch.randn(2, 16, 8)
    mask = torch.ones(2, 16, dtype=torch.bool)
    out = attn(query, keys, values, mask)
    assert out.shape == (2, 32)


def test_local_attention_masks_invalid():
    attn = LocalAttention(model_dim=16, attn_dim=8, num_heads=1)
    query = torch.randn(1, 16)
    keys = torch.randn(1, 4, 8)
    values = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    out = attn(query, keys, values, mask)
    assert out.shape == (1, 16)
    assert torch.isfinite(out).all()
