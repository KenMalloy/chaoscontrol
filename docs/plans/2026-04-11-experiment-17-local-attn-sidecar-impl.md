# Exp 17: Local Attention Sidecar — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether a tiny local attention sidecar in the top SSM block improves the fast SP-SSM backbone.

**Architecture:** Add `ChaosSSMHybridBlock` to `model.py` — wraps a `ChaosSSMBlock` with a local attention path (q/k/v projections, rolling KV cache, gated mixing). Runner trains 4 conditions × 7 seeds on SP8192 data with the compiled diag scan.

**Tech Stack:** PyTorch, sentencepiece, yaml, existing chaoscontrol training/eval infrastructure.

---

### Task 1: LocalAttention module and KV cache

**Files:**
- Create: `src/chaoscontrol/local_attn.py`
- Test: `tests/test_local_attn.py`

**Step 1: Write the failing test**

```python
# tests/test_local_attn.py
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
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_local_attn.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# src/chaoscontrol/local_attn.py
"""Local attention module and rolling KV cache for hybrid SSM blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RollingKVCache:
    """Fixed-size rolling buffer for K/V pairs."""

    def __init__(self, window: int, dim: int) -> None:
        self.window = window
        self.dim = dim
        self._keys: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []

    def write(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append (batch, dim) key/value pair."""
        self._keys.append(k.detach())
        self._values.append(v.detach())
        if len(self._keys) > self.window:
            self._keys.pop(0)
            self._values.pop(0)

    def last(self, w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return last w entries as (batch, w, dim) with validity mask."""
        n = len(self._keys)
        if n == 0:
            batch = 1
            keys = torch.zeros(batch, w, self.dim)
            values = torch.zeros(batch, w, self.dim)
            mask = torch.zeros(batch, w, dtype=torch.bool)
            return keys, values, mask
        batch = self._keys[0].shape[0]
        device = self._keys[0].device
        dtype = self._keys[0].dtype
        keys = torch.zeros(batch, w, self.dim, device=device, dtype=dtype)
        values = torch.zeros(batch, w, self.dim, device=device, dtype=dtype)
        mask = torch.zeros(batch, w, device=device, dtype=torch.bool)
        fill = min(n, w)
        start = n - fill
        for i in range(fill):
            keys[:, i] = self._keys[start + i]
            values[:, i] = self._values[start + i]
            mask[:, i] = True
        return keys, values, mask

    def reset(self) -> None:
        self._keys.clear()
        self._values.clear()


class LocalAttention(nn.Module):
    """Single-query attention over a bounded KV window."""

    def __init__(self, model_dim: int, attn_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.head_dim = attn_dim // num_heads
        self.q_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, model_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, model_dim)
            keys: (batch, w, attn_dim) — pre-projected
            values: (batch, w, attn_dim) — pre-projected
            mask: (batch, w) bool
        Returns:
            (batch, model_dim)
        """
        q = self.q_proj(query)  # (batch, attn_dim)
        B, W, _ = keys.shape
        nh, hd = self.num_heads, self.head_dim
        q = q.view(B, nh, 1, hd)
        k = keys.view(B, W, nh, hd).permute(0, 2, 1, 3)  # (B, nh, W, hd)
        v = values.view(B, W, nh, hd).permute(0, 2, 1, 3)
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, W)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)
        scores = scores.masked_fill(~attn_mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # (B, nh, 1, hd)
        out = out.view(B, nh * hd)
        return self.out_proj(out)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_local_attn.py -v`
Expected: 4 passed

**Step 5: Commit**

```
git add src/chaoscontrol/local_attn.py tests/test_local_attn.py
git commit -m "feat: LocalAttention module and RollingKVCache"
```

---

### Task 2: ChaosSSMHybridBlock

**Files:**
- Modify: `src/chaoscontrol/model.py`
- Test: `tests/test_model.py` (add tests)

**Step 1: Write the failing test**

Add to `tests/test_model.py`:

```python
def test_hybrid_block_forward_shape():
    from chaoscontrol.model import ChaosSSMHybridBlock
    block = ChaosSSMHybridBlock(
        dim=32, ff_mult=2, a_mode="diag",
        local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
    )
    x = torch.randn(2, 12, 32)
    y = block(x)
    assert y.shape == (2, 12, 32)


def test_hybrid_block_step_shape():
    from chaoscontrol.model import ChaosSSMHybridBlock
    block = ChaosSSMHybridBlock(
        dim=32, ff_mult=2, a_mode="diag",
        local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
    )
    state = torch.zeros(2, 32)
    x = torch.randn(2, 32)
    out, new_state = block.step(x, state)
    assert out.shape == (2, 32)
    assert new_state.shape == (2, 32)


def test_hybrid_block_gate_starts_near_zero():
    from chaoscontrol.model import ChaosSSMHybridBlock
    block = ChaosSSMHybridBlock(
        dim=32, ff_mult=2, a_mode="diag",
        local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
    )
    # gate_bias initialized to -4, sigmoid(-4) ~ 0.018
    assert block.gate_bias.item() < -3.0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_model.py -v -k hybrid`
Expected: FAIL with ImportError

**Step 3: Write implementation**

Add to `src/chaoscontrol/model.py` after `ChaosSSMBlock`:

```python
from chaoscontrol.local_attn import LocalAttention, RollingKVCache


class ChaosSSMHybridBlock(nn.Module):
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
        local_attn_window: int = 64,
        local_attn_heads: int = 1,
        local_attn_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.ff_norm = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_mult)
        self.core = ChaosSSMCore(
            dim, a_mode=a_mode, a_full_rank=a_full_rank,
            a_full_gamma=a_full_gamma,
        )
        self.rich_b = None  # compatibility with ChaosSSMBlock

        # Local attention sidecar
        self.local_attn_window = local_attn_window
        self.local_attn_dim = local_attn_dim
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
        x_ssm = x + y

        if kv_cache is not None:
            keys, values, mask = kv_cache.last(self.local_attn_window)
            if mask.any():
                attn_out = self.local_attn(x_ssm, keys, values, mask)
                gate = torch.sigmoid(self.gate_proj(x_ssm) + self.gate_bias)
                x_ssm = x_ssm + gate * attn_out
            kv_cache.write(self.k_proj(x_ssm), self.v_proj(x_ssm))

        x_out = x_ssm + self.ff(self.ff_norm(x_ssm))
        return x_out, new_state

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_jacobian_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Sequence-level forward. Processes token-by-token for KV cache."""
        batch, seq, dim = x.shape
        kv_cache = self._init_kv_cache()
        state = x.new_zeros((batch, dim))
        outputs = []
        for t in range(seq):
            out, state = self.step(x[:, t], state, kv_cache=kv_cache)
            outputs.append(out)
        y = torch.stack(outputs, dim=1)
        if return_jacobian_stats:
            return y, {"lambda_max": torch.tensor(0.0), "sv_log_var": torch.tensor(0.0)}
        return y
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_model.py -v -k hybrid`
Expected: 3 passed

**Step 5: Commit**

```
git add src/chaoscontrol/model.py tests/test_model.py
git commit -m "feat: ChaosSSMHybridBlock with local attention sidecar"
```

---

### Task 3: Wire hybrid block into ChaosStudentLM

**Files:**
- Modify: `src/chaoscontrol/model.py` (ChaosStudentLM.__init__ and step)

**Step 1: Write the failing test**

Add to `tests/test_model.py`:

```python
def test_student_lm_with_hybrid_top_block():
    from chaoscontrol.model import ChaosStudentLM
    model = ChaosStudentLM(
        vocab_size=64, dim=32, num_layers=4, ff_mult=2,
        a_mode="diag", outer_model_dim=0, wernicke_enabled=False,
        local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
    )
    x = torch.randint(0, 64, (2, 10))
    out = model(x)
    assert out["logits"].shape == (2, 10, 64)


def test_student_lm_hybrid_step():
    from chaoscontrol.model import ChaosStudentLM
    model = ChaosStudentLM(
        vocab_size=64, dim=32, num_layers=4, ff_mult=2,
        a_mode="diag", outer_model_dim=0, wernicke_enabled=False,
        local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
    )
    states = model.init_state(2)
    token = torch.randint(0, 64, (2, 1))
    logits, hidden, new_states = model.step(token, states)
    assert logits.shape == (2, 64)
    assert len(new_states) == 4


def test_student_lm_no_hybrid_by_default():
    from chaoscontrol.model import ChaosStudentLM, ChaosSSMBlock
    model = ChaosStudentLM(
        vocab_size=64, dim=32, num_layers=4, ff_mult=2,
        a_mode="diag", outer_model_dim=0, wernicke_enabled=False,
    )
    # All layers should be plain SSM blocks
    for layer in model.layers:
        assert isinstance(layer, ChaosSSMBlock)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_model.py -v -k "hybrid_step or hybrid_top or no_hybrid"`
Expected: FAIL — ChaosStudentLM doesn't accept local_attn_window

**Step 3: Modify ChaosStudentLM**

In `ChaosStudentLM.__init__`, add parameters and modify layer construction:

```python
# Add to __init__ signature:
local_attn_window: int = 0,
local_attn_heads: int = 1,
local_attn_dim: int = 64,

# Replace the self.layers construction:
if local_attn_window > 0:
    ssm_layers = [
        ChaosSSMBlock(dim, ff_mult, a_mode=a_mode, ...)
        for _ in range(num_layers - 1)
    ]
    hybrid_layer = ChaosSSMHybridBlock(
        dim, ff_mult, a_mode=a_mode,
        a_full_rank=a_full_rank, a_full_gamma=a_full_gamma,
        local_attn_window=local_attn_window,
        local_attn_heads=local_attn_heads,
        local_attn_dim=local_attn_dim,
    )
    self.layers = nn.ModuleList(ssm_layers + [hybrid_layer])
else:
    self.layers = nn.ModuleList([
        ChaosSSMBlock(dim, ff_mult, ...)
        for _ in range(num_layers)
    ])

# Store for KV cache management:
self.local_attn_window = local_attn_window
```

In `ChaosStudentLM.step()`, pass KV cache to hybrid block:

```python
# The step() method needs a kv_cache argument.
# Add _kv_cache as an optional attribute initialized in init_state.
```

Note: The existing `step()` calls `layer.step(x, states[i])` for
all layers. For the hybrid block, it needs `kv_cache=...`. Add a
`self._kv_caches` list that gets initialized alongside states and
passed to hybrid blocks.

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_model.py -v -k "hybrid or no_hybrid"`
Expected: 4 passed (3 new + existing tests unbroken)

**Step 5: Commit**

```
git add src/chaoscontrol/model.py tests/test_model.py
git commit -m "feat: wire hybrid block into ChaosStudentLM"
```

---

### Task 4: Exp 17 runner

**Files:**
- Create: `experiments/17_local_attn_sidecar/runner_exp17.py`
- Create: `experiments/17_local_attn_sidecar/test_exp17.py`

**Step 1: Write the failing test**

```python
# experiments/17_local_attn_sidecar/test_exp17.py
import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner_exp17 import build_model


def test_build_bare_ssm():
    config = {
        "model_type": "ssm", "vocab_size": 64, "model_dim": 32,
        "num_layers": 4, "ff_mult": 2, "a_mode": "diag",
        "local_attn_window": 0,
    }
    model = build_model(config, torch.device("cpu"), torch.float32)
    assert model.vocab_size == 64
    assert model.local_attn_window == 0


def test_build_hybrid_model():
    config = {
        "model_type": "ssm", "vocab_size": 64, "model_dim": 32,
        "num_layers": 4, "ff_mult": 2, "a_mode": "diag",
        "local_attn_window": 8, "local_attn_heads": 1,
        "local_attn_dim": 16,
    }
    model = build_model(config, torch.device("cpu"), torch.float32)
    assert model.local_attn_window == 8


def test_hybrid_model_trains_without_nan():
    config = {
        "model_type": "ssm", "vocab_size": 64, "model_dim": 16,
        "num_layers": 2, "ff_mult": 2, "a_mode": "diag",
        "local_attn_window": 4, "local_attn_heads": 1,
        "local_attn_dim": 8,
    }
    model = build_model(config, torch.device("cpu"), torch.float32)
    x = torch.randint(0, 64, (2, 16))
    out = model(x)
    logits = out["logits"]
    assert not logits.isnan().any()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 64), x[:, 1:].reshape(-1).clamp(0, 63),
    )
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert not p.grad.isnan().any()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest experiments/17_local_attn_sidecar/test_exp17.py -v`
Expected: FAIL with ImportError

**Step 3: Write runner**

The runner follows the same structure as `runner_exp16.py` but simpler
(no probe, no oracle). Key differences:

- `build_model()` passes `local_attn_window`, `local_attn_heads`,
  `local_attn_dim` to `ChaosStudentLM`
- No probe phase — just train and evaluate bpb
- Reports: bpb, steps, steps/s, params, artifact_bytes

```python
# experiments/17_local_attn_sidecar/runner_exp17.py
# Core structure: build_model -> train -> evaluate -> report
# Reuse: load_sp_data, evaluate_bpb_sp, build_sentencepiece_luts
# from runner_exp16 pattern (or better, factor shared code)
```

The runner should be self-contained (copy the SP data loading and
bpb eval helpers rather than importing from exp16).

**Step 4: Run tests**

Run: `python3 -m pytest experiments/17_local_attn_sidecar/test_exp17.py -v`
Expected: 3 passed

**Step 5: Commit**

```
git add experiments/17_local_attn_sidecar/runner_exp17.py
git add experiments/17_local_attn_sidecar/test_exp17.py
git commit -m "feat: Exp 17 runner — local attention sidecar training"
```

---

### Task 5: Exp 17 orchestrator

**Files:**
- Create: `experiments/17_local_attn_sidecar/run_exp17.py`

**Step 1: Write orchestrator**

Follows `run_exp16.py` structure. 4 conditions × 7 seeds = 28 runs.

```python
CONDITIONS = {
    "bare_fast_ssm": _base(local_attn_window=0),
    "local_w16": _base(local_attn_window=16, local_attn_heads=1, local_attn_dim=64),
    "local_w32": _base(local_attn_window=32, local_attn_heads=1, local_attn_dim=64),
    "local_w64": _base(local_attn_window=64, local_attn_heads=1, local_attn_dim=64),
}
```

Summary should report per-condition:
- mean bpb ± SEM with 95% CI
- mean steps/s
- artifact bytes
- bpb delta vs bare_fast_ssm with p-value

Go/no-go gates:
1. Any local_w beats bare by ≥ 0.02 bpb
2. Winner steps/s ≥ 50% of bare steps/s
3. Artifact < 16MB

**Step 2: Syntax check**

Run: `python3 -c "import ast; ast.parse(open('experiments/17_local_attn_sidecar/run_exp17.py').read()); print('OK')"`

**Step 3: Commit**

```
git add experiments/17_local_attn_sidecar/run_exp17.py
git commit -m "feat: Exp 17 orchestrator — 4 conditions × 7 seeds"
```

---

### Task 6: Integration test on GPU

**Files:** None new — runs existing code on pod

**Step 1: Push code to pod**

```bash
rsync src/ experiments/17_local_attn_sidecar/ tests/ to pod
```

**Step 2: Run tests on pod**

```bash
pytest tests/test_local_attn.py tests/test_model.py -v -k "hybrid or local_attn"
pytest experiments/17_local_attn_sidecar/test_exp17.py -v
```

**Step 3: Smoke test — single short run**

```bash
python experiments/17_local_attn_sidecar/runner_exp17.py \
    --config <temp_config.yaml> \
    --data-path .../fineweb10B_sp8192 \
    --sp-model-path .../fineweb_8192_bpe.model \
    --budget 60 --output-json /tmp/smoke.json
```

Verify: non-NaN bpb, reasonable steps count, finite gradients.

**Step 4: Commit any fixes**

```
git commit -m "fix: integration fixes from GPU smoke test"
```

---

### Task 7: Launch full experiment

**Step 1: Launch on pod**

```bash
nohup python experiments/17_local_attn_sidecar/run_exp17.py \
    --data-path .../fineweb10B_sp8192 \
    --sp-model-path .../fineweb_8192_bpe.model \
    --budget 600 --num-gpus 4 > /workspace/exp17_run.log 2>&1 &
```

**Step 2: Monitor with /loop**

Check every ~8 min, pull results, stop pod when done.

**Step 3: Commit results**

```
git add experiments/17_local_attn_sidecar/VERDICT.md
git commit -m "data: Exp 17 Phase A complete"
```
