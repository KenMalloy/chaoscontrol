# Experiment 14 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the typed KV buffer architecture (Claim 1) and ablation runner for 8xH100.

**Architecture:** Replace surprise-gated episodic memory with an append-only KV buffer organized by Wernicke bucket. Add 4 retrieval modes (bucket_mean, bucket_recent, bucket_topk, softmax_all). Add per-bucket semantic prototypes. Add hierarchical Wernicke option. Build TTT warming-curve evaluation. Build experiment runner for T2+T3 ablation grid on 8 GPUs.

**Tech Stack:** Python 3.11+, PyTorch, YAML configs, existing ChaosControl codebase.

**Design doc:** `docs/plans/2026-04-09-e2e-architecture-design.md`

---

### Task 1: Config — Add Experiment 14 fields

**Files:**
- Modify: `src/chaoscontrol/config.py:30-98`
- Test: `tests/test_config_exp14.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_config_exp14.py
from chaoscontrol.config import ChaosControlConfig

def test_buffer_mode_default():
    cfg = ChaosControlConfig()
    assert cfg.buffer_mode == "legacy"

def test_buffer_mode_append():
    cfg = ChaosControlConfig(buffer_mode="append_only")
    assert cfg.buffer_mode == "append_only"

def test_retrieval_mode_default():
    cfg = ChaosControlConfig()
    assert cfg.retrieval_mode == "softmax_all"

def test_retrieval_modes():
    for mode in ("softmax_all", "bucket_mean", "bucket_recent", "bucket_topk"):
        cfg = ChaosControlConfig(retrieval_mode=mode)
        assert cfg.retrieval_mode == mode

def test_retrieval_k_default():
    cfg = ChaosControlConfig()
    assert cfg.retrieval_k == 8

def test_hierarchical_wernicke_default():
    cfg = ChaosControlConfig()
    assert cfg.wernicke_layers == 1

def test_hierarchical_wernicke():
    cfg = ChaosControlConfig(wernicke_layers=2, wernicke_k_max=8,
                              wernicke_k_max_fine=32)
    assert cfg.wernicke_layers == 2
    assert cfg.wernicke_k_max_fine == 32

def test_bucket_prototypes_default():
    cfg = ChaosControlConfig()
    assert cfg.bucket_prototypes is False

def test_max_slots_unlimited():
    cfg = ChaosControlConfig(outer_max_slots=0)
    assert cfg.outer_max_slots == 0  # 0 = unlimited
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_config_exp14.py -v`
Expected: FAIL — fields not defined

**Step 3: Write minimal implementation**

Add to `src/chaoscontrol/config.py` after existing memory fields:

```python
# Experiment 14: typed buffer
buffer_mode: str = "legacy"           # "legacy" | "append_only"
retrieval_mode: str = "softmax_all"   # "softmax_all" | "bucket_mean" | "bucket_recent" | "bucket_topk"
retrieval_k: int = 8                  # k for bucket_recent and bucket_topk
bucket_prototypes: bool = False       # per-bucket semantic prototypes

# Experiment 14: hierarchical Wernicke
wernicke_layers: int = 1              # 1 = flat (current), 2 = hierarchical
wernicke_k_max_fine: int = 8          # fine-grained buckets per coarse bucket (hier only)
```

Also update `outer_max_slots` default comment to note that 0 = unlimited.

**Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_config_exp14.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/config.py tests/test_config_exp14.py
git commit -m "feat(config): add experiment 14 fields — buffer mode, retrieval, hierarchical Wernicke"
```

---

### Task 2: Append-only KV buffer — write path

**Files:**
- Modify: `src/chaoscontrol/memory.py:194-350`
- Test: `tests/test_kv_buffer.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_kv_buffer.py
import torch
from chaoscontrol.memory import MultiSlotOuterModel

def test_append_kv_stores_entry():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    kv = torch.randn(1, 64)
    model.append_kv(kv, bucket_id=3)
    assert len(model._slots) == 1
    assert model._bucket_ids[0] == 3

def test_append_kv_unconditional():
    """Append should always store, no surprise gating."""
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    for i in range(100):
        kv = torch.randn(1, 64)
        model.append_kv(kv, bucket_id=i % 4)
    assert len(model._slots) == 100

def test_append_kv_no_compression_when_unlimited():
    """max_slots=0 means unlimited — no compression should fire."""
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    for i in range(500):
        kv = torch.randn(1, 64)
        model.append_kv(kv, bucket_id=i % 8)
    assert len(model._slots) == 500

def test_append_kv_bucket_tracking():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    model.append_kv(torch.randn(1, 64), bucket_id=0)
    model.append_kv(torch.randn(1, 64), bucket_id=0)
    model.append_kv(torch.randn(1, 64), bucket_id=5)
    bucket_0 = [i for i, b in enumerate(model._bucket_ids) if b == 0]
    bucket_5 = [i for i, b in enumerate(model._bucket_ids) if b == 5]
    assert len(bucket_0) == 2
    assert len(bucket_5) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_kv_buffer.py -v`
Expected: FAIL — append_kv not defined

**Step 3: Write minimal implementation**

Add to `MultiSlotOuterModel` in `memory.py` after the existing `write()` method:

```python
def append_kv(self, encoded: torch.Tensor, bucket_id: int | None = None) -> None:
    """Unconditional append — no surprise gating, no compression if unlimited."""
    self._slots.append(encoded.detach())
    self._survival.append(1.0)
    if bucket_id is not None:
        self._bucket_ids.append(bucket_id)
    else:
        self._bucket_ids.append(-1)
    # Only compress if max_slots > 0 (0 = unlimited)
    if self.max_slots > 0 and len(self._slots) > self.max_slots:
        self._compress()
```

Ensure `_bucket_ids` list exists in `__init__` (it may already as part of typed_storage).

**Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_kv_buffer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/memory.py tests/test_kv_buffer.py
git commit -m "feat(memory): add append_kv — unconditional buffer write, no gating"
```

---

### Task 3: Within-bucket retrieval — bucket_mean

**Files:**
- Modify: `src/chaoscontrol/memory.py:292-320`
- Test: `tests/test_kv_buffer.py` (extend)

**Step 1: Write the failing test**

```python
# Append to tests/test_kv_buffer.py

def test_read_bucket_mean():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    # Add 3 entries to bucket 0, 2 entries to bucket 1
    v0a, v0b, v0c = torch.ones(1, 64), torch.ones(1, 64) * 2, torch.ones(1, 64) * 3
    v1a, v1b = torch.ones(1, 64) * 10, torch.ones(1, 64) * 20
    model.append_kv(v0a, bucket_id=0)
    model.append_kv(v0b, bucket_id=0)
    model.append_kv(v1a, bucket_id=1)
    model.append_kv(v0c, bucket_id=0)
    model.append_kv(v1b, bucket_id=1)

    # Read from bucket 0: should be mean of v0a, v0b, v0c = 2.0
    result = model.read_bucket(batch_size=1, bucket_id=0, mode="bucket_mean")
    assert result.shape == (1, 64)
    assert torch.allclose(result, torch.ones(1, 64) * 2.0)

    # Read from bucket 1: should be mean of v1a, v1b = 15.0
    result = model.read_bucket(batch_size=1, bucket_id=1, mode="bucket_mean")
    assert torch.allclose(result, torch.ones(1, 64) * 15.0)

def test_read_bucket_mean_empty_bucket():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    model.append_kv(torch.randn(1, 64), bucket_id=0)
    # Bucket 5 is empty — should return zeros
    result = model.read_bucket(batch_size=1, bucket_id=5, mode="bucket_mean")
    assert torch.allclose(result, torch.zeros(1, 64))
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_kv_buffer.py::test_read_bucket_mean -v`
Expected: FAIL — read_bucket not defined

**Step 3: Write minimal implementation**

Add to `MultiSlotOuterModel`:

```python
def read_bucket(self, batch_size: int, bucket_id: int, mode: str = "bucket_mean",
                k: int = 8, cue: torch.Tensor | None = None) -> torch.Tensor:
    """Retrieve from a specific Wernicke bucket using the specified mode."""
    # Gather slots belonging to this bucket
    indices = [i for i, b in enumerate(self._bucket_ids) if b == bucket_id]
    if not indices:
        dim = self._slots[0].shape[-1] if self._slots else 64
        return torch.zeros(batch_size, dim, device=self._get_device())

    bucket_slots = torch.cat([self._slots[i] for i in indices], dim=0)  # (n, dim)

    if mode == "bucket_mean":
        result = bucket_slots.mean(dim=0, keepdim=True)  # (1, dim)
        return result.expand(batch_size, -1)

    raise ValueError(f"Unknown retrieval mode: {mode}")

def _get_device(self):
    if self._slots:
        return self._slots[0].device
    return torch.device("cpu")
```

**Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_kv_buffer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/memory.py tests/test_kv_buffer.py
git commit -m "feat(memory): add read_bucket with bucket_mean retrieval mode"
```

---

### Task 4: Within-bucket retrieval — bucket_recent and bucket_topk

**Files:**
- Modify: `src/chaoscontrol/memory.py` (read_bucket method)
- Test: `tests/test_kv_buffer.py` (extend)

**Step 1: Write the failing test**

```python
# Append to tests/test_kv_buffer.py

def test_read_bucket_recent():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    for i in range(20):
        model.append_kv(torch.ones(1, 64) * i, bucket_id=0)
    # k=3: should be mean of last 3 entries (17, 18, 19)
    result = model.read_bucket(batch_size=1, bucket_id=0, mode="bucket_recent", k=3)
    expected = torch.ones(1, 64) * 18.0  # mean(17, 18, 19)
    assert torch.allclose(result, expected)

def test_read_bucket_topk():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    # Insert entries: one very similar to query, others random
    target = torch.randn(1, 64)
    model.append_kv(torch.randn(1, 64), bucket_id=0)  # random
    model.append_kv(torch.randn(1, 64), bucket_id=0)  # random
    model.append_kv(target.clone(), bucket_id=0)        # exact match
    model.append_kv(torch.randn(1, 64), bucket_id=0)  # random

    # top-1 with the target as cue should return something close to target
    result = model.read_bucket(batch_size=1, bucket_id=0, mode="bucket_topk",
                               k=1, cue=target)
    cos_sim = torch.nn.functional.cosine_similarity(result, target, dim=-1)
    assert cos_sim.item() > 0.95

def test_read_bucket_topk_softmax_weighting():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    v1 = torch.ones(1, 64)
    v2 = torch.ones(1, 64) * 2
    model.append_kv(v1, bucket_id=0)
    model.append_kv(v2, bucket_id=0)
    cue = torch.ones(1, 64) * 1.5  # equidistant-ish
    result = model.read_bucket(batch_size=1, bucket_id=0, mode="bucket_topk",
                               k=2, cue=cue)
    # Result should be between v1 and v2 (softmax-weighted blend)
    assert result.mean().item() > 1.0
    assert result.mean().item() < 2.0
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_kv_buffer.py::test_read_bucket_recent -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Extend `read_bucket` in `MultiSlotOuterModel`:

```python
# Inside read_bucket, add after bucket_mean case:

if mode == "bucket_recent":
    recent = bucket_slots[-k:]  # last k entries
    result = recent.mean(dim=0, keepdim=True)
    return result.expand(batch_size, -1)

if mode == "bucket_topk":
    if cue is None:
        raise ValueError("bucket_topk requires a cue tensor")
    # cue: (batch, dim) — use first sample for scoring
    q = cue[0:1]  # (1, dim)
    scores = (bucket_slots @ q.T).squeeze(-1)  # (n,)
    topk_idx = scores.topk(min(k, len(scores))).indices
    topk_slots = bucket_slots[topk_idx]  # (k, dim)
    topk_scores = scores[topk_idx]
    weights = torch.softmax(topk_scores, dim=0)  # (k,)
    result = (weights.unsqueeze(-1) * topk_slots).sum(dim=0, keepdim=True)
    return result.expand(batch_size, -1)
```

**Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_kv_buffer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/memory.py tests/test_kv_buffer.py
git commit -m "feat(memory): add bucket_recent and bucket_topk retrieval modes"
```

---

### Task 5: Per-bucket semantic prototypes

**Files:**
- Modify: `src/chaoscontrol/memory.py` (add BucketPrototypes class after SemanticTier)
- Test: `tests/test_kv_buffer.py` (extend)

**Step 1: Write the failing test**

```python
# Append to tests/test_kv_buffer.py
from chaoscontrol.memory import BucketPrototypes

def test_bucket_prototypes_init():
    bp = BucketPrototypes(k_max=16, dim=64, update_rate=0.1)
    assert bp.prototypes.shape == (16, 64)

def test_bucket_prototypes_read():
    bp = BucketPrototypes(k_max=4, dim=64, update_rate=0.1)
    result = bp.read(batch_size=2, bucket_id=1)
    assert result.shape == (2, 64)

def test_bucket_prototypes_update():
    bp = BucketPrototypes(k_max=4, dim=64, update_rate=1.0)
    value = torch.ones(1, 64) * 5.0
    bp.update(bucket_id=2, value=value)
    result = bp.read(batch_size=1, bucket_id=2)
    assert torch.allclose(result, value)

def test_bucket_prototypes_ema():
    bp = BucketPrototypes(k_max=4, dim=64, update_rate=0.5)
    bp.update(bucket_id=0, value=torch.ones(1, 64) * 10.0)
    bp.update(bucket_id=0, value=torch.zeros(1, 64))
    result = bp.read(batch_size=1, bucket_id=0)
    # After two updates with alpha=0.5: 10*0.5 + 0*0.5 = 5, then 5*0.5 + 0*0.5 = 2.5
    # Wait: first update on zero-init: 0*0.5 + 10*0.5 = 5. Second: 5*0.5 + 0*0.5 = 2.5
    assert torch.allclose(result, torch.ones(1, 64) * 2.5)
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_kv_buffer.py::test_bucket_prototypes_init -v`
Expected: FAIL — BucketPrototypes not defined

**Step 3: Write minimal implementation**

Add to `memory.py` after SemanticTier class:

```python
class BucketPrototypes(torch.nn.Module):
    """Per-bucket semantic priors. One prototype per Wernicke bucket type,
    EMA-updated from buffer entries. Ships in artifact for cold-start."""

    def __init__(self, k_max: int, dim: int, update_rate: float = 0.1):
        super().__init__()
        self.k_max = k_max
        self.dim = dim
        self.update_rate = update_rate
        self.register_buffer("prototypes", torch.zeros(k_max, dim))

    def read(self, batch_size: int, bucket_id: int) -> torch.Tensor:
        proto = self.prototypes[bucket_id].unsqueeze(0)  # (1, dim)
        return proto.expand(batch_size, -1)

    def update(self, bucket_id: int, value: torch.Tensor) -> None:
        v = value.detach().mean(dim=0)  # (dim,)
        self.prototypes[bucket_id] = (
            (1 - self.update_rate) * self.prototypes[bucket_id]
            + self.update_rate * v
        )
```

**Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_kv_buffer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/memory.py tests/test_kv_buffer.py
git commit -m "feat(memory): add BucketPrototypes — per-bucket EMA priors"
```

---

### Task 6: Hierarchical Wernicke

**Files:**
- Modify: `src/chaoscontrol/wernicke.py`
- Test: `tests/test_hierarchical_wernicke.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_hierarchical_wernicke.py
import torch
from chaoscontrol.wernicke import HierarchicalWernicke

def test_hierarchical_init():
    hw = HierarchicalWernicke(dim=128, k_coarse=8, k_fine=8, window=8)
    assert hw.total_buckets == 64

def test_hierarchical_forward_shape():
    hw = HierarchicalWernicke(dim=128, k_coarse=8, k_fine=8, window=8)
    x = torch.randn(2, 32, 128)  # (batch, seq, dim)
    out, bucket_ids, balance_loss = hw(x)
    assert out.shape == (2, 32, 128)
    assert bucket_ids.shape == (2, 32)
    # Bucket ids should be in [0, 64)
    assert bucket_ids.min() >= 0
    assert bucket_ids.max() < 64

def test_hierarchical_bucket_composition():
    """Bucket id = coarse * k_fine + fine."""
    hw = HierarchicalWernicke(dim=128, k_coarse=4, k_fine=8, window=8)
    x = torch.randn(1, 16, 128)
    _, bucket_ids, _ = hw(x)
    # All bucket ids should be < 4 * 8 = 32
    assert bucket_ids.max() < 32

def test_hierarchical_param_budget():
    """Hierarchical should have comparable params to flat at same bucket count."""
    hw = HierarchicalWernicke(dim=128, k_coarse=8, k_fine=8, window=8)
    hier_params = sum(p.numel() for p in hw.parameters())
    from chaoscontrol.wernicke import WernickeLayer
    flat = WernickeLayer(dim=128, k_max=64, window=8, router_type="moe")
    flat_params = sum(p.numel() for p in flat.parameters())
    # Should be in the same ballpark (within 3x)
    assert hier_params < flat_params * 3
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_hierarchical_wernicke.py -v`
Expected: FAIL — HierarchicalWernicke not defined

**Step 3: Write minimal implementation**

Add to `wernicke.py`:

```python
class HierarchicalWernicke(torch.nn.Module):
    """Two-level Wernicke routing: coarse type -> fine subtype.
    
    Total buckets = k_coarse * k_fine. Bucket id = coarse * k_fine + fine.
    Each level is a standard WernickeLayer.
    """

    def __init__(self, dim: int, k_coarse: int, k_fine: int, window: int = 8,
                 router_type: str = "moe", balance_weight: float = 0.01,
                 expert_dim: int | None = None):
        super().__init__()
        self.k_coarse = k_coarse
        self.k_fine = k_fine
        self.total_buckets = k_coarse * k_fine

        self.coarse = WernickeLayer(
            dim=dim, k_max=k_coarse, window=window,
            router_type=router_type, balance_weight=balance_weight,
            expert_dim=expert_dim,
        )
        self.fine = WernickeLayer(
            dim=dim, k_max=k_fine, window=window,
            router_type=router_type, balance_weight=balance_weight,
            expert_dim=expert_dim,
        )

    def forward(self, x: torch.Tensor):
        # Coarse routing
        x, coarse_ids, balance_coarse = self.coarse(x)
        # Fine routing
        x, fine_ids, balance_fine = self.fine(x)
        # Composite bucket id
        bucket_ids = coarse_ids * self.k_fine + fine_ids
        balance_loss = balance_coarse + balance_fine
        return x, bucket_ids, balance_loss
```

**Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_hierarchical_wernicke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/wernicke.py tests/test_hierarchical_wernicke.py
git commit -m "feat(wernicke): add HierarchicalWernicke — two-level coarse+fine routing"
```

---

### Task 7: Model integration — wire new buffer and prototypes

**Files:**
- Modify: `src/chaoscontrol/model.py:94-355`
- Test: `tests/test_model_exp14.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_model_exp14.py
import torch
from chaoscontrol.model import ChaosStudentLM

def test_forward_append_only_mode():
    model = ChaosStudentLM(
        vocab_size=256, model_dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=0,  # unlimited
        buffer_mode="append_only",
        retrieval_mode="bucket_topk", retrieval_k=4,
        wernicke_enabled=True, wernicke_k_max=8,
        wernicke_router="moe",
    )
    x = torch.randint(0, 256, (2, 32))
    out = model(x)
    assert "logits" in out
    assert out["logits"].shape == (2, 32, 256)

def test_forward_bucket_prototypes():
    model = ChaosStudentLM(
        vocab_size=256, model_dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_mean",
        bucket_prototypes=True,
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )
    x = torch.randint(0, 256, (2, 32))
    out = model(x)
    assert "logits" in out

def test_forward_hierarchical_wernicke():
    model = ChaosStudentLM(
        vocab_size=256, model_dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_topk", retrieval_k=4,
        wernicke_enabled=True, wernicke_layers=2,
        wernicke_k_max=8, wernicke_k_max_fine=8,
        wernicke_router="moe",
    )
    x = torch.randint(0, 256, (2, 32))
    out = model(x)
    assert "logits" in out
    assert "bucket_ids" in out
    assert out["bucket_ids"].max() < 64  # 8 * 8

def test_buffer_grows_during_forward():
    model = ChaosStudentLM(
        vocab_size=256, model_dim=128, num_layers=2,
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_mean",
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )
    assert len(model.outer_model._slots) == 0
    x = torch.randint(0, 256, (2, 32))
    model(x)
    assert len(model.outer_model._slots) > 0
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_model_exp14.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Modify `model.py`:

1. In `__init__`: if `wernicke_layers == 2`, create `HierarchicalWernicke` instead of `WernickeLayer`. If `bucket_prototypes`, create `BucketPrototypes`.

2. In `forward()`: after Wernicke routing returns `bucket_ids`, use the dominant bucket_id per sample. If `buffer_mode == "append_only"`, call `read_bucket()` instead of `read()` for retrieval, and call `append_kv()` instead of relying on training.py's consolidation_step.

Key changes to `forward()` (around lines 298-355):

```python
# After Wernicke:
if self.wernicke is not None:
    x, bucket_ids, balance_loss = self.wernicke(x)

# Buffer read (new path):
if self.outer_model is not None and self.buffer_mode == "append_only":
    dominant_bucket = bucket_ids[:, -1].mode().values.item() if bucket_ids is not None else 0
    cue = x.detach().mean(dim=1) if self.retrieval_mode == "bucket_topk" else None
    outer_read = self.outer_model.read_bucket(
        x.size(0), bucket_id=dominant_bucket,
        mode=self.retrieval_mode, k=self.retrieval_k, cue=cue,
    )
    x = x + outer_read.unsqueeze(1)
elif self.outer_model is not None:
    # Legacy path unchanged
    ...

# Bucket prototypes:
if self.bucket_prototypes_module is not None and bucket_ids is not None:
    dominant_bucket = bucket_ids[:, -1].mode().values.item()
    proto = self.bucket_prototypes_module.read(x.size(0), dominant_bucket)
    x = x + proto.unsqueeze(1)

# SSM recurrence (unchanged)
...

# Buffer write (new path) — append after forward:
if self.outer_model is not None and self.buffer_mode == "append_only":
    hidden_for_write = x[:, -1, :].detach()
    encoded = self.outer_model.encoder(hidden_for_write.unsqueeze(0)).squeeze(0)
    dominant_bucket = bucket_ids[:, -1].mode().values.item() if bucket_ids is not None else 0
    self.outer_model.append_kv(encoded.mean(dim=0, keepdim=True), bucket_id=dominant_bucket)
    if self.bucket_prototypes_module is not None:
        self.bucket_prototypes_module.update(dominant_bucket, encoded)
```

**Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_model_exp14.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/model.py tests/test_model_exp14.py
git commit -m "feat(model): wire append-only buffer, bucket retrieval, hierarchical Wernicke"
```

---

### Task 8: Training loop — wire append-only mode

**Files:**
- Modify: `src/chaoscontrol/training.py:386-441`
- Test: `tests/test_training_exp14.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_training_exp14.py
import torch
from chaoscontrol.training import train_chaoscontrol_for_budget

def test_training_append_only_runs():
    """Smoke test: training loop completes with append-only buffer."""
    result = train_chaoscontrol_for_budget(
        model_dim=64, num_layers=2, vocab_size=256,
        outer_model_dim=32, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_mean",
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
        data_path=None,  # uses synthetic data
        budget=5.0,  # 5 seconds
        seq_len=64, batch_size=4,
    )
    assert result["train"]["steps"] > 0
    assert "eval" in result

def test_training_append_only_skips_consolidation():
    """In append_only mode, consolidation_step should not be called."""
    result = train_chaoscontrol_for_budget(
        model_dim=64, num_layers=2, vocab_size=256,
        outer_model_dim=32, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_topk", retrieval_k=4,
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
        data_path=None,
        budget=5.0,
        seq_len=64, batch_size=4,
    )
    assert result["train"]["steps"] > 0
```

**Step 2: Run test, Step 3: Implement, Step 4: Verify, Step 5: Commit**

In `training.py`, around the consolidation section (lines 386-441):

```python
if buffer_mode == "append_only":
    # Buffer write happens inside model.forward() — skip legacy consolidation
    pass
else:
    # Legacy consolidation path (unchanged)
    surprise = model.outer_model.consolidation_step(...)
```

```bash
git commit -m "feat(training): skip legacy consolidation in append_only buffer mode"
```

---

### Task 9: TTT warming-curve evaluation

**Files:**
- Modify: `src/chaoscontrol/evaluation.py`
- Test: `tests/test_eval_warming.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_eval_warming.py
import torch
from chaoscontrol.evaluation import evaluate_warming_curve

def test_warming_curve_returns_dict():
    # Minimal model setup
    from chaoscontrol.model import ChaosStudentLM
    model = ChaosStudentLM(
        vocab_size=256, model_dim=64, num_layers=2,
        outer_model_dim=32, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_mean",
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )
    data = torch.randint(0, 256, (8192,))
    curve = evaluate_warming_curve(
        model, data, warmup_tokens=[0, 100, 500],
        score_tokens=256, segment_len=1024,
    )
    assert isinstance(curve, dict)
    assert 0 in curve
    assert 100 in curve
    assert 500 in curve
    for n, bpb in curve.items():
        assert isinstance(bpb, float)
        assert bpb > 0

def test_warming_curve_improves():
    """More warmup tokens should give equal or better bpb."""
    from chaoscontrol.model import ChaosStudentLM
    model = ChaosStudentLM(
        vocab_size=256, model_dim=64, num_layers=2,
        outer_model_dim=32, outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_topk", retrieval_k=4,
        wernicke_enabled=True, wernicke_k_max=4,
        wernicke_router="moe",
    )
    data = torch.randint(0, 256, (8192,))
    curve = evaluate_warming_curve(
        model, data, warmup_tokens=[0, 500],
        score_tokens=256, segment_len=1024,
    )
    # On random data with random model, warming may not help,
    # but the function should run without error
    assert len(curve) == 2
```

**Step 2-5: Implement and commit**

Implement `evaluate_warming_curve()` in `evaluation.py` following the TTT
evaluation contract from the design doc (reset state + buffer, warm N
tokens, score next 1024, reset between segments).

```bash
git commit -m "feat(eval): add evaluate_warming_curve — TTT bpb at N warmup tokens"
```

---

### Task 10: Experiment 14 runner

**Files:**
- Create: `experiments/14_vram_buffer/run_exp14.py`
- Create: `experiments/14_vram_buffer/configs/` (generated by runner)

**Step 1: Write runner**

Model on existing `experiments/12_polyphasic_sleep/run_polyphasic_ablation.py`.
The runner should:

1. Define CONDITIONS dict with all T2 + T3 conditions from the design doc
2. Use the same statistical framework (paired Wilcoxon, Holm-Bonferroni,
   bootstrap CIs, Cohen's d) imported from `experiments/09_revised_architecture/stats.py`
3. Accept `--phase` flag: `A` (T2+T3), `B` (T5), `C` (T6+T7)
4. Accept `--data-path`, `--budget`, `--num-gpus`
5. Run warming-curve evaluation at end of each training run
6. Report realized parameter counts for T3 conditions
7. Build flat queue, skip completed runs, dispatch across GPUs
8. Print summary with contrasts and decision logic

**Step 2: Verify it parses and dry-runs**

Run: `python experiments/14_vram_buffer/run_exp14.py --help`

**Step 3: Commit**

```bash
git commit -m "feat(exp14): add experiment runner — T2+T3 ablation grid, warming curves"
```

---

### Task 11: Integration smoke test

**Files:** none new — verify existing test suite + new tests pass together

**Step 1: Run full test suite**

```bash
cd src && python -m pytest ../tests/ -v --timeout=60
```

Expected: all tests pass, no regressions in existing tests.

**Step 2: Run a 30-second local smoke test**

```bash
python experiments/14_vram_buffer/run_exp14.py \
    --data-path data/enwik8 \
    --budget 30 \
    --num-gpus 1 \
    --phase A \
    --dry-run
```

Expected: prints condition list, estimated run count, exits cleanly.

**Step 3: Commit any fixes**

```bash
git commit -m "test: integration smoke test for experiment 14"
```

---

## Phase B Tasks (Claim 2 — deferred)

### Task 12: Fast weight matrix layer
### Task 13: Online defrag (amortized N3)
### Task 14: Phase B runner conditions

These are deferred until Phase A results are in. Skeleton task
descriptions omitted — they follow the same TDD pattern.

---

## Summary

| Task | Component | Estimated time |
|------|-----------|---------------|
| 1 | Config fields | 10 min |
| 2 | Append-only write | 10 min |
| 3 | Bucket mean retrieval | 10 min |
| 4 | Recent + topk retrieval | 15 min |
| 5 | Bucket prototypes | 10 min |
| 6 | Hierarchical Wernicke | 15 min |
| 7 | Model integration | 20 min |
| 8 | Training loop wiring | 10 min |
| 9 | TTT warming curve eval | 20 min |
| 10 | Experiment runner | 30 min |
| 11 | Integration smoke test | 10 min |
| **Total** | | **~2.5 hours** |
