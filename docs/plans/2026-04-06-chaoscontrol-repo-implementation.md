# ChaosControl Research Repo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-contained research repo with modular source, 65 unit tests, and 8 reproducible experiment directories — ready to run on a single GPU pod.

**Architecture:** Extract the 1796-LOC monolith from `parameter-golf/tools/chaoscontrol.py` into focused modules under `src/chaoscontrol/`. Copy utility functions from parameter-golf into `data.py` (no external dependency). Add three new components: semantic memory tier, structured projections for metabolic gate, compression-consequence signal for Wernicke typing. Scaffold 8 experiment directories with configs, run scripts, and analysis stubs.

**Tech Stack:** Python 3.14, PyTorch, dataclasses, YAML configs, pytest. Venv at `.venv/`. Run tests with `.venv/bin/python -m pytest`.

**Source repo:** `/Users/kennethmalloy/Local Documents/Developer/parameter-golf` (branch `chaoscontrol`)
**Target repo:** `/Users/kennethmalloy/Local Documents/Developer/chaoscontrol`

---

### Task 1: Repo scaffolding — pyproject.toml, .gitignore, directory structure

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/chaoscontrol/__init__.py`
- Create: all empty `__init__.py` and `.gitkeep` files for directory structure

**Step 1: Create pyproject.toml**

```toml
[project]
name = "chaoscontrol"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["torch>=2.0", "numpy", "pyyaml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = [
    "ignore::DeprecationWarning:importlib._bootstrap",
    "ignore:builtin type Swig:DeprecationWarning",
]
```

**Step 2: Create .gitignore**

```
.venv/
__pycache__/
*.pyc
*.egg-info/
results/
*.pt
*.bin
enwik8
```

**Step 3: Create directory structure**

```bash
mkdir -p src/chaoscontrol
mkdir -p tests
mkdir -p experiments/{01_baseline,02_critical_dynamics,03_state_dependent_routing,04_long_term_memory,05_typed_composition,06_metabolic_gate,07_full_system,08_gap_analysis}/{configs,results}
mkdir -p analysis
mkdir -p docs/research
touch src/chaoscontrol/__init__.py
```

**Step 4: Add .gitkeep to empty results dirs**

```bash
for d in experiments/*/results; do touch "$d/.gitkeep"; done
```

**Step 5: Commit**

```bash
git add -A
git commit -m "scaffold: repo structure, pyproject.toml, .gitignore"
```

---

### Task 2: Extract data.py — self-contained utilities

**Files:**
- Create: `src/chaoscontrol/data.py`
- Test: `tests/test_data.py`

**Step 1: Write the failing test**

```python
# tests/test_data.py
import torch
from chaoscontrol.data import (
    batch_from_starts,
    build_lm_starts,
    resolve_device,
    resolve_param_dtype,
    maybe_autocast,
    maybe_sync_cuda,
    load_enwik8_splits,
    choose_eval_starts,
)

def test_build_lm_starts():
    starts = build_lm_starts(100, seq_len=10, stride=5)
    assert len(starts) > 0
    assert all(s + 10 + 1 <= 100 for s in starts)

def test_batch_from_starts():
    tokens = torch.arange(100)
    starts = [0, 10, 20]
    inputs, targets = batch_from_starts(tokens, starts, seq_len=8, device=torch.device("cpu"))
    assert inputs.shape == (3, 8)
    assert targets.shape == (3, 8)

def test_resolve_device():
    d = resolve_device("cpu")
    assert d == torch.device("cpu")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write data.py**

Copy the following functions from parameter-golf into `src/chaoscontrol/data.py`:
- `resolve_device` from `tools/evolutionary_benchmark.py:120-123`
- `resolve_param_dtype` from `tools/evolutionary_benchmark.py:126-134`
- `maybe_autocast` from `tools/evolutionary_benchmark.py:137-141` (add `from contextlib import nullcontext`)
- `maybe_sync_cuda`: `def maybe_sync_cuda(device): ... if device.type == "cuda": torch.cuda.synchronize()`
- `load_enwik8_splits` from `tools/evolutionary_benchmark.py:954-971` (add `import numpy as np`)
- `maybe_cache_tokens_on_device` from `tools/evolutionary_benchmark.py:149-152`
- `prepare_tokenized_enwik8_splits`: simplified version that only supports bytes mode (no sentencepiece)
- `choose_eval_starts` from `tools/evolutionary_benchmark.py:1087-end`
- `build_lm_starts` from `spectral_flood_walk_v2a.py:66-70`
- `batch_from_starts` from `spectral_flood_walk_v2a.py:84-92`

Add `import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` to conftest.py for imports.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/data.py tests/test_data.py tests/conftest.py
git commit -m "feat: add data.py — self-contained data utilities"
```

---

### Task 3: Extract config.py

**Files:**
- Create: `src/chaoscontrol/config.py`
- Test: `tests/test_config.py`

**Step 1: Write tests**

Copy `TestChaosControlConfig` from parameter-golf's `tests/test_chaoscontrol.py`. Add test for new fields (`metabolic_threshold_mode`, `outer_model_type`, `wernicke_enabled`, etc.).

**Step 2: Extract ChaosControlConfig**

Copy the dataclass from `parameter-golf/tools/chaoscontrol.py:34-84` into `src/chaoscontrol/config.py`. No imports beyond stdlib + dataclass.

**Step 3: Test, commit**

```bash
git commit -m "feat: add config.py — ChaosControlConfig dataclass"
```

---

### Task 4: Extract core.py — ChaosSSMCore, RMSNorm, FeedForward

**Files:**
- Create: `src/chaoscontrol/core.py`
- Test: `tests/test_core.py`

**Step 1: Write tests**

Copy `TestChaosSSMCore`, `TestAPaired`, `TestAFull`, `TestCriticalityRegularizer` from parameter-golf tests.

**Step 2: Extract classes**

- `RMSNorm`
- `FeedForward`
- `ChaosSSMCore` (with all three A modes)
- `criticality_loss` function

**Step 3: Test, commit**

```bash
git commit -m "feat: add core.py — ChaosSSMCore with diag/paired/full A modes"
```

---

### Task 5: Extract routing.py — RichBNN, DistributedB

**Files:**
- Create: `src/chaoscontrol/routing.py`
- Test: `tests/test_routing.py`

**Step 1: Write tests**

Copy `TestRichBNN`, `TestDistributedBHub`, `TestDistributedBAssembly`, `TestDistributedBHybrid` from parameter-golf tests.

**Step 2: Extract classes**

- `RichBNN`
- `DistributedB` (with settling_norm)

**Step 3: Test, commit**

```bash
git commit -m "feat: add routing.py — RichBNN and DistributedB"
```

---

### Task 6: Extract memory.py — OuterModel, MultiSlotOuterModel

**Files:**
- Create: `src/chaoscontrol/memory.py`
- Test: `tests/test_memory.py`

**Step 1: Write tests**

Copy `TestOuterModel`, `TestMultiSlotOuterModel`, `TestCheckpointPersistence` from parameter-golf tests.

**Step 2: Extract classes**

- `OuterModel`
- `MultiSlotOuterModel` (with get/set_extra_state, typed compression, survival scoring)

**Step 3: Test, commit**

```bash
git commit -m "feat: add memory.py — OuterModel and MultiSlotOuterModel"
```

---

### Task 7: Extract wernicke.py — WernickeLayer

**Files:**
- Create: `src/chaoscontrol/wernicke.py`
- Test: `tests/test_wernicke.py`

**Step 1: Write tests**

Copy `TestWernickeLayer` from parameter-golf tests.

**Step 2: Extract WernickeLayer**

Import `RMSNorm` from `core.py`.

**Step 3: Test, commit**

```bash
git commit -m "feat: add wernicke.py — WernickeLayer with VQ/MoE routing"
```

---

### Task 8: Extract metabolic.py — metabolic_fork

**Files:**
- Create: `src/chaoscontrol/metabolic.py`
- Test: `tests/test_metabolic.py`

**Step 1: Write tests**

Copy `TestMetabolicGate` from parameter-golf tests.

**Step 2: Extract metabolic_fork function**

**Step 3: Test, commit**

```bash
git commit -m "feat: add metabolic.py — metabolic_fork with scoring modes"
```

---

### Task 9: Extract model.py — ChaosSSMBlock, ChaosStudentLM

**Files:**
- Create: `src/chaoscontrol/model.py`
- Test: `tests/test_model.py`

**Step 1: Write tests**

Copy `TestChaosStudentLM` from parameter-golf tests.

**Step 2: Extract ChaosSSMBlock and ChaosStudentLM**

Import from `core`, `routing`, `memory`, `wernicke`, `metabolic`.

**Step 3: Test, commit**

```bash
git commit -m "feat: add model.py — ChaosSSMBlock and ChaosStudentLM"
```

---

### Task 10: Extract training.py and evaluation.py

**Files:**
- Create: `src/chaoscontrol/training.py`
- Create: `src/chaoscontrol/evaluation.py`
- Test: `tests/test_training.py`

**Step 1: Write tests**

Copy `TestTraining` from parameter-golf tests.

**Step 2: Extract functions**

- `train_chaoscontrol_for_budget` → `training.py`
- `evaluate_chaoscontrol_bpb` → `evaluation.py`
- `build_chaoscontrol_matrix`, `run_chaoscontrol_matrix` → `training.py`
- `parse_chaoscontrol_args` → `training.py`

**Step 3: Test, commit**

```bash
git commit -m "feat: add training.py and evaluation.py"
```

---

### Task 11: Add baselines.py — SimpleTransformerLM

**Files:**
- Create: `src/chaoscontrol/baselines.py`
- Test: `tests/test_baselines.py`

**Step 1: Write the failing test**

```python
from chaoscontrol.baselines import SimpleTransformerLM
import torch

def test_transformer_forward():
    model = SimpleTransformerLM(vocab_size=256, dim=64, num_layers=2, num_heads=4)
    ids = torch.randint(0, 256, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, 256)

def test_transformer_param_budget():
    model = SimpleTransformerLM(vocab_size=256, dim=128, num_layers=4, num_heads=4)
    bytes_used = sum(p.numel() for p in model.parameters()) * 2
    assert bytes_used < 16 * 1024 * 1024
```

**Step 2: Implement SimpleTransformerLM**

Minimal causal transformer: embedding, N layers of (RMSNorm → causal self-attention → RMSNorm → FFN) with residuals, final norm, LM head. No rotary embeddings needed for enwik8 scale. Use `torch.nn.functional.scaled_dot_product_attention` with causal mask.

**Step 3: Test, commit**

```bash
git commit -m "feat: add baselines.py — SimpleTransformerLM"
```

---

### Task 12: Add SemanticTier to memory.py

**Files:**
- Modify: `src/chaoscontrol/memory.py`
- Test: `tests/test_memory.py` (add tests)

**Step 1: Write the failing test**

```python
def test_semantic_tier_shape():
    st = SemanticTier(model_dim=16, num_bases=8)
    bias = st.read(batch_size=2)
    assert bias.shape == (2, 16)

def test_semantic_tier_updates_from_episodes():
    st = SemanticTier(model_dim=16, num_bases=8)
    initial = st.bases.clone()
    # Feed similar episodes
    for _ in range(10):
        st.consolidate_from_episodes(torch.randn(1, 16) + 1.0)
    assert not torch.allclose(initial, st.bases)

def test_semantic_tier_always_on():
    """Semantic read should never be zero after consolidation."""
    st = SemanticTier(model_dim=16, num_bases=8)
    st.consolidate_from_episodes(torch.randn(1, 16))
    bias = st.read(batch_size=2)
    assert bias.abs().sum() > 0
```

**Step 2: Implement SemanticTier**

```python
class SemanticTier(nn.Module):
    """Neocortical knowledge layer — always-on background bias.

    Stores slowly-updated basis vectors extracted from episodic experience.
    Not cue-dependent, not gated — shapes all processing as a persistent prior.
    """
    def __init__(self, model_dim, num_bases=8, update_rate=0.01):
        super().__init__()
        self.decoder = nn.Linear(num_bases, model_dim, bias=False)
        self.register_buffer("bases", torch.zeros(1, num_bases))
        self.update_rate = update_rate
        self.encoder = nn.Linear(model_dim, num_bases, bias=False)
        self.encoder.weight.requires_grad_(False)

    def read(self, batch_size):
        return self.decoder(self.bases.expand(batch_size, -1))

    def consolidate_from_episodes(self, episode_vectors):
        """Extract shared structure from episode vectors into bases."""
        encoded = self.encoder(episode_vectors.detach()).mean(dim=0, keepdim=True)
        self.bases = (1 - self.update_rate) * self.bases.detach() + self.update_rate * encoded
```

**Step 3: Wire into ChaosStudentLM** — add `semantic_tier_bases: int = 0` config. If >0, create SemanticTier, add its read to the recurrence at every step (additive bias, separate from episodic read). In training loop, after episodic consolidation, call `semantic_tier.consolidate_from_episodes()` with the episodic tier's recent slots.

**Step 4: Test, commit**

```bash
git commit -m "feat: add SemanticTier — neocortical always-on knowledge bias"
```

---

### Task 13: Add StructuredProjections to metabolic.py

**Files:**
- Modify: `src/chaoscontrol/metabolic.py`
- Test: `tests/test_metabolic.py` (add tests)

**Step 1: Write the failing test**

```python
def test_structured_projections_produce_different_views():
    sp = StructuredProjections(dim=16, k=4)
    x = torch.randn(2, 8, 16)
    views = sp(x)
    assert len(views) == 4
    assert views[0].shape == (2, 8, 16)
    # Different projections should produce different outputs
    assert not torch.allclose(views[0], views[1])
```

**Step 2: Implement StructuredProjections**

```python
class StructuredProjections(nn.Module):
    """K learned projection heads — each emphasizes different features.
    'Choosing the question' — NFT-aligned generation mechanism.
    """
    def __init__(self, dim, k=4):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(k)
        ])

    def forward(self, x):
        return [proj(x) for proj in self.projections]
```

**Step 3: Update metabolic_fork** to accept `generation_mode="noise"` or `"structured"`. When structured, use StructuredProjections instead of noise perturbation. Add `generation_mode` to config.

**Step 4: Test, commit**

```bash
git commit -m "feat: add StructuredProjections — 'choosing the question' generation"
```

---

### Task 14: Add compression-consequence signal to wernicke.py

**Files:**
- Modify: `src/chaoscontrol/wernicke.py`
- Modify: `src/chaoscontrol/memory.py`
- Test: `tests/test_wernicke.py` (add test)

**Step 1: Write the failing test**

```python
def test_compression_consequence_updates_router():
    w = WernickeLayer(dim=16, k_max=4, window=4, router_type="moe")
    initial_weights = w.router.weight.data.clone()
    # Simulate: a bad merge happened for bucket 0
    w.compression_consequence_update(bucket_id=0, quality_delta=-0.5)
    # Router should have changed
    assert not torch.allclose(initial_weights, w.router.weight.data)
```

**Step 2: Implement compression_consequence_update**

A gradient-free nudge to the router weights. When a merge in bucket X produces a quality drop (post-merge retrieval worse than pre-merge), nudge the router to be more discriminating for bucket X — increase the separation between bucket X and its neighbors in weight space.

```python
def compression_consequence_update(self, bucket_id, quality_delta, lr=0.01):
    """Gradient-free update: bad merge → make this bucket more discriminating."""
    if self.router_type != "moe" or quality_delta >= 0:
        return  # only update on bad merges, only for MoE router
    with torch.no_grad():
        # Strengthen this bucket's router weights (make it more selective)
        self.router.weight.data[bucket_id] *= (1.0 + lr * abs(quality_delta))
```

**Step 3: Wire into MultiSlotOuterModel._compress()** — after merging, compute quality_delta (compare retrieval on a cached cue before and after merge), call `wernicke.compression_consequence_update()` if quality dropped.

**Step 4: Test, commit**

```bash
git commit -m "feat: add compression-consequence signal from memory to Wernicke"
```

---

### Task 15: Experiment 01 — Baseline configs and run script

**Files:**
- Create: `experiments/01_baseline/README.md`
- Create: `experiments/01_baseline/configs/ssm_small.yaml`
- Create: `experiments/01_baseline/configs/ssm_medium.yaml`
- Create: `experiments/01_baseline/configs/ssm_full.yaml`
- Create: `experiments/01_baseline/configs/tfm_small.yaml`
- Create: `experiments/01_baseline/configs/tfm_medium.yaml`
- Create: `experiments/01_baseline/configs/tfm_full.yaml`
- Create: `experiments/01_baseline/run.sh`
- Create: `experiments/01_baseline/analyze.py`

**Step 1: Write README.md**

```markdown
# Experiment 01: Baseline

## Hypothesis
A simple transformer outperforms a vanilla SSM at matched parameter budgets. This establishes the floor (SSM) and ceiling (transformer) for ChaosControl.

## Null hypothesis
Expected: transformer > SSM. This experiment establishes the gap.

## Predictions
- Transformer beats SSM at all three sizes
- The gap may narrow or widen with model size

## Method
6 configs: 3 SSM sizes (128/256/384 dim) + 3 transformer sizes, matched parameter budget, same enwik8 data, same training seconds.

## Dependencies
None.

## Kill criteria
None — this is the reference point.
```

**Step 2: Write YAML configs**

```yaml
# ssm_small.yaml
model_type: ssm
model_dim: 128
num_layers: 4
ff_mult: 2
a_mode: diag
rich_b_mode: none
outer_model_dim: 0
wernicke_enabled: false
metabolic_gate: false
budget_seconds: 300
```

(Similar for other configs, with model_type: transformer for tfm variants.)

**Step 3: Write run.sh**

A script that iterates configs, calls the appropriate training function, saves results JSON per config with checkpointing.

**Step 4: Write analyze.py**

Reads all results JSONs, prints comparison table (model, size, bpb, params, time).

**Step 5: Commit**

```bash
git commit -m "experiment: add 01_baseline — SSM vs transformer at 3 sizes"
```

---

### Tasks 16-22: Experiment directories 02-08

Each follows the same pattern as Task 15: README.md with hypothesis/method/predictions, YAML configs from the design doc ablation tables, run.sh that iterates configs, analyze.py that reads results.

**Task 16:** `02_critical_dynamics` — 7 configs from design doc
**Task 17:** `03_state_dependent_routing` — 10 configs
**Task 18:** `04_long_term_memory` — 10 configs (including semantic tier)
**Task 19:** `05_typed_composition` — 10 configs
**Task 20:** `06_metabolic_gate` — 11 configs (including structured_proj)
**Task 21:** `07_full_system` — 10 configs
**Task 22:** `08_gap_analysis` — 6 configs

Each commit: `experiment: add XX_name — [brief description]`

---

### Task 23: Top-level run_all.sh and README.md

**Files:**
- Create: `run_all.sh`
- Create: `README.md`

**Step 1: Write run_all.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail
ENWIK8="${1:?Usage: run_all.sh /path/to/enwik8 [--budget SECONDS]}"
shift
BUDGET="${2:-300}"

for exp in experiments/0*/; do
    echo "=== Running $(basename $exp) ==="
    bash "$exp/run.sh" "$ENWIK8" --budget "$BUDGET"
done
echo "=== All experiments complete ==="
```

**Step 2: Write README.md**

Thesis statement, component overview, how to run, experiment index with one-line descriptions.

**Step 3: Commit**

```bash
git commit -m "docs: add README.md and run_all.sh"
```

---

### Task 24: Final verification — all tests pass, dry-run smoke test

**Step 1:** Run full test suite: `.venv/bin/python -m pytest tests/ -v`
**Step 2:** Smoke test a single baseline config with 10s budget to verify the pipeline works end-to-end
**Step 3:** Final commit if needed

---

## Task Dependency Summary

```
Task 1 (scaffolding)
  └─> Task 2 (data.py)
       └─> Task 3 (config.py)
            └─> Task 4 (core.py)
            └─> Task 5 (routing.py)
            └─> Task 6 (memory.py)
            └─> Task 7 (wernicke.py)
            └─> Task 8 (metabolic.py)
                 └─> Task 9 (model.py) ← depends on 4-8
                      └─> Task 10 (training.py) ← depends on 9
                           └─> Task 11 (baselines.py)
                           └─> Task 12 (SemanticTier)
                           └─> Task 13 (StructuredProjections)
                           └─> Task 14 (compression-consequence)
                                └─> Tasks 15-22 (experiments) ← depends on 10-14
                                     └─> Task 23 (run_all + README)
                                          └─> Task 24 (verification)
```

Tasks 4-8 can be done in parallel after Task 3.
Tasks 11-14 can be done in parallel after Task 10.
Tasks 15-22 can be done in parallel after Task 14.
