# Learned Tokenizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three learned tokenizer variants, four codebook alignment mechanisms, FineWeb raw-bytes data pipeline, and update the experiment runner to support the new Layer 0 / Layer 0.5 in the experiment matrix.

**Architecture:** A new `LearnedTokenizer` module sits between raw bytes and the existing model. Three variants (fixed stride, learned boundaries, attention pooling) share a VQ codebook that serves as the learned vocabulary. A separate codebook alignment module computes coupling losses between the tokenizer and Wernicke codebooks. The data pipeline reads FineWeb text as raw UTF-8 bytes. bpb is computed as total CE nats / total raw bytes / ln(2).

**Tech Stack:** PyTorch. No new dependencies beyond what exists (torch, numpy, pyyaml, pytest).

---

## Dependency Graph

```
Task 1 (FineWeb data) ─────────────────────────────────┐
Task 2 (VQ utils) ──┬── Task 3 (Tokenizer variants) ───┤
                     └── Task 4 (Alignment losses) ──────┼── Task 6 (Wire into model) ── Task 7 (Configs) ── Task 8 (Deploy)
Task 5 (bpb calculation) ───────────────────────────────┘
```

Tasks 1, 2, 5 are independent. Task 3 depends on 2. Task 4 depends on 2. Task 6 integrates everything. Task 7 generates configs. Task 8 deploys.

---

### Task 1: FineWeb Raw Bytes Data Loader

**Files:**
- Modify: `src/chaoscontrol/data.py`
- Modify: `src/chaoscontrol/config.py`
- Test: `tests/test_data.py`

**Step 1: Write the failing tests**

```python
class TestFineWebLoader(unittest.TestCase):
    def test_load_fineweb_raw_bytes(self):
        """Load raw text file as byte tensor."""
        from chaoscontrol.data import load_raw_bytes
        import tempfile, os
        # Create a temp file with known content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello, world! This is a test.")
            path = f.name
        try:
            tokens = load_raw_bytes(path)
            assert tokens.dtype == torch.long
            assert tokens[0] == ord('H')
            assert tokens[-1] == ord('.')
            assert tokens.numel() == 28
        finally:
            os.unlink(path)

    def test_load_fineweb_splits(self):
        """Split raw bytes into train/val/test."""
        from chaoscontrol.data import load_raw_byte_splits
        import tempfile, os
        content = "A" * 1000
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            path = f.name
        try:
            train, val, test = load_raw_byte_splits(path)
            assert train.numel() + val.numel() + test.numel() == 1000
            assert train.numel() == 900  # 90%
            assert val.numel() == 50     # 5%
        finally:
            os.unlink(path)
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_data.py::TestFineWebLoader -v`

**Step 3: Implement**

Add to `src/chaoscontrol/data.py`:

```python
def load_raw_bytes(path: Path) -> torch.Tensor:
    """Load a text file as a tensor of raw UTF-8 byte values (uint8 → int64)."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"empty file: {path}")
    return torch.from_numpy(raw.astype(np.int64, copy=False))

def load_raw_byte_splits(
    path: Path,
    *,
    train_fraction: float = 0.90,
    val_fraction: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load raw bytes and split into train/val/test."""
    tokens = load_raw_bytes(path)
    train_end = int(tokens.numel() * train_fraction)
    val_end = int(tokens.numel() * (train_fraction + val_fraction))
    return tokens[:train_end], tokens[train_end:val_end], tokens[val_end:]
```

Update `config.py`: rename `enwik8_path` to `data_path` and add `data_format`:

```python
data_path: str = ""  # path to data file (enwik8 or FineWeb raw text)
data_format: str = "enwik8"  # "enwik8" or "raw_bytes"
```

Update `runner.py` to dispatch to the right loader based on `data_format`.

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```
git commit -m "feat: FineWeb raw bytes data loader"
```

---

### Task 2: VQ Utilities Module

**Files:**
- Create: `src/chaoscontrol/vq.py`
- Test: `tests/test_vq.py`

**Step 1: Write the failing tests**

```python
class TestVectorQuantize(unittest.TestCase):
    def test_quantize_shape(self):
        from chaoscontrol.vq import vector_quantize
        x = torch.randn(2, 16, 32)  # (batch, seq, dim)
        codebook = torch.randn(64, 32)  # (K, dim)
        quantized, indices, commit_loss = vector_quantize(x, codebook)
        assert quantized.shape == (2, 16, 32)
        assert indices.shape == (2, 16)
        assert indices.dtype == torch.long
        assert commit_loss.shape == ()

    def test_straight_through_gradient(self):
        """Gradient should flow through the straight-through estimator."""
        from chaoscontrol.vq import vector_quantize
        x = torch.randn(2, 8, 16, requires_grad=True)
        codebook = torch.randn(32, 16)
        quantized, _, _ = vector_quantize(x, codebook)
        loss = quantized.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_indices_are_nearest_neighbor(self):
        from chaoscontrol.vq import vector_quantize
        codebook = torch.eye(4)  # 4 entries in 4-dim space
        x = torch.tensor([[[0.9, 0.1, 0.0, 0.0]]])  # closest to entry 0
        _, indices, _ = vector_quantize(x, codebook)
        assert indices[0, 0] == 0

    def test_commitment_loss_is_positive(self):
        from chaoscontrol.vq import vector_quantize
        x = torch.randn(2, 8, 16)
        codebook = torch.randn(32, 16)
        _, _, commit_loss = vector_quantize(x, codebook)
        assert commit_loss.item() > 0
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement**

Create `src/chaoscontrol/vq.py`:

```python
"""Vector quantization utilities shared by tokenizer and Wernicke."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def vector_quantize(
    x: torch.Tensor,
    codebook: torch.Tensor,
    beta: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vector quantize x against codebook with straight-through gradient.

    Args:
        x: (batch, seq, dim) — continuous representations
        codebook: (K, dim) — codebook entries
        beta: commitment loss weight

    Returns:
        (quantized, indices, commitment_loss)
        quantized: (batch, seq, dim) — quantized with straight-through
        indices: (batch, seq) — codebook indices (int64)
        commitment_loss: scalar
    """
    # Distances: (batch, seq, K)
    dists = torch.cdist(x, codebook.unsqueeze(0).expand(x.size(0), -1, -1))
    indices = dists.argmin(dim=-1)  # (batch, seq)

    # Look up quantized vectors
    quantized = codebook[indices]  # (batch, seq, dim)

    # Commitment loss: encourage x to stay close to codebook entries
    commit_loss = F.mse_loss(x.detach(), quantized) + beta * F.mse_loss(x, quantized.detach())

    # Straight-through: forward uses quantized, backward uses x
    quantized_st = x + (quantized - x).detach()

    return quantized_st, indices, commit_loss
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```
git commit -m "feat: vector quantization utilities module"
```

---

### Task 3: LearnedTokenizer Module (3 Variants)

**Files:**
- Create: `src/chaoscontrol/tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Step 1: Write the failing tests**

```python
class TestFixedStrideTokenizer(unittest.TestCase):
    def test_output_shapes(self):
        from chaoscontrol.tokenizer import FixedStrideTokenizer
        tok = FixedStrideTokenizer(byte_dim=32, token_dim=64, codebook_size=128, stride=4)
        byte_ids = torch.randint(0, 256, (2, 32))
        embeds, ids, commit_loss = tok(byte_ids)
        assert embeds.shape == (2, 8, 64)  # 32 bytes / stride 4 = 8 tokens
        assert ids.shape == (2, 8)
        assert commit_loss.shape == ()

    def test_different_inputs_different_outputs(self):
        from chaoscontrol.tokenizer import FixedStrideTokenizer
        tok = FixedStrideTokenizer(byte_dim=32, token_dim=64, codebook_size=128, stride=4)
        ids1 = torch.zeros(1, 32, dtype=torch.long)
        ids2 = torch.ones(1, 32, dtype=torch.long) * 100
        e1, _, _ = tok(ids1)
        e2, _, _ = tok(ids2)
        assert not torch.allclose(e1, e2)

class TestLearnedBoundaryTokenizer(unittest.TestCase):
    def test_output_shapes(self):
        from chaoscontrol.tokenizer import LearnedBoundaryTokenizer
        tok = LearnedBoundaryTokenizer(byte_dim=32, token_dim=64, codebook_size=128, target_rate=4)
        byte_ids = torch.randint(0, 256, (2, 32))
        embeds, ids, commit_loss, boundary_loss = tok(byte_ids)
        assert embeds.dim() == 3  # (batch, token_seq, token_dim)
        assert embeds.size(0) == 2
        assert embeds.size(2) == 64
        assert ids.dim() == 2

    def test_boundary_loss_penalizes_deviation_from_target(self):
        from chaoscontrol.tokenizer import LearnedBoundaryTokenizer
        tok = LearnedBoundaryTokenizer(byte_dim=32, token_dim=64, codebook_size=128, target_rate=4)
        byte_ids = torch.randint(0, 256, (2, 32))
        _, _, _, boundary_loss = tok(byte_ids)
        assert boundary_loss.item() >= 0

class TestAttnPoolTokenizer(unittest.TestCase):
    def test_output_shapes(self):
        from chaoscontrol.tokenizer import AttnPoolTokenizer
        tok = AttnPoolTokenizer(byte_dim=32, token_dim=64, codebook_size=128, window_size=4)
        byte_ids = torch.randint(0, 256, (2, 32))
        embeds, ids, commit_loss = tok(byte_ids)
        assert embeds.shape == (2, 8, 64)  # 32 bytes / window 4 = 8 tokens
        assert ids.shape == (2, 8)

    def test_context_dependent(self):
        """Same bytes in different context should produce different tokens."""
        from chaoscontrol.tokenizer import AttnPoolTokenizer
        tok = AttnPoolTokenizer(byte_dim=32, token_dim=64, codebook_size=128, window_size=4)
        # Two sequences with same first window but different second window
        ids1 = torch.randint(0, 256, (1, 8))
        ids2 = ids1.clone()
        ids2[0, 4:] = (ids2[0, 4:] + 100) % 256  # change second window
        e1, _, _ = tok(ids1)
        e2, _, _ = tok(ids2)
        # First window output may differ due to causal attention seeing different context
        # But at minimum, second window should differ
        assert not torch.allclose(e1[0, 1], e2[0, 1])

class TestTokenizerGradientFlow(unittest.TestCase):
    def test_all_variants_have_gradients(self):
        from chaoscontrol.tokenizer import FixedStrideTokenizer, LearnedBoundaryTokenizer, AttnPoolTokenizer
        for Cls, kwargs in [
            (FixedStrideTokenizer, {"stride": 4}),
            (LearnedBoundaryTokenizer, {"target_rate": 4}),
            (AttnPoolTokenizer, {"window_size": 4}),
        ]:
            tok = Cls(byte_dim=16, token_dim=32, codebook_size=64, **kwargs)
            byte_ids = torch.randint(0, 256, (2, 16))
            result = tok(byte_ids)
            embeds = result[0]
            loss = embeds.sum()
            loss.backward()
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in tok.parameters())
            assert has_grad, f"{Cls.__name__} has no gradients"
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement `src/chaoscontrol/tokenizer.py`**

The file should contain three classes: `FixedStrideTokenizer`, `LearnedBoundaryTokenizer`, `AttnPoolTokenizer`. All share:
- `byte_embed: nn.Embedding(256, byte_dim)`
- `codebook: nn.Parameter(codebook_size, token_dim)` — the learned vocabulary
- Use `vector_quantize` from `chaoscontrol.vq`
- Return `(token_embeds, token_ids, commit_loss)` — boundary variant also returns `boundary_loss`

**FixedStrideTokenizer:** `Conv1d(byte_dim, token_dim, kernel_size=stride*2, stride=stride)` with causal padding, then VQ.

**LearnedBoundaryTokenizer:** `Conv1d(byte_dim, 1, kernel_size=5)` → sigmoid boundary scores. Segment-and-pool between boundaries (use differentiable soft boundaries with straight-through on the threshold). Target rate loss penalizes deviation from `target_rate` bytes/token. Pad batches to max token_seq length.

**AttnPoolTokenizer:** Reshape bytes into `(batch*n_windows, window_size, byte_dim)`. Learned query token cross-attends to each window. Project to token_dim. VQ quantize.

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```
git commit -m "feat: three learned tokenizer variants (fixed stride, learned boundary, attention pooling)"
```

---

### Task 4: Codebook Alignment Losses

**Files:**
- Create: `src/chaoscontrol/alignment.py`
- Test: `tests/test_alignment.py`

**Step 1: Write the failing tests**

```python
class TestAlignmentLosses(unittest.TestCase):
    def test_contrastive_shape_and_sign(self):
        from chaoscontrol.alignment import contrastive_alignment
        tok_codes = torch.randn(2, 8, 32)  # (batch, seq, dim)
        tok_ids = torch.randint(0, 4, (2, 8))  # assigned to 4 Wernicke buckets
        wer_codebook = torch.randn(4, 32)  # 4 Wernicke entries
        loss = contrastive_alignment(tok_codes, tok_ids, wer_codebook)
        assert loss.shape == ()
        assert loss.item() > 0  # contrastive loss is positive

    def test_diversity_penalizes_similarity(self):
        from chaoscontrol.alignment import diversity_alignment
        # Two identical codebooks should have high diversity loss
        cb1 = torch.randn(4, 32)
        cb2 = cb1.clone()
        loss_same = diversity_alignment(cb1, cb2)
        # Two orthogonal codebooks should have low diversity loss
        cb3 = torch.randn(4, 32)
        loss_diff = diversity_alignment(cb1, cb3)
        # Same should be penalized more (higher loss) than different
        assert loss_same.item() > loss_diff.item() - 0.1  # approximate

    def test_distillation_zero_for_identical(self):
        from chaoscontrol.alignment import distillation_alignment
        tok_codes = torch.randn(2, 8, 32)
        tok_ids = torch.randint(0, 4, (2, 8))
        # Wernicke codebook = exact means of tok_codes per bucket
        wer_codebook = torch.zeros(4, 32)
        for b in range(4):
            mask = tok_ids == b
            if mask.any():
                wer_codebook[b] = tok_codes[mask].mean(dim=0)
        loss = distillation_alignment(tok_codes, tok_ids, wer_codebook)
        assert loss.item() < 0.1  # near zero when perfectly aligned

    def test_no_alignment_returns_zero(self):
        from chaoscontrol.alignment import no_alignment
        loss = no_alignment()
        assert loss.item() == 0.0
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement `src/chaoscontrol/alignment.py`**

```python
"""Codebook alignment losses between tokenizer and Wernicke."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def no_alignment() -> torch.Tensor:
    return torch.tensor(0.0)


def contrastive_alignment(
    tok_codes: torch.Tensor,
    wernicke_bucket_ids: torch.Tensor,
    wer_codebook: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE: tokenizer codes should cluster by Wernicke bucket."""
    K = wer_codebook.size(0)
    # Compute mean tok code per Wernicke bucket
    means = []
    for j in range(K):
        mask = wernicke_bucket_ids == j
        if mask.any():
            means.append(tok_codes[mask].mean(dim=0))
        else:
            means.append(torch.zeros_like(wer_codebook[j]))
    means = torch.stack(means)  # (K, dim)

    # Cosine similarity matrix
    means_n = F.normalize(means, dim=-1)
    wer_n = F.normalize(wer_codebook, dim=-1)
    sim = means_n @ wer_n.T / temperature  # (K, K)

    # InfoNCE: diagonal should dominate
    labels = torch.arange(K, device=sim.device)
    return F.cross_entropy(sim, labels)


def diversity_alignment(
    codebook_a: torch.Tensor,
    codebook_b: torch.Tensor,
) -> torch.Tensor:
    """SSIM-style: penalize similarity between two codebooks."""
    a_n = F.normalize(codebook_a, dim=-1)
    b_n = F.normalize(codebook_b, dim=-1)
    sim = (a_n @ b_n.T).abs()
    return sim.mean()


def distillation_alignment(
    tok_codes: torch.Tensor,
    wernicke_bucket_ids: torch.Tensor,
    wer_codebook: torch.Tensor,
) -> torch.Tensor:
    """Cosine distillation: each Wernicke entry is teacher for its tokens."""
    K = wer_codebook.size(0)
    total = torch.tensor(0.0, device=tok_codes.device)
    count = 0
    for j in range(K):
        mask = wernicke_bucket_ids == j
        if mask.any():
            mean_tok = tok_codes[mask].mean(dim=0)
            total = total + (1 - F.cosine_similarity(mean_tok.unsqueeze(0), wer_codebook[j].unsqueeze(0)))
            count += 1
    return total / max(count, 1)


ALIGNMENT_REGISTRY = {
    "none": no_alignment,
    "contrastive": contrastive_alignment,
    "diversity": diversity_alignment,
    "distillation": distillation_alignment,
}
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```
git commit -m "feat: four codebook alignment losses (none, contrastive, diversity, distillation)"
```

---

### Task 5: bpb Calculation

**Files:**
- Modify: `src/chaoscontrol/evaluation.py`
- Test: `tests/test_evaluation.py`

**Step 1: Write the failing test**

```python
class TestBPBCalculation(unittest.TestCase):
    def test_compute_bpb_known_value(self):
        from chaoscontrol.evaluation import compute_bpb
        import math
        # 100 nats over 100 bytes = 1 nat/byte = 1/ln(2) bpb ≈ 1.4427
        bpb = compute_bpb(total_ce_nats=100.0, total_raw_bytes=100)
        assert abs(bpb - 1.0 / math.log(2.0)) < 1e-6

    def test_compute_bpb_stride_invariant(self):
        """bpb should be the same whether model uses stride 1 or stride 4."""
        from chaoscontrol.evaluation import compute_bpb
        # Stride 1: 100 tokens × 1 byte each, total CE = 100 nats
        bpb1 = compute_bpb(total_ce_nats=100.0, total_raw_bytes=100)
        # Stride 4: 25 tokens × 4 bytes each, total CE = 100 nats (same info)
        bpb4 = compute_bpb(total_ce_nats=100.0, total_raw_bytes=100)
        assert bpb1 == bpb4  # same because denominator is always raw bytes
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement**

Add to `src/chaoscontrol/evaluation.py`:

```python
def compute_bpb(total_ce_nats: float, total_raw_bytes: int) -> float:
    """Compute bits-per-byte. Tokenizer-agnostic.

    Args:
        total_ce_nats: Sum of cross-entropy loss (in nats) across all predicted tokens.
        total_raw_bytes: Count of raw bytes in the evaluation text (independent of tokenizer).

    Returns:
        Bits per byte. Lower is better.
    """
    return total_ce_nats / max(total_raw_bytes, 1) / math.log(2.0)
```

Update `evaluate_chaoscontrol_bpb` to accept `total_raw_bytes` and use `compute_bpb` instead of computing bpb from mean loss. The caller (runner.py) passes the byte count of the validation text.

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```
git commit -m "feat: tokenizer-agnostic bpb calculation"
```

---

### Task 6: Wire Tokenizer into Model + Training

**Files:**
- Modify: `src/chaoscontrol/config.py`
- Modify: `src/chaoscontrol/model.py`
- Modify: `src/chaoscontrol/runner.py`
- Modify: `src/chaoscontrol/training.py`
- Test: `tests/test_integration.py`

**Step 1: Add config fields**

```python
# Learned tokenizer
tokenizer_type: str = "none"  # "none", "fixed_stride", "learned_boundary", "attn_pool", "bpe"
tokenizer_codebook_size: int = 1024
tokenizer_byte_dim: int = 64
tokenizer_stride: int = 4  # for fixed_stride and attn_pool window_size
tokenizer_target_rate: int = 4  # for learned_boundary

# Codebook alignment
align_type: str = "none"  # "none", "contrastive", "diversity", "distillation"
align_weight: float = 0.05
```

**Step 2: Modify ChaosStudentLM**

Add an optional `tokenizer` attribute. When present, `forward()` and `step()` route through the tokenizer before the embedding layer. The tokenizer replaces `self.embed` for input processing:

```python
# In forward():
if self.tokenizer is not None:
    x, token_ids_l0, tok_commit_loss = self.tokenizer(input_ids)
    # token_ids_l0 are Level 0 type IDs
else:
    x = self.embed(input_ids)
    token_ids_l0 = None
    tok_commit_loss = None
```

The tokenizer output `x` has shape `(batch, token_seq, model_dim)` where `token_seq` may differ from `byte_seq` (downsampled). A projection layer maps `token_dim → model_dim` if they differ.

**Step 3: Modify runner.py**

Build the tokenizer based on config and pass it to the model. Compute alignment loss in the training loop when `align_type != "none"`.

**Step 4: Update bpb computation**

The runner needs to know the raw byte count of the eval text. Pass it through to the eval function. For learned tokenizers, the model predicts fewer tokens (downsampled) but each covers multiple bytes. The bpb calculation uses raw byte count as denominator.

**Step 5: Add integration test**

```python
def test_tokenizer_end_to_end(self):
    """Full training loop with learned tokenizer."""
    from chaoscontrol.tokenizer import FixedStrideTokenizer
    model = ChaosStudentLM(vocab_size=1024, dim=32, num_layers=2)
    tok = FixedStrideTokenizer(byte_dim=16, token_dim=32, codebook_size=1024, stride=4)
    model.tokenizer = tok
    # ... run short training, verify no crashes, verify bpb is finite
```

**Step 6: Commit**

```
git commit -m "feat: wire learned tokenizer + alignment into model and training loop"
```

---

### Task 7: Update Experiment 09 Configs + Runner

**Files:**
- Modify: `experiments/09_revised_architecture/run_layered.py`
- Create: additional Layer 0 and Layer 0.5 config templates
- Modify: `experiments/09_revised_architecture/README.md`

**Step 1: Add Layer 0 config generation**

Add `generate_l0_configs()` that creates 5 YAML configs for the tokenizer variants. These are standalone (no winner injection needed).

**Step 2: Add Layer 0.5 config generation**

Add `generate_l05_configs(tokenizer_settings)` that creates 4 YAML configs for alignment types, injecting the L0 winner tokenizer settings.

**Step 3: Wire into main()**

Insert L0 and L0.5 before L1 in the runner's main function. L1+ configs inherit the winning tokenizer + alignment settings.

**Step 4: Update README**

Add L0 and L0.5 to the method table. Update total config counts and time estimates.

**Step 5: Commit**

```
git commit -m "feat: Layer 0 (tokenizer) and Layer 0.5 (alignment) in experiment runner"
```

---

### Task 8: Deploy and Run

**Step 1: Provision GPU pod** (any CUDA GPU, 16GB+ VRAM)

**Step 2: Download FineWeb raw bytes**

Use the parameter-golf download script to get raw docs, then extract to bytes:

```bash
cd /workspace
# Clone parameter-golf for the download script
git clone <parameter-golf-repo>
cd parameter-golf
python data/download_hf_docs_and_tokenize.py --output-root /workspace/fineweb_raw --raw-only
```

Or alternatively, download FineWeb docs from HuggingFace directly and save as raw text.

**Step 3: Push chaoscontrol repo, set up venv, verify tests pass**

**Step 4: Run experiment 09**

```bash
bash experiments/09_revised_architecture/run.sh /workspace/fineweb_raw/fineweb.txt
```

**Step 5: Harvest results**

**Step 6: Run analyzer, write results summary**

---

## Notes for implementers

- **Do NOT add "Co-Authored-By" lines to commits.**
- **Follow TDD strictly:** write tests first, confirm failure, then implement.
- **The tokenizer's codebook IS the vocabulary.** `token_ids` from the tokenizer are used the same way Wernicke's `bucket_ids` are used — for memory typing, PQ compression, CFR regret tracking.
- **Variable-length sequences (Variant B):** the learned boundary tokenizer produces different token counts per batch element. Use padding to the max length in the batch, with an attention mask. This is the most complex variant.
- **bpb denominator is always raw bytes.** Never use token count. The `compute_bpb` function should be the ONLY place bpb is calculated.
- **The `data_path` / `data_format` change in config.py is backward-compatible.** Existing enwik8 configs still work by setting `data_format: "enwik8"`.
