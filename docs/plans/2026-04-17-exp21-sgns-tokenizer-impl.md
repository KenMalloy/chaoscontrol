# Exp 21: SGNS Semantic Tokenizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the offline SGNS tokenizer and 4-cell ablation specified in `docs/plans/2026-04-17-exp21-sgns-tokenizer-design.md`.

**Architecture:** Train SGNS (skip-gram with NCE) on FineWeb train → produce per-subword embedding table `W_in ∈ R^{V×d}` → moment-match row statistics to the Gaussian random-init baseline → load into `model.embed.weight` as the LM's embedding init. Validate via a 4-cell ablation: `{raw SSM, modded-NanoGPT lean} × {random init, SGNS init}`. Primary test: paired one-sided t-test on SSM bpb delta at `p<0.01` across 5 seeds per cell.

**Tech Stack:** PyTorch 2.x, SentencePiece (SP8192), Muon optimizer, existing ChaosControl training harness (extended). Assumed prerequisites from Exp 19: SP8192 tokenization path integrated into `src/chaoscontrol/data.py`; verify before Phase 1 Task 1.

**Operating context:** `main` branch, 2× H100 CUDA-13 pod for runs. A git worktree for the implementation work is recommended — see `superpowers:using-git-worktrees`.

---

## Preflight

### Task 0: Verify Exp 19 prerequisites

**Files:** read-only audit, no edits.

**Step 1: Verify SP8192 tokenization path exists**

Run: `rg -n "SentencePiece|SP8192|sp8192" src/chaoscontrol/data.py src/chaoscontrol/config.py src/chaoscontrol/runner.py`
Expected: at least one hit showing an SP8192 code path wired into the data pipeline. If NO hits, Exp 21 is blocked — Exp 19 hasn't landed the tokenization yet; stop and escalate.

**Step 2: Verify transformer arm dispatch**

Run: `rg -n 'model_type.*==.*"transformer"' src/chaoscontrol/runner.py`
Expected: at least one hit around `runner.py:29` (per design doc). If absent, Exp 21 is blocked.

**Step 3: Verify baseline transformer class exists**

Run: `rg -n "class.*[Tt]ransformer" src/chaoscontrol/baselines.py`
Expected: a transformer class around `baselines.py:9`. Capture its name — it will be adapted in Phase 3.

**Step 4: Verify existing quantization/artifact path (for later ship-ready work, not Exp 21 thesis)**

Run: `rg -n "int6|symmetric.*quant" src/chaoscontrol/artifact.py`
Expected: per design, symmetric int6 path around line 52. No action needed — just confirm presence.

**No commit for Task 0.** If any step fails, document the gap and stop.

---

## Phase 1: SGNS offline training infrastructure

### Task 1: SGNSModel skeleton with NCE scoring

**Files:**
- Create: `src/chaoscontrol/sgns/__init__.py` (empty)
- Create: `src/chaoscontrol/sgns/model.py`
- Create: `tests/unit/test_sgns_model.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_sgns_model.py
import torch
from chaoscontrol.sgns.model import SGNSModel


def test_sgns_model_shapes():
    model = SGNSModel(vocab_size=100, dim=16)
    assert model.input_embed.weight.shape == (100, 16)
    assert model.output_embed.weight.shape == (100, 16)


def test_sgns_score_pairs_shape():
    torch.manual_seed(0)
    model = SGNSModel(vocab_size=100, dim=16)
    center = torch.tensor([0, 1, 2])
    context = torch.tensor([3, 4, 5])
    scores = model.score_pairs(center, context)
    assert scores.shape == (3,)


def test_sgns_input_embed_nonzero_init():
    torch.manual_seed(0)
    model = SGNSModel(vocab_size=100, dim=16)
    assert not torch.allclose(model.input_embed.weight, torch.zeros_like(model.input_embed.weight))
```

**Step 2: Run tests, verify fail**

Run: `pytest tests/unit/test_sgns_model.py -v`
Expected: ImportError / ModuleNotFoundError on `chaoscontrol.sgns.model`.

**Step 3: Implement**

```python
# src/chaoscontrol/sgns/model.py
import torch
import torch.nn as nn


class SGNSModel(nn.Module):
    """Skip-gram with negative sampling. Two separate embedding tables:
    input (W_in) for center words, output (W_out) for contexts/negatives.
    Standard word2vec convention; W_out is discarded after training.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.input_embed = nn.Embedding(vocab_size, dim)
        self.output_embed = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.input_embed.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_embed.weight)

    def score_pairs(self, center: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        w_in = self.input_embed(center)
        w_out = self.output_embed(context)
        return (w_in * w_out).sum(dim=-1)
```

**Step 4: Run tests, verify pass**

Run: `pytest tests/unit/test_sgns_model.py -v`
Expected: 3 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/sgns/__init__.py src/chaoscontrol/sgns/model.py tests/unit/test_sgns_model.py
git commit -m "feat(exp21): SGNSModel skeleton with input/output embeddings"
```

---

### Task 2: NCE loss function

**Files:**
- Modify: `src/chaoscontrol/sgns/model.py` (add `nce_loss`)
- Modify: `tests/unit/test_sgns_model.py` (add loss tests)

**Step 1: Write failing tests**

```python
# Append to tests/unit/test_sgns_model.py
from chaoscontrol.sgns.model import nce_loss


def test_nce_loss_positive_attracts():
    """Identical vectors → positive score → small loss for the positive term."""
    torch.manual_seed(0)
    model = SGNSModel(vocab_size=10, dim=4)
    model.input_embed.weight.data = torch.eye(10, 4)[:, :4] * 2
    model.output_embed.weight.data = torch.eye(10, 4)[:, :4] * 2
    center = torch.tensor([0])
    context = torch.tensor([0])
    negatives = torch.tensor([[5, 6, 7]])
    loss = nce_loss(model, center, context, negatives)
    assert loss.item() < 0.5


def test_nce_loss_negative_repels():
    """Random embeddings + reasonable sample: loss is a finite scalar."""
    torch.manual_seed(0)
    model = SGNSModel(vocab_size=100, dim=16)
    center = torch.tensor([1, 2, 3])
    context = torch.tensor([4, 5, 6])
    negatives = torch.randint(0, 100, (3, 5))
    loss = nce_loss(model, center, context, negatives)
    assert torch.isfinite(loss)
    assert loss.dim() == 0
```

**Step 2: Run tests, verify fail**

Run: `pytest tests/unit/test_sgns_model.py::test_nce_loss_positive_attracts -v`
Expected: ImportError on `nce_loss`.

**Step 3: Implement**

```python
# Append to src/chaoscontrol/sgns/model.py
import torch.nn.functional as F


def nce_loss(
    model: SGNSModel,
    center: torch.Tensor,
    context: torch.Tensor,
    negatives: torch.Tensor,
) -> torch.Tensor:
    """Skip-gram NCE loss.
    center:    (B,)
    context:   (B,)
    negatives: (B, K)
    Returns scalar mean loss.
    """
    w_in = model.input_embed(center)  # (B, D)
    w_pos = model.output_embed(context)  # (B, D)
    w_neg = model.output_embed(negatives)  # (B, K, D)

    pos_score = (w_in * w_pos).sum(dim=-1)  # (B,)
    neg_score = torch.einsum("bd,bkd->bk", w_in, w_neg)  # (B, K)

    pos_loss = F.logsigmoid(pos_score).neg()
    neg_loss = F.logsigmoid(-neg_score).neg().sum(dim=-1)
    return (pos_loss + neg_loss).mean()
```

**Step 4: Run tests, verify pass**

Run: `pytest tests/unit/test_sgns_model.py -v`
Expected: 5 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/sgns/model.py tests/unit/test_sgns_model.py
git commit -m "feat(exp21): NCE loss for SGNS"
```

---

### Task 3: Negative sampler (frequency-weighted)

**Files:**
- Create: `src/chaoscontrol/sgns/sampler.py`
- Create: `tests/unit/test_sgns_sampler.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_sgns_sampler.py
import torch
from chaoscontrol.sgns.sampler import NegativeSampler, unigram_probs_from_counts


def test_unigram_probs_from_counts_normalized():
    counts = torch.tensor([100.0, 10.0, 1.0])
    probs = unigram_probs_from_counts(counts, power=0.75)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert probs[0] > probs[1] > probs[2]


def test_unigram_probs_distortion():
    counts = torch.tensor([100.0, 10.0])
    flat = unigram_probs_from_counts(counts, power=0.0)
    distorted = unigram_probs_from_counts(counts, power=0.75)
    raw = unigram_probs_from_counts(counts, power=1.0)
    # 0.75 power moves mass from high-freq toward low-freq vs raw
    assert distorted[1] > raw[1]
    # power=0 gives uniform
    assert torch.allclose(flat, torch.tensor([0.5, 0.5]))


def test_negative_sampler_shape_and_range():
    torch.manual_seed(0)
    probs = torch.tensor([0.5, 0.3, 0.2])
    sampler = NegativeSampler(probs)
    samples = sampler.sample(batch_size=4, k=5)
    assert samples.shape == (4, 5)
    assert samples.min() >= 0
    assert samples.max() < 3
```

**Step 2: Run tests, verify fail**

Run: `pytest tests/unit/test_sgns_sampler.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# src/chaoscontrol/sgns/sampler.py
import torch


def unigram_probs_from_counts(counts: torch.Tensor, power: float = 0.75) -> torch.Tensor:
    """Standard word2vec negative-sampling distribution: counts^power / sum."""
    if power == 0.0:
        return torch.full_like(counts, 1.0 / len(counts))
    weighted = counts.clamp(min=0).float().pow(power)
    return weighted / weighted.sum()


class NegativeSampler:
    """Samples K negatives per positive pair from a unigram distribution."""

    def __init__(self, probs: torch.Tensor):
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
        self.probs = probs

    def sample(self, batch_size: int, k: int) -> torch.Tensor:
        return torch.multinomial(self.probs, batch_size * k, replacement=True).view(batch_size, k)
```

**Step 4: Run tests, verify pass**

Run: `pytest tests/unit/test_sgns_sampler.py -v`
Expected: 3 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/sgns/sampler.py tests/unit/test_sgns_sampler.py
git commit -m "feat(exp21): frequency-weighted negative sampler (power=0.75)"
```

---

### Task 4: Training loop (single epoch, on in-memory token stream)

**Files:**
- Create: `src/chaoscontrol/sgns/train.py`
- Create: `tests/unit/test_sgns_train.py`

**Step 1: Write failing test**

```python
# tests/unit/test_sgns_train.py
import torch
from chaoscontrol.sgns.train import train_one_epoch
from chaoscontrol.sgns.model import SGNSModel
from chaoscontrol.sgns.sampler import NegativeSampler


def test_train_one_epoch_loss_decreases():
    torch.manual_seed(0)
    # Synthetic stream of size 200 with 10 tokens, clear co-occurrence pattern
    stream = torch.tensor([i % 10 for i in range(200)], dtype=torch.long)
    counts = torch.bincount(stream, minlength=10).float()
    probs = counts / counts.sum()
    sampler = NegativeSampler(probs)
    model = SGNSModel(vocab_size=10, dim=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_before = train_one_epoch(
        model, stream, sampler, window=2, k=5, batch_size=16, opt=opt, max_batches=1
    )
    for _ in range(10):
        loss_after = train_one_epoch(
            model, stream, sampler, window=2, k=5, batch_size=16, opt=opt, max_batches=5
        )
    assert loss_after < loss_before
```

**Step 2: Run test, verify fail**

Run: `pytest tests/unit/test_sgns_train.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# src/chaoscontrol/sgns/train.py
import torch
from torch.optim import Optimizer
from chaoscontrol.sgns.model import SGNSModel, nce_loss
from chaoscontrol.sgns.sampler import NegativeSampler


def _iterate_center_context(
    stream: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (center, context) tensors of all (i, j) pairs with 1 <= |i-j| <= window."""
    n = len(stream)
    centers, contexts = [], []
    for offset in range(1, window + 1):
        centers.append(stream[offset:])
        contexts.append(stream[:-offset])
        centers.append(stream[:-offset])
        contexts.append(stream[offset:])
    return torch.cat(centers), torch.cat(contexts)


def train_one_epoch(
    model: SGNSModel,
    stream: torch.Tensor,
    sampler: NegativeSampler,
    window: int,
    k: int,
    batch_size: int,
    opt: Optimizer,
    max_batches: int | None = None,
) -> float:
    """One pass over stream. Returns mean loss over processed batches."""
    centers, contexts = _iterate_center_context(stream, window)
    n = len(centers)
    perm = torch.randperm(n)
    centers = centers[perm]
    contexts = contexts[perm]
    total_loss, batches = 0.0, 0
    for start in range(0, n, batch_size):
        if max_batches is not None and batches >= max_batches:
            break
        end = min(start + batch_size, n)
        c = centers[start:end]
        ctx = contexts[start:end]
        negs = sampler.sample(batch_size=len(c), k=k)
        loss = nce_loss(model, c, ctx, negs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)
```

**Step 4: Run tests, verify pass**

Run: `pytest tests/unit/test_sgns_train.py -v`
Expected: 1 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/sgns/train.py tests/unit/test_sgns_train.py
git commit -m "feat(exp21): SGNS single-epoch training loop"
```

---

### Task 5: End-to-end SGNS training script

**Files:**
- Create: `scripts/train_sgns.py`

Invokes FineWeb train tokenization (via existing data path) and the SGNS training loop. Produces `W_in` tensor saved to disk. Configurable: V, d, window, k, epochs, subsampling threshold, output path.

**Step 1: Implement script (no new unit test — smoke-tested via invocation in Step 2)**

```python
# scripts/train_sgns.py
"""Offline SGNS training on FineWeb train. Produces per-subword W_in tensor.

Usage:
    python scripts/train_sgns.py \\
        --vocab-size 8192 \\
        --dim 256 \\
        --window 5 \\
        --k 10 \\
        --epochs 3 \\
        --subsample-threshold 1e-5 \\
        --out artifacts/sgns_v8192_d256.pt
"""
import argparse
import torch
from pathlib import Path

from chaoscontrol.sgns.model import SGNSModel
from chaoscontrol.sgns.sampler import NegativeSampler, unigram_probs_from_counts
from chaoscontrol.sgns.train import train_one_epoch
# NOTE: Exp 19 prerequisite. This import must resolve; verify in Task 0 Step 1.
from chaoscontrol.data import load_fineweb_train_tokens


def _subsample(stream: torch.Tensor, counts: torch.Tensor, threshold: float) -> torch.Tensor:
    """word2vec subsampling: drop frequent tokens with prob 1 - sqrt(t / f)."""
    total = counts.sum().item()
    freqs = counts / total
    keep_prob = torch.minimum(torch.sqrt(threshold / freqs.clamp(min=1e-12)), torch.ones_like(freqs))
    mask = torch.rand(len(stream)) < keep_prob[stream]
    return stream[mask]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--subsample-threshold", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load and tokenize FineWeb train (assumed from Exp 19)
    stream = load_fineweb_train_tokens(vocab_size=args.vocab_size)
    counts = torch.bincount(stream, minlength=args.vocab_size).float()
    stream = _subsample(stream, counts, args.subsample_threshold)
    probs = unigram_probs_from_counts(counts, power=0.75)

    sampler = NegativeSampler(probs)
    model = SGNSModel(vocab_size=args.vocab_size, dim=args.dim).cuda()
    stream_gpu = stream.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model, stream_gpu, sampler, args.window, args.k, args.batch_size, opt
        )
        print(f"epoch {epoch}: mean_loss = {loss:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.input_embed.weight.detach().cpu(), args.out)
    print(f"saved W_in to {args.out}")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke-test with tiny V (requires FineWeb train path working)**

Run: `python scripts/train_sgns.py --vocab-size 128 --dim 16 --epochs 1 --out /tmp/sgns_smoke.pt --batch-size 256`
Expected: prints one `mean_loss = ...` line and saves a `(128, 16)` tensor. Verify shape:
Run: `python -c "import torch; t = torch.load('/tmp/sgns_smoke.pt'); print(t.shape)"`
Expected: `torch.Size([128, 16])`

**Step 3: Commit**

```bash
git add scripts/train_sgns.py
git commit -m "feat(exp21): end-to-end SGNS training script on FineWeb train"
```

---

## Phase 2: Intrinsic validation tools

### Task 6: Nearest-neighbor sanity + k-means coherence

**Files:**
- Create: `src/chaoscontrol/sgns/intrinsic.py`
- Create: `tests/unit/test_sgns_intrinsic.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_sgns_intrinsic.py
import torch
from chaoscontrol.sgns.intrinsic import nearest_neighbors, kmeans_clusters


def test_nearest_neighbors_returns_topk_by_cosine():
    # 3 tokens: 0 and 1 are nearly identical, 2 is orthogonal
    embed = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    nn_result = nearest_neighbors(embed, query_ids=[0], k=2)
    assert nn_result[0][0] == 1  # closest non-self


def test_kmeans_clusters_shapes():
    torch.manual_seed(0)
    embed = torch.randn(50, 8)
    labels = kmeans_clusters(embed, k=5)
    assert labels.shape == (50,)
    assert labels.max().item() < 5
```

**Step 2: Run tests, verify fail**

Run: `pytest tests/unit/test_sgns_intrinsic.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# src/chaoscontrol/sgns/intrinsic.py
import torch


def nearest_neighbors(
    embed: torch.Tensor, query_ids: list[int], k: int = 5
) -> dict[int, list[int]]:
    """Return top-k nearest neighbors (excluding self) for each query_id, by cosine."""
    normed = torch.nn.functional.normalize(embed, dim=-1)
    out: dict[int, list[int]] = {}
    for q in query_ids:
        sims = normed @ normed[q]
        sims[q] = float("-inf")
        top = torch.topk(sims, k).indices.tolist()
        out[q] = top
    return out


def kmeans_clusters(embed: torch.Tensor, k: int, n_iter: int = 20) -> torch.Tensor:
    """Simple k-means over embedding rows. Returns cluster label per row."""
    n, d = embed.shape
    torch.manual_seed(0)
    centroids = embed[torch.randperm(n)[:k]].clone()
    for _ in range(n_iter):
        dists = torch.cdist(embed, centroids)
        labels = dists.argmin(dim=-1)
        for j in range(k):
            members = embed[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(dim=0)
    return labels
```

**Step 4: Run tests, verify pass**

Run: `pytest tests/unit/test_sgns_intrinsic.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/sgns/intrinsic.py tests/unit/test_sgns_intrinsic.py
git commit -m "feat(exp21): intrinsic validation — NN sanity + k-means clustering"
```

---

### Task 7: Intrinsic report script

**Files:**
- Create: `scripts/sgns_intrinsic_report.py`

Loads a saved SGNS `W_in`, prints NN for 50 common subwords, runs k=20 k-means, prints cluster-size distribution. No unit test — this is an operator-facing report.

**Step 1: Implement**

```python
# scripts/sgns_intrinsic_report.py
"""Intrinsic validation report for a saved SGNS embedding table.

Usage: python scripts/sgns_intrinsic_report.py --embed artifacts/sgns_v8192_d256.pt
"""
import argparse
from collections import Counter
from pathlib import Path
import torch

from chaoscontrol.sgns.intrinsic import nearest_neighbors, kmeans_clusters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embed", type=Path, required=True)
    p.add_argument("--n-common", type=int, default=50)
    p.add_argument("--k-clusters", type=int, default=20)
    args = p.parse_args()

    embed = torch.load(args.embed)
    V, D = embed.shape
    print(f"Loaded {args.embed}: V={V}, D={D}")

    # NN on first N_common token IDs (convention: low IDs are common under BPE)
    queries = list(range(args.n_common))
    nn = nearest_neighbors(embed, query_ids=queries, k=5)
    print(f"\\n== Nearest neighbors for first {args.n_common} token IDs ==")
    for q, nbrs in list(nn.items())[:args.n_common]:
        print(f"  {q:5d} -> {nbrs}")

    # k-means
    labels = kmeans_clusters(embed, k=args.k_clusters)
    counts = Counter(labels.tolist())
    print(f"\\n== k-means (k={args.k_clusters}) cluster sizes ==")
    for c, n in sorted(counts.items()):
        print(f"  cluster {c:2d}: {n:6d} tokens")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test after Phase 1 Task 5 run**

(Deferred — runs after SGNS is trained. See Task 11.)

**Step 3: Commit**

```bash
git add scripts/sgns_intrinsic_report.py
git commit -m "feat(exp21): intrinsic validation report script"
```

---

## Phase 3: Moment-matching + embedding init hook

### Task 8: Moment-matching utilities (mean+std, full-cov, shuffle)

**Files:**
- Create: `src/chaoscontrol/sgns/moment_match.py`
- Create: `tests/unit/test_moment_match.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_moment_match.py
import torch
from chaoscontrol.sgns.moment_match import (
    match_row_norm_moments,
    match_full_covariance,
    shuffle_rows,
)


def test_match_row_norm_moments_matches_target_moments():
    torch.manual_seed(0)
    src = torch.randn(100, 16) * 3.0 + 2.0
    target = torch.randn(100, 16) * 0.5
    out = match_row_norm_moments(src, target)
    src_row_norms = src.norm(dim=-1)
    target_row_norms = target.norm(dim=-1)
    out_row_norms = out.norm(dim=-1)
    assert torch.isclose(out_row_norms.mean(), target_row_norms.mean(), rtol=1e-3)
    assert torch.isclose(out_row_norms.std(), target_row_norms.std(), rtol=1e-3)


def test_match_row_norm_preserves_directions():
    """Rescaling rows preserves pairwise cosine similarity."""
    torch.manual_seed(1)
    src = torch.randn(50, 8)
    target = torch.randn(50, 8) * 0.01
    out = match_row_norm_moments(src, target)
    src_n = torch.nn.functional.normalize(src, dim=-1)
    out_n = torch.nn.functional.normalize(out, dim=-1)
    assert torch.allclose(src_n, out_n, atol=1e-5)


def test_match_full_covariance_matches_cov():
    torch.manual_seed(0)
    src = torch.randn(200, 8) * 3.0
    target = torch.randn(200, 8) * 0.5
    out = match_full_covariance(src, target)
    cov_out = torch.cov(out.T)
    cov_target = torch.cov(target.T)
    assert torch.allclose(cov_out, cov_target, atol=0.05)


def test_shuffle_rows_is_permutation():
    torch.manual_seed(0)
    src = torch.randn(100, 16)
    out = shuffle_rows(src, seed=42)
    # Same set of rows, different order
    sorted_src = src[torch.argsort(src[:, 0])]
    sorted_out = out[torch.argsort(out[:, 0])]
    assert torch.allclose(sorted_src, sorted_out)
    assert not torch.allclose(src, out)
```

**Step 2: Run tests, verify fail**

Run: `pytest tests/unit/test_moment_match.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# src/chaoscontrol/sgns/moment_match.py
import torch


def match_row_norm_moments(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rescale `src` per-row so the resulting row-norm distribution has the same
    mean and std as `target`'s. Preserves per-row direction (cosine) exactly.
    """
    src_norms = src.norm(dim=-1)
    target_norms = target.norm(dim=-1)
    # Z-score src norms, then re-scale to target moments
    z = (src_norms - src_norms.mean()) / src_norms.std().clamp(min=1e-8)
    new_norms = z * target_norms.std() + target_norms.mean()
    scale = (new_norms / src_norms.clamp(min=1e-12)).unsqueeze(-1)
    return src * scale


def match_full_covariance(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Whiten `src` then re-color to match `target`'s full row covariance.
    Rows of output have approximately the same covariance as rows of `target`.
    """
    src_centered = src - src.mean(dim=0)
    target_centered = target - target.mean(dim=0)
    cov_src = torch.cov(src_centered.T)
    cov_target = torch.cov(target_centered.T)
    # Whitening: L_src^{-1}, L_target applied
    L_src = torch.linalg.cholesky(cov_src + 1e-6 * torch.eye(src.shape[1]))
    L_target = torch.linalg.cholesky(cov_target + 1e-6 * torch.eye(target.shape[1]))
    whitened = torch.linalg.solve_triangular(L_src, src_centered.T, upper=False).T
    recolored = whitened @ L_target.T
    return recolored + target.mean(dim=0)


def shuffle_rows(src: torch.Tensor, seed: int) -> torch.Tensor:
    """Randomly permute rows of `src` under the given seed. Used as a control:
    preserves marginal distribution but destroys ID-to-vector mapping.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(src.shape[0], generator=g)
    return src[perm].clone()
```

**Step 4: Run tests, verify pass**

Run: `pytest tests/unit/test_moment_match.py -v`
Expected: 4 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/sgns/moment_match.py tests/unit/test_moment_match.py
git commit -m "feat(exp21): moment-matching utilities (mean+std, full-cov, shuffle)"
```

---

### Task 9: `embed_init_path` config + runner hook

**Files:**
- Modify: `src/chaoscontrol/config.py` (add field)
- Modify: `src/chaoscontrol/runner.py` (load init in `build_model`)
- Create: `tests/unit/test_embed_init_hook.py`

**Step 1: Write failing test**

```python
# tests/unit/test_embed_init_hook.py
import torch
import tempfile
from pathlib import Path
from chaoscontrol.config import build_default_config
from chaoscontrol.runner import build_model


def test_embed_init_path_loads_weights():
    """When embed_init_path is set, model.embed.weight matches the loaded tensor."""
    V, D = 64, 16
    init_weights = torch.randn(V, D) * 0.05
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "embed.pt"
        torch.save(init_weights, path)

        cfg = build_default_config(vocab_size=V, d_model=D, n_layer=2)
        cfg.embed_init_path = str(path)
        model = build_model(cfg)
        assert torch.allclose(model.embed.weight.detach().cpu(), init_weights, atol=1e-5)


def test_embed_init_path_none_uses_random():
    """When embed_init_path is None, no override occurs."""
    V, D = 64, 16
    cfg = build_default_config(vocab_size=V, d_model=D, n_layer=2)
    cfg.embed_init_path = None
    model = build_model(cfg)
    # Random init: not all zeros (sanity)
    assert not torch.allclose(model.embed.weight, torch.zeros_like(model.embed.weight))
```

NOTE: `build_default_config` is a helper that may need to be created if it doesn't exist — check `src/chaoscontrol/config.py` first; adapt to whatever the project's config entry point is.

**Step 2: Run tests, verify fail**

Run: `pytest tests/unit/test_embed_init_hook.py -v`
Expected: AttributeError on `cfg.embed_init_path`.

**Step 3: Implement config field**

Edit `src/chaoscontrol/config.py` around line 127 (per design doc's dependency note). Add field to the main model config dataclass / dict schema:

```python
# Add to the appropriate Config class:
embed_init_path: str | None = None
"""Path to a .pt tensor of shape (vocab_size, d_model) used to override random init
on model.embed.weight. If None, use default random init. See Exp 21 design doc."""
```

**Step 4: Implement runner hook**

Edit `src/chaoscontrol/runner.py:build_model`. After `model = model.to(device)`:

```python
if getattr(cfg, "embed_init_path", None):
    import torch as _t
    embed_weights = _t.load(cfg.embed_init_path, map_location=device)
    expected = model.embed.weight.shape
    assert embed_weights.shape == expected, (
        f"embed_init_path shape mismatch: got {embed_weights.shape}, expected {expected}"
    )
    with _t.no_grad():
        model.embed.weight.data.copy_(embed_weights)
```

**Step 5: Run tests, verify pass**

Run: `pytest tests/unit/test_embed_init_hook.py -v`
Expected: 2 passed.

**Step 6: Commit**

```bash
git add src/chaoscontrol/config.py src/chaoscontrol/runner.py tests/unit/test_embed_init_hook.py
git commit -m "feat(exp21): embed_init_path config + runner hook for SGNS init"
```

---

## Phase 4: Transformer arm registration

### Task 10: Audit existing transformer baseline

**Files:** `src/chaoscontrol/baselines.py` — read-only.

**Step 1: Read the baseline**

Run: `sed -n '1,80p' src/chaoscontrol/baselines.py`
(Or use the Read tool.) Capture: class name, arch hyperparams it accepts, whether it uses RoPE, RMSNorm, ReLU², Flash Attention, QK-norm, tied embeddings, auxiliary embeddings.

**Step 2: Compare to design doc requirements**

Design requires: d=256, n_head=4, n_layer=8, ffn_mult=4, RoPE, RMSNorm, ReLU², Flash Attn + QK-norm, no aux embeds, untied.

**Step 3: Decide adapt-or-add**

- If the existing baseline is close: plan a small adapter (next task).
- If it's very different: plan a new variant class (next task).

Write the audit result as a comment at the top of `experiments/21_sgns_tokenizer/NOTES.md` (create the dir if absent).

```bash
mkdir -p experiments/21_sgns_tokenizer
```

Write `experiments/21_sgns_tokenizer/NOTES.md` with 5-10 lines summarizing the audit decision.

**Step 4: Commit**

```bash
git add experiments/21_sgns_tokenizer/NOTES.md
git commit -m "docs(exp21): transformer baseline audit note"
```

---

### Task 11: Register modded-NanoGPT lean variant

**Files (determined by Task 10 audit):**
- Either modify `src/chaoscontrol/baselines.py` (adapter path)
- Or create `src/chaoscontrol/baselines_nanogpt_lean.py` (new-variant path)
- Create: `tests/unit/test_nanogpt_lean_config.py`

**Step 1: Write failing param-count test**

```python
# tests/unit/test_nanogpt_lean_config.py
import torch
from chaoscontrol.baselines import build_nanogpt_lean  # or appropriate entry point


def test_nanogpt_lean_param_count_v8192():
    model = build_nanogpt_lean(vocab_size=8192, d_model=256, n_head=4, n_layer=8, ffn_mult=4)
    n_params = sum(p.numel() for p in model.parameters())
    # Expected ≈ 10.49M per design doc param-count verification
    assert 10_300_000 <= n_params <= 10_700_000, f"got {n_params}"


def test_nanogpt_lean_forward_shapes():
    model = build_nanogpt_lean(vocab_size=8192, d_model=256, n_head=4, n_layer=8, ffn_mult=4)
    tokens = torch.randint(0, 8192, (2, 64))
    logits = model(tokens)
    assert logits.shape == (2, 64, 8192)
```

**Step 2: Run tests, verify fail**

Run: `pytest tests/unit/test_nanogpt_lean_config.py -v`
Expected: ImportError or shape mismatch.

**Step 3: Implement the variant** (scope depends on Task 10 outcome)

Adapter path: add a `build_nanogpt_lean(...)` factory that wraps the existing baseline with the design's exact hyperparams (d=256, n_head=4, n_layer=8, ffn_mult=4), ensures RoPE/RMSNorm/ReLU²/Flash+QK-norm are enabled, and disables tied embeddings + aux embeddings.

New-variant path: implement a minimal transformer module from scratch with those components. Reference `src/chaoscontrol/model.py` for style and serialization conventions.

Keep it minimal. No dropout. No positional bias.

**Step 4: Run tests, verify pass**

Run: `pytest tests/unit/test_nanogpt_lean_config.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/baselines*.py tests/unit/test_nanogpt_lean_config.py
git commit -m "feat(exp21): modded-NanoGPT lean variant (d=256, L=8, RoPE/RMSNorm/ReLU²/Flash+QK-norm)"
```

---

## Phase 5: Experiment execution

**Note:** Phase 5 tasks launch compute jobs on the CUDA-13 pod. Delegate pod orchestration to a subagent per the `feedback_delegate_ops.md` memory. Each task's "Step N: Run" may take 10 min to multiple hours of wall-clock.

### Task 12: Train the two SGNS embedding tables

**Files:**
- Create: `experiments/21_sgns_tokenizer/configs/sgns.yaml`
- Create: `experiments/21_sgns_tokenizer/run_sgns.sh`

**Step 1: Write configs**

```yaml
# experiments/21_sgns_tokenizer/configs/sgns.yaml
vocab_size: 8192
dim: 256
window: 5
k: 10
epochs: 3
subsample_threshold: 1.0e-5
batch_size: 4096
lr: 0.025
seed: 0
```

**Step 2: Launch SGNS training**

```bash
python scripts/train_sgns.py \
  --vocab-size 8192 --dim 256 --window 5 --k 10 --epochs 3 \
  --subsample-threshold 1e-5 --batch-size 4096 --lr 0.025 --seed 0 \
  --out artifacts/sgns_v8192_d256.pt
```

Expected: three "epoch N: mean_loss = ..." lines showing monotonic decrease; final tensor saved. Record the final loss in `experiments/21_sgns_tokenizer/results/sgns_training_log.txt`.

**Step 3: Derive the two init variants (mean+std, full-cov)**

Create: `scripts/prepare_sgns_inits.py` — loads `artifacts/sgns_v8192_d256.pt`, reads a reference random-init tensor (standard Gaussian matched to model init), applies both moment-matching flavors, saves:
- `artifacts/sgns_init_meanstd.pt`
- `artifacts/sgns_init_fullcov.pt`
- `artifacts/sgns_init_shuffled.pt` (shuffled-row control, based on meanstd variant, seed 42)

**Step 4: Run intrinsic report**

Run: `python scripts/sgns_intrinsic_report.py --embed artifacts/sgns_v8192_d256.pt | tee experiments/21_sgns_tokenizer/results/intrinsic_report.txt`

Manually inspect. **Kill criterion (design doc):** if >50% of the first 50 tokens' NN look incoherent → abort and retune SGNS hyperparameters.

**Step 5: Commit artifacts' paths (not weights — weights are too big)**

```bash
git add experiments/21_sgns_tokenizer/configs/sgns.yaml \
        experiments/21_sgns_tokenizer/run_sgns.sh \
        experiments/21_sgns_tokenizer/results/sgns_training_log.txt \
        experiments/21_sgns_tokenizer/results/intrinsic_report.txt \
        scripts/prepare_sgns_inits.py
git commit -m "exp21(sgns): offline training artifacts + intrinsic report"
```

---

### Task 13: Phase 0 transformer LR sweep

**Files:**
- Create: `experiments/21_sgns_tokenizer/configs/phase0_transformer_lr.yaml`
- Create: `experiments/21_sgns_tokenizer/runner_phase0.py`

**Step 1: Write sweep config**

```yaml
# 1 seed × 3 LRs on the transformer arm, random init (no SGNS)
arch: transformer_nanogpt_lean
d_model: 256
n_layer: 8
n_head: 4
ffn_mult: 4
vocab_size: 8192
seq_len: 512
batch_size_per_rank: 1024
world_size: 2
optimizer: muon
budget_seconds: 600
activation_checkpoint: true
lrs: [0.016, 0.032, 0.064]
seed: 1337
```

**Step 2: Implement a thin runner**

`experiments/21_sgns_tokenizer/runner_phase0.py` — wraps the existing training entry point (see `experiments/18_throughput_levers/runner_exp18_ssm.py` for the pattern), loops over the 3 LRs, dumps per-run bpb to `results/phase0_lr_bpb.json`.

**Step 3: Launch on pod (delegate to subagent)**

Spawn a subagent: "Launch the Exp 21 Phase 0 LR sweep on the CUDA-13 pod. 3 runs × 600 s ≈ 30 min. Report per-run bpb when finished."

**Step 4: Pick the winner**

Read `results/phase0_lr_bpb.json`, pick the LR with the lowest bpb. Record in `experiments/21_sgns_tokenizer/NOTES.md`. This is the transformer's LR for Phase 6.

**Step 5: Commit**

```bash
git add experiments/21_sgns_tokenizer/configs/phase0_transformer_lr.yaml \
        experiments/21_sgns_tokenizer/runner_phase0.py \
        experiments/21_sgns_tokenizer/results/phase0_lr_bpb.json \
        experiments/21_sgns_tokenizer/NOTES.md
git commit -m "exp21(phase0): transformer LR sweep; winner=<LR>"
```

---

### Task 14: 4-cell main ablation

**Files:**
- Create: `experiments/21_sgns_tokenizer/configs/four_cell.yaml`
- Create: `experiments/21_sgns_tokenizer/runner_4cell.py`

**Step 1: Write config**

```yaml
# 4 cells × 5 seeds = 20 runs
cells:
  - name: A_transformer_random
    arch: transformer_nanogpt_lean
    embed_init_path: null
    lr: <phase0 winner>
  - name: B_transformer_sgns
    arch: transformer_nanogpt_lean
    embed_init_path: artifacts/sgns_init_meanstd.pt
    lr: <phase0 winner>
  - name: C_ssm_random
    arch: ssm_exp18_t4b
    embed_init_path: null
    lr: 0.064
  - name: D_ssm_sgns
    arch: ssm_exp18_t4b
    embed_init_path: artifacts/sgns_init_meanstd.pt
    lr: 0.064
seeds: [1337, 42, 123, 7, 8]
seq_len: 512
batch_size_per_rank: 1024
world_size: 2
budget_seconds: 600
optimizer: muon
vocab_size: 8192
```

**Step 2: Implement runner**

`experiments/21_sgns_tokenizer/runner_4cell.py` — iterates cells × seeds (20 runs total), dumps per-run bpb to `results/four_cell_bpb.json` keyed by `(cell, seed)`.

**Step 3: Launch on pod (delegate to subagent)**

Delegate: "Launch Exp 21 4-cell main ablation, 20 runs × 10 min ≈ 3.5 hr. Monitor and report per-run bpb."

**Step 4: Verify completion and data integrity**

After subagent reports done, read `results/four_cell_bpb.json`. Verify: 20 entries, no NaNs, seeds match.

**Step 5: Commit**

```bash
git add experiments/21_sgns_tokenizer/configs/four_cell.yaml \
        experiments/21_sgns_tokenizer/runner_4cell.py \
        experiments/21_sgns_tokenizer/results/four_cell_bpb.json
git commit -m "exp21: 4-cell ablation (SSM × Txfmr × {random, SGNS}), 5 seeds each"
```

---

### Task 15: Moment-match robustness (full-cov) + shuffled-row sanity

**Files:**
- Create: `experiments/21_sgns_tokenizer/runner_controls.py`

**Step 1: Implement runner**

Runs two extra sweeps on the SSM arm only:
- 5 seeds with `embed_init_path=artifacts/sgns_init_fullcov.pt` — output to `results/fullcov_bpb.json`.
- 1 seed with `embed_init_path=artifacts/sgns_init_shuffled.pt` — output to `results/shuffled_bpb.json`.

Reuse the 4-cell runner pattern; vary only the init.

**Step 2: Launch on pod (delegate to subagent)**

Delegate: "Launch Exp 21 control runs — 5 seeds full-cov init + 1 seed shuffled-row init. 6 runs × 10 min ≈ 1 hr."

**Step 3: Verify integrity**

Check both JSON files: 5 entries + 1 entry, no NaNs.

**Step 4: Commit**

```bash
git add experiments/21_sgns_tokenizer/runner_controls.py \
        experiments/21_sgns_tokenizer/results/fullcov_bpb.json \
        experiments/21_sgns_tokenizer/results/shuffled_bpb.json
git commit -m "exp21: control runs — full-cov moment-match + shuffled-row sanity"
```

---

## Phase 6: Analysis

### Task 16: Statistical tests + report

**Files:**
- Create: `scripts/exp21_analyze.py`
- Create: `tests/unit/test_exp21_analyze.py`

**Step 1: Write failing test for the paired-t helper**

```python
# tests/unit/test_exp21_analyze.py
import pytest
from scripts.exp21_analyze import paired_t_one_sided


def test_paired_t_one_sided_obvious_effect():
    """Clear effect: x consistently > y."""
    x = [1.0, 1.1, 1.05, 1.08, 1.03]
    y = [0.5, 0.6, 0.55, 0.58, 0.53]
    p = paired_t_one_sided(x, y, alternative="greater")
    assert p < 0.01


def test_paired_t_one_sided_no_effect():
    """No effect: x ≈ y."""
    x = [1.0, 1.1, 0.9, 1.0, 1.05]
    y = [1.01, 1.09, 0.91, 1.02, 1.04]
    p = paired_t_one_sided(x, y, alternative="greater")
    assert p > 0.1
```

**Step 2: Run test, verify fail**

Run: `pytest tests/unit/test_exp21_analyze.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# scripts/exp21_analyze.py
"""Analyze Exp 21 4-cell + control results; print statistical conclusions.

Usage: python scripts/exp21_analyze.py --results-dir experiments/21_sgns_tokenizer/results
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def paired_t_one_sided(
    x: list[float], y: list[float], alternative: str = "greater"
) -> float:
    """Paired one-sided t-test p-value for H1: mean(x) > mean(y) (greater)
    or mean(x) < mean(y) (less). Seeds must be paired in the same order in x, y.
    """
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    assert len(x_arr) == len(y_arr)
    res = stats.ttest_rel(x_arr, y_arr, alternative=alternative)
    return float(res.pvalue)


def _by_seed(bpb_by_run: dict, cell: str) -> list[float]:
    """Extract bpb list ordered by seed for the given cell."""
    runs = {k: v for k, v in bpb_by_run.items() if k.startswith(cell)}
    return [runs[k] for k in sorted(runs)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    args = p.parse_args()

    four_cell = json.loads((args.results_dir / "four_cell_bpb.json").read_text())
    fullcov = json.loads((args.results_dir / "fullcov_bpb.json").read_text())
    shuffled = json.loads((args.results_dir / "shuffled_bpb.json").read_text())

    A = _by_seed(four_cell, "A_transformer_random")
    B = _by_seed(four_cell, "B_transformer_sgns")
    C = _by_seed(four_cell, "C_ssm_random")
    D = _by_seed(four_cell, "D_ssm_sgns")

    delta_ssm = [c - d for c, d in zip(C, D)]
    delta_trans = [a - b for a, b in zip(A, B)]
    diff = [ds - dt for ds, dt in zip(delta_ssm, delta_trans)]

    # Primary: Δ_SSM > 0 → C > D → paired_t_one_sided(C, D, "greater")
    p_primary = paired_t_one_sided(C, D, alternative="greater")
    # Secondary: Δ_SSM > Δ_Trans → paired on the differences
    p_secondary = paired_t_one_sided(delta_ssm, delta_trans, alternative="greater")

    print(f"== Exp 21 results ==")
    print(f"  A (transformer, random): mean={np.mean(A):.4f}, std={np.std(A, ddof=1):.4f}")
    print(f"  B (transformer, SGNS):   mean={np.mean(B):.4f}, std={np.std(B, ddof=1):.4f}")
    print(f"  C (SSM, random):         mean={np.mean(C):.4f}, std={np.std(C, ddof=1):.4f}")
    print(f"  D (SSM, SGNS):           mean={np.mean(D):.4f}, std={np.std(D, ddof=1):.4f}")
    print(f"  Δ_SSM   = C - D:  mean={np.mean(delta_ssm):.4f}")
    print(f"  Δ_Trans = A - B:  mean={np.mean(delta_trans):.4f}")
    print(f"\\n  Primary (Δ_SSM > 0) p = {p_primary:.4g}")
    print(f"  Secondary (Δ_SSM > Δ_Trans) p = {p_secondary:.4g}")

    thesis_validating = p_primary < 0.01 and p_secondary < 0.01
    thesis_weak = p_primary < 0.01 and p_secondary >= 0.01
    ship_worthy = np.mean(D) < 1.46

    print(f"\\n  thesis-validating: {thesis_validating}")
    print(f"  thesis-weak:       {thesis_weak}")
    print(f"  ship-worthy:       {ship_worthy}")

    # Controls
    D_fullcov = _by_seed(fullcov, "D_ssm_sgns_fullcov")
    p_fullcov_vs_random = paired_t_one_sided(C, D_fullcov, alternative="greater")
    delta_meanstd = np.mean(delta_ssm)
    delta_fullcov = np.mean(C) - np.mean(D_fullcov)
    print(f"\\n== Controls ==")
    print(f"  mean+std SGNS-SSM:   mean={np.mean(D):.4f}, Δ={delta_meanstd:.4f}")
    print(f"  full-cov SGNS-SSM:   mean={np.mean(D_fullcov):.4f}, Δ={delta_fullcov:.4f}")
    print(f"  full-cov vs random p = {p_fullcov_vs_random:.4g}")

    shuffled_key = next(k for k in shuffled)
    shuffled_bpb = shuffled[shuffled_key]
    print(f"  shuffled-row (1 seed): bpb = {shuffled_bpb:.4f}  (expect ≈ random mean if thesis holds: {np.mean(C):.4f})")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests, verify pass**

Run: `pytest tests/unit/test_exp21_analyze.py -v`
Expected: 2 passed.

**Step 5: Run the analysis**

Run: `python scripts/exp21_analyze.py --results-dir experiments/21_sgns_tokenizer/results | tee experiments/21_sgns_tokenizer/results/analysis.txt`
Expected: printed summary of means, deltas, p-values, thesis verdicts.

**Step 6: Commit**

```bash
git add scripts/exp21_analyze.py tests/unit/test_exp21_analyze.py \
        experiments/21_sgns_tokenizer/results/analysis.txt
git commit -m "exp21(analysis): paired t-tests on 4-cell + controls, verdict report"
```

---

### Task 17: Writeup + memory save

**Files:**
- Create: `docs/plans/2026-04-17-exp21-sgns-tokenizer-results.md`
- Create: `~/.claude/projects/-Users-kennethmalloy-Local-Documents-Developer-chaoscontrol/memory/project_exp21_results_<YYYY-MM-DD>.md`
- Modify: `~/.claude/projects/-Users-kennethmalloy-Local-Documents-Developer-chaoscontrol/memory/MEMORY.md`

**Step 1: Write the results doc**

In `docs/plans/2026-04-17-exp21-sgns-tokenizer-results.md`: mirror the design doc's structure — hypothesis, controls, method, results (tables of A/B/C/D means ± std, Δ_SSM, Δ_Trans, p-values), verdict, limitations, follow-ons.

**Step 2: Save a project memory**

Create a new project-type memory file capturing: the verdict (thesis-validating / weak / null), the headline numbers, and any new gotchas for future experiments. Keep to ~200 words.

**Step 3: Update MEMORY.md**

Add one-line entry pointing at the new memory file, following the conventions in the other entries.

**Step 4: Commit results doc (memory is outside this repo)**

```bash
git add docs/plans/2026-04-17-exp21-sgns-tokenizer-results.md
git commit -m "docs(exp21): results writeup + verdict"
```

---

## Reference: skills and verification

- Before declaring any analysis result, invoke `superpowers:verification-before-completion` — run the pytest suite and confirm green, quote the analysis output, do not claim p-values you haven't seen printed.
- If any run diverges or fails, invoke `superpowers:systematic-debugging` before patching. No band-aids.
- All commits follow `{scope}({subscope}): {what}` conventions from the repo's recent history (see `git log --oneline -n 10`).

## Out of scope (do not implement in Exp 21)

- Embedding mutation during LM training (freeze schedule, LR profile) — future post-Exp-19 session.
- Eval-time TTT adaptation of the embedding — Exp 20.
- V=16384 + int6 ship-ready variant — requires GPTQ wiring in the artifact path; separate engineering task.
- Alternative offline SSL objectives (CBOW, tiny-LM CE, encoder-based JEPA) — only if SGNS wins and we want to compare.
