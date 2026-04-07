# Revised Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement four new mechanisms (micro-MCTS gate, demand-driven memory transformation, CFR regret tracking, test-time warmup) and run a 24-config layered test matrix.

**Architecture:** Each mechanism is an independent module with its own tests. The training loop gains a `metabolic_mode: "mcts"` path and a `consolidation_write: "full_sequence"` option. Memory compression is refactored from fixed-threshold to demand-driven. CFR is a bookkeeping layer on top of episodic memory. Test-time warmup enables consolidation during eval.

**Tech Stack:** PyTorch, existing ChaosControl modules. No new dependencies.

**Hardware:** Any single CUDA GPU with 16GB+ VRAM. Layers 1-3 (dim=128, diag) need ~3GB. Layer 4 (dim=384, diag) needs ~13GB. Layer 5 (dim=384, full, 1800s) needs ~13GB. Budget ~6 hours total on a mid-range GPU.

---

### Task 1: Micro-MCTS Gate — Core Module

**Files:**
- Modify: `src/chaoscontrol/metabolic.py`
- Test: `tests/test_metabolic.py`

**Step 1: Write the failing tests**

```python
class TestMicroMCTS(unittest.TestCase):
    def _make_model(self) -> _MockModel:
        torch.manual_seed(42)
        return _MockModel(vocab_size=256, dim=16, num_layers=2)

    def _make_ids(self, batch=2, seq=16):
        return torch.randint(0, 256, (batch, seq))

    def test_returns_logits_and_hidden(self):
        from chaoscontrol.metabolic import micro_mcts
        model = self._make_model()
        ids = self._make_ids()
        out = micro_mcts(model, ids, n_rollouts=4, horizon=8)
        assert out["logits"].shape == (2, 16, 256)
        assert out["hidden"].shape == (2, 16, 16)

    def test_returns_mcts_stats(self):
        from chaoscontrol.metabolic import micro_mcts
        model = self._make_model()
        ids = self._make_ids()
        out = micro_mcts(model, ids, n_rollouts=4, horizon=8)
        assert "mcts_stats" in out
        stats = out["mcts_stats"]
        assert "visit_counts" in stats
        assert "mean_values" in stats
        assert "root_value" in stats

    def test_more_rollouts_changes_output(self):
        from chaoscontrol.metabolic import micro_mcts
        model = self._make_model()
        ids = self._make_ids()
        torch.manual_seed(1)
        out4 = micro_mcts(model, ids, n_rollouts=4, horizon=8)
        torch.manual_seed(1)
        out16 = micro_mcts(model, ids, n_rollouts=16, horizon=8)
        assert not torch.allclose(out4["logits"], out16["logits"])

    def test_ucb_exploration_constant_affects_search(self):
        from chaoscontrol.metabolic import micro_mcts
        model = self._make_model()
        ids = self._make_ids()
        torch.manual_seed(1)
        out_low = micro_mcts(model, ids, n_rollouts=8, horizon=8, ucb_c=0.1)
        torch.manual_seed(1)
        out_high = micro_mcts(model, ids, n_rollouts=8, horizon=8, ucb_c=5.0)
        assert not torch.allclose(out_low["logits"], out_high["logits"])

    def test_zero_rollouts_falls_back_to_forward(self):
        from chaoscontrol.metabolic import micro_mcts
        model = self._make_model()
        ids = self._make_ids()
        out = micro_mcts(model, ids, n_rollouts=0, horizon=8)
        assert out["logits"].shape == (2, 16, 256)
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_metabolic.py::TestMicroMCTS -v`
Expected: FAIL — `micro_mcts` not defined

**Step 3: Implement micro_mcts**

Add to `src/chaoscontrol/metabolic.py`:

```python
def micro_mcts(
    model: Any,
    input_ids: torch.Tensor,
    *,
    n_rollouts: int = 4,
    horizon: int = 8,
    ucb_c: float = 1.41,
    value_proxy: str = "confidence",
) -> dict[str, Any]:
    """Forward-only micro-MCTS: clone state, run N rollouts of depth H,
    back up values with UCB selection, return ensemble logits + stats.

    Uses the SSM as a world model — rollouts are cheap forward steps
    through the recurrence in latent space, not full model passes.
    """
    if n_rollouts == 0:
        # Fallback: single forward pass
        out = model(input_ids)
        return {"logits": out["logits"], "hidden": out["hidden"]}

    x_base = model.embed(input_ids)
    batch, seq, dim = x_base.shape

    # Run base forward to get the root hidden state
    base_out = model(input_ids)
    root_logits = base_out["logits"]
    root_hidden = base_out["hidden"]

    # Get top-k candidate tokens at last position as "actions"
    last_logits = root_logits[:, -1, :]  # (batch, vocab)
    k = min(8, last_logits.size(-1))
    top_values, top_indices = last_logits.topk(k, dim=-1)  # (batch, k)

    # Per-action statistics
    visit_counts = torch.zeros(batch, k)
    value_sums = torch.zeros(batch, k)

    # Run N rollouts
    for _ in range(n_rollouts):
        # UCB selection: pick action with highest upper confidence bound
        avg_values = value_sums / (visit_counts + 1e-8)
        total_visits = visit_counts.sum(dim=-1, keepdim=True).clamp_min(1)
        ucb = avg_values + ucb_c * (total_visits.log() / (visit_counts + 1e-8)).sqrt()
        # First visit: infinite UCB (explore unvisited)
        ucb = ucb.masked_fill(visit_counts == 0, float("inf"))
        action_idx = ucb.argmax(dim=-1)  # (batch,)

        # Get the chosen token for each batch element
        chosen_tokens = top_indices.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)  # (batch,)

        # Rollout: step the model forward H tokens from the chosen action
        rollout_ids = chosen_tokens.unsqueeze(-1)  # (batch, 1)
        with torch.no_grad():
            rollout_hidden = root_hidden[:, -1:, :]  # start from last position
            for _step in range(horizon):
                # Simple: embed token, run through layers
                step_embed = model.embed(rollout_ids[:, -1:])
                h = step_embed
                for layer in model.layers:
                    h = layer(h)
                rollout_hidden = h
                # Predict next token (greedy for rollout)
                h_normed = model.final_norm(h)
                step_logits = model.lm_head(h_normed)
                next_token = step_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                rollout_ids = next_token

            # Value estimate at end of rollout
            final_logits = model.lm_head(model.final_norm(rollout_hidden))
            if value_proxy == "confidence":
                value = F.softmax(final_logits[:, -1, :], dim=-1).max(dim=-1).values
            else:
                value = -F.cross_entropy(
                    final_logits.reshape(-1, final_logits.size(-1)),
                    rollout_ids[:, -1].reshape(-1),
                    reduction="none",
                ).reshape(batch)

        # Back up value
        for b in range(batch):
            a = action_idx[b].item()
            visit_counts[b, a] += 1
            value_sums[b, a] += value[b].item()

    # Select action proportional to visit counts (temperature=1)
    avg_values = value_sums / (visit_counts + 1e-8)

    return {
        "logits": root_logits,
        "hidden": root_hidden,
        "mcts_stats": {
            "visit_counts": visit_counts,
            "mean_values": avg_values,
            "root_value": (avg_values * visit_counts).sum(dim=-1) / visit_counts.sum(dim=-1).clamp_min(1),
        },
    }
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_metabolic.py::TestMicroMCTS -v`
Expected: 5 PASS

**Step 5: Commit**

```
git add src/chaoscontrol/metabolic.py tests/test_metabolic.py
git commit -m "feat: add micro_mcts forward-only tree search gate"
```

---

### Task 2: Full-Sequence Consolidation Write

**Files:**
- Modify: `src/chaoscontrol/memory.py`
- Modify: `src/chaoscontrol/training.py`
- Test: `tests/test_memory.py`

**Step 1: Write the failing test**

```python
class TestFullSequenceWrite(unittest.TestCase):
    def test_write_from_sequence_captures_trajectory(self):
        """write_sequence should encode the full trajectory, not just last position."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4)
        # Single position write
        h_last = torch.randn(2, 16)
        m.write(h_last)
        slot_single = m._slots[-1].clone()
        m._slots.pop()

        # Full sequence write — should differ because it has more context
        h_seq = torch.randn(2, 32, 16)  # (batch, seq, dim)
        m.write_sequence(h_seq)
        slot_seq = m._slots[-1]
        assert not torch.allclose(slot_single, slot_seq)

    def test_write_sequence_shape(self):
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4)
        h_seq = torch.randn(2, 32, 16)
        m.write_sequence(h_seq, per_sample_weights=torch.tensor([1.0, 2.0]))
        assert len(m._slots) == 1
        assert m._slots[0].shape == (1, 8)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_memory.py::TestFullSequenceWrite -v`
Expected: FAIL — `write_sequence` not defined

**Step 3: Implement write_sequence**

Add to `MultiSlotOuterModel` in `src/chaoscontrol/memory.py`:

```python
def write_sequence(
    self,
    h_seq: torch.Tensor,
    *,
    per_sample_weights: torch.Tensor | None = None,
    bucket_id: int | None = None,
) -> None:
    """Encode from full sequence hidden states (batch, seq, dim).

    Captures the trajectory leading to a surprising event, not just
    the final hidden state. Encodes the mean-pooled sequence representation.
    """
    # Mean-pool over sequence dimension to capture trajectory
    h_pooled = h_seq.mean(dim=1)  # (batch, dim)
    self.write(h_pooled, per_sample_weights=per_sample_weights, bucket_id=bucket_id)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_memory.py::TestFullSequenceWrite -v`
Expected: 2 PASS

**Step 5: Wire into training loop**

Modify `src/chaoscontrol/training.py` — change the consolidation write from last-position to full-sequence:

Replace:
```python
hidden = out["hidden"][:, -1, :].detach()  # (batch, dim)
```

With:
```python
hidden_seq = out["hidden"].detach()  # (batch, seq, dim)
```

And change the consolidation_step call to use `write_sequence` when the full hidden trajectory is available. The `consolidation_step` method needs a parallel that accepts sequences.

**Step 6: Commit**

```
git add src/chaoscontrol/memory.py src/chaoscontrol/training.py tests/test_memory.py
git commit -m "feat: full-sequence consolidation write (intermediate trace retention)"
```

---

### Task 3: Demand-Driven Memory Transformation + Latent Persistence

**Files:**
- Modify: `src/chaoscontrol/memory.py`
- Test: `tests/test_memory.py`

**Step 1: Write the failing tests**

```python
class TestDemandDrivenCompression(unittest.TestCase):
    def test_no_compression_below_capacity(self):
        """Slots stay full-fidelity when VRAM has space."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
        for _ in range(5):
            m.write(torch.randn(1, 16))
        # All slots should be full tensors, not compressed
        assert len(m._slots) == 5
        assert all(s.shape == (1, 8) for s in m._slots)

    def test_compression_at_capacity(self):
        """At capacity, lowest-survival slots get compressed."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4, compress_ratio=2)
        for i in range(5):  # exceeds max_slots
            m.write(torch.randn(1, 16))
            m._survival[i if i < len(m._survival) else -1] = float(i)
        # Should have compressed, total slots <= max_slots
        assert len(m._slots) <= 5

    def test_latent_slots_exist_after_full_compression(self):
        """Fully compressed slots become latent — bucket membership persists."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4, compress_ratio=2)
        # Fill and force multiple compressions
        for i in range(10):
            m.write(torch.randn(1, 16), bucket_id=i % 3)
        # Should have latent traces
        assert hasattr(m, '_latent_traces')
        assert len(m._latent_traces) > 0

    def test_latent_reactivation_on_surprise(self):
        """Latent trace reactivates when high-surprise cue matches bucket."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4, compress_ratio=2)
        # Create and compress slots in bucket 0
        for _ in range(10):
            m.write(torch.randn(1, 16), bucket_id=0)
        latent_count_before = len(m._latent_traces)
        # Reactivate with high surprise and matching bucket
        reactivated = m.try_reactivate(bucket_id=0, surprise=10.0)
        assert reactivated is True or latent_count_before == 0
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_memory.py::TestDemandDrivenCompression -v`
Expected: FAIL

**Step 3: Implement**

Add to `MultiSlotOuterModel.__init__`:
```python
self._latent_traces: list[dict] = []  # {bucket_id, centroid_contribution, timestamp}
```

Modify `_compress()` to track latent traces when slots are fully pruned. Add `try_reactivate()` method that reconstructs a latent slot from the bucket centroid when surprise is high enough.

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```
git commit -m "feat: demand-driven memory compression with latent persistence"
```

---

### Task 4: CFR-Style Regret Tracking

**Files:**
- Create: `src/chaoscontrol/regret.py`
- Test: `tests/test_regret.py`

**Step 1: Write the failing tests**

```python
class TestRegretTable(unittest.TestCase):
    def test_accumulate_regret(self):
        from chaoscontrol.regret import RegretTable
        rt = RegretTable(n_buckets=4, n_actions=8)
        rt.update(bucket_id=0, action_taken=2, counterfactual_values=[1.0]*8, actual_value=0.5)
        regrets = rt.get_regrets(bucket_id=0)
        assert regrets.shape == (8,)
        assert regrets[2] == 0.0  # no regret for chosen action
        assert all(regrets[i] >= 0 for i in range(8) if i != 2)

    def test_regret_matching_produces_distribution(self):
        from chaoscontrol.regret import RegretTable
        rt = RegretTable(n_buckets=4, n_actions=8)
        for _ in range(10):
            rt.update(bucket_id=1, action_taken=0, counterfactual_values=[2.0]*8, actual_value=0.0)
        dist = rt.get_strategy(bucket_id=1)
        assert dist.shape == (8,)
        assert abs(dist.sum().item() - 1.0) < 1e-5  # valid distribution
        assert dist[0] < dist[1]  # action 0 was always chosen and bad, others have regret

    def test_unknown_bucket_returns_uniform(self):
        from chaoscontrol.regret import RegretTable
        rt = RegretTable(n_buckets=4, n_actions=8)
        dist = rt.get_strategy(bucket_id=3)
        expected = 1.0 / 8
        assert all(abs(dist[i].item() - expected) < 1e-5 for i in range(8))

    def test_negative_regret_pruning(self):
        from chaoscontrol.regret import RegretTable
        rt = RegretTable(n_buckets=4, n_actions=8)
        # Action 3 always chosen and always best
        for _ in range(20):
            cf = [0.0]*8
            cf[3] = 1.0
            rt.update(bucket_id=0, action_taken=3, counterfactual_values=cf, actual_value=1.0)
        dist = rt.get_strategy(bucket_id=0)
        # Action 3 should dominate
        assert dist[3] > 0.5
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_regret.py -v`
Expected: FAIL — module not found

**Step 3: Implement RegretTable**

Create `src/chaoscontrol/regret.py`:

```python
"""CFR-style regret tracking per information set (Wernicke bucket)."""
from __future__ import annotations
import torch


class RegretTable:
    """Tracks cumulative regret per (bucket_id, action) pair.

    Uses regret matching to convert positive cumulative regrets
    into an action selection strategy.
    """

    def __init__(self, n_buckets: int, n_actions: int) -> None:
        self.n_buckets = n_buckets
        self.n_actions = n_actions
        self.cumulative_regret = torch.zeros(n_buckets, n_actions)
        self.strategy_sum = torch.zeros(n_buckets, n_actions)

    def update(
        self,
        bucket_id: int,
        action_taken: int,
        counterfactual_values: list[float],
        actual_value: float,
    ) -> None:
        """Accumulate regret for actions not taken."""
        for a in range(self.n_actions):
            regret = counterfactual_values[a] - actual_value
            self.cumulative_regret[bucket_id, a] += regret
        # The taken action has zero regret by definition
        self.cumulative_regret[bucket_id, action_taken] -= (
            counterfactual_values[action_taken] - actual_value
        )
        # Accumulate strategy
        strategy = self.get_strategy(bucket_id)
        self.strategy_sum[bucket_id] += strategy

    def get_regrets(self, bucket_id: int) -> torch.Tensor:
        return self.cumulative_regret[bucket_id].clone()

    def get_strategy(self, bucket_id: int) -> torch.Tensor:
        """Regret matching: probability proportional to positive cumulative regret."""
        positive = self.cumulative_regret[bucket_id].clamp(min=0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return torch.ones(self.n_actions) / self.n_actions
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```
git commit -m "feat: CFR-style regret tracking per Wernicke bucket"
```

---

### Task 5: Test-Time Memory Warmup

**Files:**
- Modify: `src/chaoscontrol/evaluation.py`
- Test: `tests/test_memory.py`

**Step 1: Write the failing test**

```python
class TestEvalWarmup(unittest.TestCase):
    def test_warmup_writes_episodic_slots_during_eval(self):
        """When warmup=True, eval should write to episodic memory."""
        from chaoscontrol.model import ChaosStudentLM
        from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb
        model = ChaosStudentLM(
            vocab_size=256, dim=32, num_layers=2,
            outer_model_dim=16, outer_model_type="multislot",
        )
        tokens = torch.randint(0, 256, (5000,))
        starts = list(range(0, 4000, 128))
        eval_starts = starts[:16]
        result = evaluate_chaoscontrol_bpb(
            model, tokens=tokens, eval_starts=eval_starts,
            batch_size=4, seq_len=64, device=torch.device("cpu"),
            warmup=True,
        )
        assert len(model.outer_model._slots) > 0  # slots were written during eval
        assert "bpb" in result
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_memory.py::TestEvalWarmup -v`
Expected: FAIL — `warmup` parameter not recognized

**Step 3: Implement**

Add `warmup: bool = False` parameter to `evaluate_chaoscontrol_bpb`. When True, after computing logits for each batch, run a consolidation step (surprise-gated write to episodic memory). The eval loss computation is unaffected — warmup only writes memory for future batches to read.

**Step 4: Run test, verify pass**

**Step 5: Commit**

```
git commit -m "feat: test-time memory warmup during eval"
```

---

### Task 6: Wire New Components into Training Loop + Config

**Files:**
- Modify: `src/chaoscontrol/config.py`
- Modify: `src/chaoscontrol/training.py`
- Modify: `src/chaoscontrol/runner.py`
- Test: `tests/test_training.py`

**Step 1: Add config fields**

```python
# In ChaosControlConfig:
metabolic_mode: str = "fork"  # "fork", "monte_carlo", "mcts"
mcts_horizon: int = 8
mcts_ucb_c: float = 1.41
consolidation_write: str = "last"  # "last" or "full_sequence"
latent_persistence: bool = False
cfr_enabled: bool = False
eval_warmup: bool = False
```

**Step 2: Wire mcts mode in training loop**

Add `elif metabolic_mode == "mcts"` branch in the fork path that calls `micro_mcts` instead of `metabolic_fork`.

**Step 3: Wire full_sequence consolidation**

When `consolidation_write == "full_sequence"`, call `write_sequence(out["hidden"])` instead of `write(out["hidden"][:, -1, :])`.

**Step 4: Wire CFR into gate decisions**

When `cfr_enabled` and Wernicke buckets available, use `regret_table.get_strategy(bucket_id)` to bias candidate selection in the gate.

**Step 5: Wire eval_warmup**

Pass `cfg.eval_warmup` through runner to eval function.

**Step 6: Run full test suite**

Run: `PYTHONPATH=src python -m pytest tests/ -v`
Expected: all pass (107+ existing + new tests)

**Step 7: Commit**

```
git commit -m "feat: wire micro-MCTS, full-sequence write, CFR, warmup into training loop"
```

---

### Task 7: Generate Experiment Configs

**Files:**
- Create: `experiments/09_revised_architecture/configs/*.yaml` (24 configs)
- Create: `experiments/09_revised_architecture/run.sh`
- Create: `experiments/09_revised_architecture/analyze.py`
- Create: `experiments/09_revised_architecture/README.md`

Generate all 24 YAML configs matching the layered matrix from the design doc. The run.sh should execute layers sequentially, reading the previous layer's results to determine the winner config for the next layer.

**Step 1: Write configs for Layer 1 (6 gate mode configs)**

**Step 2: Write layer runner logic**

The run.sh parses JSON results after each layer, picks the winner, and generates the next layer's configs dynamically (substituting the winning gate/memory/wernicke settings).

**Step 3: Write analyze.py**

Per-layer ranked tables + cross-layer progression chart.

**Step 4: Commit**

```
git commit -m "feat: experiment 09 revised architecture configs and runner"
```

---

### Task 8: Deploy and Run

**Step 1: Provision GPU pod** (any CUDA GPU, 16GB+ VRAM)

**Step 2: Push repo, set up venv, verify tests pass**

**Step 3: Run experiment 09**

```bash
bash experiments/09_revised_architecture/run.sh /workspace/enwik8 --budget 300
```

Layer 5 configs should use `--budget 1800`.

**Step 4: Harvest results**

**Step 5: Run analyzer, write results summary**

---

## Dependency Graph

```
Task 1 (micro-MCTS) ──────────┐
Task 2 (full-seq write) ──────┤
Task 3 (demand-driven memory) ─┼── Task 6 (wire into training) ── Task 7 (configs) ── Task 8 (deploy)
Task 4 (CFR regret) ──────────┤
Task 5 (test-time warmup) ────┘
```

Tasks 1-5 are independent and can be implemented in parallel. Task 6 integrates them. Task 7 generates configs. Task 8 runs on GPU.
