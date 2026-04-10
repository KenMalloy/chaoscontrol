# Experiment 15: ChaosPiece -- SSM with Tokenized Front-End

## Status

Design phase. Pending Phase A implementation.

**Artifact accounting assumption:** SP8192 is provided by the competition
at evaluation time (it is part of the eval harness, not the 16 MB
artifact). We do not ship the tokenizer. This matches competition rules
and existing top submissions. All param budget calculations below exclude
the SentencePiece model file.

## Motivation

### Experiment 14 Results

Experiment 14 tested whether a typed KV buffer with Wernicke MoE routing
could close the gap between our bare SSM and transformer-based
competition leaders. It failed decisively:

| Condition           | Mean bpb | 95% CI           |
|---------------------|----------|------------------|
| **bare_ssm**        | **2.396**| [2.383, 2.409]   |
| softmax_all_32      | 2.414    | [2.402, 2.426]   |
| topk_uncapped_16    | 2.559    | [2.546, 2.571]   |
| flat_16             | 2.564    | [2.535, 2.590]   |
| mean_uncapped       | 2.563    | [2.550, 2.579]   |
| topk_uncapped_8     | 2.583    | [2.560, 2.605]   |
| topk_uncapped_4     | 2.585    | [2.558, 2.612]   |
| flat_64             | 2.599    | [2.578, 2.619]   |
| recent_uncapped     | 2.614    | [2.593, 2.636]   |
| softmax_all_uncapped| 2.618    | [2.592, 2.637]   |
| topk_8k_8           | 2.623    | [2.609, 2.637]   |
| flat_8              | 2.631    | [2.608, 2.653]   |

Every buffer condition hurt. The best buffer condition (softmax_all_32 at
2.414) was worse than the bare SSM (2.396) by +0.018 bpb. Hierarchical
Wernicke conditions (hier_8x8, hier_8x32) failed to run entirely.

### Why the Buffer Thesis Failed

Wernicke was trying to discover byte-level structure online: routing raw
bytes into semantic buckets via VQ/MoE, then using those noisy bucket IDs
as memory keys. This approach has three compounding problems:

1. **Unstable keys.** Byte-level bucket assignments are context-sensitive
   and noisy. The same substring routes differently depending on position
   and surrounding bytes. Memory keyed by unstable IDs produces
   unreliable retrieval.

2. **Expensive online discovery.** Learning byte-level types from scratch
   within a 10-minute training budget costs gradient steps that would be
   better spent on the core model. The 16-expert MoE Wernicke layer adds
   parameters and FLOPS without a proportional return.

3. **No compression prior.** Raw bytes give the model no leverage over
   repeated structure. "the" appears thousands of times in training data
   but must be recognized from scratch at each of its 3 byte positions.
   Transformers with SP8192 see "the" as a single known token.

### The Competition Gap

Competition leaders score 1.06-1.08 bpb. Our best is 2.396 bpb. The
gap is ~1.3 bpb. The dominant factor is the tokenizer:

- SP8192 provides 8192 discrete symbols covering frequent substrings.
- Each token compresses ~4 raw bytes on average (measured on FineWeb).
- The model sees ~4x fewer timesteps for the same text, meaning the SSM
  state covers ~4x more context per position.
- Frequent patterns ("the ", "ing", "tion") are single symbols. The
  model predicts one token instead of 3-5 conditional bytes.

SP8192 is not a tokenizer trick. It is a representation strategy:
compress repeated structure into stable symbols early, then model on
top. This is the insight that motivates Experiment 15.

## Architecture

### Current (Exp 14, failed)

```
raw bytes (256 vocab)
  |
  v
byte_embed(256, 128)          32,768 params
  |
  v
WernickeLayer                  16-expert MoE, causal conv
  |                            Discovers types online from bytes
  v                            Noisy bucket IDs
KV buffer read                 Retrieval keyed by bucket ID
  |
  v
SSM backbone                   4 layers, dim=128, diag A
  |
  v
KV buffer write                Append per-token KV pair
  |
  v
lm_head(128, 256)             Predict next byte
```

### ChaosPiece (Exp 15)

```
raw bytes (256 vocab)
  |
  v
SP8192 tokenizer               Pre-trained, not shipped.
  |                            ~4x compression. Stable symbols.
  v
token_embed(8192, D)           8192 * D params (D TBD)
  |
  v
[Phase C only]
contextual_typer(D, K)         Tiny linear head. K << 16.
  |                            Assigns a context-dependent type
  v                            to each stable token.
SSM backbone                   N layers, dim=D, diag A
  |
  v
[Phase B only]
token_memory.read(             Keyed by discrete token ID,
  key=token_id,                not by noisy bucket.
  context=ssm_state            Stable keys = reliable retrieval.
)
  |
  v
lm_head(D, 8192)              Predict next token
  |
  v
bpb = CE(logits, target)      Token CE in nats, divided by
      / raw_byte_count         raw bytes in the batch,
      / ln(2)                  divided by ln(2).
```

### Key Architectural Difference

The Wernicke layer tried to do two things at once: discover structure
AND provide typed routing. ChaosPiece separates these concerns:

1. **Structure discovery** is handled by SP8192 (pre-trained, free).
2. **Typed routing** (Phase C) operates on already-stable token
   representations, where a simple linear head suffices.

This is analogous to the biological distinction between peripheral
perception (fast, hardwired) and cortical categorization (learned,
contextual). Wernicke tried to be both.

### Ontological Framing

Experiment 15 can be read as a lightweight ontology stack for an SSM:

- **Level 1: lexical identity** -- SP8192 provides stable symbols
  ("what unit is this?").
- **Level 2: contextual role** -- the Phase C typer assigns a small
  role inventory on top of those symbols ("what is this unit doing
  here?").
- **Level 3: memory addressability** -- token-keyed memory stores and
  retrieves by these stable units rather than by noisy latent buckets.

This is the deeper thesis behind ChaosPiece: semantic depth in an SSM
does not come only from longer recurrence. It comes from choosing the
right units of meaning and preserving them across perception, memory,
and retrieval.

## Phase Structure

Each phase is go/no-go on the next. A phase fails if the best condition
does not beat the control (with statistical significance at p < 0.05,
two-sided bootstrap test).

The phases are also layered ontologically:

- **Phase A** establishes stable lexical units.
- **Phase B** tests whether stable units are enough to support useful
  memory keys.
- **Phase C** tests whether adding lightweight contextual roles on top
  of stable units improves memory access and semantic reuse.

### Phase A: SP8192 Baseline

**Question:** What is the SSM ceiling with proper front-end compression?

**Design:** Drop in SP8192 tokenizer, run the bare SSM on tokenized
input. No memory, no Wernicke, no typing heads. Pure SSM + tokenizer.

**Conditions (5 configs x 7 seeds = 35 runs):**

| Condition          | dim | layers | ff_mult | vocab  | Notes                     |
|--------------------|-----|--------|---------|--------|---------------------------|
| `sp_d128_L4`       | 128 | 4      | 2       | 8192   | Match Exp 14 backbone     |
| `sp_d192_L4`       | 192 | 4      | 2       | 8192   | Wider to absorb embed cost|
| `sp_d128_L6`       | 128 | 6      | 2       | 8192   | Deeper variant            |
| `sp_d192_L6`       | 192 | 6      | 2       | 8192   | Wide + deep               |
| `sp_d256_L4`       | 256 | 4      | 2       | 8192   | Maximum width             |

**Control:** `bare_ssm_byte256` -- the Exp 14 bare SSM winner (vocab=256,
dim=128, 4 layers, ~2.396 bpb). Run with 7 fresh seeds to get a
concurrent measurement on the same hardware.

**Total runs:** 42 (35 SP + 7 control)

**Parameter budget analysis (16 MB = 8,388,608 params at fp16):**

| Condition    | Embed     | SSM+FF          | LM head   | Total     | Artifact % |
|--------------|-----------|-----------------|-----------|-----------|------------|
| sp_d128_L4   | 1,048,576 | 4*(128^2*5)=327K| 1,048,576 | ~2.42M    | 29%        |
| sp_d192_L4   | 1,572,864 | 4*(192^2*5)=737K| 1,572,864 | ~3.88M    | 46%        |
| sp_d128_L6   | 1,048,576 | 6*(128^2*5)=491K| 1,048,576 | ~2.59M    | 31%        |
| sp_d192_L6   | 1,572,864 | 6*(192^2*5)=1.1M| 1,572,864| ~4.25M    | 51%        |
| sp_d256_L4   | 2,097,152 | 4*(256^2*5)=1.3M| 2,097,152 | ~5.49M    | 65%        |

Note: "SSM+FF" is approximate -- each ChaosSSMBlock has ChaosSSMCore
(A diag + B + C + D projections ~= dim^2 * 3) plus FeedForward
(dim * dim * ff_mult * 2) plus two RMSNorm (dim * 2). Total per block
~= dim^2 * (3 + 2*ff_mult) + 2*dim. The precise count will be
verified at runtime via `model.artifact_bytes()`.

All conditions fit comfortably within the 16 MB budget even before
int8 quantization (which halves the artifact size). Weight tying between
embed and lm_head is an option if budget gets tight at d256, but Phase A
should test without tying first (tying constrains the representations).

**Config template:**

```yaml
model_type: ssm
vocab_size: 8192
model_dim: 128        # varies per condition
num_layers: 4         # varies per condition
ff_mult: 2
seq_len: 512          # tokens, not bytes. ~2048 raw bytes at ~4x compression.
stride: 256
batch_size: 32
base_lr: 2e-3
a_mode: diag
crit_target_coupling: 0.92
outer_model_dim: 0    # no memory
wernicke_enabled: false
tokenizer_type: sp8192
```

**seq_len rationale:** With SP8192 compressing ~4 bytes per token,
seq_len=512 tokens covers ~2048 raw bytes -- an 8x increase in effective
context compared to the byte-level seq_len=256. This is a major
advantage. The seq_len may need tuning; Phase A will also try seq_len=256
tokens (covering ~1024 raw bytes) as a secondary sweep if the primary
runs complete quickly.

**Primary comparison:** Phase A must answer two questions, not one:

1. **Does the tokenizer close the gap vs raw bytes?**
   Compare SP+SSM against bare_ssm_byte256 (Exp 14 winner, ~2.396 bpb).
2. **How far is the SSM from tokenized transformers?**
   Compare SP+SSM against the competition baseline transformer
   (`train_gpt.py` default config, same SP8192, same budget). Run the
   baseline on the same pod/data as a proper control.

Without (2), a result of 1.5 bpb is ambiguous — it could mean "the
tokenizer explains most of the gap" or "the SSM is still 0.4 bpb behind
a tokenized transformer of the same size." Only direct comparison
against a tokenized transformer answers the motivating question.

**Conditions (updated):** Add 1 transformer control:

| Condition          | Type | dim | layers | vocab | Notes                      |
|--------------------|------|-----|--------|-------|----------------------------|
| `sp_d128_L4`       | SSM  | 128 | 4      | 8192  | Match Exp 14 backbone      |
| `sp_d192_L4`       | SSM  | 192 | 4      | 8192  | Wider                      |
| `sp_d128_L6`       | SSM  | 128 | 6      | 8192  | Deeper                     |
| `sp_d192_L6`       | SSM  | 192 | 6      | 8192  | Wide + deep                |
| `sp_d256_L4`       | SSM  | 256 | 4      | 8192  | Maximum width              |
| `bare_ssm_byte256` | SSM  | 128 | 4      | 256   | Exp 14 winner control      |
| `gpt_matched`      | GPT  | *   | *      | 8192  | Matched-param transformer  |

**Total runs:** 49 (35 SP-SSM + 7 byte control + 7 transformer control)

**Transformer control specification:** `gpt_matched` is a true
matched control, NOT a leaderboard reference. It runs inside the
ChaosControl training/eval stack (same optimizer, same budget, same
data pipeline, same eval windows) with a transformer backbone instead
of the SSM. Param count is matched to the Phase A SSM winner by
adjusting dim/layers. This isolates the backbone question: "is the SSM
competitive with a transformer *holding everything else constant*?"

Leaderboard scores (1.06-1.08 bpb) use Muon optimizer, depth
recurrence, and TTT — none of which we run. Those are aspirational
targets, not scientific controls.

**Success criteria (two-part):**
- **Tokenizer helps:** Best SP-SSM < bare_ssm_byte256 by >= 0.1 bpb
  (p < 0.05). Expected: large effect.
- **SSM competitive:** Best SP-SSM within 0.15 bpb of gpt_matched.
  If the gap is larger, the SSM backbone itself is the bottleneck and
  memory cannot close it.

**Failure response:**
- If SP-SSM > 2.0 bpb: tokenizer alone doesn't help. Revisit backbone.
- If SP-SSM is competitive with byte but far from gpt_matched: SSM
  backbone is the bottleneck. Pivot to depth recurrence / attention
  hybrid.

### Phase B: Token-Keyed Memory (contingent on Phase A)

**Question:** Does token-keyed memory help a tokenized SSM under
reset-per-segment TTT evaluation? (The competition resets all runtime
state between segments. Memory must rebuild from scratch each time.
This is a stronger test than persistent-state eval.)

**Design:** Add episodic buffer keyed by discrete token identity.
Unlike Exp 14's bucket-keyed buffer, the keys here are deterministic
(the SP8192 token ID) rather than learned (Wernicke bucket assignment).

**Memory architecture:**

**Implementation constraint:** The memory MUST be tensorized for GPU.
A Python dict of deques in the forward path would repeat Exp 14's
failure mode — mechanism costs more compute than it earns. The design
below uses a fixed-size tensor buffer with scatter/gather ops.

```python
class TokenKeyedMemory(nn.Module):
    """Episodic buffer keyed by discrete token identity.

    GPU-friendly: all storage is a pair of fixed-size tensors, indexed
    by token ID. No Python dicts, no per-example loops.

    Storage shape:
      values:  (vocab_size, max_entries, value_dim)  -- pre-allocated
      write_ptr: (vocab_size,)                       -- circular index

    Read: gather rows by token_id, score against context, return weighted sum.
    Write: scatter new values into the next slot per token (circular FIFO).
    All ops are batched scatter/gather — no host-device round trips.
    """
    def __init__(self, vocab_size, value_dim, max_entries):
        super().__init__()
        self.max_entries = max_entries
        # Pre-allocated buffer (runtime state, not parameters)
        self.register_buffer(
            "values", torch.zeros(vocab_size, max_entries, value_dim))
        self.register_buffer(
            "write_ptr", torch.zeros(vocab_size, dtype=torch.long))
        self.register_buffer(
            "occupancy", torch.zeros(vocab_size, dtype=torch.long))
        # Retrieval scoring
        self.query_proj = nn.Linear(value_dim, value_dim, bias=False)

    def read(self, token_ids, context):
        """Batched read. token_ids: (batch, seq). context: (batch, seq, dim).
        Returns: (batch, seq, value_dim)."""
        # Gather: (batch, seq, max_entries, value_dim)
        entries = self.values[token_ids]
        # Score: dot product between query and stored values
        query = self.query_proj(context).unsqueeze(-2)  # (B, S, 1, D)
        scores = (query * entries).sum(-1)  # (B, S, max_entries)
        # Mask empty slots
        occ = self.occupancy[token_ids].clamp(max=self.max_entries)
        mask = torch.arange(self.max_entries, device=scores.device) < occ.unsqueeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * entries).sum(-2)

    def write(self, token_ids_flat, values_flat):
        """Fully batched write. token_ids_flat: (N,). values_flat: (N, dim).
        No Python loops — uses scatter_ for GPU-friendly FIFO writes."""
        # Compute per-token write slot: current ptr + intra-batch offset
        # For tokens appearing multiple times in one batch, offset ensures
        # each occurrence writes to a different slot.
        _, inverse, counts = token_ids_flat.unique(
            return_inverse=True, return_counts=True)
        # Intra-batch offset per occurrence of each token
        offsets = torch.zeros_like(token_ids_flat)
        for i in range(1, token_ids_flat.size(0)):
            if inverse[i] == inverse[i - 1]:
                offsets[i] = offsets[i - 1] + 1
        # Global write position per element
        ptrs = self.write_ptr[token_ids_flat] + offsets
        slots = ptrs % self.max_entries  # circular FIFO
        # Scatter values into buffer
        self.values[token_ids_flat, slots] = values_flat.detach()
        # Update write pointers (advance by count per token)
        self.write_ptr.scatter_add_(0, token_ids_flat, torch.ones_like(token_ids_flat))
        self.occupancy = self.write_ptr.clamp(max=self.max_entries)
```

**Both read and write paths are tensorized.** The offset computation
has a sequential dependency (tracking intra-batch position per token)
but operates on CPU-resident index tensors, not model activations.
If profiling shows this matters, it can be replaced with a
`torch.cumsum` over a sorted-by-token permutation. Exp 14 taught us:
mechanisms die when they steal steps.

**Conditions (4 configs x 7 seeds = 28 runs):**

| Condition            | Memory entries/token | Value dim | Notes                   |
|----------------------|---------------------|-----------|-------------------------|
| `sp_mem_4`           | 4                   | 64        | Minimal memory          |
| `sp_mem_16`          | 16                  | 64        | Moderate capacity       |
| `sp_mem_64`          | 64                  | 64        | Large per-token buffer  |
| `sp_no_mem`          | 0                   | --        | Phase A winner, control |

Phase A winner is the control. Only the Phase A winner's dim/layers
carry forward; the other 4 dim/layer variants are dropped.

**Parameter cost:** TokenKeyedMemory adds only query_proj (D * value_dim).
The buffer contents are runtime state, not parameters. At D=128,
value_dim=64: 8,192 params. Negligible.

**Evaluation protocol:** Token memory is runtime state, not artifact
state. Phase B MUST use cold-start warming-curve evaluation (the repo's
`evaluate_warming_curve()`) to distinguish two scenarios:

1. **Memory helps as deployable TTT state:** Cold-start bpb improves
   with more warmup tokens. The warming curve slopes downward. This
   means memory is earning its keep at test time.
2. **Memory only benefits from persistent eval state:** Pooled bpb
   improves but warming curve is flat. This means memory accumulated
   useful state during the eval loop but wouldn't help in the
   competition's reset-per-segment protocol.

Only scenario (1) is a real win. The warming curve is the primary
metric, not pooled bpb.

**Success criterion:** Best memory condition shows >= 0.02 bpb
improvement at warmup=5000 vs warmup=0 on the warming curve (p < 0.05),
AND the warming curve slopes more steeply than the no-memory control.

**Failure response:** If memory does not help under cold-start eval,
skip Phase C. Write up: "Token-keyed memory does not benefit language
modeling under reset-per-segment evaluation, regardless of key
stability." Pivot to depth recurrence / TTT.

### Phase C: Lightweight Ontological Typer (contingent on Phase B)

**Question:** Does adding a lightweight contextual role system on top of
stable token identities improve memory access and semantic reuse?

**Design:** Add a tiny post-tokenization type head. This is Wernicke
shrunk from a 16-expert MoE to a small linear projection.

**Placement:** The typer runs **post-SSM** on contextual hidden states,
not pre-SSM on raw token embeddings. Rationale: type assignment needs
context ("the" as determiner vs part of "theorem" depends on
surrounding tokens), which is only available after the SSM processes
the sequence. Pre-SSM typing would be positional at best.

The typer's output modulates memory keys:
`memory_key = token_id * K + type_id`. This means the memory buffer
effectively has `vocab * K` virtual slots, but most are empty
(only contextually observed combinations get filled).

```python
class ContextualTyper(nn.Module):
    """Post-SSM type head. Operates on SSM hidden states.

    Input:  ssm_output (batch, seq, dim) — contextual representations
    Output: type_weights (batch, seq, K) — straight-through one-hot
            type_ids (batch, seq) — hard type assignments
    """
    def __init__(self, dim: int, num_types: int):
        super().__init__()
        self.num_types = num_types
        self.proj = nn.Linear(dim, num_types, bias=False)

    def forward(self, ssm_output: torch.Tensor):
        logits = self.proj(ssm_output)  # (batch, seq, K)
        type_ids = logits.argmax(dim=-1)
        one_hot = F.one_hot(type_ids, self.num_types).to(ssm_output.dtype)
        soft = F.softmax(logits, dim=-1)
        return one_hot + soft - soft.detach(), type_ids
```

**Conditions (3 configs x 7 seeds = 21 runs):**

| Condition            | Num types | Type modulation           | Notes            |
|----------------------|-----------|---------------------------|------------------|
| `sp_type_4`          | 4         | Memory key = token*K+type | Minimal typing   |
| `sp_type_8`          | 8         | Memory key = token*K+type | Moderate typing  |
| `sp_no_type`         | --        | Phase B winner, control   | No typing        |

**Parameter cost:** Linear(D, K) = D * K. At D=128, K=8: 1,024 params.
Trivial.

**Ontological interpretation:** The token ID answers "what lexical unit
is this?" The type ID answers "what role is this unit playing in this
context?" This is the minimal ontology Experiment 15 is trying to learn.

**Ontology diagnostics (tracked alongside bpb):**

1. **Type collapse rate:** fraction of tokens assigned to the dominant
   type. If this is near 1.0, the typer learned nothing.
2. **Per-token role entropy:** for common tokens (`the`, `of`, `.`,
   frequent subwords), measure whether the model uses a small set of
   repeatable roles instead of random switching.
3. **Contextual consistency:** nearby hidden states with similar local
   context should often get the same type.
4. **Cross-context separation:** the same token in clearly different
   contexts should split across different types more often than chance.
5. **Retrieval selectivity:** compare token-only vs token+type memory
   reads. If typing helps, token+type should produce more concentrated
   and more useful retrieval weights.

These are not primary ranking metrics, but they determine whether a bpb
gain corresponds to actual ontological structure or just accidental
optimization noise.

**Success criterion:** Best typed condition beats Phase B winner by
>= 0.01 bpb with p < 0.05, while also avoiding trivial collapse in the
ontology diagnostics above.

## bpb Calculation

Correct bpb with a tokenizer requires care. The competition standard:

```
bpb = total_CE_nats / total_raw_bytes / ln(2)
```

Where:
- `total_CE_nats` = sum of cross-entropy loss (in nats) over all
  predicted tokens, using `reduction="sum"`.
- `total_raw_bytes` = count of raw UTF-8 bytes in the evaluation text.
  This is a property of the text, independent of the tokenizer.

The existing `compute_bpb()` in `evaluation.py` implements the formula
correctly, but there is a measurement contract issue:
`evaluate_chaoscontrol_bpb()` only scores a *sampled subset* of windows
(via `choose_eval_starts`), not the entire validation split. If
`total_raw_bytes` is set to the full split size but CE is summed only
over sampled windows, the reported bpb will be artificially low.

**Measurement contract (MUST be enforced):**

The byte denominator must count exactly the raw bytes corresponding
to the scored windows, not the entire validation split. Two options:

1. **Score the full split.** Set `eval_batches` high enough to cover
   all non-overlapping windows. Denominator = full split byte count.
   Expensive but unambiguous.
2. **Sample windows, match denominator.** Keep `choose_eval_starts`
   sampling, but compute `total_raw_bytes` as
   `len(eval_starts) * seq_len * bytes_per_token` (where
   `bytes_per_token` is looked up from the SentencePiece model for
   each token in the scored windows, not averaged). This requires
   `evaluation.py` to return the exact scored byte count alongside CE.

**Decision:** Option 2 is faster and sufficient for go/no-go
comparisons (all conditions use the same sampled windows). But
`evaluate_chaoscontrol_bpb()` must be modified to return
`total_scored_bytes` so the runner computes bpb correctly. This IS
a required change to `evaluation.py`.

What does need to change: the data pipeline. Currently,
`prepare_fineweb_splits()` returns raw byte tensors. With SP8192, we
need pre-tokenized shards. The competition already provides these as
`fineweb_train_*.bin` and `fineweb_val_*.bin` (uint16 SentencePiece
token IDs). The existing `load_fineweb_tokens()` in `data.py` already
loads these. We just need to wire the config to select between raw bytes
and pre-tokenized data.

### Data Pipeline Change

```python
# In prepare_fineweb_splits or a new prepare_fineweb_sp8192:
if config.tokenizer_type == "sp8192":
    train_tokens, val_tokens = load_fineweb_tokens(data_dir)
    # These are already uint16 SP8192 token IDs
    # Split val into val + test
    test_boundary = int(val_tokens.numel() * 0.95)
    ...
    # raw_byte_count for bpb must be computed separately
    # Option 1: store as metadata alongside shards
    # Option 2: count from docs_val_raw.txt
    # Option 3: use the sentencepiece base_bytes_lut to sum per-token byte counts
```

The critical invariant: `total_raw_bytes` must match the competition's
definition. The competition's `build_sentencepiece_luts()` computes
per-token byte counts from the SentencePiece model file. We must use
the same method.

## What Carries Forward from Exp 14

### Code to Keep

| Module               | File                     | Status       |
|----------------------|--------------------------|--------------|
| ChaosSSMCore         | `core.py`                | Unchanged    |
| ChaosSSMBlock        | `model.py`               | Unchanged    |
| ChaosControlConfig   | `config.py`              | Extend       |
| Runner               | `runner.py`              | Extend       |
| Training loop        | `training.py`            | Minor changes|
| Evaluation           | `evaluation.py`          | Extend (bpb denominator contract) |
| Artifact pipeline    | `artifact.py`            | Extend       |
| Data loading         | `data.py`                | Extend       |
| Experiment runner    | `run_exp14.py` (template)| Fork to `run_exp15.py` |

### Code to Change

**`config.py`** -- Add new config fields:

```python
# SP8192 tokenizer integration
tokenizer_type: str = "none"   # "none", "fixed_stride", "sp8192"
sp_model_path: str = ""        # path to .model file (resolved at runtime)

# Token-keyed memory (Phase B)
token_memory_enabled: bool = False
token_memory_entries: int = 16
token_memory_value_dim: int = 64

# Contextual typer (Phase C)
contextual_typer_enabled: bool = False
contextual_typer_num_types: int = 4
```

**`model.py` (ChaosStudentLM.__init__)** -- The constructor currently
hardcodes `self.embed = nn.Embedding(vocab_size, dim)`. With SP8192,
vocab_size becomes 8192. No structural change needed -- the existing
`vocab_size` parameter already handles this. The constructor also builds
the `lm_head = nn.Linear(dim, vocab_size)`. Same: already parameterized.

New modules to add to the constructor:
- `self.token_memory` (Phase B)
- `self.contextual_typer` (Phase C)

**`data.py`** -- Add `prepare_fineweb_sp8192()` or extend
`prepare_fineweb_splits()` with a `tokenizer_type` parameter to select
between raw byte loading and pre-tokenized shard loading.

**`runner.py` (build_model)** -- No changes needed. The existing
`build_model()` passes `cfg.vocab_size` through. Setting
`vocab_size=8192` in the config is sufficient.

**`artifact.py`** -- May need to handle the SentencePiece model file.
The competition does NOT require shipping the tokenizer in the artifact
(it is provided at evaluation time). So the artifact only contains
model weights. No change needed unless we add the token memory buffers
to the artifact, which we would only do if Phase B succeeds.

### Code to Remove or Bypass

For ChaosPiece experiments, the following are disabled via config:

- Wernicke layer (`wernicke_enabled: false`)
- Bucket-keyed memory (no `outer_model_dim`)
- Bucket prototypes
- Hierarchical Wernicke
- Posterior modules
- Sleep/polyphasic sleep
- Metabolic gate

These remain in the codebase for reference but are not exercised.

## Finalized Implementation Plan

The implementation is intentionally phase-gated. We do not build token
memory or contextual typing until Phase A proves that the tokenized SSM
backbone is worth extending.

### Phase 0: Measurement + Control Contract

**Goal:** Lock the evaluation contract before writing model code.

**Files:**
- `src/chaoscontrol/evaluation.py`
- `src/chaoscontrol/data.py`
- `experiments/09_revised_architecture/stats.py` (reuse only)

**Actions:**
1. Define the exact bpb contract for sampled-window evaluation:
   CE is summed only over scored windows, so the byte denominator must
   count exactly those scored windows.
2. Decide that `evaluate_chaoscontrol_bpb()` returns both:
   - `total_ce_nats`
   - `total_scored_bytes`
3. Preserve `compute_bpb()` as the final conversion function:
   `bpb = total_ce_nats / total_scored_bytes / ln(2)`.
4. Treat the competition GPT run as a **matched control in data/tokenizer
   space**, not necessarily identical in optimizer internals. The Phase A
   question is "how far is tokenized SSM from a tokenized transformer
   control under the same overall benchmark setup?"

**Verification:**
- Add or update evaluation tests so sampled-window eval reports the byte
  denominator for exactly the scored region.
- Add one regression test proving that using full-split bytes with
  sampled-window CE would change the reported bpb.

### Phase 1: SP8192 Data + Config Plumbing

**Goal:** Make the repo able to run on pre-tokenized SentencePiece shards
without disturbing the existing raw-byte or fixed-stride paths.

**Files:**
- `src/chaoscontrol/config.py`
- `src/chaoscontrol/data.py`
- `src/chaoscontrol/runner.py`
- `tests/test_config.py` or `tests/test_config_exp15.py`
- `tests/test_data.py`

**Actions:**
1. Extend config with:
   - `tokenizer_type: "sp8192"`
   - `sp_model_path`
2. Add config validation:
   - reject `tokenizer_type="sp8192"` when `sp_model_path` is empty
   - preserve current behavior for `"none"` and `"fixed_stride"`
3. Add a dedicated SP8192 loader path in `data.py` that:
   - loads `fineweb_train_*.bin` and `fineweb_val_*.bin`
   - derives train/val/test tensors
   - computes per-token byte counts via `build_sentencepiece_luts()`
     logic reused from the competition baseline
4. Make `runner.py` choose between:
   - raw bytes
   - fixed-stride learned tokenizer
   - SP8192 pre-tokenized shards
5. When `tokenizer_type="sp8192"`, set `cfg.vocab_size = 8192` and
   bypass `FixedStrideTokenizer` construction entirely.

**Verification:**
- Run config tests covering all three tokenizer modes.
- Run data tests confirming SP8192 tensors load as token IDs and return
  compatible byte-count metadata.

### Phase 2: Evaluation Path Refactor

**Goal:** Make evaluation mathematically correct for tokenized sampled-window runs.

**Files:**
- `src/chaoscontrol/evaluation.py`
- `src/chaoscontrol/runner.py`
- `tests/test_evaluation.py`

**Actions:**
1. Refactor `evaluate_chaoscontrol_bpb()` so it returns:
   - summed CE in nats
   - `total_scored_bytes`
   - reported bpb computed from those exact values
2. Remove the old runner pattern that passes full-split
   `total_raw_bytes` into sampled eval.
3. Keep raw-byte behavior unchanged:
   when tokens are bytes, `total_scored_bytes` should simply equal the
   number of scored byte targets.
4. For SP8192, compute `total_scored_bytes` by summing the raw-byte LUT
   entries for the exact scored target tokens.
5. Keep warming-curve evaluation compatible with the same denominator rule.

**Verification:**
- Add eval tests for:
  - raw-byte sampled windows
  - fixed-stride tokenizer path
  - SP8192 sampled windows
- Verify that the reported bpb changes correctly when the scored window set changes.

### Phase 3: Phase A Experiment Runner

**Goal:** Ship the minimum runnable experiment that answers the core pivot question.

**Files:**
- `experiments/15_chaospiece/run_exp15.py`
- `experiments/15_chaospiece/` configs and summaries
- `src/chaoscontrol/runner.py` if small helper extraction is needed

**Actions:**
1. Fork the Exp 14 runner into `run_exp15.py`.
2. Encode the Phase A matrix:
   - 5 SP8192 SSM configs
   - 1 raw-byte bare SSM control
   - 1 tokenized GPT control
3. Reuse the stats helpers for:
   - mean
   - SEM
   - bootstrap CI
   - Welch-style significance if needed for gate decisions
4. Write summaries that explicitly answer both Phase A questions:
   - tokenizer gain vs raw-byte SSM
   - remaining gap vs tokenized transformer
5. Encode a strict go/no-go rule:
   - if best SP-SSM is not meaningfully better than byte SSM, stop
   - if best SP-SSM remains far from GPT control, stop and pivot backbone
   - only continue to Phase B if both conditions are met

**Verification:**
- Dry-run the runner locally on a tiny sample/config set.
- Verify summary JSONs contain enough information to automate the phase gate.

### Phase 4: Phase A Result Review Gate

**Goal:** Prevent Phase B implementation unless the backbone is worth extending.

**Decision rule:**
- Proceed to Phase B only if:
  - best SP-SSM beats `bare_ssm_byte256` by the planned margin
  - best SP-SSM is within the planned competitive band of `gpt_baseline`

**Deliverable:**
- One short Phase A report with:
  - ranked table
  - confidence intervals
  - explicit go/no-go decision
  - chosen backbone config for carry-forward

### Phase 5: TokenKeyedMemory Implementation

**Goal:** Add stable-key episodic memory without reintroducing Exp 14’s step-cost failure mode.

**Files:**
- `src/chaoscontrol/token_memory.py`
- `src/chaoscontrol/model.py`
- `src/chaoscontrol/config.py`
- `tests/test_token_memory.py`
- `tests/test_model_exp15.py` or equivalent integration tests

**Actions:**
1. Implement `TokenKeyedMemory` as a GPU-resident tensor module.
2. Keep the read path fully batched from the first implementation.
3. Keep the write path simple but measurable:
   start with grouped writes if needed, but profile immediately.
4. Wire token memory into `ChaosStudentLM.forward()` after SSM layers,
   before `lm_head`.
5. Add config switches for:
   - enable/disable
   - entries per token
   - value dimension
6. Ensure token memory is runtime state only and resets correctly during
   warming-curve evaluation.

**Verification:**
- Add unit tests for read/write/reset semantics.
- Add integration tests showing no state leak across evaluation resets.
- Profile one training run to confirm the write path does not dominate wall time.

### Phase 6: Phase B Runner + Warming-Curve Evaluation

**Goal:** Judge token memory as deployable runtime state, not accidental eval carryover.

**Files:**
- `experiments/15_chaospiece/run_exp15.py`
- `src/chaoscontrol/evaluation.py`
- `tests/test_eval_warming.py` and/or new Exp 15 warming tests

**Actions:**
1. Carry forward only the winning Phase A backbone.
2. Add Phase B memory conditions.
3. Use `evaluate_warming_curve()` as the primary metric.
4. Report:
   - warmup=0
   - warmup=100
   - warmup=500
   - warmup=1000
   - warmup=5000
5. Gate Phase C on cold-start warming improvement, not pooled eval alone.

**Verification:**
- Confirm all stateful modules reset between segments.
- Confirm the no-memory control has the expected flatter warming curve.

### Phase 7: ContextualTyper Implementation

**Goal:** Reintroduce "typing" only after stable segmentation and stable keys are proven.

**Files:**
- `src/chaoscontrol/contextual_typer.py`
- `src/chaoscontrol/model.py`
- `src/chaoscontrol/config.py`
- `tests/test_contextual_typer.py`
- `tests/test_model_exp15.py`

**Actions:**
1. Implement `ContextualTyper` as a post-SSM type head operating on
   contextual hidden states.
2. Use the typer only to modulate token-memory keys:
   `composite_key = token_id * K + type_id`.
3. Keep the type counts small (`K=4`, `K=8`) and do not reintroduce MoE,
   causal conv routing, or expert branches.
4. Carry forward only the winning Phase B memory configuration.

**Verification:**
- Add shape/gradient tests for the type head.
- Add integration tests that composite keys change retrieval partitions as intended.

### Final Deliverables

At the end of the full plan, the repo should contain:

1. A correct SP8192 evaluation path with sampled-window bpb accounting.
2. A runnable Phase A experiment answering whether tokenization rescues
   the SSM enough to justify further work.
3. A Phase B memory path evaluated under reset-per-segment warming curves.
4. A Phase C typer path that is explicitly lightweight and contingent on
   proven value from Phases A and B.
5. A short report after each phase with a hard go/no-go decision.

## Risk Analysis

### Risk 1: SSM Cannot Match Transformers Even With Tokenizer

**Probability:** Medium.

**Evidence for:** Competition leaders use transformers with depth
recurrence and TTT. Our SSM has neither. The diag A-mode SSM may lack
the representational capacity for token-level modeling even with stable
inputs.

**Evidence against:** Mamba and Mamba2 achieve competitive perplexity
with transformers on standard benchmarks. Our SSM is simpler than Mamba
but the gap may close with proper tokenization.

**Mitigation:** Phase A is explicitly designed to measure this. If
bpb > 2.0, the SSM backbone is the bottleneck and we pivot to backbone
changes (depth recurrence, learned A-mode, attention hybrid).

### Risk 2: Parameter Budget Crunch

**Probability:** Low for Phase A, medium for wider configs.

The SP8192 embedding table at dim=256 is 2.1M params (4.2 MB at fp16).
With weight tying (embed = lm_head.T), this drops to 2.1M shared.
Without tying, embed + lm_head = 4.2M (8.4 MB at fp16). At dim=256
without tying, that is 50% of the 16 MB budget before the SSM backbone.

**Mitigation:**
1. Try without weight tying first (more expressive).
2. If artifact exceeds 16 MB, enable weight tying.
3. int8 quantization halves the effective size.
4. At dim=128 (2.4M total params), budget is not a concern.

### Risk 3: bpb Calculation Error

**Probability:** Low, but catastrophic if wrong.

Getting bpb wrong with a tokenizer means our scores are not comparable
to the competition. The formula is simple but the denominator (raw bytes)
must match exactly.

**Mitigation:**
1. `evaluate_chaoscontrol_bpb()` returns `total_scored_bytes` matching
   exactly the scored windows. Runner divides CE by this, not full split.
2. Cross-check per-token byte counts against SentencePiece base_bytes_lut.
3. Sanity check: train gpt_matched and our SSM on identical data, verify
   both report the same bpb formula (same CE / same denominator).

### Risk 4: seq_len Tuning

**Probability:** Medium.

Token-level seq_len=512 covers ~2048 bytes. The SSM state has O(dim)
capacity regardless of sequence length. Longer sequences may exceed
what the SSM can track, leading to degradation at the end of sequences.
Shorter sequences waste compute on padding and provide less context.

**Mitigation:** Phase A sweeps seq_len as a secondary variable if time
permits. If the primary sweep (seq_len=512) shows degradation at late
positions, reduce to 256 or 384.

### Risk 5: FixedStrideTokenizer Confusion

**Probability:** Low but worth noting.

The codebase already has a `FixedStrideTokenizer` that does learned
VQ tokenization (byte_embed -> causal conv -> VQ codebook). This is
a different thing from SP8192. The `tokenizer_type` config must cleanly
distinguish between "none" (raw bytes), "fixed_stride" (learned VQ),
and "sp8192" (pre-trained SentencePiece). The code paths must not
cross-contaminate.

**Mitigation:** Phase A uses `tokenizer_type="sp8192"` which goes
through a completely different data loading path
(`load_fineweb_tokens` vs `load_fineweb_raw_bytes`). The
`FixedStrideTokenizer` is never instantiated. No ambiguity.

## Operational Plan

### Hardware

8x H100 GPUs on RunPod. Same pod configuration as Exp 14.

### Estimated Runtime

Phase A: 49 runs at 600s each. On 8 GPUs with parallelism:
49 / 8 = 6.125 waves = 7 waves * ~13 min = ~1.5 hours.
(Includes the gpt_baseline transformer control.)

Phase B (contingent): 28 runs = 4 waves = 2400s + overhead = ~1 hour.

Phase C (contingent): 21 runs = 3 waves = 1800s + overhead = ~45 min.

Total worst case (all phases): ~3.5 hours.

### Data Requirements

SP8192 pre-tokenized shards must be present on the pod. These are
downloaded by `cached_challenge_fineweb.py` with the
`fineweb10B_byte260` variant (which includes `fineweb_train_*.bin` and
`fineweb_val_*.bin` as uint16 SentencePiece token IDs).

The SentencePiece model file (`fineweb_1024_bpe.model` or equivalent
8192-vocab model) must also be present for raw byte counting.

**Pre-flight check:** Verify shard files exist, verify SentencePiece
model loads, verify raw byte count matches expected value before
launching the sweep.

## Decision Record

| Date       | Decision                                       | Rationale                     |
|------------|-------------------------------------------------|-------------------------------|
| 2026-04-10 | Drop Wernicke, adopt SP8192                     | Exp 14: every buffer hurts    |
| 2026-04-10 | Three-phase go/no-go structure                  | Don't build memory on a bad backbone |
| 2026-04-10 | Keep SSM backbone, don't switch to transformer  | SSM + tokenizer is the untested combo |
| 2026-04-10 | Token-keyed memory over bucket-keyed            | Stable keys = reliable retrieval |
| 2026-04-10 | Contextual typer is Phase C, not Phase A        | Type on stable units, not raw bytes |
