# Exp24 Training-Time Bundle Brain Dump

Working title. Renumber freely if the experiment log drifts.

## Boundary

This experiment is about training-time mechanisms only.

Evaluation is just the fixed measurement harness after training:

- train under a fixed 600s wall-clock budget on the current Exp23 fast SSM base
- save the checkpoint
- score with the same full fixed validation evaluation
- compare BPB, train wall time, tokens processed, and any mechanism diagnostics

Do not include eval-time TTT, temporal heads, eval-time polyphasic schedules, or
any scoring-time behavior change in this bundle.

## Fixed Control

The control must be apples-to-apples with every mechanism arm:

- same current fastest SSM base
- same hardware target
- same timed 600s wall-clock training budget
- same final full-validation scorer
- same artifact/export path when relevant

Smoke results are useful diagnostics, but they are not controls for Exp24.

Ring 0 control must establish the numerical noise floor before mechanism arms
are interpreted:

- target: 3 control seeds
- minimum: 2 control seeds if budget forces a triage run
- seeds: use the Exp23 seed ladder (`1337`, `2674`, `4011`) unless the runner
  has a stronger local convention
- report: mean BPB, sample standard deviation, min/max, train elapsed, eval
  elapsed, tokens/sec, and artifact size per seed

Single-run mechanism deltas smaller than the Ring 0 BPB standard deviation are
not paper-quality wins. They can still motivate a follow-up, but the report must
label them as within the current noise floor.

The Exp23 full-corpus run proved the fast path can process the full SP16384
training corpus in about 530s on 8xH100. That matters because it creates room
for training-time complexity. It does not mean full corpus completion is itself
the goal. Full corpus completion is a metric, not a gate.

## Budget Axis

The primary budget axis is wall-clock training time.

Every training arm gets the same timed 600s training budget on the same hardware
class. Mechanism overhead counts against that 600s. A slower mechanism can win
only by producing a better final full-validation BPB under the same wall-clock
training budget.

Secondary equal-cost views are allowed for diagnosis, but they are not the main
Exp24 claim:

- equal-step comparisons explain optimizer/update dynamics
- equal-token comparisons explain data-efficiency
- wall-clock comparisons decide whether a mechanism belongs in the submission
  path

The docs and result summaries should use "matched 600s wall-clock" rather than
ambiguous "matched budget" wording.

## Artifact-Size Bookkeeping

Every arm must be tagged with its 16MB submission-artifact impact before it is
run:

- `artifact_neutral`: no extra parameters or persisted buffers at eval/export
- `artifact_changes_weights_only`: final learned weights differ, but no new
  tensors are added
- `artifact_adds_export_param`: the arm adds parameters that would need to fit
  in the submitted artifact
- `artifact_training_only`: extra state exists during training but is dropped
  before export/eval
- `artifact_invalid_until_stripped`: the arm adds training-only state, but the
  export path does not yet prove it is stripped

Expected tags:

| Candidate | Artifact impact |
|---|---|
| Fast/slow weights | `artifact_training_only` if exporting one copy; `artifact_changes_weights_only` for the chosen final copy |
| Spectral regularization | `artifact_changes_weights_only` |
| Predictive coding auxiliary | `artifact_training_only`; invalid unless aux head is excluded from export and optimizer checkpoints |
| SemanticOptimizer | `artifact_changes_weights_only` |
| SGNS init | `artifact_changes_weights_only`; precomputed init provenance must be recorded separately |
| SGNS freeze | `artifact_changes_weights_only` |
| Replay/sampling policy | `artifact_changes_weights_only` |
| Dreamworld / hidden-state replay | `artifact_training_only`; buffer and any retrieval transformer exist only during training |
| Neurogenesis | `artifact_adds_export_param`; must re-check compressed artifact size |
| STDP supplement | `artifact_changes_weights_only` if no learned gate; `artifact_adds_export_param` if gated by learned parameters |

Every result JSON should record `artifact_impact`, final compressed artifact
bytes when available, and whether the export path is submit-valid.

## Operating Assumptions

Assume historical implementations are not performant enough.

Old code and older experiment docs are idea references, not runnable machinery.
Every candidate has to earn a fast-path implementation plan before pod time.

Drop AdamW from this bundle. It is not a serious near-term candidate unless a
historical table needs it as background context. The optimizer question is
Muon versus the custom optimizer family.

## Mechanism Brief Requirement

Before testing any candidate, write a short mechanism brief:

- What is the idea?
- Why might it help this fast SSM specifically?
- What old implementation, if any, failed or was too slow?
- What is the minimal performant version?
- What logs/diagnostics prove the mechanism actually fired?
- What result kills it?
- What result promotes it?

This prevents Exp24 from becoming another loose matrix of knobs whose meaning
gets reconstructed after the run.

## Candidate Queue

### 1. Fast/Slow Weights With Scheduled Consolidation

Priority: first.

This is the most natural near-term substitute for the old sleep machinery.
Keep a fast copy that trains normally and a slow copy that interpolates toward
the fast weights every N steps. This is Lookahead-shaped, but the mechanism
framing is hippocampal/cortical consolidation: the fast model absorbs noisy
wake updates, while the slow model becomes the stabilized substrate.

Why it fits:

- training-time only
- cheap relative to the current fast path
- wraps the existing model/optimizer rather than changing the fused hot loop
- compatible with Muon and SemanticOptimizer
- gives the sleep/consolidation intuition a performant test

Old implementation status:

- no old fast-path implementation exists
- old sleep machinery is the conceptual ancestor, but it is not the code path
  to reuse
- implement as a small training-loop wrapper, not as a revival of
  `sleep.py`

Minimal version:

```python
if step % sync_interval == 0:
    for slow, fast in zip(slow_params, fast_params):
        slow.lerp_(fast, alpha=sync_alpha)
```

Open design choice: which weights are evaluated at the end?

- evaluate fast weights: measures whether slow stabilization improves training
- evaluate slow weights: treats slow as the consolidated final model
- evaluate both: cheap after a run, useful for interpretation

Kill condition: no interpolation interval/ratio improves full-val BPB or
stability versus single-copy Muon at matched 600s wall-clock.

Promote condition: equal or better BPB with modest overhead, especially if the
slow copy beats the fast copy at final eval.

### 2. Spectral Regularization On Diagonal Recurrence

Priority: first wave.

Revive criticality as a cheap differentiable penalty instead of an expensive
architecture or gradient-free side loop. The recurrence is diagonal, so this
costs O(channels). The mechanism is to shape the distribution of
`sigmoid(log_a)` away from dead channels and away from unstable edge cases.

Why it fits:

- SSM-specific
- cheap enough for the fast path
- directly targets recurrence dynamics
- easy to diagnose by logging the A distribution

Old implementation status:

- older criticality work lived outside the final fast path and should not be
  reused as machinery
- this version is a plain differentiable loss term on diagonal recurrence
  parameters
- implementation should be local to the Exp23/Exp24 training step and disabled
  by default

Sketch:

```python
a = torch.sigmoid(log_a.float())
loss = ce_loss
loss = loss + lambda_dead * torch.relu(a_min - a).square().mean()
loss = loss + lambda_sticky * torch.relu(a - a_max).square().mean()
```

The exact target band should be chosen conservatively. This is not a return to
the old criticality machinery; it is a fast-path recurrence-shaping regularizer.

Kill condition: it moves the A distribution but does not improve BPB, or it
improves diagnostics while harming token prediction.

Promote condition: measurable BPB improvement, lower gradient/pathology tails,
or a better train-loss-to-val-BPB conversion at matched 600s wall-clock.

### 3. Predictive Coding Auxiliary

Priority: first wave, after a short implementation risk check.

Add a small auxiliary loss that predicts future hidden state. The state is the
SSM's natural substrate, so this gives a training signal that does not route
only through the LM head. This is a plausible fix for underfit state dynamics
without changing eval behavior.

Why it fits:

- training-time only
- SSM-native state-shaping signal
- can be disabled entirely for artifact/eval
- likely cheaper than replay if implemented inside the existing forward

Old implementation status:

- no current fast-path implementation exists
- older state/memory machinery is only conceptual background
- must be implemented so the auxiliary head and hidden-state buffers are
  training-only and do not affect eval/export

Sketch:

```python
h = model.encode(inputs, return_hidden_sequence=True)
pred = aux_proj(h[:, :-k])
target = h[:, k:].detach()
aux = F.mse_loss(pred, target)
loss = ce_loss + aux_weight * aux
```

Risks:

- pretty hidden trajectories may not improve token prediction
- storing hidden sequences may increase VRAM or fight the fused head path
- auxiliary head parameters must not contaminate final artifact accounting

Kill condition: no auxiliary horizon/weight improves full-val BPB under matched
600s wall-clock.

Promote condition: BPB improves without a large throughput hit, or the same BPB
is reached with less training wall time.

### 4. SemanticOptimizer / Custom Optimizer

Priority: first wave, but do not run the current implementation blindly.

The custom optimizer idea is not dropped. The current `SemanticOptimizer` couples
optimizer momentum time constants to the SSM diagonal A channels. That is the
right conceptual family, but its current implementation is explicitly unfused
and may be too slow for Exp24 as-is.

Mechanism:

- slow recurrence channels get longer optimizer memory
- fast recurrence channels forget optimizer momentum sooner
- matrix geometry remains Muon-like; temporal geometry becomes channel-aware

Fast-path requirement:

- thread an optimizer mode into `runner_fast_path.py`
- bind real model parameter names
- log beta/tau summaries
- benchmark overhead before spending 8xH100 time

Pre-run overhead gate:

- benchmark against Muon on the same 1xH100 fast-path smoke shape
- kill before provisioning an 8xH100 run if steady-state step time overhead is
  greater than 8% without a compensating short-smoke loss/BPB signal
- promote to 8xH100 only if overhead is less than or equal to 8%, or if a
  smaller smoke shows enough quality lift to justify explicitly buying that
  overhead

Possible minimal test shape:

```python
semantic_cfg = resolve_semantic_optimizer_channels(model)
optimizer = SemanticOptimizer(
    model.parameters(),
    lr=base_lr,
    weight_decay=weight_decay,
    momentum_min=0.5,
    **semantic_cfg,
)
optimizer.bind_param_names(list(model.named_parameters()))
```

The channel map needs an audit against the actual Exp23 SSM module names. This
should be done once and tested so a typo cannot silently reduce to Muon.

Kill condition: overhead is too high to be competitive, or beta coupling has no
BPB benefit versus Muon.

Promote condition: equal throughput with better BPB, or slightly slower
throughput with a clear enough BPB gain to justify the trade.

### 5. SGNS Init / Critical Period

Priority: first wave if artifacts are available for the current vocab.

SGNS already showed signal in Exp21, but controls were incomplete and the run
was not against the final fast path. The clean fast-path question is whether
semantic geometry at initialization still helps once the base can see a large
amount of data quickly.

Exp21 relationship:

- Exp24 does not replace the historical Exp21 4-cell thesis result
- Exp24 does subsume the practical submission question: should SGNS-derived
  initialization be used in the fastest SSM training recipe?
- Any Exp24 SGNS win should be reported as "fast-path training recipe lift";
  semantic-vs-distributional causality still depends on the Exp21-style
  controls (`fullcov`, shuffled, zero/norm-only) if we need a paper claim

Arms worth discussing:

- random init control
- SGNS mean/std matched init
- SGNS full-cov matched init
- SGNS critical-period freeze/unfreeze

The freeze schedule is the interesting addition. If SGNS helps because it gives
early semantic anchors, freezing the embedding briefly may protect that geometry
while the recurrence learns to use it. If SGNS helps only as optimizer
conditioning, freezing may hurt.

Sketch:

```python
if step < embed_freeze_steps:
    model.embed.weight.requires_grad_(False)
else:
    model.embed.weight.requires_grad_(True)
```

Implementation detail: toggling `requires_grad` mid-run can interact with
optimizer state. A cleaner fast-path version may leave gradients computed and
zero the embedding grad until the unfreeze step.

Kill condition: SGNS arms do not beat random under the fixed 600s wall-clock
plus full-val contract, or freeze only preserves geometry while hurting BPB.

Promote condition: SGNS improves BPB on the fast base, especially if full-cov
or freeze differentiates semantic geometry from row-scale conditioning.

### 6. Training Data Exposure / Replay Policy

Priority: design carefully; do not let it become a separate data-order project.

Full corpus completion is no longer sacred. The objective is final BPB after
600s. Training order is legal to change, but the comparison must be honest:
matched 600s wall-clock, same eval, same base.

Candidate policies:

- random windows, current smoke-like behavior
- deterministic sequential epoch
- shuffled epoch
- mixed policy: mostly coverage, some high-loss replay
- short-horizon repeat: replay hard windows within a bounded freshness window

This is where "sleep/replay" can become performant. Token replay is cheap and
may approximate consolidation without old semantic-engine overhead. Hidden-state
replay is more SSM-native, but it has the state/weight staleness problem.

Old implementation status:

- old sleep/replay implementations are not assumed performant
- token replay can be implemented as a sampling policy first
- hidden-state replay needs a separate mechanism brief because cached states
  can go stale as weights move

Kill condition: no exposure/replay policy beats the simple control on full-val
BPB at matched 600s wall-clock.

Promote condition: a policy lets us keep most of the fast-path data exposure
while improving final BPB.

### 7. Dreamworld / Hidden-State Replay

Priority: parallel to §6 replay policy; needs its own mechanism brief before
entering the first implementation pass.

The conceptual ancestor is the Exp11 sleep design
(`docs/plans/2026-04-08-sleep-cycle-design.md`), where the SSM dreamed by
generating token sequences from memory-slot centroids and scored them
teacher-forced against cached real continuations. That design depended on
apparatus the fast path does not have: Wernicke buckets, memory slots,
semantic bases, latent traces, a regret table, and a full-tier `dream_step()`
on `ChaosStudentLM`. Exp11 killed that machinery at the 600s budget; Exp20's
notes make the non-port explicit.

The SSM-native strip-down keeps only what the compact diagonal state provides:

- **Seed:** cache `(state_before_seed, seed_plus_M_targets)` pairs at document
  boundaries (or every K waking steps) into a buffer. For a window with
  `prefix_tokens = p`, encode tokens `[0, p - 1)` to get the cached state, then
  cache tokens `[p - 1, p + M)` as the replay sequence. This preserves the
  boundary transition: the replay seed token is consumed once and predicts the
  first cached continuation token. State is O(channels × inner); M is the dream
  rollout length. The SSM's compact state makes this storable at scale — a
  transformer's equivalent would require caching the full KV prefix.
- **Dream replay (v0):** sample a cached pair, run the fast-path SSM forward
  once over the cached real tokens with the cached state as initial hidden
  state. Teacher-forced, same batched forward as waking — no autoregressive
  sampling in v0.
- **Training signal:** cross-entropy against the cached real continuation
  tokens. Same primary loss as waking; no auxiliary objective added. Only the
  input distribution changes.

Autoregressive dream generation — sampling the model's own tokens forward from
the cached state, rather than teacher-forcing against the cached real
continuation — is a v1+ variant. It tests whether self-generated trajectories
add signal beyond replay of real continuations, but it breaks fast-path parity
because generation is inherently sequential. Do not run it as the first
measured version.

Why it fits:

- training-time only, does not touch the submission artifact or eval path
- reuses the existing fast-path forward and loss; v0 teacher-forces against
  the cached real tokens, so per-step throughput matches waking training — no
  sequential inference loop in the hot path
- consumes cached waking data rather than fresh FineWeb tokens — the "SSM
  running without new stream data" framing
- SSM-native in a way transformers cannot cheaply replicate, because
  transformer "state" equals the full KV prefix

Storage ladder for the `(state, continuation)` cache:

1. **Ring buffer, uniform sampling.** No model. Cache pairs, shuffle, replay.
   Baseline; run first.
2. **Vector store (FAISS or equivalent).** Nearest-neighbor retrieval over a
   state-derived embedding. Still no additional model.
3. **Transformer-based associative memory.** Query = current wake-time context;
   keys = cached state descriptors; values = pointers into the
   `(state, continuation)` buffer. The transformer picks which cached states
   to replay based on current regime. Fits the project thesis that "database
   is the transformer answer to a question SSMs don't ask": storage and
   retrieval is that kind of question, and offloading it to transformer
   machinery keeps the SSM on the streaming side.

Option 3 earns its seat only by beating 1-2. If the ring buffer does not lift
BPB, learned retrieval will not rescue it. If option 3 is run, the retrieval
transformer should be pre-trained offline against state traces from a
preliminary control run and frozen during the measured arm. Training a
transformer inside the 600s budget steals SSM compute and confuses the
comparison axis. Offline pre-training is analogous to the
SGNS-is-offline-legal precedent on the submission side; here the transformer
is training-time infrastructure that never ships.

Old implementation status:

- `sleep.py` and `dream_step()` exist but require the heavy model path and
  memory apparatus; do not revive
- no current fast-path implementation exists
- implement as a training-loop wrapper over `runner_fast_path.py`, with the
  buffer as an in-memory deque in the simplest version

Sketch (ring-buffer baseline):

```python
if step % cache_interval == 0:
    _hidden, prefix_states = model.encode(
        tokens[:, : prefix_tokens - 1],
        return_final_states=True,
    )
    replay_window = tokens[:, prefix_tokens - 1 : prefix_tokens + replay_len]
    buffer.append((
        [state.detach().clone() for state in prefix_states],
        replay_window.detach().clone(),
    ))
    if len(buffer) > buffer_cap:
        buffer.popleft()

if step % dream_interval == 0 and len(buffer) >= min_dream_size:
    states, tokens = random.choice(buffer)
    # Teacher-forced fast-path forward with cached state as initial hidden
    # state. Same batched forward as waking; no autoregressive loop in v0.
    hidden = model.encode(tokens[:, :-1], initial_states=states)
    logits = model.lm_head(model.final_norm(hidden))
    loss_dream = F.cross_entropy(
        logits.reshape(-1, V),
        tokens[:, 1:].reshape(-1),
    )
    (dream_weight * loss_dream).backward()
```

Risks:

- cached states drift out of distribution as the SSM weights move; the buffer
  must age out or be refreshed
- dream gradients can dominate fresh gradients if `dream_interval` and
  `dream_weight` are poorly tuned
- storage option 3 can quietly cost more than the replay lift if the retrieval
  transformer is trained inside the 600s budget

Staleness policy choices:

- hard age-out: drop entries older than `S` optimizer steps
- soft decay: weight replay gradient by `exp(-age / tau)`
- refresh: periodically re-run the cached prefix to produce a fresh
  `(state, continuation)` pair (expensive; only if hard age-out and soft decay
  both fail)

V0 should implement hard age-out only. Soft decay and refresh are follow-ups if
hard age-out looks too brittle or too wasteful in the first smoke.

Artifact impact: `artifact_training_only`. The buffer and any retrieval
transformer exist only during training and are not exported. Export path must
assert their exclusion.

Mechanism diagnostics:

- cache fill rate and turnover
- replay-vs-fresh gradient ratio
- staleness distribution at replay time
- BPB of replay-only vs fresh-only vs combined arms
- if storage option 3: retrieval hit distribution over cached contexts

Kill condition: no cache policy plus dream-weight combination improves
full-val BPB over the §6 replay/sampling baseline at matched 600s wall-clock.
Option 3 additionally kills if it does not beat option 1 at matched 600s.

Promote condition: dream arm lifts BPB over both the Exp23 control and the
best §6 sampling policy, or matches them at clearly lower train wall-clock.

## Parked For 24b

### Neurogenesis: Channel Addition During Training

Very appealing, but probably a few hours of engineering at minimum. Start the
SSM narrow, then widen partway through by appending new diagonal channels.
Because the recurrence is diagonal, this is more plausible for an SSM than a
transformer. It is also full of sharp edges: DDP shape changes, optimizer state,
fused kernels, checkpoint export, and dead-channel initialization.

Keep for 24b unless a prototype looks simpler than expected.

Kill condition: wider-from-start beats grow-during-train at matched 600s
wall-clock.

### STDP-Inspired Hebbian Supplement

Good idea, wrong force level for the first bundle. B/C gradient surgery is a
hammer. If resurrected, it should be gated: fire only on high-surprise or
high-confidence windows, and probably with a learned or scheduled coefficient.

Park until simpler recurrence/state-shaping methods fail.

Kill condition: any mixing coefficient degrades versus pure backprop.

## Measurement Contract

Every promoted arm should log:

- final full-validation BPB
- train elapsed seconds
- aggregate tokens/sec
- steps
- token slots processed
- sampling policy and corpus coverage
- peak VRAM
- mechanism-specific diagnostics

Mechanism diagnostics:

- fast/slow: final fast-vs-slow parameter distance and BPB for both copies
- spectral: `sigmoid(log_a)` min/max/mean/quantiles before and after
- predictive coding: auxiliary loss curve and throughput penalty
- SemanticOptimizer: beta/tau summaries and overhead versus Muon
- SGNS: embedding cosine drift and row-norm drift
- replay policy: replay fraction, unique-window count, hard-window selection rule
- Dreamworld: buffer fill/turnover, replay-vs-fresh gradient ratio, and
  staleness distribution at replay time

## Suggested Execution Shape

Phase 0: Ring 0 control and noise floor.

- run the current fixed-control recipe for 2-3 seeds
- report BPB mean/std and wall-clock throughput spread
- use this noise floor for Phase A interpretation

Phase A: training-data exposure policy gate.

- run no-extra-mechanism sampling/exposure policies under matched 600s
  wall-clock
- compare against the Phase 0 control noise floor
- choose the default policy for Phase B
- if Phase A promotes a new policy, run at least 2 seeds of that policy before
  using it as the control for mechanism arms

Candidate Phase A policies:

- current fixed-control policy
- deterministic sequential epoch
- shuffled epoch
- random windows
- mixed coverage plus bounded hard-window replay

Phase B: mechanism briefs and implementation sorting.

For each candidate, write the mechanism brief and decide whether the candidate
is ready for fast-path scaffolding. The brief is not decorative; it must include
the old-implementation status, artifact-impact tag, mechanism diagnostics, and
pre-run kill gate where applicable.

Phase C: implement only the lowest-risk fast-path scaffolding:

- fast/slow weight wrapper
- spectral regularizer
- predictive auxiliary if hidden access is cheap
- SemanticOptimizer runner wiring if overhead looks bounded
- SGNS/freeze hooks if artifacts are ready
- replay/sampling policy hooks

Phase D: run short 1xH100 smokes only to prove runtime and diagnostics.

Phase E: spend 8xH100 time only on arms whose mechanism fired and whose overhead
does not obviously erase the matched 600s wall-clock budget.

Phase F: compose only mechanisms that individually helped or were nearly free.

## Current Recommendation

Order the first implementation pass:

1. fast/slow weights
2. spectral recurrence regularization
3. predictive coding auxiliary
4. SemanticOptimizer fast-path wiring
5. SGNS critical-period freeze
6. replay/sampling policy

The order is not a claim of scientific importance. It is a risk-adjusted path
based on expected lift, implementation risk, and how directly the mechanism
answers the current failure mode.

| Candidate | Expected lift | Main risk | Why this position |
|---|---|---|---|
| Fast/slow weights | medium | low/medium overhead from shadow copy and sync | Most direct performant replacement for old sleep/consolidation ideas |
| Spectral regularization | low/medium | wrong band can improve diagnostics while hurting BPB | Cheapest SSM-native recurrence-shaping test |
| Predictive coding auxiliary | medium/high | hidden storage and aux head can slow or contaminate export | Best state-shaping idea if hidden access is cheap |
| SemanticOptimizer | medium | current implementation may be too slow unfused | Conceptually important, but must pass the overhead gate first |
| SGNS critical-period freeze | medium | artifact/provenance and optimizer-state details | Good prior signal, but Exp21 causality remains separate |
| Replay/sampling policy | medium/high | can become an open-ended data-order project | Phase A handles sampling first; extra replay waits for a mechanism brief |

## Implementation Plan

Implementation checklist:
`docs/superpowers/plans/2026-04-22-exp24-training-time-bundle-implementation.md`
