# Exp24 Training-Time Bundle Brain Dump

Working title. Renumber freely if the experiment log drifts.

## Boundary

This experiment is about training-time mechanisms only.

Evaluation is just the fixed measurement harness after training:

- train under a fixed 600s budget on the current Exp23 fast SSM base
- save the checkpoint
- score with the same full fixed validation evaluation
- compare BPB, train wall time, tokens processed, and any mechanism diagnostics

Do not include eval-time TTT, temporal heads, eval-time polyphasic schedules, or
any scoring-time behavior change in this bundle.

## Fixed Control

The control must be apples-to-apples with every mechanism arm:

- same current fastest SSM base
- same hardware target
- same train budget
- same final full-validation scorer
- same artifact/export path when relevant

Smoke results are useful diagnostics, but they are not controls for Exp24.

The Exp23 full-corpus run proved the fast path can process the full SP16384
training corpus in about 530s on 8xH100. That matters because it creates room
for training-time complexity. It does not mean full corpus completion is itself
the goal. Full corpus completion is a metric, not a gate.

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
stability versus single-copy Muon at matched 600s.

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
or a better train-loss-to-val-BPB conversion at the same budget.

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

Kill condition: no auxiliary horizon/weight improves full-val BPB under a fair
600s budget.

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

Kill condition: SGNS arms do not beat random under the fixed 600s/full-val
contract, or freeze only preserves geometry while hurting BPB.

Promote condition: SGNS improves BPB on the fast base, especially if full-cov
or freeze differentiates semantic geometry from row-scale conditioning.

### 6. Training Data Exposure / Replay Policy

Priority: design carefully; do not let it become a separate data-order project.

Full corpus completion is no longer sacred. The objective is final BPB after
600s. Training order is legal to change, but the comparison must be honest:
same budget, same eval, same base.

Candidate policies:

- random windows, current smoke-like behavior
- deterministic sequential epoch
- shuffled epoch
- mixed policy: mostly coverage, some high-loss replay
- short-horizon repeat: replay hard windows within a bounded freshness window

This is where "sleep/replay" can become performant. Token replay is cheap and
may approximate consolidation without old semantic-engine overhead. Hidden-state
replay is more SSM-native, but it has the state/weight staleness problem.

Kill condition: no exposure/replay policy beats the simple control on full-val
BPB at matched budget.

Promote condition: a policy lets us keep most of the fast-path data exposure
while improving final BPB.

## Parked For 24b

### Neurogenesis: Channel Addition During Training

Very appealing, but probably a few hours of engineering at minimum. Start the
SSM narrow, then widen partway through by appending new diagonal channels.
Because the recurrence is diagonal, this is more plausible for an SSM than a
transformer. It is also full of sharp edges: DDP shape changes, optimizer state,
fused kernels, checkpoint export, and dead-channel initialization.

Keep for 24b unless a prototype looks simpler than expected.

Kill condition: wider-from-start beats grow-during-train at matched wall clock.

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

## Suggested Execution Shape

Phase A: write mechanism briefs and sort candidates.

Phase B: implement only the lowest-risk fast-path scaffolding:

- fast/slow weight wrapper
- spectral regularizer
- predictive auxiliary if hidden access is cheap
- SemanticOptimizer runner wiring if overhead looks bounded
- SGNS/freeze hooks if artifacts are ready
- replay/sampling policy hooks

Phase C: run short 1xH100 smokes only to prove runtime and diagnostics.

Phase D: spend 8xH100 time only on arms whose mechanism fired and whose overhead
does not obviously erase the budget.

Phase E: compose only mechanisms that individually helped or were nearly free.

## Current Recommendation

Order the first implementation pass:

1. fast/slow weights
2. spectral recurrence regularization
3. predictive coding auxiliary
4. SemanticOptimizer fast-path wiring
5. SGNS critical-period freeze
6. replay/sampling policy

The order is not a claim of scientific importance. It is a risk-adjusted path:
start with mechanisms that are cheap, training-only, SSM-native, and unlikely to
break the fast path.
