# Polyphasic Partitioned Sleep Design Note

## Thesis

Dolphins sleep with one hemisphere at a time. One side consolidates while the other remains awake, keeps sensing the world, and keeps swimming.

But dolphins have two hemispheres because they have two hemispheres. An 8×H100 node has 8 GPUs. The generalization is **polyphasic sleep across N partitions**: at any moment, K partitions are awake and N−K are sleeping. The wake:sleep ratio becomes a schedulable fraction of the partition pool, not a binary toggle.

The ChaosControl analogue is:

- keep the **SSM world model** awake (data-parallel across all GPUs)
- partition the **semantic engine** into N slices (one per GPU, bucket-owned)
- at each step, K partitions process new data while N−K run `N2/N3/REM`
- a scheduler rotates which partitions are awake vs sleeping

This turns sleep from a full offline pause into a **distributed background process**. More hardware doesn't just buy more data parallelism — it buys deeper concurrent consolidation.

The key claim is not "free sleep." The claim is:

> Polyphasic partitioned sleep changes the wake:sleep tradeoff from "lost training steps" to "reduced live semantic capacity plus synchronization overhead." Adding GPUs increases consolidation depth, not just throughput.

That is a much better scaling story for `8xH100` than globally pausing training every sleep cycle. And it scales beyond 8 — the architecture doesn't care how many partitions you have.

## Why This Fits ChaosControl

The natural place to split is not the recurrent trunk. The trunk in [model.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/model.py) and [core.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/core.py) should remain coherent and online.

The natural place to split is the **semantic substrate**:

- typed routing in [wernicke.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/wernicke.py)
- episodic slots and latent traces in [memory.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/memory.py)
- wake-time high-signal cache in [wake_cache.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/wake_cache.py)
- offline consolidation in [sleep.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/sleep.py)
- training-time orchestration in [training.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/training.py)

This is important because CCSSM's scaling edge is not "become a better transformer." It is:

- longer context through an SSM-native trunk
- more typed memory
- more structured consolidation
- more branch/search on hard moments
- richer training-time semantics than the final artifact can carry

Unihemispheric sleep fits that story cleanly.

## Core Architecture

### High-Level Split

Treat the semantic engine as N partitions (one per GPU in the simplest mapping):

- **Partition 0 .. Partition N−1**

Each partition owns:

- a subset of Wernicke bucket families (the natural ownership boundary)
- episodic slots assigned to those buckets
- associated latent traces
- associated semantic bases or semantic subspace
- a local regret/gate policy table

At any training instant, a scheduler assigns each partition to one of two modes:

- **awake** (K partitions)
- **sleeping** (N−K partitions)

The K:N ratio is the polyphasic wake fraction. K=6, N=8 means 75% semantic capacity awake, 25% consolidating. The scheduler rotates assignments so every partition gets regular sleep.

The trunk remains globally active (data-parallel across all GPUs).

### What "Awake" Means

An awake partition:

- accepts new episodic writes
- participates in normal retrieval for wake batches
- updates wake-time diagnostics
- records high-signal moments into the cache
- contributes to forward prediction on real data

### What "Sleeping" Means

A sleeping partition:

- accepts **no new canonical writes**
- performs no wake-time gradient learning
- runs `N2` utility scoring on its own slots
- runs `N3` rewrite/compression on its own slots
- runs `REM` on its own cached scenes
- may export **read-only summaries** to the awake side

The sleeping side is internally active but externally closed, which is exactly the right computational interpretation of dolphin sleep.

## 8×H100 Mapping

Each GPU owns a partition. With 8 GPUs and `wernicke_k_max=16` buckets:

- GPU 0 → buckets 0-1
- GPU 1 → buckets 2-3
- ...
- GPU 7 → buckets 14-15

Each GPU holds the trunk replica (data-parallel) plus its partition's semantic state (slots, traces, bases, regret table for its buckets).

### Partition Scheme: Bucket-Owned

The natural partition boundary is **Wernicke bucket ownership**:

- buckets are assigned to partitions (round-robin or load-balanced)
- each partition owns the episodic slots, latent traces, and semantic bases for its bucket families
- sleep/consolidation happens over semantic territory, not arbitrary memory indices

This is the CCSSM-native design because:

- it aligns with typed storage
- each partition has a coherent semantic domain
- it enables semantically sparse distributed execution
- the scheduler can sleep the least-active partition first

### Fallback: Slot-Halving

If bucket-owned partitions prove too complex to prototype:

- slots are assigned round-robin to partitions regardless of bucket
- simpler but semantically arbitrary
- still tests the core economic claim (wake:sleep tradeoff)

### Wake:Sleep Scheduling

With N=8 partitions, the scheduler chooses K awake at each step:

| K (awake) | N−K (sleeping) | Wake fraction | Character |
|-----------|----------------|---------------|-----------|
| 7 | 1 | 87.5% | Light nap — one partition consolidates at a time |
| 6 | 2 | 75% | Default — good consolidation depth, most capacity online |
| 4 | 4 | 50% | Deep sleep — aggressive consolidation, halved capacity |

The scheduler rotates sleeping assignments round-robin, optionally biased by per-partition fatigue (bucket activity, slot pressure).

## Training Semantics

### Wake Path

For a real training batch:

1. run the full trunk
2. query awake partitions normally
3. optionally query sleeping partitions in **read-only summary mode**
4. update model parameters from the wake loss
5. record high-signal moments into awake partitions' caches

### Sleep Path

In parallel with wake training:

1. gather the sleeping partition's cached moments
2. run `N2` on its slots
3. run `N3` provisional rewrite / compression
4. run `REM` dream validation / CFR / reactivation on that same partition
5. publish compact summary updates at swap boundaries

The sleep side is not idle. It is just not participating as a writable wake-time memory.

## Invariants

These are the rules that keep the design scientifically interpretable.

### Invariant 1: The Trunk Never Sleeps

Do not split or alternate the SSM recurrence itself. The trunk is the online world model.

### Invariant 2: Sleeping Memory Does Not Admit New Experience

No new episodic writes into sleeping partitions.

### Invariant 3: Dreams Are Not Canonical Episodes

REM may update:

- slot survival
- merge decisions
- compression penalties
- regret / gate policy
- latent reactivation state

REM must not write dreamed sequences as normal wake episodes.

### Invariant 4: Cross-Partition Exchange Is Summarized

Do not copy full slot banks between partitions every step. Exchange should be limited to compact summaries:

- semantic basis deltas
- accepted merge metadata
- protection / penalty signals
- optional regret priors

### Invariant 5: Sleep Cost Is Still Counted

Polyphasic sleep is not free. The cost shifts from lost wake steps to:

- reduced active semantic capacity (N−K partitions offline)
- background compute on sleeping partitions
- synchronization overhead at swap boundaries
- stale reads from recently-sleeping partitions

All comparisons should be at fixed hardware-hours.

## Reference Execution Model

```python
partitions = [Partition(gpu=i, buckets=bucket_assignment[i]) for i in range(N)]
scheduler = PolyphasicScheduler(partitions, K=6, swap_interval=256)

for wake_step, batch in enumerate(train_loader):
    awake, sleeping = scheduler.current_assignment()

    # Wake path: real data, writable semantic state (K partitions).
    out = model.forward_with_partitions(batch, awake_partitions=awake)
    loss = compute_loss(out, batch.targets)
    loss.backward()
    optimizer.step()
    for p in awake:
        p.cache.record(batch, out)

    # Sleep path: no new writes, only offline maintenance (N-K partitions).
    for p in sleeping:
        p.run_sleep_cycle(
            cache=p.cache,
            validate_merges=True,
            rem_cfr=True,
            rem_reactivate=True,
            writable=False,
        )

    # Scheduler rotates sleeping assignments, syncs at swap boundaries.
    scheduler.step()
```

This is intentionally schematic. The important point is the mode boundary and the K-of-N scheduling, not the exact API.

## Where This Lives In The Codebase

This is a **training-time orchestration feature**, not an artifact feature.

Natural integration points:

- [training.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/training.py)
  - wake/sleep scheduling
  - partition swap cadence
  - distributed execution orchestration
- [sleep.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/sleep.py)
  - partition-scoped `SleepCycle`
  - per-partition `N2/N3/REM`
- [memory.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/memory.py)
  - slot ownership
  - typed merge discipline
  - latent trace recovery per partition
- [wake_cache.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/wake_cache.py)
  - per-partition high-signal caches
- [wernicke.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/src/chaoscontrol/wernicke.py)
  - bucket-to-partition ownership

The artifact should not know whether training used full offline sleep or unihemispheric sleep. It should simply inherit better-rested memory-trained weights.

## Main Scientific Claim

The central claim is:

> Polyphasic partitioned sleep preserves most of the consolidation benefit of full offline sleep while converting the wake-step tax into a semantic capacity tradeoff. Adding GPUs increases consolidation depth, not just throughput.

That can be tested cleanly.

## Required Ablations

### Primary Wake/Sleep Economics

1. `no_sleep`
2. `full_offline_sleep` (Experiment 11 baseline)
3. `polyphasic_K6_N8` (6 awake, 2 sleeping)

This is the key experiment. If polyphasic matches or beats full offline sleep while recovering wake throughput, the idea is real.

### Wake Fraction (K-of-N)

1. `K7_N8` — light nap (87.5% capacity, 1 consolidating)
2. `K6_N8` — default (75% capacity, 2 consolidating)
3. `K4_N8` — deep sleep (50% capacity, 4 consolidating)

### Partition Strategy

1. arbitrary slot-halving (fallback)
2. bucket-owned partitions (CCSSM-native)

If bucket-owned wins, the result is genuinely distinctive.

### Cross-Partition Read Policy

1. no reads from sleeping partitions
2. read-only summary from sleeping partitions
3. full live reads from sleeping partitions

The likely sweet spot is `read-only summary`. Full live reads risk conflating wake and sleep.

### Swap Interval

1. short (every 64 steps)
2. medium (every 256 steps)
3. long (every 1024 steps)

Too fast will thrash. Too slow will let sleeping partitions' state go stale.

### Sleep Payload

1. `N2/N3` only
2. `N2/N3/REM`
3. `N2/N3/REM` + reactivation

This tells you what actually earns its keep in background consolidation. Results from Experiment 11 will inform which stages to include.

## Metrics

Primary:

- validation `bpb`
- `16MB` artifact `bpb`
- `bpb` at fixed `8xH100-hours`

Secondary:

- wake throughput
- percentage of training wall time spent on active wake updates
- seeded vs cold delta
- slot utility distribution by partition
- merge acceptance / rejection rate
- reactivation success rate
- gate ROI on sleeping vs awake-trained policy

Diagnostic:

- cross-partition drift
- bucket imbalance
- summary sync cost
- retrieval entropy

## What Would Count As Success

The idea is working if all of these are true:

1. `polyphasic_K6_N8` beats `no_sleep`
2. `polyphasic_K6_N8` is at least competitive with `full_offline_sleep`
3. wake throughput is materially higher than `full_offline_sleep`
4. the effect survives artifact compression, not just live training eval
5. scaling from K6 to K7 or K4 shows the expected capacity/consolidation tradeoff

The strongest version would be:

> CCSSM uses distributed hardware to consolidate continuously, converting sleep from a sequential training interruption into a parallel semantic maintenance process. More GPUs buy deeper consolidation, not just more data parallelism.

## Main Risks

### Risk 1: Semantic Drift

A sleeping partition may drift away from the awake partitions and return stale or incompatible summaries. With N partitions this risk is distributed but not eliminated.

### Risk 2: Reduced Live Capacity Hurts More Than Sleep Helps

If halving writable semantic capacity damages wake learning too much, the idea fails on its own terms.

### Risk 3: Synchronization Overhead Eats The Gain

If summary exchange is too expensive, the "free overlap" story collapses.

### Risk 4: The Split Is Too Arbitrary

If partition assignment is not semantically meaningful, the model may just experience random memory fragmentation.

That is why the bucket-owned version is more interesting than raw slot-halving. With N=8 partitions, arbitrary assignment is even riskier than with 2.

## Prototype vs Supercar

The prototype version is:

- N partitions with round-robin slot assignment
- fixed K-of-N wake:sleep ratio
- fixed swap interval
- summary-only synchronization
- testable on 4×A40 (N=4, K=3)

The supercar version is:

- bucket-owned partitions (8 GPUs, 16 buckets, 2 per GPU)
- adaptive K selection based on per-partition fatigue
- partition-local REM policy learning
- read-only semantic exchange during sleep
- distributed semantic engine scaling on top of a stable SSM trunk
- built for 8×H100

The prototype is enough to test the core economics. The supercar is what makes the scaling story special.

## Bottom Line

Polyphasic partitioned sleep is compelling because it keeps the CCSSM thesis intact:

- the SSM trunk remains the online heart
- the semantic engine does the heavy background work
- more hardware buys deeper semantic maintenance, not just more dense parameters

That is the right direction for `8xH100` — and beyond.
