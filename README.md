# SemanticEngine SSM

*Codename: ChaosControl*

**A research repo for efficient SSM architectures built from biological principles.**

16MB artifact target. FineWeb validation. Bits-per-byte metric. Built for the OpenAI Parameter Golf competition. **Competition deadline: April 30, 2026.**

## Current Status

As of **April 20, 2026**:

- **Throughput is the dominant variable.** torch.compile on the diagonal recurrence gives ~32× speedup (14K+ training steps in 600s vs 447 before Exp 16). Everything else is downstream of this.
- **Submission regime is locked** from Exp 18: V=16384 SP tokenization, seq=512, bs=1024/rank, Muon LR=0.064, bf16, chunked scan (763K tok/s on 1×H100), ws=8 target.
- **Chunked LM backward is implemented** in `train_ssm.py`. Logits peak VRAM drops 8× at V=16384, enabling bs=1024 without OOM.
- **Exp 13 constants are locked:** `crit_target=0.92`, `max_slots=32`.
- **Score-only floor (Exp 20):** 1.5326 BPB on the full 50k-doc FineWeb validation in 163.7s on 4×H100; ~518s TTT slack at 8×H100.
- **Active parallel tracks (Exp 19–22):** training-time optimization matrix, SSM-native TTT, SGNS embedding init, and temporal heads.

## Competition Context

| | |
|---|---|
| **Task** | OpenAI Parameter Golf — compress a language model into 16MB |
| **Metric** | BPB (bits per byte) on 50,000 FineWeb validation documents |
| **Budget** | 600s train + 600s eval, 8×H100 |
| **Artifact** | ≤16MB serialized checkpoint |
| **Deadline** | April 30, 2026 |
| **Known SOTA** | ~1.0208 BPB (open PR, GDN-Hybrid) |
| **Our floor** | 1.5326 BPB score-only (Exp 20, 4×H100, preliminary) |

TTT is legal. Offline FineWeb pretraining is legal (SP tokenizers are in accepted submissions). The score-first TTT protocol (warmup steps + state restore before the 600s timer) is the production path.

## Thesis

**An SSM trunk trained at maximum throughput can serve as a strong competition base. Eval-time adaptation (TTT) and complementary initialization priors (SGNS) may push below 1.1 BPB within the 600s budget.**

The biological framing — criticality coupling, episodic memory, sleep cycles — motivated the early ablations (Exp 09–14) and produced the locked constants. The current competition arc replaces biological mechanisms with the highest-throughput SSM configuration that still fits in 16MB.

## Architecture

### Submission configuration (locked)

```text
SP8192/16384 tokenization
  -> byte embedding
  -> 4-layer × 256d SSM trunk (diag A, torch.compile, chunked scan)
  -> chunked cross-entropy head (8× VRAM reduction at V=16384)
  -> 16MB artifact
```

Key implementation details:
- **Diag A-mode:** diagonal state matrix, no full matrix_exp overhead
- **torch.compile:** JIT fuses the recurrence loop; 32× speedup vs eager sequential
- **Muon optimizer:** `LR=0.064`, `ws=2` base, `ws=8` for submission
- **bf16 throughout:** tensor core path on H100; TE (transformer_engine) IS available on CUDA 13 pod
- **Chunked scan:** CUDA-extension-backed chunked parallel scan, 763K tok/s on 1×H100
- **Chunked LM backward:** recompute logits in backward; peak VRAM for logits drops from ~9GB to ~1GB at V=16384, bs=1024

### Historical architecture (Experiments 09–14)

The earlier experiments tested biologically-inspired additions on top of this trunk:

```text
raw bytes
  -> Wernicke typed routing (flat or hierarchical)
  -> surprise-gated episodic memory (multi-slot, typed buffer)
  -> sleep-time consolidation (N1/N2/N3/REM)
  -> semantic tier
  -> SSM trunk
  -> LM head
```

Findings from this phase: the full stack is throughput-starved at 600s on A40/H100. Bare SSM wins the baseline sweep. Only `crit_target=0.92` and `max_slots=32` survived as locked constants. The biological mechanisms are **not retired** — they're parked pending a longer budget or efficient approximation.

## Key Findings

| Finding | Evidence |
|---|---|
| **Bare SSM wins the 600s baseline sweep** | Exp 09 / `experiments/baselines/` |
| **`crit_target=0.92`, `max_slots=32` are the best constants** | Exp 13 (`experiments/13_constants_validation/`) |
| **Typed KV buffer fails at 600s budget** | Exp 14 Phase A (`experiments/14_vram_buffer/REPORT_phase_a.md`) |
| **SSM state oracle falsified; residual stream is the retrieval signal** | Exp 16 (`experiments/16_entropy_sparse_attention/VERDICT.md`) |
| **torch.compile diag: 32× throughput, bpb 1.63 at 600s** | Exp 16 |
| **Chunked scan: 763K tok/s on 1×H100** | Exp 18 Test 1 |
| **LR=0.064, seq=512, bs=1024/rank, ws=2 optimal at single-node** | Exp 18 Test 5b |
| **seq=512 wins at matched wall-clock; ws=8 target for submission** | Exp 18 Test 8 / Test 4b |
| **bf16 beats fp8 at 10.7M params; bf16 is submission regime** | Exp 18 Test 10 |
| **Score-only floor: 1.5326 BPB (4×H100, 163.7s, ~518s TTT slack)** | Exp 20 (`experiments/20_ssm_native_ttt/`) |
| **TTT pilot (128-doc, steps=1): no cell beats reset floor** | Exp 20 pilot |

## Experiments

| Experiment | What it tests | Status |
|---|---|---|
| **09: Revised architecture** | Layered ablation: tokenizer, gate, memory, Wernicke, CFR | Complete |
| **10: Scaling laws** | Size and architecture scaling configs | Code ready, not run |
| **10b: Quantization robustness** | Delta-bpb curves under int8/int6 | Designed, not run |
| **11: Sleep cycle ablation** | 9 sleep conditions with REM isolation | Complete |
| **12: Polyphasic sleep** | K-of-N partition scheduling | Code ready, not run |
| **13: Constants validation** | crit_target, max_slots, semantic tier | Complete; constants locked |
| **14: VRAM typed buffer** | Typed KV buffer, retrieval ablations | Phase A complete; verdict: fails at 600s |
| **15: ChaosPiece** | SP8192 tokenizer-first SSM vs byte SSM | Code ready; Phase A matrix pending |
| **16: Entropy sparse attention** | SSM state as sparse retrieval oracle | Complete; oracle falsified; 32× speedup confirmed |
| **17: Local attention sidecar** | Local attention window on top of fast SP-SSM | Code scaffolded; not run |
| **18: Throughput advantage** | Chunked scan, LR/batch/seq/precision sweep | Tests 1–10 complete; submission regime locked |
| **18: Throughput levers** | Fused Muon, fused grad clip, compile flags | Code complete; Phase 1A bench results pending |
| **19: Phase 1** | Phase 1A lever bench; Phase 1C full training matrix | Persistent DDP runner ready; bench results pending |
| **19: Prereqs** | Persistent-DDP multi-seed launcher | Infrastructure complete |
| **20: SSM-native TTT** | Test-time training on validation stream | Score floor measured; first-wave TTT matrix pending |
| **21: SGNS tokenizer** | SGNS embedding init + 4-cell SSM×transformer ablation | In progress |
| **22: Temporal heads** | Parallel recurrent states at multiple memory horizons | Code scaffolded; Phase 0/A pending |

*Experiments 01–08 are in `archive/round1/`. They ran on H100 with the full biological stack (300s budget); the final report is in `archive/round1/FINAL_REPORT.md`.*

## Quick Start

```bash
# Install
python -m venv .venv
.venv/bin/pip install torch numpy pyyaml sentencepiece

# Run tests
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q

# On the training pod — activate the persistent venv first
source /workspace/venv/bin/activate

# Build SP16384 shards (competition submission path)
cd baselines/parameter_golf
python build_sp_shards.py --variant sp16384 --train-shards 80

# Run a single throughput bench at the submission regime
PYTHONPATH=src python experiments/19_prereqs/run_persistent_launcher.py \
    --data-path baselines/parameter_golf/datasets/fineweb10B_sp16384 \
    --sp-model-path baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
    --output-dir /tmp/bench_out \
    --world-size 4 --base-lr 0.064 --budget 600

# Score-only floor pass (Exp 20)
python scripts/run_exp20_fast_score.py \
    --cache-dir /workspace/cache/exp20_val_8192 \
    --checkpoint-path /workspace/results/final.pt \
    --output-path experiments/20_ssm_native_ttt/results_floor/score_only.jsonl \
    --chunk-size 256 --doc-batch-size 4096 \
    --max-forward-tokens auto --budget-seconds 600 --device cuda
```

**Pod note:** The persistent Python environment lives at `/workspace/venv` on the RunPod volume. Always activate it before running Python on the pod. Never `pip install` into the system interpreter — it is wiped on every pod restart.

## Project Structure

```text
src/chaoscontrol/
  core.py              SSM recurrence (diag A-mode, RMSNorm)
  core_fused.py        Fused SSM forward
  model.py             ChaosStudentLM — attention + Wernicke + typed buffer wiring
  train_ssm.py         SP-tokenized training loop with chunked LM backward
  training.py          Original training loop (byte-level)
  evaluation.py        Eval + bpb / warming-curve calculation
  artifact.py          16MB artifact serialization
  tokenizer.py         SentencePiece byte-LUT and competition-correct bpb
  data.py              FineWeb loading (raw bytes and SP tokens)
  distributed.py       DDP all-reduce helpers
  precision.py         bf16/fp8 precision management
  config.py            ChaosControlConfig dataclass
  baselines.py         SimpleTransformerLM, Mamba2LM
  baselines_nanogpt_lean.py  NanoGPTLeanLM (Exp 21, in progress)
  routing.py           Wernicke routing variants
  wernicke.py          Flat + hierarchical typed routing
  memory.py            Multi-slot memory, typed buffer, prototypes
  sleep.py             Sleep cycle (N1/N2/N3/REM) — historical
  wake_cache.py        Wake cache
  fatigue.py           Fatigue tracker
  partition.py         Polyphasic partitions + scheduler
  metabolic.py         Metabolic gate experiments
  regret.py            Counterfactual regret table
  local_attn.py        Local attention module (Exp 17)
  vq.py                Vector quantization
  posterior.py         Posterior estimation utilities
  alignment.py         Alignment utilities
  paper_results.py     Paper results extraction
  optim/               Muon and fused Muon optimizers
  kernels/             CUDA kernel extensions (_ssm_scan)
  quantization/        Quantization utilities
  eval_stream/         Streaming eval pipeline
  sgns/                SGNS embedding tools (Exp 21)

experiments/
  09_revised_architecture/   Layered ablation (complete)
  10_scaling_laws/           Scaling sweep scaffolding
  11_sleep_cycle/            Sleep stage ablation (complete)
  12_polyphasic_sleep/       Partitioned sleep scheduling
  13_constants_validation/   Constants sweep (complete; crit=0.92, slots=32 locked)
  14_vram_buffer/            Typed buffer (phase A complete; verdict: fails at 600s)
  15_chaospiece/             SP8192 tokenizer-first SSM (code ready)
  16_entropy_sparse_attention/ SSM state oracle (complete; oracle falsified; 32× speedup)
  17_local_attn_sidecar/     Local attention hybrid (scaffolded)
  18_throughput_advantage/   Throughput sweep (complete; submission regime locked)
  18_throughput_levers/      Fused Muon / compile levers (bench pending)
  19_phase1/                 Phase 1A bench + Phase 1C matrix (pending)
  19_prereqs/                Persistent-DDP multi-seed launcher (complete)
  20_ssm_native_ttt/         SSM-native TTT (floor measured; matrix pending)
  21_sgns_tokenizer/         SGNS init + transformer baseline (in progress)
  22_temporal_heads/         Temporal heads eval-time strategy (scaffolded)
  baselines/                 Bare SSM, Wernicke variants, full stack

archive/round1/            Exp 01–08 results (300s, full biological stack)
baselines/parameter_golf/  Competition harness, SP shard builder, data manifests
docs/plans/                Design documents and implementation plans
scripts/                   RunPod setup, eval scripts, shard builders
tools/                     RunPod lease-aware pod management
tests/                     Unit and integration coverage
```

## References

- McClelland, McNaughton & O'Reilly 1995 — Complementary learning systems
- Doya 1999 — Cerebellum/cortex/basal ganglia computational trichotomy
- Herculano-Houzel et al. 2014 — Elephant brain neuron distribution
- Favila et al. 2016 — Hippocampal pattern differentiation
- Molitor et al. 2021 — Simultaneous DG separation + CA1 integration
- Schuck et al. 2016 — OFC task-state representation
- Leutgeb et al. 2007 — Pattern separation in dentate gyrus and CA3
