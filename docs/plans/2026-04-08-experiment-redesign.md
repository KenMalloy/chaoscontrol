# Experiment Redesign: Separating Training from Inference

## Problem with the current design

The layered ablation confounds training-time cost with inference-time value. The metabolic gate eats training steps (fork runs model K times, cutting effective throughput to ~60%), making it look bad -- but it's designed for inference. The sequential winner-cascade means one bad early call (e.g., gate loses L1) propagates into all later layers, preventing us from ever testing gate + memory + Wernicke together.

## Principle

**Training phase:** Maximize learning. No metabolic gate (it costs training steps). Memory writes ARE training-time mechanisms -- they piggyback on the forward pass at zero extra step cost and are integral to how the model learns. Every training second goes to gradient updates + memory consolidation.

**Eval phase:** Test all inference-time mechanisms on the trained checkpoints. Gate, warmup mode, memory state (seeded/cold/ttt), CFR bias -- these are eval-time ablations that don't cost training steps.

**Key distinction:** Memory writes during training = training-time feature (episodic writes, semantic consolidation, latent traces are part of the learning algorithm). Metabolic gate during training = wasted training steps (fork/MCTS run model K times per firing, reducing total gradient updates). CFR = gate-dependent (regret updates only fire when gate fires; no gate -> no CFR data).

---

## Phase 1: Training Matrix

Each config trains for 150s with NO metabolic gate and NO CFR. All training budget goes to gradient updates. Memory writes proceed normally as part of the training algorithm.

### Layer 0: Tokenizer (4 configs)
| Config | Description |
|--------|-------------|
| `bytes` | Raw bytes, vocab 256 |
| `fixed_k512` | Learned VQ tokenizer, stride 4, K=512 |
| `fixed_k1024` | Learned VQ tokenizer, stride 4, K=1024 |
| `bytes_tfm` | Raw bytes, transformer baseline |

### Layer 1: Memory tier (3 configs, using L0 winning tokenizer)
| Config | Description |
|--------|-------------|
| `mem_none` | No episodic memory (no outer_model) |
| `mem_epi` | Episodic memory, multislot, survival scoring |
| `mem_epi_sem` | Episodic + semantic tier |

### Layer 2: Wernicke (3 configs, using L0 winner + best memory)
| Config | Description |
|--------|-------------|
| `wer_none` | No Wernicke |
| `wer_vq` | Wernicke VQ routing |
| `wer_moe` | Wernicke MoE routing |

### Layer 3: Scaling (3 configs, full winning stack)
| Config | Description |
|--------|-------------|
| `dim_128` | 128d, 4 layers |
| `dim_256` | 256d, 6 layers |
| `dim_384` | 384d, 8 layers |

**Total training configs:** 4 + 3 + 3 + 3 = 13
**Seeds:** 3
**Training runs:** 39
**Wall time (3 GPUs):** ~33 minutes

Each run saves a full checkpoint: model state_dict + tokenizer state + memory state (episodic slots, survival scores, latent traces, semantic bases). Phase 2/3 load these directly.

Still sequential layer dependencies -- each group picks a winner via Welch t-test + bootstrap CI and feeds into the next. But crucially: no gate, no CFR, no throughput confound.

---

## Phase 2: Eval-Time Ablation Matrix

Run on each trained checkpoint from Phase 1's winning stack at each scale (dim_128, dim_256, dim_384). Forward passes only -- cheap.

### Gate mode (5 levels)
| Mode | Description |
|------|-------------|
| `none` | Standard forward pass |
| `fork_k4` | metabolic_fork, 4 candidates |
| `mc_k4` | Monte Carlo, 4 samples |
| `mcts_k4` | Micro-MCTS, 4 rollouts |
| `mcts_k8` | Micro-MCTS, 8 rollouts |

### Memory eval state (3 levels)
| State | Description |
|-------|-------------|
| `seeded` | Training memory preserved in checkpoint |
| `cold` | Memory wiped before eval |
| `ttt` | Memory wiped, then reconstituted via forward pass over training data |

### CFR bias (2 levels, only with gate != none)
| Mode | Description |
|------|-------------|
| `cfr_off` | Gate selects without regret bias (uniform prior) |
| `cfr_on` | Short warmup pass populates regret table, then gate uses CFR bias |

### Warmup strategy (3 levels, only with non-cold memory)
| Strategy | Description |
|----------|-------------|
| `none` | No warmup |
| `last` | Write last hidden state on surprise |
| `full_seq_latent` | Full-sequence write + latent reactivation |

**Meaningful combos after pruning redundant:**
- gate=none: 3 memory states x {warmup relevant for seeded/ttt only} ~ 5 combos
- gate!=none: 4 gate modes x 3 memory states x 2 CFR x {warmup} ~ 20 combos
- Total per checkpoint: ~25 meaningful eval configs

**Checkpoints to eval:** 3 scales x 3 seeds = 9
**Total eval runs:** 9 x 25 = 225
**Time per eval:** ~10-15 seconds (forward pass only)
**Total eval time:** ~45 minutes with 3 GPUs

---

## Phase 3: Artifact + Quantization (experiment 10b)

Using the best checkpoint from Phase 2:

### Quantization grid
| Level | Description |
|-------|-------------|
| `bf16` | No quantization (baseline) |
| `int8` | Per-tensor symmetric int8 |
| `int6` | Per-tensor symmetric int6 |

### Model variants for mechanism isolation
| Variant | Description |
|---------|-------------|
| `full_ssm` | Full winning stack |
| `full_ssm_no_reactivation` | latent_persistence off |
| `bare_ssm` | No memory components |
| `our_tfm` | Param-matched transformer |

### Eval protocol
For each (variant x quant level):
1. Serialize artifact (quantize + LZMA compress)
2. Load artifact (dequantize + decompress)
3. Eval (bpb_artifact)
4. TTT: Phase A forwards training data with memory writes, Phase B fresh eval on val
5. Report: bpb_ttt

Report: delta_bpb (artifact vs pretrain), ttt_recovery (artifact vs ttt), reactivation_gain (full vs no_reactivation)

---

## Total budget

| Phase | Runs | Time (3 GPUs) | Cost @ $1.20/hr |
|-------|------|---------------|-----------------|
| Training | 39 | ~33 min | $0.66 |
| Eval ablation | 225 | ~45 min | $0.90 |
| Artifact + quant | ~36 | ~15 min | $0.30 |
| **Total** | **300** | **~1.6 hours** | **~$1.86** |

---

## What this replaces

The old 7-layer sequential design (L0-L6) with 117 training runs is replaced by:
- 13 training configs (no gate overhead, no CFR no-op)
- 25 eval configs per checkpoint (gate + CFR tested where they belong)
- Clean mechanism isolation in Phase 3

The old design tested the gate during training (unfair throughput cost), cascaded winners sequentially (fragile), included CFR as a training ablation (no-op without gate), and never tested the artifact pipeline (incomplete). This design fixes all four.

## What changes in code

1. `run_layered.py` kept as-is (archived). New `run_training.py` for Phase 1.
2. New `run_eval_ablation.py` for Phase 2.
3. New `run_artifact_grid.py` for Phase 3.
4. `runner.py`: `run_experiment()` gains `save_checkpoint=True` → saves `.pt` file with full state.
5. `evaluation.py`: gains `gate_mode` + `cfr_warmup` parameters so gate/CFR can be applied at eval time without being in the training config.
6. Gate and CFR configs move from training to eval exclusively.
