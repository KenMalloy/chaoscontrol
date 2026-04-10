# Experiment 15 Phase A Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Self-contained Phase A experiment answering: does SP8192 tokenization rescue the SSM, and how far is a tokenized SSM from a tokenized transformer?

**Architecture:** Two files in `experiments/15_chaospiece/`. `runner_exp15.py` is the single-run engine (data loading, training, eval with correct bpb). `run_exp15.py` is the multi-GPU matrix launcher (5 SSM configs + byte control + GPT control, 7 seeds). All code reuses shared chaoscontrol modules via import, no modifications to files outside this directory.

**Tech Stack:** PyTorch, sentencepiece (for byte LUT), existing chaoscontrol.{model,training,data,evaluation,baselines} modules.

---

### Task 1: runner_exp15.py — Single-Run Engine

**Files:**
- Create: `experiments/15_chaospiece/runner_exp15.py`

**What it does:**
- CLI: accepts --config (YAML), --data-path, --budget, --output-json, --sp-model-path
- Two data paths: SP8192 (load_fineweb_tokens + val/test split) and raw bytes (prepare_fineweb_splits)
- Builds ChaosStudentLM or SimpleTransformerLM based on config `model_type`
- Trains via train_chaoscontrol_for_budget() with all bolt-ons disabled
- Evaluates with competition-correct bpb using per-token byte LUT from SentencePiece
- Writes result JSON

**Critical correctness requirement:**
- byte LUT: replicate `build_sentencepiece_luts()` from `baselines/parameter_golf/train_gpt.py:180-204` exactly
- bpb formula: `token_bytes = base_bytes[tgt] + (has_leading_space[tgt] & ~is_boundary[prev])`, sum all, then `bpb = total_ce_nats / total_bytes / ln(2)`
- For raw-byte mode: `bpb = total_ce_nats / total_scored_tokens / ln(2)` (each byte = 1 byte)

### Task 2: run_exp15.py — Phase A Matrix Launcher

**Files:**
- Create: `experiments/15_chaospiece/run_exp15.py`

**What it does:**
- Phase A Stage 1: 5 SP-SSM configs + 1 byte control (42 runs = 6 conditions x 7 seeds)
- Phase A Stage 2: 1 GPT-matched transformer (7 runs), param-matched to Stage 1 SSM winner
- Launches runner_exp15.py subprocesses across GPUs (same pattern as run_exp14.py)
- Summarizes results: ranked table, bootstrap CIs, go/no-go decision
- Reuses stats.py from experiments/09_revised_architecture/

**Conditions:**
| Condition | Type | dim | layers | ff_mult | vocab | seq_len |
|-----------|------|-----|--------|---------|-------|---------|
| sp_d128_L4 | SSM | 128 | 4 | 2 | 8192 | 512 |
| sp_d192_L4 | SSM | 192 | 4 | 2 | 8192 | 512 |
| sp_d128_L6 | SSM | 128 | 6 | 2 | 8192 | 512 |
| sp_d192_L6 | SSM | 192 | 6 | 2 | 8192 | 512 |
| sp_d256_L4 | SSM | 256 | 4 | 2 | 8192 | 512 |
| bare_ssm_byte256 | SSM | 128 | 4 | 2 | 256 | 256 |
| gpt_matched | GPT | * | * | 2 | 8192 | 512 |

**Go/no-go criteria:**
- Tokenizer helps: best SP-SSM < bare_ssm_byte256 by >= 0.1 bpb (p < 0.05)
- SSM competitive: best SP-SSM within 0.15 bpb of gpt_matched

### Task 3: Minimal Tests

**Files:**
- Create: `experiments/15_chaospiece/test_exp15.py`

**Tests:**
1. Byte LUT: known SP token -> expected byte count
2. bpb formula: synthetic CE + known bytes -> expected bpb
3. Dry run: tiny config, 2 training steps, verify JSON output has expected keys

### Pod Prerequisites (not code — documented in runner_exp15.py docstring)

```bash
# Download SP8192 data + tokenizer model
cd baselines/parameter_golf
python cached_challenge_fineweb.py --variant sp8192 --train-shards 80
# This creates: datasets/fineweb10B_sp8192/ and tokenizers/fineweb_8192_bpe.model
```
