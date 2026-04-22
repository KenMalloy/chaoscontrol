# Exp23 Full-Corpus 8x Run

Run date: 2026-04-21 local / 2026-04-22 UTC.

Hardware: 8x H100 SXM in India (`AP-IN-1`) on the official
`runpod/parameter-golf:latest` template. Pod id was `kn59k28vo3bdji`; it was
stopped after artifacts were harvested.

Environment:

```text
torch 2.9.1+cu128
CUDA runtime 12.8
CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan
CHAOSCONTROL_POST_SCAN_BACKEND=eager
lm_head_backward_mode=fused_streaming_cached
lm_head_tile_size=8192
batch_size=1024 per GPU
seq_len=512
stride=512
model_dim=256
num_layers=4
activation_checkpoint=false
grad_allreduce_mode=bulk
train_sampling_mode=sequential_epoch
```

Dataset artifact: `Natooka/parameter-golf-sp-tokenizers` revision
`e9d696d1592d884dbb97e754efb2a7203aca3080` for SP16384 train/val bins and
tokenizer. Full validation cache was built from
`willdepueoai/parameter-golf:datasets/docs_selected.jsonl` using the same
SP16384 tokenizer.

## Training

Config:
`experiments/23_fast_path/configs/base_seq_epoch_lr0064_full_corpus.yaml`.

```text
steps:                3163
elapsed:              530.4976 s
aggregate throughput: 25,007,809 tok/s
per-GPU throughput:   3,125,976 tok/s
epoch complete:       true
unique starts:        25,903,968
token slots:          13,262,831,616
final train loss:     4.075619
peak VRAM:            53,139.6 MB
```

With `stride=512`, the deterministic pass covers every full non-overlapping
512-token window in the 13,262,831,920-token SP16384 train corpus, leaving only
the final 304-token tail outside the fixed-shape windows.

## Full Validation

The full 50k-document validation cache contained:

```text
docs:       50,000
tokens:     42,266,034
raw bytes: 151,080,645
```

Score result:

```text
docs scored:     50,000 / 50,000
tokens scored:   42,216,034
aggregate BPB:   1.5695891623
elapsed:         85.3559 s
timed out:       false
world size:      8
doc batch size:  256
max fwd tokens:  65,536
```

This is the first deterministic full-corpus Exp23 base run plus full fixed
validation score for the current fastest SSM base path.

## Artifacts

```text
experiments/23_fast_path/results_full_corpus_8x_20260421/
experiments/23_fast_path/checkpoints_full_corpus_8x_20260421/base_seq_epoch_lr0064_full_corpus.pt
experiments/23_fast_path/results_full_eval_8x_20260421/
```
