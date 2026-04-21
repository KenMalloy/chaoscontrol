# Exp23 Fresh Pod Runbook

This is the fastest known prep path for Exp23 Stage A/B on a new RunPod GPU pod.
Do not rebuild SP16384 from `docs_selected.jsonl` unless the tokenizer itself is
being intentionally regenerated; that path is CPU-bound and can leave expensive
GPUs idle for about an hour.

## Fresh Boot Checklist

1. Start the 8xH100 pod and record the pod id plus SSH host/port.
2. Sync the repo to `/workspace/chaoscontrol`.
3. Create the venv with `--system-site-packages` so it reuses the image's
   CUDA-compatible PyTorch.
4. Download/link SP16384 from `Natooka/parameter-golf-sp-tokenizers`.
5. Verify `133` train shards, `1` val shard, and CUDA visibility.
6. Confirm the Exp23 runner has lazy start sampling. It must not call
   `build_lm_starts()` on the full training corpus during startup.
7. Let `run_stage_a.py` prebuild `.train_cache.bin` / `.val_cache.bin` once
   before DDP fans out.
8. Launch Stage A under `nohup`, then monitor `orchestrator.log`.
9. Harvest results and stop the pod if Stage B is not starting immediately.

## Required Artifacts

Use the prebuilt Hugging Face dataset:

```text
Natooka/parameter-golf-sp-tokenizers
revision e9d696d1592d884dbb97e754efb2a7203aca3080
```

Expected files after prep:

```text
baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model
baselines/parameter_golf/tokenizers/fineweb_16384_bpe.vocab
baselines/parameter_golf/datasets/fineweb10B_sp16384/fineweb_train_000000.bin ... fineweb_train_000132.bin
baselines/parameter_golf/datasets/fineweb10B_sp16384/fineweb_val_000000.bin
```

Expected counts:

```text
train shards: 133
val shards:   1
download:     about 24.8 GiB
```

At the observed RunPod/HF throughput, this download/link path took about
3 to 4 minutes. If it is much slower, set `HF_TOKEN` before retrying.

## Minimal Environment

On the official Parameter Golf template, prefer the image's CUDA-good PyTorch
over a fresh PyPI torch install. The currently verified official template stack
is:

```text
image: runpod/parameter-golf:latest
torch: 2.9.1+cu128
CUDA runtime: 12.8
toolkit: /usr/local/cuda-12.8
```

`nvcc` exists in the official template but is not on `PATH` by default. Set
`CUDA_HOME=/usr/local/cuda-12.8` and prepend `$CUDA_HOME/bin` before building
extensions.

CUDA 13 is still worth a speed comparison, especially on H100, but it is a
separate ABI path. Do not install a CUDA 13 PyPI torch into the official
CUDA 12.8 template, and do not reuse a cu130 extension wheel on the cu128
template.

Two fresh-boot blockers observed on 2026-04-21:

- `runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404` can fail before SSH with
  Docker Hub `toomanyrequests` if the selected machine has exhausted
  unauthenticated pulls. Prefer a node that already has this image cached, or
  configure RunPod registry auth for Docker Hub before creating the pod.
- `runpod/comfyui:cuda13.0` booted from the official CUDA 13 template, but one
  RTX 4090 host had driver `12070`; Torch `2.11.0+cu130` saw one GPU but
  reported CUDA unavailable. Always run the CUDA visibility check before doing
  any setup work.

```bash
cd /workspace/chaoscontrol
rm -rf .venv
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy pyyaml sentencepiece huggingface-hub tqdm
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="/workspace/chaoscontrol/.venv/bin:$CUDA_HOME/bin:$PATH"

# Fast path: install the prebuilt official-template wheel.
.venv/bin/pip install --force-reinstall --no-deps \
  /workspace/artifacts/semanticengine_ssm-0.2.0-1cu128sm89sm90-cp312-cp312-linux_x86_64.whl

# Fallback path if the wheel is not present:
MAX_JOBS=6 TORCH_CUDA_ARCH_LIST="8.9;9.0" \
  .venv/bin/pip install -e . --no-build-isolation

.venv/bin/python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
PY
```

Use `tools/runpod.py deploy <pod_id>` only for repo sync. The command now skips
`tools/pod_bootstrap.sh` by default and excludes local `.git`, `.claude`, and
`.venv` state from rsync. Pass `--bootstrap` only when you intentionally want
the broad bootstrap path, because it installs optional Mamba dependencies and
can replace the working image torch if used carelessly.

The exact failure signature from the bad bootstrap was a venv torch reporting
CUDA unavailable on a CUDA-visible host. The system torch reported
`2.8.0+cu128`, CUDA `12.8`, and all 8 GPUs correctly.

## Fast-Path Sanity Checks

Before launching Stage A, make sure the pod has the current Exp23 files:

```bash
cd /workspace/chaoscontrol
grep -R "sample_sharded_lm_starts" -n experiments/23_fast_path
grep -R "build_lm_starts" -n experiments/23_fast_path/runner_fast_path.py && exit 1 || true
```

The second command should print nothing. A stale runner that calls
`build_lm_starts()` on the full 10B-token training split spends minutes building
tens of millions of Python integers before the first batch, leaving all GPUs
idle.

`run_stage_a.py` also prebuilds the mmap cache files in a single process before
starting `torchrun`. Do not pass `--skip-cache-prep` on a fresh pod. If DDP
ranks are allowed to discover a missing `.train_cache.bin` themselves, they can
race the shard-concatenation path and multiply disk work.

Also check the SSM scan backend:

```bash
cd /workspace/chaoscontrol
.venv/bin/python3 - <<'PY'
import torch
from chaoscontrol.core import verify_diag_recurrence
from chaoscontrol.core_fused import get_post_scan_backend
print("diag", verify_diag_recurrence(torch.device("cuda:0")))
print("post", get_post_scan_backend())
PY
```

Preferred: `CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan` with the `_ssm_scan`
extension installed from an ABI-matched wheel. Use the cu128 wheel on the
official Parameter Golf template and the cu130 wheel only on a CUDA 13/Torch
cu130 image. On runtime-only pod images without `nvcc`, the extension cannot be
built there; use
`CHAOSCONTROL_DIAG_SCAN_BACKEND=chunked` and `CHAOSCONTROL_POST_SCAN_BACKEND=eager`
only as a fallback to avoid expensive inductor startup.

## Official Template Smoke

Verified on 2026-04-21 using `runpod/parameter-golf:latest` on a 1x RTX 4090:

```text
torch 2.9.1+cu128
CUDA runtime 12.8
CUDA available: yes
nvcc: /usr/local/cuda-12.8/bin/nvcc after PATH setup
_ssm_scan tests: 38 passed
tiny fast-path smoke: 664 steps in 2.002s, 169,799 tokens/s, 60.7MB peak VRAM
```

Current local artifacts from that run:

```text
experiments/23_fast_path/results_smoke_pg/smoke_pg_1gpu_tiny.json
experiments/23_fast_path/artifacts/ssm_scan_wheels/semanticengine_ssm-0.2.0-1cu128sm89sm90-cp312-cp312-linux_x86_64.whl
experiments/23_fast_path/artifacts/ssm_scan_wheels/build_ssm_scan_pg_cu128.log
experiments/23_fast_path/artifacts/ssm_scan_wheels/wheel_build_pg_cu128.log
```

Verified on 2026-04-21 using the same official template on a 1x H100 SXM
(`NVIDIA H100 80GB HBM3`):

```text
torch 2.9.1+cu128
CUDA runtime 12.8
GPU capability: sm_90
wheel install + source hydration: works
_ssm_scan tests: 38 passed
```

Short synthetic fast-path probes, all 4-layer/256-dim SSM, SP16384,
seq_len=512, budget=10s unless noted:

```text
batch 1024, chunk  64: 2.12M tok/s, 42.8GB peak VRAM
batch 1024, chunk 128: 2.14M tok/s, 57.2GB peak VRAM
batch 1024, chunk 256: OOM, 16GB CE allocation
batch 1024, chunk 512: OOM, 32GB CE allocation
batch 1536, chunk 128, ckpt: 1.88M tok/s, 45.8GB peak VRAM
batch 2048, chunk  64, ckpt: 1.85M tok/s, 32.1GB peak VRAM
batch 2048, chunk 128, ckpt: OOM, 16GB CE allocation
batch 2048, chunk 256, ckpt: OOM, 32GB CE allocation
```

After adding native fused linear+CE, the same 1xH100 `batch=1024, seq=512`
smoke improved from 2.13M tok/s / 42.8GB peak VRAM to 2.56M tok/s / 30.9GB
peak VRAM. Stage A now defaults `lm_head_backward_mode: fused`; keep the
chunked rows above as the historical pre-fused baseline.

Current local H100 probe artifacts:

```text
experiments/23_fast_path/results_h100_smoke/
```

## `_ssm_scan` Without Reusable Images

If RunPod images/disks are disposable, do not treat `_ssm_scan` as something to
rebuild on every expensive pod. Treat it as a binary artifact. Build wheels per
ABI: cu128 for the official Parameter Golf template, cu130 for the CUDA 13
speed-comparison path.

Build once on a cheap CUDA-devel pod whose ABI matches the target H100 image.
Observed good scratch image:

```text
runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404
torch 2.9.1+cu130
CUDA 13.0
nvcc /usr/local/cuda-13.0/bin/nvcc
```

If H100 scratch capacity is unavailable, a 1x RTX 4090 secure pod is enough for
the wheel build as long as both Ada and H100 cubins are emitted:

```bash
cd /workspace/chaoscontrol
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip wheel setuptools ninja
.venv/bin/pip install numpy pyyaml sentencepiece -e . --no-build-isolation

rm -rf dist wheelhouse build
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="/workspace/chaoscontrol/.venv/bin:$CUDA_HOME/bin:$PATH"
MAX_JOBS=6 TORCH_CUDA_ARCH_LIST="8.9;9.0" \
  python -m pip wheel . \
  --no-build-isolation \
  --no-deps \
  -w wheelhouse \
  -v
```

Verify the wheel before uploading it:

```bash
tmpenv=/workspace/cc-wheel-test
rm -rf "$tmpenv"
python3 -m venv --system-site-packages "$tmpenv"
"$tmpenv/bin/pip" install wheelhouse/*.whl
CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan "$tmpenv/bin/python" - <<'PY'
import torch
from chaoscontrol.core import verify_diag_recurrence
info = verify_diag_recurrence(torch.device("cuda:0"))
print(info)
assert info["backend"] == "ssm_scan", info
PY
```

Confirm the wheel contains both scratch-GPU and H100 cubins:

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
for so in "$tmpenv"/lib/python3.12/site-packages/chaoscontrol/kernels/_ssm_scan/_C*.so \
          "$tmpenv"/lib/python3.12/site-packages/chaoscontrol/kernels/_cublaslt/_C*.so; do
  cuobjdump --list-elf "$so" | grep -E "sm_89|sm_90" | sort -u
done
```

Store the wheel somewhere cheap and durable, for example a Hugging Face dataset
artifact repo or GitHub release. Name it with the ABI because this wheel is not
generic:

```text
semanticengine_ssm-0.2.0-cp312-torch2.9.1-cu130-sm89-sm90-linux_x86_64.whl
```

Current local artifact path from the CUDA 13 scratch build:

```text
experiments/23_fast_path/artifacts/ssm_scan_wheels/semanticengine_ssm-0.2.0-1cu130sm89sm90-cp312-cp312-linux_x86_64.whl
```

On a fresh expensive pod, install the wheel instead of installing `nvcc`:

```bash
cd /workspace/chaoscontrol
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy pyyaml sentencepiece huggingface-hub tqdm

# Download the prebuilt wheel from the artifact repo, then:
.venv/bin/pip install --force-reinstall --no-deps /workspace/artifacts/*.whl

# Exp23 scripts insert /workspace/chaoscontrol/src at sys.path[0], so hydrate
# the source checkout with the installed extension .so files.
.venv/bin/python3 - <<'PY'
from pathlib import Path
import shutil
import sysconfig

site = Path(sysconfig.get_paths()["purelib"])
repo = Path("/workspace/chaoscontrol")
for package in ("_ssm_scan", "_cublaslt"):
    source_dir = site / "chaoscontrol" / "kernels" / package
    so = next(source_dir.glob("_C*.so"))
    dest_dir = repo / "src" / "chaoscontrol" / "kernels" / package
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(so, dest_dir / so.name)
    print(dest_dir / so.name)
PY

CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan .venv/bin/python3 - <<'PY'
import torch
from chaoscontrol.core import verify_diag_recurrence
info = verify_diag_recurrence(torch.device("cuda:0"))
print(info)
assert info["backend"] == "ssm_scan", info
PY
```

Only install `nvcc` directly on the expensive pod as a fallback. That path is
acceptable when boxed in, but it converts setup mistakes into H100 burn. The
artifact path makes fresh boot prep a download plus a smoke test.

## Download SP16384

Run this from the pod after the repo is synced. It downloads into an HF cache and
hard-links files into the layout used by `load_sp_data`, so disk is not doubled.

```bash
cd /workspace/chaoscontrol
mkdir -p baselines/parameter_golf/datasets/fineweb10B_sp16384
mkdir -p baselines/parameter_golf/tokenizers

.venv/bin/python3 - <<'PY'
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

repo_id = "Natooka/parameter-golf-sp-tokenizers"
revision = "e9d696d1592d884dbb97e754efb2a7203aca3080"
cache_dir = Path("/workspace/hf_cache_natooka")
repo = Path("/workspace/chaoscontrol")
data_dir = repo / "baselines/parameter_golf/datasets/fineweb10B_sp16384"
tok_dir = repo / "baselines/parameter_golf/tokenizers"
data_dir.mkdir(parents=True, exist_ok=True)
tok_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

files = sorted(s.rfilename for s in HfApi().repo_info(repo_id, repo_type="dataset").siblings)
wanted = [
    f for f in files
    if f in {"fineweb_16384_bpe.model", "fineweb_16384_bpe.vocab"}
    or (f.startswith("shards/fineweb_train_") and f.endswith(".bin"))
    or (f.startswith("shards/fineweb_val_") and f.endswith(".bin"))
]

def dest_for(remote_name: str) -> Path:
    name = Path(remote_name).name
    return (data_dir / name) if remote_name.startswith("shards/") else (tok_dir / name)

def fetch(remote_name: str) -> tuple[str, int]:
    dest = dest_for(remote_name)
    if dest.exists() and dest.stat().st_size > 0:
        return remote_name, dest.stat().st_size
    cached = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=remote_name,
        revision=revision,
        cache_dir=str(cache_dir),
    )).resolve(strict=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(cached, tmp)
    except OSError:
        import shutil
        shutil.copy2(cached, tmp)
    os.replace(tmp, dest)
    return remote_name, dest.stat().st_size

start = time.monotonic()
done_bytes = 0
with ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(fetch, f) for f in wanted]
    for i, future in enumerate(as_completed(futures), start=1):
        name, size = future.result()
        done_bytes += size
        if i <= 5 or i % 10 == 0 or i == len(futures):
            rate = done_bytes / 1024**2 / max(time.monotonic() - start, 1e-9)
            print(f"{i}/{len(futures)} {name} total={done_bytes/1024**3:.2f}GiB rate={rate:.1f}MiB/s", flush=True)

train = list(data_dir.glob("fineweb_train_*.bin"))
val = list(data_dir.glob("fineweb_val_*.bin"))
assert len(train) == 133, len(train)
assert len(val) == 1, len(val)
assert (tok_dir / "fineweb_16384_bpe.model").is_file()
print("SP16384 ready")
PY
```

## Verify And Launch Stage A

```bash
cd /workspace/chaoscontrol
find baselines/parameter_golf/datasets/fineweb10B_sp16384 -name 'fineweb_train_*.bin' | wc -l
find baselines/parameter_golf/datasets/fineweb10B_sp16384 -name 'fineweb_val_*.bin' | wc -l
ls -lh baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model

.venv/bin/python experiments/23_fast_path/run_stage_a.py \
  --data-path /workspace/chaoscontrol/baselines/parameter_golf/datasets/fineweb10B_sp16384 \
  --sp-model-path-16384 /workspace/chaoscontrol/baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
  --results-dir /workspace/chaoscontrol/experiments/23_fast_path/results_stage_a \
  --world-size 8 \
  --budget 90
```

For overnight or disconnected runs, launch with `nohup` and save the orchestrator
PID:

```bash
cd /workspace/chaoscontrol
mkdir -p experiments/23_fast_path/results_stage_a
nohup .venv/bin/python experiments/23_fast_path/run_stage_a.py \
  --data-path /workspace/chaoscontrol/baselines/parameter_golf/datasets/fineweb10B_sp16384 \
  --sp-model-path-16384 /workspace/chaoscontrol/baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
  --results-dir /workspace/chaoscontrol/experiments/23_fast_path/results_stage_a \
  --world-size 8 \
  --budget 90 \
  > experiments/23_fast_path/results_stage_a/orchestrator.log 2>&1 &
echo $! > experiments/23_fast_path/results_stage_a/orchestrator.pid
```

On a pod without `_ssm_scan`, use the fallback env prefix:

```bash
CHAOSCONTROL_DIAG_SCAN_BACKEND=chunked CHAOSCONTROL_POST_SCAN_BACKEND=eager \
nohup .venv/bin/python experiments/23_fast_path/run_stage_a.py \
  --data-path /workspace/chaoscontrol/baselines/parameter_golf/datasets/fineweb10B_sp16384 \
  --sp-model-path-16384 /workspace/chaoscontrol/baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
  --results-dir /workspace/chaoscontrol/experiments/23_fast_path/results_stage_a \
  --world-size 8 \
  --budget 90 \
  > experiments/23_fast_path/results_stage_a/orchestrator.log 2>&1 &
echo $! > experiments/23_fast_path/results_stage_a/orchestrator.pid
```

Monitor it with:

```bash
cd /workspace/chaoscontrol
ps -p "$(cat experiments/23_fast_path/results_stage_a/orchestrator.pid)" -o pid,etime,pcpu,pmem,cmd || true
tail -80 experiments/23_fast_path/results_stage_a/orchestrator.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
cat experiments/23_fast_path/results_stage_a/summary.json 2>/dev/null || true
```

When the run finishes, harvest
`experiments/23_fast_path/results_stage_a/summary.json` and the per-cell JSON
files before stopping the pod.

## CUDA Graph Probe Gate

CUDA graph training is a probe, not a default. Count graph warmup and capture
inside the same wall-clock budget as training. Before a graph-mode result can
replace eager/fused as the Stage A winner, record:

```text
capture_seconds
warmup_seconds
warmup_steps
eager_step_seconds
graph_step_seconds
break_even_seconds
projected_total_speedup
```

Use `fast_path.summarize_cuda_graph_gate(...)` with the submission budget. The
default Stage A policy requires at least 5% projected total-budget throughput
gain after overhead, and rejects capture taking more than 30 seconds. If graph
capture is slow or brittle, keep `cuda_graph_mode: none` and move on.

## Stage B Reminder

Stage B can run immediately from the best Stage A speed config, but the current
HF artifact set above only covers SP16384. If Stage B compares SP8192 against
SP16384, prepare the SP8192 tokenizer and bin shards first or start with the
SP16384-only base-lock cells.

## Notes

- The downloader warning about unauthenticated HF requests is harmless, but an
  `HF_TOKEN` can improve rate limits.
- Keep `/workspace` as the cache/data root. The container overlay is small and
  should not receive dataset or temp files.
- If stale local-build files exist, remove them before download:

```bash
rm -rf /workspace/tmp/*
rm -f baselines/parameter_golf/docs_selected.jsonl
rm -rf baselines/parameter_golf/datasets/fineweb10B_sp16384
```
