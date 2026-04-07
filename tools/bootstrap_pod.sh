#!/usr/bin/env bash
# Bootstrap a RunPod instance for ChaosControl experiments.
# Run this ON the pod after rsync.
set -euo pipefail

cd /workspace/chaoscontrol

# Set up venv
echo "=== Setting up venv ==="
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch numpy pyyaml sentencepiece huggingface-hub

# Download FineWeb data if not present
FINEWEB_DIR=/workspace/fineweb_data
if [ ! -d "$FINEWEB_DIR/datasets" ]; then
    echo "=== Downloading FineWeb data (sp1024 + raw bytes) ==="
    mkdir -p "$FINEWEB_DIR"
    cd "$FINEWEB_DIR"
    .venv/bin/pip install huggingface-hub
    # Download competition tokenized shards (for BPE baseline reference)
    python3 /workspace/chaoscontrol/baselines/parameter_golf/cached_challenge_fineweb.py \
        --variant sp1024 --train-shards 80
    cd /workspace/chaoscontrol
else
    echo "=== FineWeb data already present ==="
fi

# Verify GPU count
echo ""
echo "=== GPU check ==="
python3 -c "import torch; n = torch.cuda.device_count(); print(f'{n} GPU(s) available'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(n)]"

# Run smoke test
echo ""
echo "=== Smoke test ==="
PYTHONPATH=src .venv/bin/python -c "
import torch
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.tokenizer import FixedStrideTokenizer
from chaoscontrol.vq import vector_quantize
from chaoscontrol.alignment import compute_alignment_loss

# Model smoke test
model = ChaosStudentLM(vocab_size=256, dim=16, num_layers=2, ff_mult=2, a_mode='diag', rich_b_mode='none', outer_model_dim=0)
ids = torch.randint(0, 256, (2, 32))
out = model(ids)
print(f'Model OK: logits {out[\"logits\"].shape}')

# Tokenizer smoke test
tok = FixedStrideTokenizer(byte_dim=16, token_dim=32, codebook_size=64, stride=4)
tok_out = tok(ids)
print(f'Tokenizer OK: tokens {tok_out[\"token_ids\"].shape}, commit_loss={tok_out[\"commit_loss\"]:.4f}')

# VQ smoke test
x = torch.randn(2, 8, 32)
cb = torch.randn(64, 32)
q, idx, cl = vector_quantize(x, cb)
print(f'VQ OK: quantized {q.shape}, indices {idx.shape}')

print('All smoke tests passed.')
"

# Run unit tests
echo ""
echo "=== Unit tests ==="
PYTHONPATH=src .venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -5

echo ""
echo "=== Ready ==="
echo "Run experiment 09:"
echo "  bash experiments/09_revised_architecture/run.sh $FINEWEB_DIR --num-gpus 3"
