#!/usr/bin/env bash
# Bootstrap a RunPod instance for ChaosControl experiments.
# Run this ON the pod after rsync.
set -euo pipefail

cd /workspace/chaoscontrol

# Set up venv
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch numpy pyyaml

# Download enwik8 if not present
if [ ! -f /workspace/enwik8 ]; then
    echo "Downloading enwik8..."
    curl -L -o /workspace/enwik8.zip http://mattmahoney.net/dc/enwik8.zip
    unzip /workspace/enwik8.zip -d /workspace/
    rm /workspace/enwik8.zip
fi

# Verify
echo "=== Smoke test ==="
PYTHONPATH=src .venv/bin/python -c "
import torch
from chaoscontrol.model import ChaosStudentLM
model = ChaosStudentLM(vocab_size=256, dim=16, num_layers=2, ff_mult=2, a_mode='diag', rich_b_mode='none', outer_model_dim=0)
ids = torch.randint(0, 256, (2, 8))
out = model(ids)
print(f'Smoke test passed: logits shape {out[\"logits\"].shape}')
"

echo ""
echo "=== Ready ==="
echo "Run experiments with:"
echo "  bash run_all.sh /workspace/enwik8 --budget 300"
