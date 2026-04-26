"""Offline pretraining of the CPU SSM controller (Phase D4).

Trains a small SSM (D_global=128, 4 diagonal layers) on heuristic
trace data with two heads:
  - Policy head: imitate heuristic top-K selections via cross-entropy
  - Value head: predict delayed replay reward via MSE

Output: weights checkpoint that warm-starts the C++ runtime so the
online controller (Phase C) does not pay the thousand-step random-init
exploration tax during the 600s training window.

Phase D4 scaffolds the pipeline against synthetic data that matches
the trace schemas from D1 (admission), D2 (eviction), and the existing
replay-outcome NDJSON (see
``src/chaoscontrol/kernels/_cpu_ssm_controller/src/wire_events.h``).
Real trace data lands once D1-D3 traces are harvested from a
heuristic-only training run; the same ``train`` function will operate
on real data via a different ``dataset_fn`` -- only the data source
changes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PretrainConfig:
    """Hyperparameters for the offline BC + value-prediction pretrain.

    The architectural defaults (``d_global=128``, ``n_layers=4``,
    diagonal recurrence) mirror the CPU SSM controller's C++ runtime
    so the trained weights drop into the deployed substrate without
    a shape adapter. ``feature_dim=16`` is the placeholder featurized
    event width; when D1-D3 traces land the loader will project the
    256-dim ``key_rep`` plus scalar context features into this width
    (or the value can be raised at that point).
    """

    d_global: int = 128
    n_layers: int = 4
    feature_dim: int = 16            # featurized event input
    n_slots_per_query: int = 16      # top-K candidate slots per query event
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    seed: int = 1337


class DiagonalSsmTrunk(nn.Module):
    """Stack of ``n_layers`` diagonal SSM layers.

    Mirror of the C++ runtime's reference forward pass: each layer
    applies a diagonal recurrence ``h <- decay * h + h @ W_in + b``
    followed by ``GELU`` and an output projection. Single-step (no
    sequence dimension) -- the controller consumes one event at a
    time. The diagonal ``decay`` is the only recurrent parameter; the
    matmuls are channel mixers rather than time mixers, which is what
    keeps the C++ kernel's hot path cheap.
    """

    def __init__(self, d_global: int, n_layers: int, feature_dim: int):
        super().__init__()
        self.d_global = d_global
        self.n_layers = n_layers
        self.in_proj = nn.Linear(feature_dim, d_global)
        # Diagonal recurrence: h_next = decay * h + W_in @ x; y = W_out @ h
        # Per-layer parameters
        self.decay = nn.Parameter(torch.ones(n_layers, d_global) * 0.95)
        self.w_in = nn.Parameter(torch.randn(n_layers, d_global, d_global) * 0.01)
        self.w_out = nn.Parameter(torch.randn(n_layers, d_global, d_global) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_layers, d_global))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (batch, feature_dim) -> (batch, d_global)`` (single-step)."""
        h = self.in_proj(x)
        for layer in range(self.n_layers):
            h = self.decay[layer] * h + h @ self.w_in[layer] + self.bias[layer]
            h = F.gelu(h)
            h = h @ self.w_out[layer]
        return h


class ControllerPretrainModel(nn.Module):
    """Two-headed controller: shared SSM trunk, policy + value heads.

    Policy head outputs ``cfg.n_slots_per_query`` logits (the heuristic's
    top-K candidate slots for a given query event). Value head outputs
    a scalar predicting the eventual ``reward_shaped`` for the chosen
    slot. Both heads are linear; the trunk does the heavy lifting.
    """

    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = DiagonalSsmTrunk(cfg.d_global, cfg.n_layers, cfg.feature_dim)
        self.policy_head = nn.Linear(cfg.d_global, cfg.n_slots_per_query)
        self.value_head = nn.Linear(cfg.d_global, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.policy_head(h)            # (B, K)
        value = self.value_head(h).squeeze(-1)  # (B,)
        return logits, value


def synthetic_dataset(
    n_examples: int, cfg: PretrainConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ``(features, target_slot, target_reward)`` tuples.

    ``target_slot`` is the argmax of a hidden linear function of the
    features (so the policy task is recoverable by a linear probe over
    a sufficiently expressive trunk). ``target_reward`` is a noisy
    function of the same features (so the value task is regressible
    but not trivially constant). Used by tests + as a sanity-check
    default before real D1/D2/D3 traces land.
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    features = torch.randn(n_examples, cfg.feature_dim, generator=gen)
    # Hidden function: choose slot based on a fixed linear projection
    hidden_proj = torch.randn(cfg.feature_dim, cfg.n_slots_per_query, generator=gen)
    target_slot = (features @ hidden_proj).argmax(dim=-1)
    # Reward = 0.5 * dominant feature magnitude + small noise
    noise = torch.randn(n_examples, generator=gen)
    target_reward = 0.5 * features.abs().max(dim=-1).values + 0.1 * noise
    return features, target_slot, target_reward


DatasetFn = Callable[
    [int, PretrainConfig], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]


def train(cfg: PretrainConfig, dataset_fn: DatasetFn = synthetic_dataset) -> dict:
    """Train the model. Returns ``{final_policy_acc, final_value_r2, weights}``.

    ``dataset_fn`` is the pluggable data source; defaults to the
    synthetic generator so this same function will train on real
    NDJSON trace data once D1/D2/D3 ship -- only swap the loader.
    """
    torch.manual_seed(cfg.seed)
    model = ControllerPretrainModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    features, target_slot, target_reward = dataset_fn(2048, cfg)
    n_examples = features.size(0)
    steps_per_epoch = max(1, n_examples // cfg.batch_size)

    sample_gen = torch.Generator().manual_seed(cfg.seed + 1)
    for _epoch in range(cfg.epochs):
        # One full pass per epoch -- shuffle, then iterate fixed-size
        # batches. Without this, a single-batch-per-epoch loop only
        # executes ``cfg.epochs`` total gradient steps which is too
        # few for the policy head to learn the hidden linear-then-
        # argmax map within the synthetic-test threshold.
        perm = torch.randperm(n_examples, generator=sample_gen)
        for step in range(steps_per_epoch):
            start = step * cfg.batch_size
            idx = perm[start : start + cfg.batch_size]
            x = features[idx]
            y_slot = target_slot[idx]
            y_reward = target_reward[idx]

            logits, value = model(x)
            loss_policy = F.cross_entropy(logits, y_slot)
            loss_value = F.mse_loss(value, y_reward)
            loss = loss_policy + loss_value

            opt.zero_grad()
            loss.backward()
            opt.step()

    # Final eval on the full synthetic set
    model.eval()
    with torch.no_grad():
        logits, value = model(features)
        policy_acc = (logits.argmax(dim=-1) == target_slot).float().mean().item()
        # R^2 = 1 - SS_res / SS_tot; constant predictor scores 0.0
        ss_res = ((value - target_reward) ** 2).sum().item()
        ss_tot = ((target_reward - target_reward.mean()) ** 2).sum().item()
        value_r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    return {
        "final_policy_acc": policy_acc,
        "final_value_r2": value_r2,
        "weights": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/25_controller_pretrain/checkpoint.pt",
    )
    args = parser.parse_args()

    cfg = PretrainConfig(epochs=args.epochs, seed=args.seed)
    result = train(cfg)
    print(f"final_policy_acc={result['final_policy_acc']:.4f}")
    print(f"final_value_r2={result['final_value_r2']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.__dict__,
            "weights": result["weights"],
            "metrics": {
                "final_policy_acc": result["final_policy_acc"],
                "final_value_r2": result["final_value_r2"],
            },
        },
        out_path,
    )


if __name__ == "__main__":
    main()
