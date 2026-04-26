"""Phase S4 -- offline pretrain for the simplex CPU SSM controller.

The controller pivot replaces per-slot regression with a barycentric
softmax policy over a 16-vertex simplex (top-K candidate slots). This
script bootstraps the policy near the heuristic argmax via behavior
cloning so on-pod online learning starts from a sensible prior rather
than uniform-random.

Architecture (mirrors the C++ forward graph in
``docs/plans/2026-04-26-simplex-controller-design.md``):

    vertex_h = gelu(V @ W_vp + b_vp)
    attn_logits[i, j] = (vertex_h[i] . vertex_h[j]) / sqrt(H)
                        + alpha * E[i, j]
    attn = softmax_j(attn_logits)
    mixed_h = attn @ vertex_h + vertex_h          # residual
    logits = mixed_h @ W_lh + b_lh                # [N]
    p = softmax((logits + simplex_features @ W_sb) / T)

Synthetic BC target: heuristic_argmax_idx = argmax(V[:, 0]) (column 0
== utility). Edge matrix uses cosine over V[:, 1:] so the edges carry
information independent of the target label. The controller is trained
to softmax-match the heuristic.

Output: a CSWG v2 binary at ``simplex_policy_v1.cswg`` which the F1
matrix loads via ``EPISODIC_CONTROLLER_V1_WEIGHTS_PATH``.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Dimensions are locked by the design doc (Forward architecture +
# Artifact budget). They are NOT configurable: the C++ forward kernel,
# the CSWG v2 header, and the heuristic retrieval pipeline all assume
# these exact shapes.
N_VERTICES = 16    # simplex size (top-K slots per query)
K_V = 16           # vertex feature dim
K_E = 1            # edge feature dim per pair
K_S = 4            # simplex (global) feature dim
H = 32             # hidden dim
N_BUCKETS = 8      # bucket_embed first axis
BUCKET_EMBED_DIM = 8


@dataclass
class PretrainConfig:
    """Hyperparameters for the simplex BC pretrain.

    Architectural dims are exposed for tests but default to the locked
    values. ``temperature`` is held fixed at 1.0 in V1; future versions
    may train it.
    """

    n_vertices: int = N_VERTICES
    k_v: int = K_V
    k_e: int = K_E
    k_s: int = K_S
    h: int = H
    n_buckets: int = N_BUCKETS
    bucket_embed_dim: int = BUCKET_EMBED_DIM
    temperature: float = 1.0
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    n_batches: int = 1000
    seed: int = 1337


class SimplexPolicy(nn.Module):
    """PyTorch reimplementation of the C++ simplex forward kernel.

    Parameter shapes match the CSWG v2 export layout exactly so the
    state_dict can be dumped without reshaping. ``temperature`` and
    ``bucket_embed`` are buffers (not trained in V1) but exported for
    forward compatibility with the C++ runtime that consumes them.
    """

    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.cfg = cfg
        # Layer 1 -- vertex projection (per-vertex linear)
        # W_vp shape (K_v, H) row-major: V @ W_vp -> (N, H)
        self.W_vp = nn.Parameter(
            torch.empty(cfg.k_v, cfg.h).normal_(0.0, 1.0 / math.sqrt(cfg.k_v))
        )
        self.b_vp = nn.Parameter(torch.zeros(cfg.h))
        # Layer 2 -- single learned alpha mixes content with geometry.
        self.alpha = nn.Parameter(torch.zeros(()))
        # Layer 3 -- logit head and simplex-features bias.
        # W_lh shape (H,): mixed_h @ W_lh -> (N,)
        self.W_lh = nn.Parameter(
            torch.empty(cfg.h).normal_(0.0, 1.0 / math.sqrt(cfg.h))
        )
        self.b_lh = nn.Parameter(torch.zeros(()))
        # W_sb shape (K_s,): simplex_features @ W_sb -> scalar bias
        # (added uniformly to all 16 logits; affects softmax entropy
        # only -- gradient through CE is zero by construction since
        # softmax is shift-invariant. Kept as a Parameter so AdamW
        # tracks it but its update is a no-op.)
        self.W_sb = nn.Parameter(torch.zeros(cfg.k_s))
        # Buffers -- exported but not trained in V1.
        self.register_buffer(
            "temperature",
            torch.tensor(float(cfg.temperature), dtype=torch.float32),
        )
        self.register_buffer(
            "bucket_embed",
            torch.empty(cfg.n_buckets, cfg.bucket_embed_dim).normal_(0.0, 0.1),
        )

    def forward(
        self,
        V: torch.Tensor,                  # (B, N, K_v)
        E: torch.Tensor,                  # (B, N, N)
        simplex_features: torch.Tensor,   # (B, K_s)
    ) -> torch.Tensor:
        """Returns logits over the 16 simplex vertices: shape (B, N)."""
        cfg = self.cfg
        # Layer 1: per-vertex projection + GELU.
        vertex_h = V @ self.W_vp + self.b_vp                       # (B, N, H)
        vertex_h = F.gelu(vertex_h)
        # Layer 2: edge-aware attention.
        # content[i, j] = (vertex_h[i] . vertex_h[j]) / sqrt(H)
        scale = 1.0 / math.sqrt(cfg.h)
        content = torch.matmul(vertex_h, vertex_h.transpose(-1, -2)) * scale
        attn_logits = content + self.alpha * E                     # (B, N, N)
        attn = F.softmax(attn_logits, dim=-1)
        mixed_h = attn @ vertex_h + vertex_h                       # residual
        # Layer 3: logit head + simplex bias.
        logits = mixed_h @ self.W_lh + self.b_lh                   # (B, N)
        simplex_bias = simplex_features @ self.W_sb                # (B,)
        # Note: simplex_bias broadcasts uniformly over the N axis;
        # softmax is shift-invariant so this scalar does not change
        # ranking and contributes zero gradient through CE. The temp
        # divide is applied to logits + bias before the softmax
        # consumed by callers; for CE we apply it directly.
        scaled = (logits + simplex_bias.unsqueeze(-1)) / self.temperature
        return scaled

    @torch.no_grad()
    def predict_probs(
        self,
        V: torch.Tensor,
        E: torch.Tensor,
        simplex_features: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper -- returns p over the simplex."""
        return F.softmax(self.forward(V, E, simplex_features), dim=-1)


def synthetic_batch(
    batch_size: int,
    cfg: PretrainConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one synthetic training batch.

    Returns ``(V, E, simplex_features, target)`` where:
      - ``V[b, i, :]`` is the vertex feature vector for slot i in
        batch element b. Column 0 is "utility" -- the heuristic argmax
        target.
      - ``E[b, i, j] = cosine(V[b, i, 1:], V[b, j, 1:])`` -- pairwise
        cosine over V[:, 1:] so the edge matrix carries information
        independent of the target label. Diagonal is 1.0 by
        construction.
      - ``simplex_features[b, :]`` packs four global stats
        ``(top1_utility, mean_utility, std_utility, max_minus_min)``
        derived from V[:, 0]. Matches the design-doc K_s=4 inventory.
      - ``target[b]`` = argmax(V[b, :, 0]).
    """
    V = torch.randn(
        batch_size, cfg.n_vertices, cfg.k_v, generator=generator
    )
    # Edge matrix uses cosine over V[:, 1:] (skip utility column 0).
    V_for_edges = V[..., 1:]                                          # (B, N, K_v - 1)
    norm = V_for_edges.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    V_norm = V_for_edges / norm
    E = V_norm @ V_norm.transpose(-1, -2)                             # (B, N, N)
    # Simplex features: global summary stats over the utility column.
    utility = V[..., 0]                                               # (B, N)
    top1 = utility.max(dim=-1).values
    mean_u = utility.mean(dim=-1)
    std_u = utility.std(dim=-1)
    spread = top1 - utility.min(dim=-1).values
    simplex_features = torch.stack([top1, mean_u, std_u, spread], dim=-1)
    target = utility.argmax(dim=-1)                                   # (B,)
    return V, E, simplex_features, target


def train(cfg: PretrainConfig | None = None) -> dict:
    """Train ``SimplexPolicy`` via behavior cloning on synthetic data.

    Returns ``{"final_policy_acc", "final_loss", "model"}``. Tests use
    ``final_policy_acc`` to gate convergence; ``main`` writes the
    state_dict out to a CSWG v2 binary.
    """
    if cfg is None:
        cfg = PretrainConfig()
    torch.manual_seed(cfg.seed)
    model = SimplexPolicy(cfg)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Use a dedicated generator for data so the model init RNG stays
    # decoupled from the sample stream. Both seeded off cfg.seed for
    # determinism.
    data_gen = torch.Generator().manual_seed(cfg.seed + 1)
    last_loss = float("nan")
    last_acc = 0.0
    model.train()
    for _step in range(cfg.n_batches):
        V, E, sf, target = synthetic_batch(cfg.batch_size, cfg, data_gen)
        logits = model(V, E, sf)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()

    # Final eval on a fresh held-out batch so the reported accuracy is
    # not the same data the optimizer just saw.
    eval_gen = torch.Generator().manual_seed(cfg.seed + 9999)
    model.eval()
    with torch.no_grad():
        V, E, sf, target = synthetic_batch(512, cfg, eval_gen)
        logits = model(V, E, sf)
        last_acc = (logits.argmax(dim=-1) == target).float().mean().item()

    return {
        "final_policy_acc": last_acc,
        "final_loss": last_loss,
        "model": model,
        "config": cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-batches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "simplex_policy_v1.cswg",
    )
    args = parser.parse_args()

    cfg = PretrainConfig(n_batches=args.n_batches, seed=args.seed)
    result = train(cfg)
    print(
        f"final_policy_acc={result['final_policy_acc']:.4f} "
        f"final_loss={result['final_loss']:.4f}"
    )

    # Local import to avoid a circular dependency at module load time:
    # tests import pretrain_controller without needing the dumper.
    from dump_to_cpp import dump_model_to_cswg_v2  # type: ignore

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = dump_model_to_cswg_v2(result["model"], args.out)
    print(f"wrote {manifest['path']} ({manifest['file_bytes']} bytes)")


if __name__ == "__main__":
    main()
