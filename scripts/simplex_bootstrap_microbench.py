"""Local microbench: does REINFORCE on near-uniform simplex policy actually
move weights, and does sharpening initial temperature unblock learning?

Reproduces the production setup as closely as possible WITHOUT a pod:
- 16-vertex simplex, H=32, K_v=16, K_e=1, K_s=4 (production geometry)
- Near-uniform initial weights (head weights scaled down so initial p is
  ~uniform, matching the CSWG-loaded condition observed in 2026-04-27 v2:
  max(p)=0.0667 vs uniform=0.0625)
- Random V/E/simplex_features per event (synthetic, doesn't match the
  trained-from-experience case but stresses the learning loop)
- Synthetic 'reward' = advantage with bucket-baseline subtraction, drawn
  from N(mean_per_vertex, std) with mean_per_vertex sampled once at the
  start so ONE specific vertex has consistently positive reward
- Run ~5000 events, watch entropy + W_lh L2 over time

Three configs sweep `temperature` ∈ {1.0, 0.2, 0.05}. If the 1.0 case
fails to move and lower temperatures DO move, we've validated the
"sharpen the bootstrap" hypothesis.

Usage:
    .venv/bin/python scripts/simplex_bootstrap_microbench.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Path setup matches test files
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from chaoscontrol.kernels import _cpu_ssm_controller as _ext  # noqa: E402

# Production geometry
N = 16
K_V = 16
K_E = 1
K_S = 4
H = 32

# Microbench knobs
N_EVENTS = 5000
SGD_INTERVAL = 256
LEARNING_RATE = 1e-3
ENTROPY_BETA = 0.05
WEIGHT_INIT_SCALE = 0.001  # near-uniform initial p; produces |logit| << 1


def near_uniform_weights(seed: int = 1337) -> dict:
    """Initial weights with head weights small so initial p ≈ uniform.

    Matches production CSWG init shape (max p ≈ 0.0667, uniform = 0.0625
    on 16 vertices). The encoder layers (W_vp, b_vp) are slightly larger
    so vertex_h is non-degenerate, but the head onto logits (W_lh, b_lh,
    W_sb, alpha) is tiny.
    """
    g = torch.Generator().manual_seed(seed)
    return {
        "W_vp": (torch.randn(K_V, H, generator=g) * 0.1),
        "b_vp": (torch.randn(H, generator=g) * 0.05),
        # Head weights: SMALL — produces near-uniform p.
        "W_lh": (torch.randn(H, generator=g) * WEIGHT_INIT_SCALE),
        "b_lh": float(torch.randn((), generator=g) * WEIGHT_INIT_SCALE),
        "W_sb": (torch.randn(K_S, generator=g) * WEIGHT_INIT_SCALE),
        "alpha": float(torch.randn((), generator=g) * WEIGHT_INIT_SCALE),
        "temperature": 1.0,
        "bucket_embed": torch.zeros(8, 8),
        "lambda_hxh": 0.0,
    }


def build_weights_struct(weights_dict, n_heads: int = 0):
    w = _ext.SimplexWeights()
    w.K_v = K_V
    w.K_e = K_E
    w.K_s = K_S
    w.H = H
    w.N = N
    w.n_heads = int(n_heads)
    w.W_vp = weights_dict["W_vp"].flatten().tolist()
    w.b_vp = weights_dict["b_vp"].tolist()
    w.W_lh = weights_dict["W_lh"].tolist()
    w.b_lh = float(weights_dict["b_lh"])
    w.W_sb = weights_dict["W_sb"].tolist()
    w.alpha = float(weights_dict["alpha"])
    w.temperature = float(weights_dict["temperature"])
    w.bucket_embed = weights_dict["bucket_embed"].flatten().tolist()
    w.lambda_hxh = float(weights_dict["lambda_hxh"])
    return w


def synth_inputs(seed: int):
    """Random simplex inputs that mimic the production candidate distribution."""
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(N, K_V, generator=g)
    raw = torch.randn(N, K_V, generator=g)
    raw_n = F.normalize(raw, dim=1)
    E = (raw_n @ raw_n.T).clamp(-1.0, 1.0)
    sf = torch.randn(K_S, generator=g) * 0.5
    return V, E, sf


def _simulate_forward(V, E, simplex_features, w_dict):
    """Pure-torch reference forward; gives p[16] for sampling chosen_idx."""
    pre_gelu = V @ w_dict["W_vp"] + w_dict["b_vp"]
    vertex_h = F.gelu(pre_gelu, approximate="none")
    attn_logits = (vertex_h @ vertex_h.T) / math.sqrt(H) + w_dict["alpha"] * E
    attn = F.softmax(attn_logits, dim=1)
    mixed_h = attn @ vertex_h + vertex_h
    logits = mixed_h @ w_dict["W_lh"] + w_dict["b_lh"]
    sb = simplex_features @ w_dict["W_sb"]
    p = F.softmax((logits + sb) / w_dict["temperature"], dim=0)
    return p, logits


def replay_outcome_dict(*, slot_id, gpu_step, selection_step, ce_delta_raw):
    return {
        "event_type": 3,
        "selected_rank": 0,
        "outcome_status": 0,
        "replay_id": gpu_step,
        "gpu_step": gpu_step,
        "query_event_id": gpu_step,
        "source_write_id": gpu_step,
        "slot_id": slot_id,
        "policy_version": 1,
        "selection_step": selection_step,
        "teacher_score": 0.0,
        "controller_logit": 0.0,
        "ce_before_replay": 0.0,
        "ce_after_replay": 0.0,
        "ce_delta_raw": float(ce_delta_raw),
        "bucket_baseline": 0.0,
        "reward_shaped": 0.0,
        "grad_cos_rare": math.nan,
        "grad_cos_total": math.nan,
        "flags": 0,
    }


def run_one_config(*, temperature: float, label: str, seed: int = 1337) -> dict:
    """Run N_EVENTS through the simplex learner with the given starting
    temperature override. Track entropy + W_lh L2 every 256 events.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ONE specific vertex has consistently +0.5 mean reward; others have
    # ±0 mean. Std is large so individual events are noisy — exactly the
    # production "find the signal in the noise" scenario.
    favored_vertex = 7
    vertex_reward_mean = np.zeros(N, dtype=np.float32)
    vertex_reward_mean[favored_vertex] = 0.5
    vertex_reward_std = 1.0

    w_dict = near_uniform_weights(seed=seed)
    w_dict["temperature"] = float(temperature)
    learner = _ext.SimplexOnlineLearner(
        num_slots=8,
        max_entries_per_slot=64,
        gamma=1.0,
        learning_rate=LEARNING_RATE,
        sgd_interval=SGD_INTERVAL,
        ema_alpha=0.25,
        ema_interval=999_999,  # disable slow blend; isolate REINFORCE
        gerber_c=0.0,  # no Gerber gating to isolate the SGD signal
        lambda_hxh_warmup_events=999_999,
        lambda_hxh_clip=0.0,
        entropy_beta=ENTROPY_BETA,
    )
    learner.initialize_simplex_weights(build_weights_struct(w_dict))
    if temperature != 1.0:
        learner.set_temperature(float(temperature))

    history = {
        "step": [],
        "entropy_avg": [],  # behavior entropy (over recent window)
        "w_lh_l2": [],
        "p_favored": [],  # current policy's prob mass on favored vertex
        "sgd_steps": [],
    }

    rng = np.random.default_rng(seed)

    # Take a snapshot of the initial w_dict so we can compute initial p
    # and inspect its uniformity.
    V0, E0, sf0 = synth_inputs(seed=999)
    w_dict["temperature"] = float(temperature)
    p0, _ = _simulate_forward(V0, E0, sf0, w_dict)
    print(f"[{label}] T={temperature}  init max(p)={p0.max().item():.4f}  "
          f"min(p)={p0.min().item():.4f}  entropy={-(p0*p0.log()).sum().item():.4f}")

    recent_entropies = []
    for ev in range(N_EVENTS):
        # Synthetic forward to get p[16]
        V, E, sf = synth_inputs(seed=ev)
        # Get the LIVE policy via the learner's current fast_weights
        fast = learner.fast_weights()
        live_w = {
            "W_vp": torch.tensor(fast.W_vp, dtype=torch.float32).reshape(K_V, H),
            "b_vp": torch.tensor(fast.b_vp, dtype=torch.float32),
            "W_lh": torch.tensor(fast.W_lh, dtype=torch.float32),
            "b_lh": float(fast.b_lh),
            "W_sb": torch.tensor(fast.W_sb, dtype=torch.float32),
            "alpha": float(fast.alpha),
            "temperature": float(fast.temperature),
        }
        p, _ = _simulate_forward(V, E, sf, live_w)
        p_np = p.detach().numpy()
        # Sample chosen_idx ~ p
        chosen_idx = int(rng.choice(N, p=p_np / p_np.sum()))
        p_chosen = float(p_np[chosen_idx])
        recent_entropies.append(float(-(p * p.log()).sum().item()))
        if len(recent_entropies) > SGD_INTERVAL:
            recent_entropies.pop(0)

        # Record decision
        slot_id = ev % 8
        learner.record_simplex_decision(
            chosen_slot_id=slot_id,
            gpu_step=ev,
            policy_version=1,
            chosen_idx=chosen_idx,
            p_chosen_decision=p_chosen,
            V=V.flatten().tolist(),
            E=E.flatten().tolist(),
            simplex_features=sf.tolist(),
            n_actual=N,
            write_bucket=0,
        )
        # Synthetic reward: chosen vertex's mean + noise
        ce_delta_raw = float(rng.normal(
            vertex_reward_mean[chosen_idx], vertex_reward_std))
        learner.on_replay_outcome(replay_outcome_dict(
            slot_id=slot_id,
            gpu_step=ev,
            selection_step=ev,
            ce_delta_raw=ce_delta_raw,
        ))

        # Snapshot every SGD_INTERVAL events
        if (ev + 1) % SGD_INTERVAL == 0:
            fast = learner.fast_weights()
            w_lh_l2 = float(np.linalg.norm(fast.W_lh))
            t = learner.telemetry()
            history["step"].append(ev + 1)
            history["entropy_avg"].append(
                sum(recent_entropies) / len(recent_entropies))
            history["w_lh_l2"].append(w_lh_l2)
            history["p_favored"].append(float(p_np[favored_vertex]))
            history["sgd_steps"].append(int(t.sgd_steps))

    return history


def main():
    print(f"=== simplex bootstrap microbench (N_EVENTS={N_EVENTS}, "
          f"SGD_INTERVAL={SGD_INTERVAL}, LR={LEARNING_RATE}, "
          f"entropy_beta={ENTROPY_BETA}) ===\n")
    print(f"Synthetic reward: vertex 7 has mean +0.5; others ±0; std=1.0")
    print(f"Initial weights: head scale={WEIGHT_INIT_SCALE} (near-uniform "
          f"initial p)\n")

    results = {}
    for T in (1.0, 0.2, 0.05):
        label = f"T={T}"
        results[label] = run_one_config(temperature=T, label=label)
        print()

    # Compare final state across configs
    print("\n=== FINAL STATE BY CONFIG ===")
    print(f'{"config":<12}{"sgd_steps":>11}{"avg_entropy":>14}'
          f'{"w_lh_l2":>10}{"p_favored":>12}')
    for label, h in results.items():
        print(f'{label:<12}'
              f'{h["sgd_steps"][-1]:>11}'
              f'{h["entropy_avg"][-1]:>14.4f}'
              f'{h["w_lh_l2"][-1]:>10.6f}'
              f'{h["p_favored"][-1]:>12.4f}')

    # Drift over training (last - first)
    print("\n=== DRIFT (last snapshot - first snapshot) ===")
    print(f'{"config":<12}{"Δ entropy":>14}{"Δ w_lh_l2":>14}'
          f'{"Δ p_favored":>14}')
    for label, h in results.items():
        d_ent = h["entropy_avg"][-1] - h["entropy_avg"][0]
        d_w = h["w_lh_l2"][-1] - h["w_lh_l2"][0]
        d_p = h["p_favored"][-1] - h["p_favored"][0]
        print(f'{label:<12}{d_ent:>+14.4f}{d_w:>+14.6f}{d_p:>+14.4f}')

    print("\n=== TRAJECTORY (every snapshot) ===")
    for label, h in results.items():
        print(f"\n{label}:")
        print(f'  {"step":>8}{"entropy":>11}{"w_lh_l2":>10}{"p_favored":>12}')
        for i in range(len(h["step"])):
            print(f'  {h["step"][i]:>8}'
                  f'{h["entropy_avg"][i]:>11.4f}'
                  f'{h["w_lh_l2"][i]:>10.6f}'
                  f'{h["p_favored"][i]:>12.4f}')


if __name__ == "__main__":
    main()
