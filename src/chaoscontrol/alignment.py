"""Codebook alignment losses: tokenizer codebook <-> Wernicke codebook.

Four alignment mechanisms that couple the learned tokenizer's VQ codebook
(Level 0: morpheme-like types) with the Wernicke codebook (Level 1:
semantic types).  A Linear(token_dim -> model_dim) projection bridges the
two spaces before alignment is computed.

See docs/plans/2026-04-07-learned-tokenizer-design.md, section
"Codebook Coupling: Tokenizer <-> Wernicke" for the design rationale.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual alignment loss functions
# ---------------------------------------------------------------------------

def no_alignment(**kwargs) -> torch.Tensor:
    """No explicit coupling.  Return zero loss."""
    return torch.tensor(0.0)


def contrastive_alignment(
    projected_tok_embeds: torch.Tensor,  # (batch, token_seq, model_dim)
    wernicke_entries: torch.Tensor,       # (K_wer, model_dim)
    wernicke_assignments: torch.Tensor,   # (batch, token_seq)
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss.

    For each Wernicke bucket *j*, compute the mean of projected tokenizer
    codes that were routed to bucket *j* in this batch.  That mean should
    be closer to Wernicke entry *j* than to all other entries.

    L_align = -mean_j[log(exp(sim(j,j)/tau) / sum_i(exp(sim(i,j)/tau)))]

    Buckets with no assignments in the batch are skipped.
    """
    flat_embeds = projected_tok_embeds.reshape(-1, projected_tok_embeds.size(-1))
    flat_assigns = wernicke_assignments.reshape(-1)

    K = wernicke_entries.size(0)

    # Accumulate per-bucket sums and counts
    bucket_sums = torch.zeros(K, flat_embeds.size(-1), device=flat_embeds.device)
    bucket_counts = torch.zeros(K, device=flat_embeds.device)
    bucket_sums.scatter_add_(0, flat_assigns.unsqueeze(-1).expand_as(flat_embeds), flat_embeds)
    bucket_counts.scatter_add_(0, flat_assigns, torch.ones_like(flat_assigns, dtype=torch.float))

    # Mask out empty buckets
    active = bucket_counts > 0
    if active.sum() == 0:
        return torch.tensor(0.0, device=flat_embeds.device)

    bucket_means = bucket_sums[active] / bucket_counts[active].unsqueeze(-1)  # (A, dim)
    active_entries = wernicke_entries[active]  # (A, dim)

    # Cosine similarity of each bucket mean against ALL wernicke entries: (A, K)
    bucket_means_norm = F.normalize(bucket_means, dim=-1)
    all_entries_norm = F.normalize(wernicke_entries, dim=-1)
    sim_matrix = bucket_means_norm @ all_entries_norm.T  # (A, K)

    # For each active bucket, the positive is its own wernicke entry
    active_indices = torch.where(active)[0]  # (A,)
    pos_sim = sim_matrix[torch.arange(len(active_indices), device=sim_matrix.device), active_indices]

    # InfoNCE: -log(exp(pos/tau) / sum(exp(all/tau)))
    logits = sim_matrix / temperature  # (A, K)
    log_sum_exp = torch.logsumexp(logits, dim=-1)  # (A,)
    loss = -(pos_sim / temperature - log_sum_exp).mean()
    return loss


def diversity_alignment(
    projected_tok_entries: torch.Tensor,  # (K_tok, model_dim)
    wernicke_entries: torch.Tensor,        # (K_wer, model_dim)
) -> torch.Tensor:
    """SSIM-style diversity loss.

    Penalize similarity between codebooks to encourage complementary info.
    L_diverse = mean(|cosine_similarity(projected_tok, wer)|)

    Computes pairwise cosine similarity between all tokenizer codebook
    entries and all Wernicke codebook entries, then takes the mean
    absolute value.
    """
    tok_norm = F.normalize(projected_tok_entries, dim=-1)  # (K_tok, dim)
    wer_norm = F.normalize(wernicke_entries, dim=-1)        # (K_wer, dim)
    cos_sim = tok_norm @ wer_norm.T  # (K_tok, K_wer)
    return cos_sim.abs().mean()


def distillation_alignment(
    projected_tok_embeds: torch.Tensor,  # (batch, token_seq, model_dim)
    wernicke_entries: torch.Tensor,       # (K_wer, model_dim)
    wernicke_assignments: torch.Tensor,   # (batch, token_seq)
) -> torch.Tensor:
    """Cosine distillation loss.

    Each Wernicke bucket acts as teacher.  Tokenizer codes feeding into
    bucket *j* should predict the Wernicke entry.

    L_distill = 1 - mean_j[cosine(mean_tok_for_bucket_j, wer_entry_j)]

    Buckets with no assignments in the batch are skipped.
    """
    flat_embeds = projected_tok_embeds.reshape(-1, projected_tok_embeds.size(-1))
    flat_assigns = wernicke_assignments.reshape(-1)

    K = wernicke_entries.size(0)

    bucket_sums = torch.zeros(K, flat_embeds.size(-1), device=flat_embeds.device)
    bucket_counts = torch.zeros(K, device=flat_embeds.device)
    bucket_sums.scatter_add_(0, flat_assigns.unsqueeze(-1).expand_as(flat_embeds), flat_embeds)
    bucket_counts.scatter_add_(0, flat_assigns, torch.ones_like(flat_assigns, dtype=torch.float))

    active = bucket_counts > 0
    if active.sum() == 0:
        return torch.tensor(0.0, device=flat_embeds.device)

    bucket_means = bucket_sums[active] / bucket_counts[active].unsqueeze(-1)  # (A, dim)
    active_entries = wernicke_entries[active]  # (A, dim)

    cos = F.cosine_similarity(bucket_means, active_entries, dim=-1)  # (A,)
    return 1.0 - cos.mean()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_ALIGNMENT_FNS = {
    "none": no_alignment,
    "contrastive": contrastive_alignment,
    "diversity": diversity_alignment,
    "distillation": distillation_alignment,
}


def compute_alignment_loss(
    align_type: str,  # "none", "contrastive", "diversity", "distillation"
    **kwargs,
) -> torch.Tensor:
    """Dispatch to the correct alignment function."""
    if align_type not in _ALIGNMENT_FNS:
        raise ValueError(
            f"Unknown alignment type {align_type!r}. "
            f"Expected one of {list(_ALIGNMENT_FNS.keys())}"
        )
    return _ALIGNMENT_FNS[align_type](**kwargs)
