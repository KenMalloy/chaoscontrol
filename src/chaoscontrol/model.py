"""ChaosSSMBlock and ChaosStudentLM — full model assembly."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from chaoscontrol.core import RMSNorm, FeedForward, ChaosSSMCore
from chaoscontrol.routing import RichBNN, DistributedB
from chaoscontrol.memory import OuterModel, MultiSlotOuterModel, SemanticTier
from chaoscontrol.wernicke import WernickeLayer


class ChaosSSMBlock(nn.Module):
    """Single block: input_norm -> ChaosSSMCore (with optional rich_b) -> residual -> ff_norm -> FeedForward -> residual."""

    def __init__(
        self,
        dim: int,
        ff_mult: int = 2,
        *,
        a_mode: str = "diag",
        a_full_rank: int = 8,
        a_full_gamma: float = 0.05,
        rich_b_mode: str = "none",
        rich_b_bottleneck: int = 32,
        rich_b_num_subnets: int = 4,
        rich_b_settling_steps: int = 2,
    ) -> None:
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.ff_norm = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_mult)
        self.core = ChaosSSMCore(dim, a_mode=a_mode, a_full_rank=a_full_rank, a_full_gamma=a_full_gamma)

        if rich_b_mode == "none":
            self.rich_b: nn.Module | None = None
        elif rich_b_mode == "nn":
            self.rich_b = RichBNN(dim, bottleneck=rich_b_bottleneck)
        elif rich_b_mode in ("hub", "assembly", "hybrid"):
            self.rich_b = DistributedB(
                dim,
                num_subnets=rich_b_num_subnets,
                bottleneck=rich_b_bottleneck,
                topology=rich_b_mode,
                settling_steps=rich_b_settling_steps,
            )
        else:
            raise ValueError(f"unsupported rich_b_mode: {rich_b_mode}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_jacobian_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        normed = self.input_norm(x)
        result = self.core(normed, rich_b=self.rich_b, return_jacobian_stats=return_jacobian_stats)
        if return_jacobian_stats:
            y, stats = result
        else:
            y = result
        x = x + y
        x = x + self.ff(self.ff_norm(x))
        if return_jacobian_stats:
            return x, stats
        return x


class ChaosStudentLM(nn.Module):
    """Full ChaosControl student language model wiring all components."""

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        num_layers: int,
        ff_mult: int = 2,
        a_mode: str = "diag",
        a_full_rank: int = 8,
        a_full_gamma: float = 0.05,
        rich_b_mode: str = "none",
        rich_b_bottleneck: int = 32,
        rich_b_num_subnets: int = 4,
        rich_b_settling_steps: int = 2,
        outer_model_dim: int = 0,
        consolidation_mode: str = "symmetric",
        consolidation_ema_decay: float = 0.99,
        consolidation_trigger: str = "immediate",
        consolidation_window: int = 8,
        outer_model_type: str = "single",
        outer_max_slots: int = 64,
        outer_compress_ratio: int = 2,
        wernicke_enabled: bool = False,
        wernicke_k_max: int = 16,
        wernicke_window: int = 8,
        wernicke_router: str = "vq",
        wernicke_balance_weight: float = 0.01,
        semantic_tier_bases: int = 0,
        typed_storage: bool = False,
        typed_consolidation: bool = False,
        compression_consequence: bool = False,
        cue_projection: bool = True,
        dynamic_crit_per_layer: bool = False,
        compression_selection: str = "survival",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)

        # Wernicke layer: typed compositional preprocessing
        self.wernicke: WernickeLayer | None = None
        self.wernicke_balance_weight = wernicke_balance_weight
        if wernicke_enabled:
            self.wernicke = WernickeLayer(
                dim,
                k_max=wernicke_k_max,
                window=wernicke_window,
                router_type=wernicke_router,
                balance_weight=wernicke_balance_weight,
            )

        self.layers = nn.ModuleList([
            ChaosSSMBlock(
                dim,
                ff_mult,
                a_mode=a_mode,
                a_full_rank=a_full_rank,
                a_full_gamma=a_full_gamma,
                rich_b_mode=rich_b_mode,
                rich_b_bottleneck=rich_b_bottleneck,
                rich_b_num_subnets=rich_b_num_subnets,
                rich_b_settling_steps=rich_b_settling_steps,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.outer_model_type = outer_model_type
        self.outer_model: OuterModel | MultiSlotOuterModel | None = None
        if outer_model_dim > 0:
            common_kw = dict(
                consolidation_mode=consolidation_mode,
                ema_decay=consolidation_ema_decay,
                trigger=consolidation_trigger,
                trigger_window=consolidation_window,
            )
            if outer_model_type == "multislot":
                self.outer_model = MultiSlotOuterModel(
                    dim, outer_dim=outer_model_dim,
                    max_slots=outer_max_slots,
                    compress_ratio=outer_compress_ratio,
                    compression_selection=compression_selection,
                    **common_kw,
                )
            else:
                self.outer_model = OuterModel(
                    dim, outer_dim=outer_model_dim, **common_kw,
                )

        # Semantic tier: always-on background bias from episodic experience
        self.semantic_tier: SemanticTier | None = None
        if semantic_tier_bases > 0:
            self.semantic_tier = SemanticTier(dim, num_bases=semantic_tier_bases)

        # Gap-analysis flags
        self.typed_storage = typed_storage
        self.typed_consolidation = typed_consolidation
        self.compression_consequence = compression_consequence
        self.cue_projection = cue_projection
        self.dynamic_crit_per_layer = dynamic_crit_per_layer

    def artifact_bytes(self) -> int:
        return int(sum(p.numel() for p in self.parameters()) * 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_jacobian_stats: bool = False,
    ) -> dict[str, Any]:
        x = self.embed(input_ids)

        # Wernicke layer: compose bytes into typed units before SSM recurrence
        balance_loss = None
        bucket_ids = None
        if self.wernicke is not None:
            x, bucket_ids, balance_loss = self.wernicke(x)

        if self.outer_model is not None:
            if isinstance(self.outer_model, MultiSlotOuterModel):
                # Cue-dependent retrieval. If cue_projection=False (gap analysis),
                # pass cue=None so retrieval uses uniform weighting over slots
                # instead of the learned cue projection — tests whether the
                # recurrence state naturally serves as an address.
                # Normal mode: mean pool over input embeddings as cue.
                cue = x.detach().mean(dim=1) if self.cue_projection else None
                outer_read = self.outer_model.read(x.size(0), cue=cue)
            else:
                outer_read = self.outer_model.read(x.size(0))
            x = x + outer_read.unsqueeze(1)  # broadcast across seq dim

        if self.semantic_tier is not None:
            semantic_bias = self.semantic_tier.read(x.size(0))
            x = x + semantic_bias.unsqueeze(1)

        all_stats: list[dict] = []
        for layer in self.layers:
            result = layer(x, return_jacobian_stats=return_jacobian_stats)
            if return_jacobian_stats:
                x, stats = result
                all_stats.append(stats)
            else:
                x = result

        hidden = x
        x = self.final_norm(x)
        logits = self.lm_head(x)

        out: dict[str, Any] = {"logits": logits, "hidden": hidden}
        if balance_loss is not None:
            out["balance_loss"] = balance_loss
        if bucket_ids is not None:
            out["bucket_ids"] = bucket_ids
        if return_jacobian_stats:
            # Average stats across layers
            merged: dict[str, torch.Tensor] = {}
            for key in all_stats[0]:
                merged[key] = torch.stack([s[key] for s in all_stats]).mean()
            out["jacobian_stats"] = merged
            if self.dynamic_crit_per_layer:
                out["per_layer_jacobian_stats"] = all_stats
        return out
