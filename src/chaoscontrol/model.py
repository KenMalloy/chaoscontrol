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

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-token step through the block.

        Args:
            x: (batch, dim) — single token
            state: (batch, dim) — previous recurrence state for this block's core

        Returns:
            (output, new_state) — output is (batch, dim), new_state is (batch, dim)
        """
        normed = self.input_norm(x)
        y, new_state = self.core.step(normed, state, rich_b=self.rich_b)
        x = x + y
        x = x + self.ff(self.ff_norm(x))
        return x, new_state

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
        wernicke_expert_dim: int = 0,
        semantic_tier_bases: int = 0,
        semantic_tier_update_rate: float = 0.01,
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
                expert_dim=wernicke_expert_dim if wernicke_expert_dim > 0 else None,
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
            self.semantic_tier = SemanticTier(dim, num_bases=semantic_tier_bases, update_rate=semantic_tier_update_rate)

        # Gap-analysis flags
        self.typed_storage = typed_storage
        self.typed_consolidation = typed_consolidation
        self.compression_consequence = compression_consequence
        self.cue_projection = cue_projection
        self.dynamic_crit_per_layer = dynamic_crit_per_layer

    def init_state(self, batch_size: int) -> list[torch.Tensor]:
        """Initialize recurrence states for all layers."""
        device = self.embed.weight.device
        return [torch.zeros(batch_size, self.dim, device=device) for _ in range(len(self.layers))]

    def step(
        self,
        token_ids: torch.Tensor,
        states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Single-token forward step through the SSM recurrence only.

        Intentionally simpler than forward() — skips Wernicke typed composition,
        episodic/semantic memory reads, and outer model interactions. This is the
        "world model" for MCTS planning: analogous to System 2 deliberation
        operating on a compressed internal model rather than full perception.

        For components that require the full forward path (Wernicke routing,
        memory cue-dependent retrieval), use forward() on complete sequences.

        Args:
            token_ids: (batch, 1) — single token ids
            states: list of (batch, dim) per layer

        Returns:
            (logits, hidden, new_states)
            logits: (batch, vocab)
            hidden: (batch, dim)
            new_states: list of (batch, dim) per layer
        """
        x = self.embed(token_ids).squeeze(1)  # (batch, dim)

        new_states = []
        for i, layer in enumerate(self.layers):
            x, new_s = layer.step(x, states[i])
            new_states.append(new_s)

        hidden = x
        x = self.final_norm(x.unsqueeze(1)).squeeze(1)  # RMSNorm expects (..., dim)
        logits = self.lm_head(x)

        return logits, hidden, new_states

    def dream_step(
        self,
        token_ids: torch.Tensor,
        states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Full-tier single-token forward step for REM dream generation.

        Like step() but includes ALL tiers (Wernicke, memory, semantic) that
        step() intentionally skips. During REM dream generation the model needs
        to experience the full forward path so that dream sequences reflect the
        complete perception pipeline.

        Args:
            token_ids: (batch, 1) — single token ids
            states: list of (batch, dim) per layer

        Returns:
            (logits, hidden, new_states)
            logits: (batch, vocab)
            hidden: (batch, dim)
            new_states: list of (batch, dim) per layer
        """
        x = self.embed(token_ids).squeeze(1)  # (batch, dim)

        # Wernicke (NOT skipped, unlike step)
        if self.wernicke is not None:
            x_seq = x.unsqueeze(1)  # Wernicke expects (batch, seq, dim)
            x_seq, bucket_ids, _ = self.wernicke(x_seq)
            x = x_seq.squeeze(1)

        # Memory read (NOT skipped)
        if self.outer_model is not None:
            batch_size = x.size(0)
            if hasattr(self.outer_model, "_slots"):
                outer_read = self.outer_model.read(batch_size, cue=x)
            else:
                outer_read = self.outer_model.read(batch_size)
            x = x + outer_read

        # Semantic tier (NOT skipped)
        if self.semantic_tier is not None:
            x = x + self.semantic_tier.read(x.size(0))

        # SSM recurrence — use ChaosSSMBlock.step()
        new_states = []
        for i, layer in enumerate(self.layers):
            x, new_s = layer.step(x, states[i])
            new_states.append(new_s)

        hidden = x
        logits = self.lm_head(self.final_norm(x.unsqueeze(1)).squeeze(1))
        return logits, hidden, new_states

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
