from __future__ import annotations
import torch
import torch.nn as nn
from typing import Iterable


# Pattern forms:
#   "substr:X"    — match if X appears anywhere in the param's FQN
#   "exact:X"     — match only if the FQN equals X exactly
# (Plain strings in the list are treated as substr: for backward compatibility.)
ADAPT_SET_PATTERNS: dict[str, list[str]] = {
    "none": [],
    "log_a": ["substr:log_a"],
    "delta_proj": ["substr:delta_proj"],
    "log_a+delta_proj": ["substr:log_a", "substr:delta_proj"],
    "B_side": ["substr:in_proj", "substr:select_proj"],
    "C_side": ["substr:out_proj", "substr:gate_proj"],
    # embed_rows_seen: exact match to avoid collisions with any future
    # "embed_dim" / "embedding_norm" / etc. parameter names.
    "embed_rows_seen": ["exact:embed.weight"],
    "lm_head": ["substr:lm_head"],
    "lora_r8": ["substr:lora_"],  # lora adapters named lora_A_<name> / lora_B_<name>
    "trainable_h0": ["substr:_trainable_h0"],  # Axis 2 trainable h0, see Task 7
    "all": ["*"],
}


def _matches(name: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if pat.startswith("exact:"):
            if name == pat[len("exact:"):]:
                return True
        elif pat.startswith("substr:"):
            if pat[len("substr:"):] in name:
                return True
        else:  # legacy plain string == substr
            if pat in name:
                return True
    return False


def select_adapt_params(module: nn.Module, *, adapt_set: str) -> list[nn.Parameter]:
    """Return the list of parameters that match the adapt_set filter."""
    if adapt_set not in ADAPT_SET_PATTERNS:
        raise ValueError(f"unknown adapt_set: {adapt_set}")
    patterns = ADAPT_SET_PATTERNS[adapt_set]
    if not patterns:
        return []
    if patterns == ["*"]:
        return list(module.parameters())
    out: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, p in module.named_parameters():
        if _matches(name, patterns):
            if id(p) not in seen:
                out.append(p)
                seen.add(id(p))
    return out
