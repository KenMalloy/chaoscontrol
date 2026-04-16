"""Unit tests for ``chaoscontrol.distributed``.

Only covers helpers that don't require a live process group:
``resolve_ddp_context`` decodes (explicit args, env vars, fallback) and
a typo in that decode ladder would silently route DDP runs into the
single-rank path. The collective helpers (``broadcast_params``,
``allreduce_grads``, ``should_stop_now``) are exercised end-to-end by
``test_ddp_integration.py`` and the pod smoke runs.
"""
from __future__ import annotations

import os

import pytest

from chaoscontrol.distributed import resolve_ddp_context


class TestResolveDDPContext:
    def test_explicit_args_passed_through(self) -> None:
        assert resolve_ddp_context(rank=2, world_size=4) == (2, 4)

    def test_one_of_explicit_raises(self) -> None:
        with pytest.raises(ValueError, match="both"):
            resolve_ddp_context(rank=1, world_size=None)
        with pytest.raises(ValueError, match="both"):
            resolve_ddp_context(rank=None, world_size=2)

    def test_env_vars_picked_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "8")
        assert resolve_ddp_context(rank=None, world_size=None) == (3, 8)

    def test_fallback_single_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Strip any inherited torchrun-style env so the fallback actually fires.
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        assert resolve_ddp_context(rank=None, world_size=None) == (0, 1)
