"""Foundation tests for ``chaoscontrol.eval.ttt_eval`` — registry + dispatch.

These tests cover the contract only; calc_type bodies are implemented in
follow-up commits and tested in ``test_exp27_calc_types.py``.
"""
from __future__ import annotations

import pytest


def test_registry_module_imports() -> None:
    from chaoscontrol.eval import ttt_eval as mod

    assert hasattr(mod, "CALC_TYPE_REGISTRY")
    assert hasattr(mod, "register_calc_type")
    assert hasattr(mod, "evaluate_with_calc_types")
    assert hasattr(mod, "CalcTypeContext")
    assert hasattr(mod, "CalcTypeResult")


def test_calc_type_subpackage_registers_three_names() -> None:
    # Import the subpackage so registration happens.
    import chaoscontrol.eval.calc_types  # noqa: F401
    from chaoscontrol.eval.ttt_eval import (
        CALC_TYPE_REGISTRY,
        list_registered_calc_types,
    )

    expected = {
        "score_only_reset",
        "carry_state",
        "dreamworld_eval",
    }
    registered = set(list_registered_calc_types())
    assert expected <= registered, (
        f"missing calc_types: {expected - registered}"
    )
    for name in expected:
        assert name in CALC_TYPE_REGISTRY
    # state_replay_within_doc was removed for causality reasons; ensure
    # nothing brought it back through a stray re-register.
    assert "state_replay_within_doc" not in registered


def test_metadata_records_source_order_and_grad_flags() -> None:
    import chaoscontrol.eval.calc_types  # noqa: F401
    from chaoscontrol.eval.ttt_eval import calc_type_metadata

    # carry_state breaks reset-commutativity → requires source_order
    assert calc_type_metadata("carry_state")["requires_source_order"] is True
    # score_only_reset is the floor → reset-commutative
    assert calc_type_metadata("score_only_reset")["requires_source_order"] is False
    # dreamworld_eval needs autograd
    assert calc_type_metadata("dreamworld_eval")["requires_grad"] is True


def test_register_rejects_duplicate_name() -> None:
    from chaoscontrol.eval.ttt_eval import register_calc_type

    @register_calc_type("__test_dup_a__")
    def _a(ctx):  # pragma: no cover
        raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        @register_calc_type("__test_dup_a__")
        def _b(ctx):  # pragma: no cover
            raise NotImplementedError


def test_evaluate_with_calc_types_raises_on_unknown_name() -> None:
    import chaoscontrol.eval.calc_types  # noqa: F401
    from chaoscontrol.eval.ttt_eval import evaluate_with_calc_types

    with pytest.raises(ValueError, match="unknown calc_type"):
        evaluate_with_calc_types(
            model=None,  # never reached
            val_cache=None,  # type: ignore[arg-type]
            calc_types=["nope_does_not_exist"],
            calc_type_configs={},
            device=None,  # type: ignore[arg-type]
            base_bytes_lut=None,  # type: ignore[arg-type]
            has_leading_space_lut=None,  # type: ignore[arg-type]
            is_boundary_token_lut=None,  # type: ignore[arg-type]
        )


def test_calc_type_metadata_unknown_raises() -> None:
    from chaoscontrol.eval.ttt_eval import calc_type_metadata

    with pytest.raises(ValueError, match="unknown calc_type"):
        calc_type_metadata("nope_does_not_exist")


def test_evaluate_rejects_order_sensitive_when_source_order_not_preserved() -> None:
    """Tripwire: carry_state needs source order; refuse if caller signals shuffled."""
    import chaoscontrol.eval.calc_types  # noqa: F401
    from chaoscontrol.eval.ttt_eval import evaluate_with_calc_types

    with pytest.raises(ValueError, match="require source-ordered docs"):
        evaluate_with_calc_types(
            model=None,  # never reached
            val_cache=None,  # type: ignore[arg-type]
            calc_types=["score_only_reset", "carry_state"],
            calc_type_configs={},
            device=None,  # type: ignore[arg-type]
            base_bytes_lut=None,  # type: ignore[arg-type]
            has_leading_space_lut=None,  # type: ignore[arg-type]
            is_boundary_token_lut=None,  # type: ignore[arg-type]
            source_order_preserved=False,
        )


def test_evaluate_allows_reset_commutative_calc_types_when_shuffled() -> None:
    """score_only_reset is reset-commutative, so source_order_preserved=False is OK
    as long as no order-sensitive calc_type is requested. We can't run a real eval
    here without a model; we only assert the gate doesn't fire pre-dispatch."""
    import chaoscontrol.eval.calc_types  # noqa: F401
    from chaoscontrol.eval.ttt_eval import evaluate_with_calc_types

    # Will fail later (None model), but the source-order check passes first.
    with pytest.raises(Exception) as excinfo:
        evaluate_with_calc_types(
            model=None,  # type: ignore[arg-type]
            val_cache=None,  # type: ignore[arg-type]
            calc_types=["score_only_reset"],
            calc_type_configs={},
            device=None,  # type: ignore[arg-type]
            base_bytes_lut=None,  # type: ignore[arg-type]
            has_leading_space_lut=None,  # type: ignore[arg-type]
            is_boundary_token_lut=None,  # type: ignore[arg-type]
            source_order_preserved=False,
        )
    # The error must NOT be the source-order tripwire — that means the gate
    # correctly let the reset-commutative calc_type through.
    assert "require source-ordered docs" not in str(excinfo.value)
