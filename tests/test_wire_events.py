"""Verify the wire-event structs from src/wire_events.h have the
documented sizes and alignment so the C-side ShmRing slot layout
matches the design contract (Phase A1 of CPU SSM controller plan).

Sizes come from `docs/plans/2026-04-26-cpu-ssm-controller.md` (Task A1)
after the size-math correction (568 / 544 / 96 — the original 552 / 528
targets were arithmetically unreachable for the documented field set
under `#pragma pack(push, 1)`).

Note on alignment: the structs are declared under `#pragma pack(push, 1)`
so `alignof(T)` is 1 by definition. The number we care about is the
alignment ShmRing slots need so a `uint64_t` load from any field is
naturally aligned. That is `alignof(uint64_t)` = 8 on every platform we
target. The binding therefore reports the largest-member natural
alignment, not `alignof(struct)`.
"""
from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_wire_event_sizes_match_design():
    sizes = _ext.wire_event_sizes()
    assert sizes["WriteEvent"] == 568, sizes
    assert sizes["QueryEvent"] == 544, sizes
    assert sizes["ReplayOutcome"] == 96, sizes


def test_wire_event_alignments_are_8_byte():
    aligns = _ext.wire_event_alignments()
    assert aligns["WriteEvent"] == 8, aligns
    assert aligns["QueryEvent"] == 8, aligns
    assert aligns["ReplayOutcome"] == 8, aligns


def test_wire_event_constants_exposed():
    constants = _ext.wire_event_constants()
    assert constants["KEY_REP_DIM_DEFAULT"] == 256
    assert constants["SPAN_LENGTH_DEFAULT"] == 4
