"""CRCT distributed coordination layer.

See docs/plans/2026-04-27-crct-distributed-design.md.
"""
from __future__ import annotations
__all__ = [
    "create_crct_process_groups",
    "WeightMailbox",
    "TeacherResultMailbox",
    "TeacherPayload",
    "Rank3MemoryCoprocessor",
    "rank3_coprocessor_loop",
    "fail_open_controller_term",
]


def create_crct_process_groups(world_size):
    raise NotImplementedError


class WeightMailbox:  # noqa: D101 — fleshed out in Task 3
    def __init__(self, *args, **kwargs): raise NotImplementedError


class TeacherPayload:  # filled in Task 5
    pass


class TeacherResultMailbox:
    def __init__(self, *args, **kwargs): raise NotImplementedError


class Rank3MemoryCoprocessor:
    def __init__(self, *args, **kwargs): raise NotImplementedError


def rank3_coprocessor_loop(*args, **kwargs):
    raise NotImplementedError


def fail_open_controller_term(*args, **kwargs):
    raise NotImplementedError
