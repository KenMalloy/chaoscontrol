"""Mocked-primitive coverage for crct_distributed.

End-to-end gloo-spawn 4-rank coverage rides on the existing
test_runner_3plus1_skip_main.py pattern as a follow-up integration test;
this file unit-tests the coordination layer with FakeProcessGroup so the
suite stays CPU-only and fast.
"""
import unittest
from chaoscontrol import crct_distributed as crct


class FakeWork:
    def __init__(self, completed=False):
        self._completed = completed
    def is_completed(self):
        return self._completed
    def wait(self):
        self._completed = True


class FakeProcessGroup:
    def __init__(self, ranks):
        self.ranks = list(ranks)
        self.broadcasts = []
        self.gathers = []
    def __repr__(self):
        return f"FakePG({self.ranks})"


class TestModuleImports(unittest.TestCase):
    def test_module_exposes_documented_api(self):
        for name in (
            "create_crct_process_groups",
            "WeightMailbox",
            "TeacherResultMailbox",
            "TeacherPayload",
            "Rank3MemoryCoprocessor",
            "rank3_coprocessor_loop",
            "fail_open_controller_term",
        ):
            self.assertTrue(
                hasattr(crct, name),
                f"crct_distributed must expose {name}",
            )
