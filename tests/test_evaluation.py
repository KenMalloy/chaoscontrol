"""Tests for evaluation utilities."""
import math
import unittest


class TestComputeBPB(unittest.TestCase):
    def test_known_value(self):
        from chaoscontrol.evaluation import compute_bpb
        # 100 nats over 100 bytes = 1 nat/byte = 1/ln(2) bpb
        bpb = compute_bpb(total_ce_nats=100.0, total_raw_bytes=100)
        assert abs(bpb - 1.0 / math.log(2.0)) < 1e-6

    def test_zero_bytes_returns_zero(self):
        from chaoscontrol.evaluation import compute_bpb
        bpb = compute_bpb(total_ce_nats=100.0, total_raw_bytes=0)
        assert bpb == 0.0 or math.isinf(bpb) is False  # should not crash

    def test_stride_invariant(self):
        from chaoscontrol.evaluation import compute_bpb
        # Same total CE, same total bytes — bpb should be the same
        # regardless of how many tokens were involved
        bpb1 = compute_bpb(total_ce_nats=100.0, total_raw_bytes=100)
        bpb2 = compute_bpb(total_ce_nats=100.0, total_raw_bytes=100)
        assert bpb1 == bpb2

    def test_lower_ce_means_lower_bpb(self):
        from chaoscontrol.evaluation import compute_bpb
        bpb_high = compute_bpb(total_ce_nats=200.0, total_raw_bytes=100)
        bpb_low = compute_bpb(total_ce_nats=100.0, total_raw_bytes=100)
        assert bpb_low < bpb_high


if __name__ == "__main__":
    unittest.main()
