"""Tests for chaoscontrol.routing — RichBNN and DistributedB."""
from __future__ import annotations

import unittest

import torch

from chaoscontrol.routing import DistributedB, RichBNN


class TestRichBNN(unittest.TestCase):
    def test_output_shape(self) -> None:
        b = RichBNN(dim=16, bottleneck=8)
        x = torch.randn(2, 16)
        h = torch.randn(2, 16)
        out = b(x, h)
        assert out.shape == (2, 16)

    def test_state_dependent(self) -> None:
        b = RichBNN(dim=16, bottleneck=8)
        x = torch.randn(2, 16)
        h1 = torch.randn(2, 16)
        h2 = torch.randn(2, 16)
        out1 = b(x, h1)
        out2 = b(x, h2)
        assert not torch.allclose(out1, out2)


class TestDistributedBHub(unittest.TestCase):
    def test_output_shape(self) -> None:
        b = DistributedB(dim=16, num_subnets=4, bottleneck=8, topology="hub")
        out = b(torch.randn(2, 16), torch.randn(2, 16))
        assert out.shape == (2, 16)

    def test_state_dependent(self) -> None:
        b = DistributedB(dim=16, num_subnets=4, bottleneck=8, topology="hub")
        x = torch.randn(2, 16)
        assert not torch.allclose(b(x, torch.randn(2, 16)), b(x, torch.randn(2, 16)))

    def test_subnets_see_different_views(self) -> None:
        b = DistributedB(dim=16, num_subnets=4, bottleneck=8, topology="hub")
        views = [b.view_projs[i].weight.data for i in range(4)]
        all_same = all(torch.allclose(views[0], v) for v in views[1:])
        assert not all_same


class TestDistributedBAssembly(unittest.TestCase):
    def test_output_shape(self) -> None:
        b = DistributedB(dim=16, num_subnets=4, bottleneck=8, topology="assembly", settling_steps=2)
        out = b(torch.randn(2, 16), torch.randn(2, 16))
        assert out.shape == (2, 16)

    def test_more_settling_steps_changes_output(self) -> None:
        torch.manual_seed(42)
        b1 = DistributedB(dim=16, num_subnets=4, bottleneck=8, topology="assembly", settling_steps=1)
        torch.manual_seed(42)
        b2 = DistributedB(dim=16, num_subnets=4, bottleneck=8, topology="assembly", settling_steps=3)
        x = torch.randn(2, 16)
        h = torch.randn(2, 16)
        assert not torch.allclose(b1(x, h), b2(x, h))

    def test_settling_norm_normalizes_lateral_update(self) -> None:
        b = DistributedB(dim=16, num_subnets=4, bottleneck=8, topology="assembly", settling_steps=2)
        x = torch.randn(2, 16)
        h = torch.randn(2, 16)

        views = [proj(h) for proj in b.view_projs]
        partials = [
            subnet(torch.cat([x, view], dim=-1))
            for subnet, view in zip(b.subnets, views)
        ]
        raw_update = b.lateral(torch.cat(partials, dim=-1))
        normed_update = b.settling_norm(raw_update)
        rms = normed_update.pow(2).mean(dim=-1).sqrt()

        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4)


class TestDistributedBHybrid(unittest.TestCase):
    def test_output_shape(self) -> None:
        b = DistributedB(dim=16, num_subnets=4, bottleneck=8, topology="hybrid", settling_steps=2)
        out = b(torch.randn(2, 16), torch.randn(2, 16))
        assert out.shape == (2, 16)


if __name__ == "__main__":
    unittest.main()
