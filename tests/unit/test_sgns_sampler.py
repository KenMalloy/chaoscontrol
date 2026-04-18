import torch
from chaoscontrol.sgns.sampler import NegativeSampler, unigram_probs_from_counts


def test_unigram_probs_from_counts_normalized():
    counts = torch.tensor([100.0, 10.0, 1.0])
    probs = unigram_probs_from_counts(counts, power=0.75)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert probs[0] > probs[1] > probs[2]


def test_unigram_probs_distortion():
    counts = torch.tensor([100.0, 10.0])
    flat = unigram_probs_from_counts(counts, power=0.0)
    distorted = unigram_probs_from_counts(counts, power=0.75)
    raw = unigram_probs_from_counts(counts, power=1.0)
    # 0.75 power moves mass from high-freq toward low-freq vs raw
    assert distorted[1] > raw[1]
    # power=0 gives uniform
    assert torch.allclose(flat, torch.tensor([0.5, 0.5]))


def test_negative_sampler_shape_and_range():
    torch.manual_seed(0)
    probs = torch.tensor([0.5, 0.3, 0.2])
    sampler = NegativeSampler(probs)
    samples = sampler.sample(batch_size=4, k=5)
    assert samples.shape == (4, 5)
    assert samples.min() >= 0
    assert samples.max() < 3


def test_negative_sampler_returns_samples_on_probs_device():
    probs = torch.tensor([0.5, 0.3, 0.2])
    sampler = NegativeSampler(probs)
    samples = sampler.sample(batch_size=2, k=3)
    assert samples.device == probs.device
