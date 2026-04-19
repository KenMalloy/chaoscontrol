import torch
from chaoscontrol.sgns.moment_match import (
    frequency_bucket_ids,
    match_row_norm_moments,
    match_full_covariance,
    sample_full_covariance,
    sample_with_row_norms,
    shuffle_rows,
    shuffle_rows_within_buckets,
    token_class_bucket_ids,
)


def test_match_row_norm_moments_matches_target_moments():
    torch.manual_seed(0)
    src = torch.randn(100, 16) * 3.0 + 2.0
    target = torch.randn(100, 16) * 0.5
    out = match_row_norm_moments(src, target)
    src_row_norms = src.norm(dim=-1)
    target_row_norms = target.norm(dim=-1)
    out_row_norms = out.norm(dim=-1)
    assert torch.isclose(out_row_norms.mean(), target_row_norms.mean(), rtol=1e-3)
    assert torch.isclose(out_row_norms.std(), target_row_norms.std(), rtol=1e-3)


def test_match_row_norm_preserves_directions():
    """Rescaling rows preserves pairwise cosine similarity."""
    torch.manual_seed(1)
    src = torch.randn(50, 8)
    target = torch.randn(50, 8) * 0.01
    out = match_row_norm_moments(src, target)
    src_n = torch.nn.functional.normalize(src, dim=-1)
    out_n = torch.nn.functional.normalize(out, dim=-1)
    assert torch.allclose(src_n, out_n, atol=1e-5)


def test_match_full_covariance_matches_cov():
    torch.manual_seed(0)
    src = torch.randn(200, 8) * 3.0
    target = torch.randn(200, 8) * 0.5
    out = match_full_covariance(src, target)
    cov_out = torch.cov(out.T)
    cov_target = torch.cov(target.T)
    assert torch.allclose(cov_out, cov_target, atol=0.05)


def test_shuffle_rows_is_permutation():
    torch.manual_seed(0)
    src = torch.randn(100, 16)
    out = shuffle_rows(src, seed=42)
    # Same set of rows, different order
    sorted_src = src[torch.argsort(src[:, 0])]
    sorted_out = out[torch.argsort(out[:, 0])]
    assert torch.allclose(sorted_src, sorted_out)
    assert not torch.allclose(src, out)


def test_sample_with_row_norms_preserves_reference_norms_without_directions():
    torch.manual_seed(0)
    reference = torch.randn(128, 16) * 2.0
    out_a = sample_with_row_norms(reference, seed=11)
    out_b = sample_with_row_norms(reference, seed=11)
    out_c = sample_with_row_norms(reference, seed=12)

    torch.testing.assert_close(out_a.norm(dim=-1), reference.norm(dim=-1))
    torch.testing.assert_close(out_a, out_b)
    assert not torch.allclose(out_a, out_c)

    # The control carries per-token norm information, not SGNS directions.
    cos = torch.nn.functional.cosine_similarity(reference, out_a, dim=-1)
    assert cos.abs().mean() < 0.5


def test_sample_with_row_norms_can_shuffle_norm_assignment():
    reference = torch.arange(1, 17, dtype=torch.float32).unsqueeze(1).repeat(1, 4)
    aligned = sample_with_row_norms(reference, seed=3)
    shuffled = sample_with_row_norms(reference, seed=3, shuffle_norms_seed=99)

    torch.testing.assert_close(
        shuffled.norm(dim=-1).sort().values,
        reference.norm(dim=-1).sort().values,
    )
    assert not torch.allclose(shuffled.norm(dim=-1), aligned.norm(dim=-1))


def test_shuffle_rows_within_buckets_preserves_each_bucket_multiset():
    src = torch.arange(10, dtype=torch.float32).unsqueeze(1)
    buckets = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    out = shuffle_rows_within_buckets(src, buckets, seed=7)

    for bucket in buckets.unique():
        mask = buckets == bucket
        torch.testing.assert_close(
            out[mask, 0].sort().values,
            src[mask, 0].sort().values,
        )
    assert not torch.allclose(out, src)


def test_frequency_bucket_ids_assigns_equal_size_frequency_ranks():
    counts = torch.tensor([100.0, 90.0, 80.0, 5.0, 4.0, 3.0, 0.0, 0.0])
    buckets = frequency_bucket_ids(counts, num_buckets=4)
    torch.testing.assert_close(
        buckets,
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
    )


def test_token_class_bucket_ids_groups_sentencepiece_style_pieces():
    pieces = [
        "<unk>",
        "\u2581hello",
        "world",
        "\u2581",
        "123",
        "9th",
        ".",
        "\u2581!?",
        "hello-world",
    ]
    buckets = token_class_bucket_ids(pieces)

    assert buckets[1] == buckets[3] == buckets[7]  # leading-space/whitespace
    assert buckets[2] != buckets[4]  # alpha vs numeric
    assert buckets[4] != buckets[5]  # numeric vs mixed
    assert buckets[6] != buckets[8]  # punctuation vs mixed
    assert len(set(buckets.tolist())) >= 5


def test_sample_full_covariance_is_deterministic_and_matches_moments():
    torch.manual_seed(0)
    scale = torch.tensor([2.0, 0.5, 1.5, 0.25])
    reference = torch.randn(4096, 4) * scale + torch.tensor([1.0, -2.0, 0.5, 3.0])

    out_a = sample_full_covariance(reference, seed=101)
    out_b = sample_full_covariance(reference, seed=101)
    torch.testing.assert_close(out_a, out_b)

    torch.testing.assert_close(out_a.mean(dim=0), reference.mean(dim=0), atol=0.08, rtol=0.08)
    torch.testing.assert_close(torch.cov(out_a.T), torch.cov(reference.T), atol=0.15, rtol=0.15)
