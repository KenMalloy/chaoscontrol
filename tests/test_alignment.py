"""Tests for codebook alignment losses."""
import unittest
import torch


class TestNoAlignment(unittest.TestCase):
    def test_no_alignment_returns_zero(self):
        from chaoscontrol.alignment import no_alignment
        loss = no_alignment()
        assert loss.item() == 0.0
        assert loss.shape == ()


class TestContrastiveAlignment(unittest.TestCase):
    def test_contrastive_positive(self):
        """Non-zero loss for random embeddings."""
        from chaoscontrol.alignment import contrastive_alignment
        torch.manual_seed(42)
        B, T, D, K = 4, 16, 32, 8
        projected = torch.randn(B, T, D)
        wernicke = torch.randn(K, D)
        assigns = torch.randint(0, K, (B, T))
        loss = contrastive_alignment(projected, wernicke, assigns)
        assert loss.item() > 0.0

    def test_contrastive_perfect_alignment(self):
        """Loss approaches 0 when tok embeds exactly match wernicke entries."""
        from chaoscontrol.alignment import contrastive_alignment
        K, D = 4, 16
        wernicke = torch.randn(K, D)
        # Each token embedding is exactly the wernicke entry it's assigned to
        B, T = 2, 8
        assigns = torch.randint(0, K, (B, T))
        projected = wernicke[assigns]  # (B, T, D) — perfect match
        loss = contrastive_alignment(projected, wernicke, assigns, temperature=0.1)
        # With perfect alignment, each bucket mean equals its wernicke entry,
        # so the positive similarity is maximal.  Loss should be very small.
        assert loss.item() < 0.1


class TestDiversityAlignment(unittest.TestCase):
    def test_diversity_high_when_similar(self):
        """Loss is higher when codebooks are similar."""
        from chaoscontrol.alignment import diversity_alignment
        torch.manual_seed(7)
        K, D = 8, 32
        tok = torch.randn(K, D)
        # Wernicke entries are near-copies of tokenizer entries
        wer_similar = tok + 0.01 * torch.randn(K, D)
        loss_similar = diversity_alignment(tok, wer_similar)

        # Wernicke entries are random (less correlated)
        wer_random = torch.randn(K, D)
        loss_random = diversity_alignment(tok, wer_random)

        assert loss_similar.item() > loss_random.item()

    def test_diversity_low_when_orthogonal(self):
        """Loss is lower when codebooks are orthogonal."""
        from chaoscontrol.alignment import diversity_alignment
        D = 32
        # Orthogonal codebook entries
        tok = torch.zeros(2, D)
        tok[0, :16] = 1.0
        tok[1, :16] = -1.0
        wer = torch.zeros(2, D)
        wer[0, 16:] = 1.0
        wer[1, 16:] = -1.0
        loss_ortho = diversity_alignment(tok, wer)
        # These are perfectly orthogonal, so cosine sim = 0
        assert loss_ortho.item() < 0.01

        # Similar codebooks
        loss_similar = diversity_alignment(tok, tok)
        assert loss_similar.item() > loss_ortho.item()


class TestDistillationAlignment(unittest.TestCase):
    def test_distillation_positive(self):
        """Non-zero loss for random embeddings."""
        from chaoscontrol.alignment import distillation_alignment
        torch.manual_seed(99)
        B, T, D, K = 4, 16, 32, 8
        projected = torch.randn(B, T, D)
        wernicke = torch.randn(K, D)
        assigns = torch.randint(0, K, (B, T))
        loss = distillation_alignment(projected, wernicke, assigns)
        assert loss.item() > 0.0


class TestDispatch(unittest.TestCase):
    def test_dispatch(self):
        """compute_alignment_loss dispatches correctly for all 4 types."""
        from chaoscontrol.alignment import compute_alignment_loss
        torch.manual_seed(0)
        B, T, D, K = 2, 8, 16, 4
        projected = torch.randn(B, T, D)
        wernicke = torch.randn(K, D)
        assigns = torch.randint(0, K, (B, T))
        tok_entries = torch.randn(K, D)

        # none
        loss_none = compute_alignment_loss("none")
        assert loss_none.item() == 0.0

        # contrastive
        loss_c = compute_alignment_loss(
            "contrastive",
            projected_tok_embeds=projected,
            wernicke_entries=wernicke,
            wernicke_assignments=assigns,
        )
        assert loss_c.shape == ()

        # diversity
        loss_d = compute_alignment_loss(
            "diversity",
            projected_tok_entries=tok_entries,
            wernicke_entries=wernicke,
        )
        assert loss_d.shape == ()

        # distillation
        loss_dist = compute_alignment_loss(
            "distillation",
            projected_tok_embeds=projected,
            wernicke_entries=wernicke,
            wernicke_assignments=assigns,
        )
        assert loss_dist.shape == ()

        # unknown type raises
        with self.assertRaises(ValueError):
            compute_alignment_loss("bogus")


if __name__ == "__main__":
    unittest.main()
