"""Tests for FixedStrideTokenizer."""
import unittest
import torch


class TestFixedStrideTokenizer(unittest.TestCase):
    def _make_tokenizer(self, **kwargs):
        from chaoscontrol.tokenizer import FixedStrideTokenizer
        return FixedStrideTokenizer(**kwargs)

    def test_output_shapes(self):
        """Check token_embeds, token_ids shapes for known input."""
        tok = self._make_tokenizer(
            byte_dim=32, token_dim=64, codebook_size=128, stride=4,
        )
        batch, byte_seq = 2, 64
        byte_ids = torch.randint(0, 256, (batch, byte_seq))
        out = tok(byte_ids)

        token_seq = byte_seq // 4  # stride=4
        assert out["token_embeds"].shape == (batch, token_seq, 64)
        assert out["token_ids"].shape == (batch, token_seq)
        assert out["token_ids"].dtype == torch.long
        assert out["commit_loss"].shape == ()
        assert out["recon_loss"].shape == ()
        assert out["codebook"] is tok.codebook

    def test_stride_reduces_sequence(self):
        """token_seq should equal byte_seq // stride."""
        for stride in [2, 4, 8]:
            tok = self._make_tokenizer(
                byte_dim=16, token_dim=32, codebook_size=64, stride=stride,
            )
            byte_seq = 128
            byte_ids = torch.randint(0, 256, (1, byte_seq))
            out = tok(byte_ids)
            expected_token_seq = byte_seq // stride
            assert out["token_embeds"].shape[1] == expected_token_seq, (
                f"stride={stride}: expected {expected_token_seq}, "
                f"got {out['token_embeds'].shape[1]}"
            )

    def test_reconstruction_loss_positive(self):
        """Reconstruction loss should be > 0 for random input."""
        tok = self._make_tokenizer(
            byte_dim=16, token_dim=32, codebook_size=64, stride=4,
        )
        byte_ids = torch.randint(0, 256, (2, 64))
        out = tok(byte_ids)
        assert out["recon_loss"].item() > 0

    def test_commit_loss_positive(self):
        """Commitment loss should be > 0 for random input."""
        tok = self._make_tokenizer(
            byte_dim=16, token_dim=32, codebook_size=64, stride=4,
        )
        byte_ids = torch.randint(0, 256, (2, 64))
        out = tok(byte_ids)
        assert out["commit_loss"].item() > 0

    def test_gradients_flow(self):
        """Backward through commit_loss + recon_loss updates encoder and decoder."""
        tok = self._make_tokenizer(
            byte_dim=16, token_dim=32, codebook_size=64, stride=4,
        )
        byte_ids = torch.randint(0, 256, (2, 64))
        out = tok(byte_ids)
        loss = out["commit_loss"] + out["recon_loss"]
        loss.backward()

        # Encoder (downsample conv) should receive gradients
        assert tok.downsample.weight.grad is not None
        assert tok.downsample.weight.grad.abs().sum() > 0

        # Decoder (transposed conv) should receive gradients
        assert tok.decoder.weight.grad is not None
        assert tok.decoder.weight.grad.abs().sum() > 0

        # Byte embedding should receive gradients
        assert tok.byte_embed.weight.grad is not None
        assert tok.byte_embed.weight.grad.abs().sum() > 0

    def test_causal_conv(self):
        """Changing a future byte must not affect earlier token embeddings."""
        tok = self._make_tokenizer(
            byte_dim=16, token_dim=32, codebook_size=64, stride=4,
        )
        tok.eval()

        byte_ids = torch.randint(0, 256, (1, 64))

        with torch.no_grad():
            out1 = tok(byte_ids)
            embeds1 = out1["token_embeds"].clone()

            # Mutate the last 4 bytes (affects only the last token at most)
            byte_ids_mut = byte_ids.clone()
            byte_ids_mut[0, -4:] = (byte_ids_mut[0, -4:] + 1) % 256
            out2 = tok(byte_ids_mut)
            embeds2 = out2["token_embeds"].clone()

        # All tokens except the last must be identical
        n_tokens = embeds1.shape[1]
        if n_tokens > 1:
            assert torch.equal(embeds1[0, :-1], embeds2[0, :-1]), (
                "Changing future bytes affected earlier token embeddings — "
                "conv is not causal"
            )


if __name__ == "__main__":
    unittest.main()
