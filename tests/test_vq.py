"""Tests for vector quantization utilities."""
import unittest
import torch


class TestVectorQuantize(unittest.TestCase):
    def test_quantize_shape(self):
        from chaoscontrol.vq import vector_quantize
        x = torch.randn(2, 16, 32)
        codebook = torch.randn(64, 32)
        quantized, indices, commit_loss = vector_quantize(x, codebook)
        assert quantized.shape == (2, 16, 32)
        assert indices.shape == (2, 16)
        assert indices.dtype == torch.long
        assert commit_loss.shape == ()

    def test_straight_through_gradient(self):
        from chaoscontrol.vq import vector_quantize
        x = torch.randn(2, 8, 16, requires_grad=True)
        codebook = torch.randn(32, 16)
        quantized, _, _ = vector_quantize(x, codebook)
        loss = quantized.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_indices_are_nearest_neighbor(self):
        from chaoscontrol.vq import vector_quantize
        codebook = torch.eye(4)
        x = torch.tensor([[[0.9, 0.1, 0.0, 0.0]]])
        _, indices, _ = vector_quantize(x, codebook)
        assert indices[0, 0] == 0

    def test_commitment_loss_is_positive(self):
        from chaoscontrol.vq import vector_quantize
        x = torch.randn(2, 8, 16)
        codebook = torch.randn(32, 16)
        _, _, commit_loss = vector_quantize(x, codebook)
        assert commit_loss.item() > 0

    def test_codebook_gradient(self):
        """Codebook should receive gradients through the commitment loss."""
        from chaoscontrol.vq import vector_quantize
        x = torch.randn(2, 8, 16)
        codebook = torch.randn(32, 16, requires_grad=True)
        _, _, commit_loss = vector_quantize(x, codebook)
        commit_loss.backward()
        assert codebook.grad is not None


if __name__ == "__main__":
    unittest.main()
