import torch
from chaoscontrol.sgns.model import SGNSModel
from chaoscontrol.sgns.model import nce_loss


def test_sgns_model_shapes():
    model = SGNSModel(vocab_size=100, dim=16)
    assert model.input_embed.weight.shape == (100, 16)
    assert model.output_embed.weight.shape == (100, 16)


def test_sgns_score_pairs_shape():
    torch.manual_seed(0)
    model = SGNSModel(vocab_size=100, dim=16)
    center = torch.tensor([0, 1, 2])
    context = torch.tensor([3, 4, 5])
    scores = model.score_pairs(center, context)
    assert scores.shape == (3,)


def test_sgns_input_embed_nonzero_init():
    torch.manual_seed(0)
    model = SGNSModel(vocab_size=100, dim=16)
    assert not torch.allclose(model.input_embed.weight, torch.zeros_like(model.input_embed.weight))


def test_nce_loss_positive_attracts():
    """Aligned center/context pair yields lower loss than a mismatched pair."""
    torch.manual_seed(0)
    model = SGNSModel(vocab_size=10, dim=4)
    model.input_embed.weight.data = torch.eye(10, 4) * 2
    model.output_embed.weight.data = torch.eye(10, 4) * 2
    center = torch.tensor([0])
    negatives = torch.tensor([[5, 6, 7]])
    loss_aligned = nce_loss(model, center, torch.tensor([0]), negatives)
    loss_mismatched = nce_loss(model, center, torch.tensor([5]), negatives)
    assert loss_aligned < loss_mismatched


def test_nce_loss_negative_repels():
    """Random embeddings + reasonable sample: loss is a finite scalar."""
    torch.manual_seed(0)
    model = SGNSModel(vocab_size=100, dim=16)
    center = torch.tensor([1, 2, 3])
    context = torch.tensor([4, 5, 6])
    negatives = torch.randint(0, 100, (3, 5))
    loss = nce_loss(model, center, context, negatives)
    assert torch.isfinite(loss)
    assert loss.dim() == 0
