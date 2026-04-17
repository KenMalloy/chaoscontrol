import torch
from chaoscontrol.sgns.model import SGNSModel


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
