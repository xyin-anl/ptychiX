import ptychointerim.maths as pmath

import torch


def test_orthgonalize_gs():
    x = torch.rand(2, 3, 4, 5)
    x = pmath.orthogonalize_gs(x, dim=(-2, -1))
    assert torch.sum(x[0, 0] * x[0, 1]) < 1e-5


def test_orthgonalize_svd():
    x = torch.rand(2, 3, 4, 5)
    x = pmath.orthogonalize_svd(x, dim=(-2, -1))
    assert torch.sum(x[0, 0] * x[0, 1]) < 1e-5
