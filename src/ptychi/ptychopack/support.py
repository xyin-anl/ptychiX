import torch

from .api import ComplexTensor, RealTensor


def squared_modulus(values: ComplexTensor) -> RealTensor:
    return torch.abs(values) ** 2
