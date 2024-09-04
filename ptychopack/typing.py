from typing import TypeAlias

import torch

DeviceType: TypeAlias = str | torch.device
BooleanTensor: TypeAlias = torch.Tensor
ComplexTensor: TypeAlias = torch.Tensor
RealTensor: TypeAlias = torch.Tensor
