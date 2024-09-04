import torch

from .algorithm import squared_modulus
from .typing import ComplexTensor


def proj(u, v):
    return u * torch.vdot(u, v) / torch.vdot(u, u)


def gram_schmidt(V):
    U = torch.copy(V)
    for i in range(1, V.shape[0]):
        for j in range(i):
            U[i, :] -= proj(U[j, :], V[i, :])
    return U


def orthogonalize_probe(self, probes: ComplexTensor) -> ComplexTensor:
    probes_temp = gram_schmidt(probes.reshape(N_probe, N_roi**2))
    probes[:, :, :] = probes_temp.reshape(N_probe, N_roi, N_roi)
    #sort probes based on power
    power = torch.sum(squared_modulus(probes), dim=(-2, -1))
    power_ind = torch.argsort(-power)
    probes[:, :, :] = probes[power_ind, :, :]
    return probes
