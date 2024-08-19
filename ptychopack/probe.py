def center_probe():  # TODO
    pass


def proj(u, v):
    return u * torch.vdot(u, v) / torch.vdot(u, u)


def gramschmidt(V):
    U = torch.copy(V)
    for i in range(1, V.shape[0]):
        for j in range(i):
            U[i, :] -= proj(U[j, :], V[i, :])
    return U


def orthoProbe(self, probes):
    probes_temp = gramschmidt(probes.reshape(paraDict['N_probe'], paraDict['N_roi']**2))
    probes[:, :, :] = probes_temp.reshape(paraDict['N_probe'], paraDict['N_roi'],
                                          paraDict['N_roi'])
    #sort probes based on power
    power = sum(abs(probes)**2, axis=(1, 2))
    power_ind = argsort(-power)
    probes[:, :, :] = probes[power_ind, :, :]
    return probes
