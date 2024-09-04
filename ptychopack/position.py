import torch
from torch.fft import fft, fft2, ifft, fftshift


def correct_positions_serial_cross_correlation(im1, im2, scale):
    N, M = im1.shape

    # fourier transform images
    F = fft2(im1)
    G = fft2(im2)

    # start setting up cross-correlation
    FG = F * torch.conj(G)

    # set up spatial and frequency coordinates
    x = torch.linspace(-10. / scale, 10. / scale, 21).reshape((1, 21))
    y = torch.linspace(-10. / scale, 10. / scale, 21).reshape((21, 1))
    u = torch.linspace(-.5 + 1. / M * .5, .5 - 1. / M * .5, M).reshape((M, 1))
    v = torch.linspace(-.5 + 1. / N * .5, .5 - 1. / N * .5, N).reshape((1, N))

    # perform the inverse fourier transform to obtain zoomed-in correlation
    corr2 = torch.dot(torch.exp(2j * torch.pi * torch.dot(y, v)),
                      torch.dot(FG, torch.exp(2j * torch.pi * torch.dot(u, x))))

    # find the peak of the correlation
    xmax2 = torch.argmax(torch.max(torch.abs(corr2), axis=0))
    ymax2 = torch.argmax(torch.max(torch.abs(corr2), axis=1))

    shift = torch.array([x[0, xmax2], y[ymax2, 0]])

    return shift


def correct_positions_gradient(self, probe, object_, ind_dp, delta_psi):
    Ny, Nx = object_.shape
    kx = fftshift(torch.linspace(0, Nx - 1, Nx) * 1.0 / Nx - 0.5)
    ky = fftshift(torch.linspace(0, Ny - 1, Ny) * 1.0 / Ny - 0.5)
    [kX, kY] = torch.meshgrid(kx, ky)

    object_fx = fft(object_, axis=1)
    object_fy = fft(object_, axis=0)

    object_dx = ifft(object_fx * kX * 2j * pi, axis=1)
    object_dy = ifft(object_fy * kY * 2j * pi, axis=0)

    dx_OP = dx_O * probe
    shift_x = torch.sum(real(conj(dx_OP) * delta_psi)) / torch.sum(squared_modulus(dx_OP))

    dy_OP = dy_O * probe
    shift_y = torch.sum(real(conj(dy_OP) * delta_psi)) / torch.sum(squared_modulus(dy_OP))

    #update position
    ppY[ind_dp] = ppY[ind_dp] + shift_y * dx_y
    ppX[ind_dp] = ppX[ind_dp] + shift_x * dx_x

    #position = pi + pf = integer + fraction
    py_i = torch.round(ppY[ind_dp] / dx_y)
    py_f[ind_dp] = ppY[ind_dp] - py_i * dx_y
    px_i = torch.round(ppX[ind_dp] / dx_x)
    px_f[ind_dp] = ppX[ind_dp] - px_i * dx_x

    #calculate ROI indices in the whole fov
    ind_x_lb_s[ind_dp] = (px_i - torch.floor(N_roi / 2.0) + center_index_image).astype(torch.int)
    ind_x_ub_s[ind_dp] = (px_i + torch.ceil(N_roi / 2.0) + center_index_image).astype(torch.int)
    ind_y_lb_s[ind_dp] = (py_i - torch.floor(N_roi / 2.0) + center_index_image).astype(torch.int)
    ind_y_ub_s[ind_dp] = (py_i + torch.ceil(N_roi / 2.0) + center_index_image).astype(torch.int)
