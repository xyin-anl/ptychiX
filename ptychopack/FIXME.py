import numpy


def shift(input, dx, px, py):
    N_image = input.shape[0]

    dk = 1.0 / (dx * N_image)
    kx = linspace(-floor(N_image / 2.0), ceil(N_image / 2.0) - 1, N_image)
    [kX, kY] = meshgrid(kx, kx)
    kX = kX * dk
    kY = kY * dk

    f = fftshift(fft2(ifftshift(input)))
    f = f * exp(-2 * pi * 1j * px * kX) * exp(-2 * pi * 1j * py * kY)
    f = fftshift(ifft2(ifftshift(f)))

    return f


def recenter(input):
    N = input.shape[0]
    center_index = N // 2
    [yy, xx] = where(abs(input) == numpy.max(abs(input)))

    output = roll(input, int(-(yy[0] - center_index)), axis=0)
    output = roll(output, int(-(xx[0] - center_index)), axis=1)

    return output


def calculateUpdate(self, psi, delta_psi, type):
    psi_mag = abs(psi)**2
    if type == 'o':  #calculate update for object
        w = self.alpha
    elif type == 'p':  #calculate update for probe/psi
        w = self.beta
    else:
        raise RuntimeError('Invalid input!')

    return w * conj(psi) / amax(psi_mag) * delta_psi


def calculateMixedStatesUpdate(self, psi, delta_psi, type):
    psi_tot = sum(abs(psi)**2, axis=0)
    #psi_mag = abs(psi)**2
    if type == 'o':  #calculate update for object
        w = self.alpha
    elif type == 'p':  #calculate update for probe/psi
        w = self.beta
    else:
        raise RuntimeError('Invalid input!')

    return w * sum(conj(psi) * delta_psi, axis=0) / amax(psi_tot)


def shiftProb(self, probe, ind_dp, direction, checkFilter=False):
    if direction == 'toScanPosition':  #from origin to scan position
        px = self.px_f[ind_dp]
        py = self.py_f[ind_dp]
    elif direction == 'toOrigin':  #from scan position back to origin
        px = -self.px_f[ind_dp]
        py = -self.py_f[ind_dp]
    else:
        raise RuntimeError('Invalid input!')

    self.r[:, :] = probe
    self.fft_forward.update_arrays(self.r, self.f)
    self.fft_forward.execute()
    self.f = self.f * exp(-2 * pi * 1j * px * self.kX) * exp(-2 * pi * 1j * py * self.kY)
    if checkFilter and 'filter_f_probe' in self.paraDict:
        self.f = self.f * self.paraDict['filter_f_probe']
    self.fft_inverse.update_arrays(self.f, self.r)
    self.fft_inverse.execute()
    return self.r / self.N_tot  #fix normalization


def updateFourierIntensity(self, psi_f, dp, denominator):
    #psi_f: wave function in Fourier space
    self.f[:, :] = psi_f
    f_cbed = self.f[self.ind_dp_lb:self.ind_dp_ub, self.ind_dp_lb:self.ind_dp_ub]

    f_cbed[self.paraDict['badPixels'] == 0] = f_cbed[self.paraDict['badPixels'] == 0] / (
        denominator[self.paraDict['badPixels'] == 0] + 1e-16) * dp[self.paraDict['badPixels'] == 0]
    self.f[self.ind_dp_lb:self.ind_dp_ub, self.ind_dp_lb:self.ind_dp_ub] = f_cbed
    self.f = ifftshift(self.f)
    if 'filter_f_psi' in self.paraDict: self.f = self.f * self.paraDict['filter_f_psi']
    self.fft_inverse.update_arrays(self.f, self.r)
    self.fft_inverse.execute()
    return self.r / self.N_tot


def reconPIE_mixed_state(dp, paraDict):
    for k in range(startNiter, Niter):
        probes_temp = gramschmidt(probes.reshape(N_probe, N_roi * N_roi))
        probes[:, :, :] = probes_temp.reshape(N_probe, N_roi, N_roi)

        update_order = random.permutation(paraDict['N_scan'])  #random order

        for i in update_order:
            O_old = auxiFunc.getObjectROI(O, i)
            for p in range(N_probe_recon):
                probes_shifted[p, :, :] = auxiFunc.shiftProb(probes[p, :, :], i, 'toScanPosition')

                #overlap projection
                psi[p, :, :] = O_old * probes_shifted[p, :, :]

                #Fourier projection
                psi_old[p, :, :] = psi[p, :, :]
                psi[p, :, :], cbed_region_mag[p, :, :] = auxiFunc.FFTpsi(psi[p, :, :])

            psi_f_mag_tot = sqrt(torch.sum(cbed_region_mag**2, axis=0))
            dp_error[i] = torch.sum((psi_f_mag_tot - dp[i, :, :])**2)

            #Fourier projection
            for p in range(N_probe_recon):
                psi[p, :, :] = auxiFunc.updateFourierIntensity(psi[p, :, :], dp[i, :, :],
                                                               psi_f_mag_tot)
                if 'filter_r_psi' in paraDict:
                    psi[p, :, :] = psi[p, :, :] * paraDict['filter_r_psi']

            delta_psi = psi - psi_old
            probe_old = probes_shifted[:, :, :]
            O_update = auxiFunc.calculateMixedStatesUpdate(probe_old, delta_psi, 'o')
            auxiFunc.updateObj(O, O_update, i)

            if k >= paraDict['Niter_update_probe']:
                O_tot_max = amax(abs(O_old)**2)
                for p in range(N_probe_recon):
                    probe_update = auxiFunc.calculateUpdate(O_old, delta_psi[p, :, :], 'p')
                    probes_shifted[p, :, :] += probe_update
                    probes[p, :, :] = auxiFunc.shiftProb(probes_shifted[p, :, :],
                                                         i,
                                                         'toOrigin',
                                                         checkFilter=True)

            if k >= paraDict['Niter_update_position']:
                auxiFunc.gradPositionCorrection(probe_old[0, :, :], O_old, i, delta_psi[0, :, :])

        ############################### orthogonalise probe #######################################
        if k >= paraDict['Niter_update_states']:
            probes_temp = gramschmidt(probes.reshape(N_probe, N_roi * N_roi))
            probes[:, :, :] = probes_temp.reshape(N_probe, N_roi, N_roi)

        ############################### calcuate data error #######################################
        s[k] = torch.sum(dp_error) / dp_tot
        dp_error_old = dp_error.copy()


### normalize probe ###
probe = probe0 * sqrt(torch.sum(dp_avg) / torch.sum(squared_modulus(probe0)) / N_tot)
