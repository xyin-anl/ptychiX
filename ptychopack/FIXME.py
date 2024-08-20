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
