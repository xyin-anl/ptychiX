from typing import Sequence

import torch

from .algorithm import squared_modulus, CorrectionPlan, IterativeAlgorithm
from .data import DataProduct, DetectorData
from .position import correct_positions_serial_cross_correlation


class PtychographicIterativeEngine(IterativeAlgorithm):

    def __init__(self, detector_data: DetectorData, product: DataProduct) -> None:
        self._detector_data = detector_data
        self._product = product

        self._iteration = 0
        self._object_relaxation = 1.
        self._alpha = 0.05
        self._probe_relaxation = 1.
        self._beta = 1.

        self._pc_probe_threshold = 0.1
        self._pc_feedback = 0.  # FIXME

    def iterate(self, plan: CorrectionPlan) -> Sequence[float]:
        # TODO transfer to device
        layer = 0
        good_pixels = torch.logical_not(self._detector_data.bad_pixels)
        positions_px = self._product.positions_px
        probe = self._product.probe[0]  # TODO support OPR modes
        object_ = self._product.object_[layer]  # TODO support multislice
        propagators = self._product.propagators
        iteration_data_error = list()

        for iteration in range(plan.number_of_iterations):
            data_error = 0.

            for idx in torch.randperm(positions_px.shape[0]):
                diffraction_pattern = self._detector_data.diffraction_patterns[idx]

                # top left corner of object support
                xmin = positions_px[idx, -1] - probe.shape[-1] / 2
                ymin = positions_px[idx, -2] - probe.shape[-2] / 2

                # whole components (pixel indexes)
                xmin_wh = int(xmin)
                ymin_wh = int(ymin)

                # fractional (subpixel) components
                xmin_fr = xmin - xmin_wh
                ymin_fr = ymin - ymin_wh

                # bottom right corner of object patch support
                xmax_wh = xmin_wh + probe.shape[-1] + 1
                ymax_wh = ymin_wh + probe.shape[-2] + 1

                # extract patch support region from full object
                object_support = object_[ymin_wh:ymax_wh, xmin_wh:xmax_wh].detach().clone()

                # reused quantities
                xmin_fr_c = 1. - xmin_fr
                ymin_fr_c = 1. - ymin_fr

                # barycentric interpolant weights
                weight00 = ymin_fr_c * xmin_fr_c
                weight01 = ymin_fr_c * xmin_fr
                weight10 = ymin_fr * xmin_fr_c
                weight11 = ymin_fr * xmin_fr

                # interpolate object support to obtain patch
                object_patch = weight00 * object_support[:-1, :-1]
                object_patch += weight01 * object_support[:-1, 1:]
                object_patch += weight10 * object_support[1:, :-1]
                object_patch += weight11 * object_support[1:, 1:]
                corrected_object_patch = object_patch.detach().clone()
                wavefield = torch.zeros_like(probe)

                for imode in range(probe.shape[0]):
                    # exit wave is the outcome of the probe-object interation
                    exit_wave = probe[imode] * object_patch
                    # propagate exit wave to the detector plane
                    wavefield[imode] = propagators[layer].propagate_forward(exit_wave)

                # propagated wavefield intensity is incoherent sum of mixed-state modes
                wavefield_intensity = torch.sum(squared_modulus(wavefield), dim=0)

                # calculate data error (TODO: normalize)
                intensity_diff = wavefield_intensity - diffraction_pattern
                data_error += torch.mean(intensity_diff[good_pixels]).numpy()

                # intensity correction coefficient
                correctable_pixels = torch.logical_and(good_pixels, wavefield_intensity > 0.)
                correction = torch.where(correctable_pixels,
                                         torch.sqrt(diffraction_pattern / wavefield_intensity), 1.)
                exit_wave_diff = torch.zeros_like(probe)

                for imode in range(probe.shape[0]):
                    # exit wave is the outcome of the probe-object interation
                    exit_wave = probe[imode] * object_patch

                    # propagate corrected wavefield to object plane
                    corrected_exit_wave = propagators[layer].propagate_backward(correction *
                                                                                wavefield[imode])

                    # probe and object updates depend on exit wave difference
                    exit_wave_diff[imode] = corrected_exit_wave - exit_wave

                if plan.object_correction.is_enabled(iteration):
                    object_update_upper = torch.zeros_like(object_patch)

                    for imode in range(probe.shape[0]):
                        object_update_upper += torch.conj(probe[imode]) * exit_wave_diff[imode]

                    probe_abssq = torch.sum(squared_modulus(probe), dim=0)
                    object_update_lower = torch.lerp(probe_abssq, torch.max(probe_abssq),
                                                     self._alpha)
                    object_update = object_update_upper / object_update_lower
                    object_update *= self._object_relaxation
                    corrected_object_patch += object_update

                    # update object support
                    object_support[:-1, :-1] += weight00 * object_update
                    object_support[:-1, 1:] += weight01 * object_update
                    object_support[1:, :-1] += weight10 * object_update
                    object_support[1:, 1:] += weight11 * object_update

                    # overwrite support region in full object
                    object_[ymin_wh:ymax_wh, xmin_wh:xmax_wh] = object_support

                if plan.probe_correction.is_enabled(iteration):
                    object_abssq = squared_modulus(corrected_object_patch)
                    probe_update_lower = torch.lerp(object_abssq, torch.max(object_abssq),
                                                    self._beta)

                    # FIXME orthogonalize

                    for imode in range(probe.shape[0]):
                        probe_update_upper = torch.conj(
                            corrected_object_patch) * exit_wave_diff[imode]
                        probe_update = probe_update_upper / probe_update_lower
                        probe_update *= self._probe_relaxation
                        probe[imode] += probe_update

                if plan.position_correction.is_enabled(iteration):
                    # indicate object pixels where the illumination significantlly
                    # contributes to the diffraction pattern
                    probe_amplitude = torch.sqrt(torch.sum(squared_modulus(probe), dim=0))
                    probe_amplitude_max = torch.max(probe_amplitude)
                    probe_mask = (probe_amplitude > self._pc_probe_threshold * probe_amplitude_max)

                    # mask low illumination regions
                    masked_object_patch = object_patch * probe_mask
                    masked_corrected_object_patch = corrected_object_patch * probe_mask

                    # use serial cross correlation to determine correcting shift
                    shift = correct_positions_serial_cross_correlation(
                        masked_corrected_object_patch, masked_object_patch, 1.)  # FIXME scale

                    # update position (FIXME verify sign)
                    positions_px[idx, :] += self._pc_feedback * shift

            iteration_data_error.append(data_error)

            self._product = DataProduct(
                positions_px,
                torch.unsqueeze(probe, 0),
                torch.unsqueeze(object_, 0),
                propagators,
            )
            self._iteration += 1
            print(f'{self._iteration} -> {data_error}')  # FIXME

        return iteration_data_error

    def get_product(self) -> DataProduct:
        return self._product

# FIXME BEGIN

    def adjust_pc(self):
        # check scale
        if (torch.max(torch.abs(self.shiftXNew)) <= 2 * self._pc_feedback / self.scale
                and torch.max(torch.abs(self.shiftYNew)) <= 2 * self._pc_feedback / self.scale):
            self.scale *= 10
        elif (torch.mean(torch.abs(self.shiftXNew)) >= 5 * self._pc_feedback / self.scale
              or torch.mean(torch.abs(self.shiftYNew)) >= 5 * self._pc_feedback / self.scale):
            self.scale /= 10

        # check gamma
        shiftXCorr = signal.correlate(self.shiftXNew, self.shiftXOld, mode='valid')
        shiftYCorr = signal.correlate(self.shiftYNew, self.shiftYOld, mode='valid')

        aveCorr = (shiftXCorr + shiftYCorr) / 2

        # adjust gamma depending on the sign of the correlation
        if aveCorr < 0:
            self._pc_feedback *= 0.9
        elif aveCorr > 0.3:
            self._pc_feedback *= 1.1

        return aveCorr

    def fullePIEstep(self, indexOrder):
        errorOut = 0
        oldx = torch.copy(self.xcoords)
        oldy = torch.copy(self.ycoords)
        self.shiftXOld = torch.copy(self.shiftXNew)
        self.shiftYOld = torch.copy(self.shiftYNew)

        for i in range(self.P):

            errorTemp = self.ptychography_step(indexOrder[i])

            errorOut += errorTemp

        self.shiftXNew = self.xcoords - oldx
        self.shiftYNew = self.ycoords - oldy

        if self._pc_feedback > 0:
            ave_corr = self.adjust_pc()
        else:
            ave_corr = 0

        metadata = {'error': errorOut, 'ave_corr': ave_corr}

        return metadata


# FIXME END
