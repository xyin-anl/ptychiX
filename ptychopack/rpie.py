from types import TracebackType
from typing import overload, Sequence

import torch

from .common import squared_modulus, CorrectionPlan, IterativeAlgorithm
from .data import DataProduct, DetectorData
from .position import correct_positions_cross_correlation


class PtychographicIterativeEngine(IterativeAlgorithm):

    def __init__(self, detector_data: DetectorData, product: DataProduct,
                 plan: CorrectionPlan) -> None:
        self._detector_data = detector_data
        self._product = product
        self._plan = plan

        self._iteration = 0
        # TODO look up good defaults
        self._object_damping = 1.
        self._object_alpha = 0.8
        self._probe_damping = 1.
        self._probe_alpha = 0.8

    def iterate(self, repeat: int = 1) -> Sequence[float]:
        # TODO transfer to device
        layer = 0
        good_pixels = torch.logical_not(self._detector_data.bad_pixels)
        positions_px = self._product.positions_px
        probe = self._product.probe[0]  # TODO support OPR modes
        object_ = self._product.object_[layer]  # TODO support multislice
        propagators = self._product.propagators
        iteration_data_error = list()

        exit_wave = torch.zeros_like(probe)
        wavefield = torch.zeros_like(probe)
        corrected_exit_wave = torch.zeros_like(probe)

        for it in range(repeat):
            is_correcting_positions = self._plan.is_position_correction_enabled(self._iteration)
            is_correcting_probe = self._plan.is_probe_correction_enabled(self._iteration)
            is_correcting_object = self._plan.is_object_correction_enabled(self._iteration)
            data_error = 0.

            for idx in torch.randperm(positions_px.shape[0]):
                diffraction_pattern = self._detector_data.diffraction_patterns[idx, :, :]

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
                ymax_wh = xmin_wh + probe.shape[-2] + 1

                # extract patch support region from full object
                object_support = object_[ymin_wh:ymax_wh, xmin_wh:xmax_wh]  # TODO copy or view?

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

                for imode in range(probe.shape[0]):
                    exit_wave[imode, :, :] = probe[imode, :, :] * object_patch
                    wavefield[imode, :, :] = propagators[layer].propagate_forward(
                        exit_wave[imode, :, :])

                wavefield_intensity = torch.sum(squared_modulus(wavefield), dim=0)
                intensity_diff = wavefield_intensity - diffraction_pattern
                data_error += torch.sum(intensity_diff[good_pixels]).numpy()[0]
                correctable_pixels = (good_pixels and wavefield_intensity > 0.)
                correction = torch.where(correctable_pixels,
                                         torch.sqrt(diffraction_pattern / wavefield_intensity), 1.)

                for imode in range(probe.shape[0]):
                    wavefield[imode, :, :] *= correction
                    corrected_exit_wave[imode, :, :] = propagators[layer].propagate_backward(
                        wavefield[imode, :, :])

                exit_wave_diff = exit_wave - corrected_exit_wave

                if is_correcting_object:
                    alpha = self._object_alpha
                    probe_abssq = torch.sum(squared_modulus(probe), dim=0)
                    probe_abssq_max = torch.max(probe_abssq)
                    object_update_upper = torch.zeros_like(object_patch)

                    for imode in range(probe.shape[0]):
                        object_update_upper += torch.conj(
                            probe[imode, :, :]) * exit_wave_diff[imode, :, :]

                    object_update_lower = (1 - alpha) * probe_abssq + alpha * probe_abssq_max
                    object_update = -self._object_damping * object_update_upper / object_update_lower

                    # update object support
                    object_support[:-1, :-1] += weight00 * object_update
                    object_support[:-1, 1:] += weight01 * object_update
                    object_support[1:, :-1] += weight10 * object_update
                    object_support[1:, 1:] += weight11 * object_update

                    # overwrite support region in full object
                    object_[ymin_wh:ymax_wh, xmin_wh:xmax_wh] = object_support

                if is_correcting_probe:
                    alpha = self._probe_alpha
                    object_abssq = squared_modulus(object_)
                    object_abssq_max = torch.max(object_abssq)

                    # vvv FIXME vvv
                    step_size_upper = torch.tensor(self._probe_step_size)
                    step_size_lower = 2 * torch.max(squared_modulus(object_patch))
                    step_size = step_size_upper / step_size_lower

                    exit_wave_diff = exit_wave - corrected_exit_wave
                    probe -= step_size * torch.conj(object_patch) * exit_wave_diff

                    for imode in range(probe.shape[0]):
                        probe_update_upper = torch.conj(object_) * exit_wave_diff
                        probe_update_lower = 0.  # FIXME XXX
                        probe_update = probe_update_upper / probe_update_lower
                        # FIXME save/update

                if is_correcting_positions:
                    # FIXME object layers? probe modes?
                    # calculate next guess for coordinates
                    # find probe threshold mask at 10%
                    mask = torch.abs(probeSubShift) > 0.1 * probeDivide

                    # mask off object guesses
                    subObject = subObject * mask
                    subObjectNew = subObjectNew * mask

                    # do subpixel cross-correlation between new and old guesses
                    shift = correct_positions_cross_correlation(subObjectNew, subObject,
                                                                self.scale)

                    # get shift in pixels
                    shift = self.gamma * shift

                    # update current positions
                    positions_px[idx, :] -= shift

            iteration_data_error.append(data_error)

            self._product = DataProduct(
                positions_px,
                torch.unsqueeze(probe, 0),
                torch.unsqueeze(object_, 0),
                propagators,
            )
            self._iteration += 1

        return iteration_data_error

    def get_product(self) -> DataProduct:
        return self._product

# FIXME BEGIN

    def adjust_pc(self):
        # check scale
        if (torch.max(torch.abs(self.shiftXNew)) <= 2 * self.gamma / self.scale
                and torch.max(torch.abs(self.shiftYNew)) <= 2 * self.gamma / self.scale):
            self.scale *= 10
        elif (torch.mean(torch.abs(self.shiftXNew)) >= 5 * self.gamma / self.scale
              or torch.mean(torch.abs(self.shiftYNew)) >= 5 * self.gamma / self.scale):
            self.scale /= 10

        # check gamma
        shiftXCorr = signal.correlate(self.shiftXNew, self.shiftXOld, mode='valid')
        shiftYCorr = signal.correlate(self.shiftYNew, self.shiftYOld, mode='valid')

        aveCorr = (shiftXCorr + shiftYCorr) / 2

        # adjust gamma depending on the sign of the correlation
        if aveCorr < 0:
            self.gamma *= 0.9
        elif aveCorr > 0.3:
            self.gamma *= 1.1

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

        if self.gamma > 0:
            ave_corr = self.adjust_pc()
        else:
            ave_corr = 0

        metadata = {'error': errorOut, 'ave_corr': ave_corr}

        return metadata


# FIXME END
