from typing import Sequence
import logging

import torch

from .core import squared_modulus, CorrectionPlan, DataProduct, DetectorData, IterativeAlgorithm
from .position import correct_positions_serial_cross_correlation
from .utilities import Device

logger = logging.getLogger(__name__)


class PtychographicIterativeEngine(IterativeAlgorithm):

    def __init__(self, device: Device, detector_data: DetectorData, product: DataProduct) -> None:
        self._good_pixels = torch.logical_not(detector_data.bad_pixels).to(device.torch_device)
        self._diffraction_patterns = detector_data.diffraction_patterns.to(device.torch_device)
        self._positions_px = product.positions_px.to(device.torch_device)
        self._probe = product.probe[0].to(device.torch_device)  # TODO support OPR modes
        self._object = product.object_[0].to(device.torch_device)  # TODO support multislice
        self._propagators = [propagator.to(device) for propagator in product.propagators]

        self._iteration = 0
        self._object_relaxation = 1.
        self._alpha = 1.
        self._probe_power = 0.
        self._probe_relaxation = 1.
        self._beta = 1.

        self._pc_probe_threshold = 0.1
        self._pc_feedback = 50.

    def set_object_relaxation(self, value: float) -> None:
        self._object_relaxation = value

    def get_object_relaxation(self) -> float:
        return self._object_relaxation

    def set_alpha(self, value: float) -> None:
        self._alpha = value

    def get_alpha(self) -> float:
        return self._alpha

    def set_probe_power(self, value: float) -> None:
        self._probe_power = value

    def get_probe_power(self) -> float:
        return self._probe_power

    def set_probe_relaxation(self, value: float) -> None:
        self._probe_relaxation = value

    def get_probe_relaxation(self) -> float:
        return self._probe_relaxation

    def set_beta(self, value: float) -> None:
        self._beta = value

    def get_beta(self) -> float:
        return self._beta

    def set_pc_probe_threshold(self, value: float) -> None:
        self._pc_probe_threshold = value

    def get_pc_probe_threshold(self) -> float:
        return self._pc_probe_threshold

    def set_pc_feedback(self, value: float) -> None:
        self._pc_feedback = value

    def get_pc_feedback(self) -> float:
        return self._pc_feedback

    def correct_probe_power(self) -> None:
        layer = 0  # TODO support multislice
        # propagate probe to the detector plane
        propagated_probe = self._propagators[layer].propagate_forward(self._probe)
        # calculate probe power
        propagated_probe_power = torch.sum(squared_modulus(propagated_probe))
        # calculate power correction
        power_correction = torch.sqrt(self._probe_power / propagated_probe_power)
        # appply power correction
        self._probe = self._probe * power_correction

    def iterate(self, plan: CorrectionPlan) -> Sequence[float]:
        iteration_data_error = list()
        layer = 0  # TODO support multislice

        if plan.probe_power_correction.is_enabled(0):
            self.correct_probe_power()

        for iteration in range(plan.number_of_iterations):
            shuffled_indexes = torch.randperm(self._positions_px.shape[0])
            data_error = 0.

            for idx in shuffled_indexes:
                diffraction_pattern = self._diffraction_patterns[idx]

                # top left corner of object support
                xmin = self._positions_px[idx, -1] - self._probe.shape[-1] / 2
                ymin = self._positions_px[idx, -2] - self._probe.shape[-2] / 2

                # whole components (pixel indexes)
                xmin_wh = xmin.int()
                ymin_wh = ymin.int()

                # fractional (subpixel) components
                xmin_fr = xmin - xmin_wh
                ymin_fr = ymin - ymin_wh

                # bottom right corner of object patch support
                xmax_wh = xmin_wh + self._probe.shape[-1] + 1
                ymax_wh = ymin_wh + self._probe.shape[-2] + 1

                # extract patch support region from full object
                object_support = self._object[ymin_wh:ymax_wh, xmin_wh:xmax_wh]

                # reused quantities
                xmin_fr_c = 1. - xmin_fr
                ymin_fr_c = 1. - ymin_fr

                # barycentric interpolant weights
                weight00 = ymin_fr_c * xmin_fr_c
                weight01 = ymin_fr_c * xmin_fr
                weight10 = ymin_fr * xmin_fr_c
                weight11 = ymin_fr * xmin_fr

                # interpolate object support to extract patch
                object_patch = weight00 * object_support[:-1, :-1]
                object_patch = object_patch + weight01 * object_support[:-1, 1:]
                object_patch = object_patch + weight10 * object_support[1:, :-1]
                object_patch = object_patch + weight11 * object_support[1:, 1:]
                corrected_object_patch = object_patch.detach().clone()

                # exit wave is the outcome of the probe-object interation
                exit_wave = self._probe * object_patch
                # propagate exit wave to the detector plane
                wavefield = self._propagators[layer].propagate_forward(exit_wave)
                # propagated wavefield intensity is incoherent sum of mixed-state modes
                wavefield_intensity = torch.sum(squared_modulus(wavefield), dim=-3)

                # calculate data error
                intensity_diff = torch.abs(wavefield_intensity - diffraction_pattern)
                data_error += torch.mean(intensity_diff[self._good_pixels]).item()

                # intensity correction
                correctable_pixels = torch.logical_and(self._good_pixels, wavefield_intensity > 0.)
                corrected_wavefield = wavefield * torch.where(
                    correctable_pixels, torch.sqrt(diffraction_pattern / wavefield_intensity), 1.)

                # propagate corrected wavefield to object plane
                corrected_exit_wave = self._propagators[layer].propagate_backward(
                    corrected_wavefield)

                # probe and object updates depend on exit wave difference
                exit_wave_diff = corrected_exit_wave - exit_wave

                # FIXME pre-fftshift

                if plan.probe_power_correction.is_enabled(iteration):
                    self.correct_probe_power()

                if plan.object_correction.is_enabled(iteration):
                    probe_abssq = torch.sum(squared_modulus(self._probe), dim=-3)
                    object_update_upper = torch.sum(self._probe.conj() * exit_wave_diff, dim=-3)
                    object_update_lower = torch.lerp(probe_abssq, probe_abssq.max(), self._alpha)
                    object_update = object_update_upper / object_update_lower
                    object_update = object_update * self._object_relaxation
                    corrected_object_patch = corrected_object_patch + object_update

                    # update object support
                    object_support[:-1, :-1] += weight00 * object_update
                    object_support[:-1, 1:] += weight01 * object_update
                    object_support[1:, :-1] += weight10 * object_update
                    object_support[1:, 1:] += weight11 * object_update

                if plan.probe_correction.is_enabled(iteration):
                    object_abssq = squared_modulus(corrected_object_patch)
                    probe_update_upper = corrected_object_patch.conj() * exit_wave_diff
                    probe_update_lower = torch.lerp(object_abssq, object_abssq.max(), self._beta)
                    probe_update = probe_update_upper / probe_update_lower
                    probe_update = probe_update * self._probe_relaxation
                    # TODO orthogonalize, center
                    self._probe = self._probe + probe_update

                if plan.position_correction.is_enabled(iteration):
                    # indicate object pixels where the illumination significantlly
                    # contributes to the diffraction pattern
                    probe_amplitude = torch.sqrt(torch.sum(squared_modulus(self._probe),
                                                           dim=-3))  # FIXME
                    probe_amplitude_max = torch.max(probe_amplitude)
                    probe_mask = (probe_amplitude > self._pc_probe_threshold * probe_amplitude_max)

                    # mask low illumination regions
                    masked_object_patch = object_patch * probe_mask
                    masked_corrected_object_patch = corrected_object_patch * probe_mask

                    # use serial cross correlation to determine correcting shift
                    shift = correct_positions_serial_cross_correlation(
                        masked_corrected_object_patch, masked_object_patch, 1.)  # FIXME scale

                    # update position
                    self._positions_px[idx, :] += self._pc_feedback * shift

            iteration_data_error.append(data_error)
            self._iteration += 1
            logger.info(f"iteration={self._iteration} error={data_error}")

        return iteration_data_error

    def get_product(self) -> DataProduct:
        return DataProduct(
            self._positions_px.cpu(),
            torch.unsqueeze(self._probe.cpu(), 0),
            torch.unsqueeze(self._object.cpu(), 0),
            [propagator.cpu() for propagator in self._propagators],
        )
