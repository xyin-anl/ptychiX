from typing import Sequence
import logging

import torch

from .core import squared_modulus, CorrectionPlan, DataProduct, DetectorData, IterativeAlgorithm
from .utilities import Device

logger = logging.getLogger(__name__)


class DifferenceMap(IterativeAlgorithm):

    def __init__(self, device: Device, detector_data: DetectorData, product: DataProduct) -> None:
        self._good_pixels = torch.logical_not(detector_data.bad_pixels).to(device.torch_device)
        self._diffraction_patterns = detector_data.diffraction_patterns.to(device.torch_device)
        self._positions_px = product.positions_px.to(device.torch_device)
        self._probe = product.probe[0].to(device.torch_device)  # TODO support OPR modes
        self._object = product.object_[0].to(device.torch_device)  # TODO support multislice
        self._propagators = [propagator.to(device) for propagator in product.propagators]

        self._iteration = 0
        # FIXME apply/use beta
        self._beta = 0.8  # FIXME 0.6 - 0.95, 1.0 = DM
        self._probe_power = 0.

    def set_beta(self, value: float) -> None:
        self._beta = value

    def get_beta(self) -> float:
        return self._beta

    def set_probe_power(self, value: float) -> None:
        self._probe_power = value

    def get_probe_power(self) -> float:
        return self._probe_power

    def iterate(self, plan: CorrectionPlan) -> Sequence[float]:
        number_of_positions = self._positions_px.shape[0]
        iteration_data_error = list()
        layer = 0  # TODO support multislice

        psi = self._probe.unsqueeze(0).repeat(number_of_positions, 1, 1, 1)

        for iteration in range(plan.number_of_iterations):
            data_error = 0.

            if self._probe_power > 0.:
                # calculate probe power correction
                propagated_probe = self._propagators[layer].propagate_forward(self._probe)
                propagated_probe_power = torch.sum(squared_modulus(propagated_probe))
                power_correction = torch.sqrt(self._probe_power / propagated_probe_power)

                # apply power correction
                self._probe = self._probe * power_correction
                self._object = self._object / power_correction

            for idx in range(number_of_positions):
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
                wavefield = self._propagators[layer].propagate_forward(2 * exit_wave - psi[idx])
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

                # update exit wave
                psi[idx] += corrected_exit_wave - exit_wave

            if plan.probe_correction.is_enabled(iteration):
                probe_update_upper = torch.zeros_like(self._probe)
                probe_update_lower = torch.zeros_like(self._probe)
                object_abssq = squared_modulus(self._object)
                object_conj = self._object.conj()

                for idx in range(number_of_positions):
                    probe_update_upper += object_patch_conj * psi[idx]  # FIXME
                    probe_update_lower += object_patch_abssq  # FIXME

                # FIXME orthogonalize probe
                self._probe = divide_where_nonzero(probe_update_upper, probe_update_lower)

            if plan.object_correction.is_enabled(iteration):
                object_update_upper = torch.zeros_like(self._object)
                object_update_lower = torch.zeros_like(self._object)
                probe_abssq = torch.sum(squared_modulus(self._probe), dim=-3)
                probe_conj = self._probe.conj()

                for idx in range(number_of_positions):
                    object_update_upper += probe_conj * psi[idx]  # FIXME update support region
                    object_update_lower += probe_abssq  # FIXME update support region

                self._object = divide_where_nonzero(object_update_upper, object_update_lower)

            # FIXME position correction

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
