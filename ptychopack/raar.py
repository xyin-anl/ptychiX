from typing import Sequence
import logging

import torch

from .core import (squared_modulus, CorrectionPlan, DataProduct, DetectorData, IterativeAlgorithm,
                   ObjectPatchInterpolator)
from .utilities import ComplexTensor, Device

logger = logging.getLogger(__name__)


def divide_where_nonzero(upper: ComplexTensor, lower: ComplexTensor) -> ComplexTensor:
    return torch.where(lower.abs() > 1e-10, upper / lower, 0.)


class RelaxedAveragedAlternatingReflections(IterativeAlgorithm):

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
                interpolator = ObjectPatchInterpolator(self._object, self._positions_px[idx],
                                                       self._probe.size())
                object_patch = interpolator.get_patch()

                # exit wave is the outcome of the probe-object interation
                exit_wave = self._probe * object_patch
                # FIXME BEGIN: UPDATE FOR RAAR
                # propagate exit wave to the detector plane
                wavefield = self._propagators[layer].propagate_forward(2 * exit_wave - psi[idx])
                # propagated wavefield intensity is incoherent sum of mixed-state modes
                wavefield_intensity = torch.sum(squared_modulus(wavefield), dim=-3)

                # calculate data error
                diffraction_pattern = self._diffraction_patterns[idx]
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
                # FIXME END

            if plan.probe_correction.is_enabled(iteration):
                probe_upper = torch.zeros_like(self._probe)
                probe_lower = torch.zeros(self._probe.shape[1:])

                for idx in range(number_of_positions):
                    interpolator = ObjectPatchInterpolator(self._object, self._positions_px[idx],
                                                           self._probe.size())
                    object_patch = interpolator.get_patch()

                    probe_upper += object_patch.conj() * psi[idx]
                    probe_lower += squared_modulus(object_patch)

                # FIXME orthogonalize probe
                self._probe = divide_where_nonzero(probe_upper, probe_lower)

            if plan.object_correction.is_enabled(iteration):
                object_upper = torch.zeros_like(self._object)
                object_lower = torch.zeros(self._object.size())

                for idx in range(number_of_positions):
                    interpolator_upper = ObjectPatchInterpolator(object_upper,
                                                                 self._positions_px[idx],
                                                                 self._probe.size())
                    interpolator_upper.update_patch(
                        torch.sum(self._probe.conj() * psi[idx], dim=-3))

                    interpolator_lower = ObjectPatchInterpolator(object_lower,
                                                                 self._positions_px[idx],
                                                                 self._probe.size())
                    interpolator_lower.update_patch(torch.sum(squared_modulus(self._probe),
                                                              dim=-3))

                self._object = divide_where_nonzero(object_upper, object_lower)

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
