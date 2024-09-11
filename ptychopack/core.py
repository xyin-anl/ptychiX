from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .propagate import FourierPropagator, WavefieldPropagator
from .utilities import BooleanTensor, ComplexTensor, RealTensor


def squared_modulus(values: ComplexTensor) -> RealTensor:
    return torch.abs(values)**2


@dataclass(frozen=True)
class CorrectionPlanElement:
    start: int
    stop: int
    stride: int

    def is_enabled(self, iteration: int) -> bool:
        if self.start <= iteration and iteration < self.stop:
            return ((iteration - self.start) % self.stride == 0)

        return False


@dataclass(frozen=True)
class CorrectionPlan:
    object_correction: CorrectionPlanElement
    probe_correction: CorrectionPlanElement
    position_correction: CorrectionPlanElement

    @classmethod
    def create_simple(
        cls,
        num_iterations: int,
        *,
        correct_object: bool = False,
        correct_probe: bool = False,
        correct_positions: bool = False,
    ) -> CorrectionPlan:
        object_correction = CorrectionPlanElement(
            start=0,
            stop=num_iterations if correct_object else 0,
            stride=1,
        )
        probe_correction = CorrectionPlanElement(
            start=0,
            stop=num_iterations if correct_probe else 0,
            stride=1,
        )
        position_correction = CorrectionPlanElement(
            start=0,
            stop=num_iterations if correct_positions else 0,
            stride=1,
        )
        return cls(object_correction, probe_correction, position_correction)

    @property
    def number_of_iterations(self) -> int:
        return max(
            self.object_correction.stop,
            self.probe_correction.stop,
            self.position_correction.stop,
        )


@dataclass(frozen=True)
class DetectorData:
    diffraction_patterns: RealTensor  # [N, H, W]
    """diffraction patterns"""
    bad_pixels: BooleanTensor  # [H, W]
    """bad pixel mask to exclude detector dead regions or saturated pixels"""

    @classmethod
    def create_simple(cls,
                      diffraction_patterns: RealTensor,
                      bad_pixels: BooleanTensor | None = None) -> DetectorData:
        return cls(
            torch.fft.ifftshift(diffraction_patterns, dim=(-2, -1)),
            torch.full(diffraction_patterns.shape[1:], False)
            if bad_pixels is None else bad_pixels,
        )


@dataclass(frozen=True)
class DataProduct:
    positions_px: RealTensor  # [N, 2]
    """positions are in pixel units"""
    probe: ComplexTensor  # [C, I, H, W]
    """probe shape is the number of coherent (OPR) modes, number of incoherent
    (mixed-states) modes, height in pixels, and width in pixels"""
    object_: ComplexTensor  # [L, H, W]
    """object shape is the number of (multi-slice) layers, height in pixels,
    and width in pixels"""
    propagators: Sequence[WavefieldPropagator]  # [L]
    """sequence of wavefield propagators, one for each object layer"""

    @classmethod
    def create_simple(
        cls,
        positions_px: RealTensor,
        probe: ComplexTensor,
        object_: ComplexTensor,
    ) -> DataProduct:
        propagators = [FourierPropagator()]
        return cls(positions_px, probe, object_, propagators)


class ObjectPatchInterpolator:

    def __init__(self, object_: ComplexTensor, position_px: RealTensor, size: torch.Size) -> None:
        # top left corner of object support
        xmin = position_px[-1] - size[-1] / 2
        ymin = position_px[-2] - size[-2] / 2

        # whole components (pixel indexes)
        xmin_wh = xmin.int()
        ymin_wh = ymin.int()

        # fractional (subpixel) components
        xmin_fr = xmin - xmin_wh
        ymin_fr = ymin - ymin_wh

        # bottom right corner of object patch support
        xmax_wh = xmin_wh + size[-1] + 1
        ymax_wh = ymin_wh + size[-2] + 1

        # reused quantities
        xmin_fr_c = 1. - xmin_fr
        ymin_fr_c = 1. - ymin_fr

        # barycentric interpolant weights
        self._weight00 = ymin_fr_c * xmin_fr_c
        self._weight01 = ymin_fr_c * xmin_fr
        self._weight10 = ymin_fr * xmin_fr_c
        self._weight11 = ymin_fr * xmin_fr

        # extract patch support region from full object
        self._object_support = object_[ymin_wh:ymax_wh, xmin_wh:xmax_wh]

    def get_patch(self) -> ComplexTensor:
        '''interpolate object support to extract patch'''
        object_patch = self._weight00 * self._object_support[:-1, :-1]
        object_patch += self._weight01 * self._object_support[:-1, 1:]
        object_patch += self._weight10 * self._object_support[1:, :-1]
        object_patch += self._weight11 * self._object_support[1:, 1:]
        return object_patch

    def update_patch(self, object_update: ComplexTensor) -> None:
        '''add patch update to object support'''
        self._object_support[:-1, :-1] += self._weight00 * object_update
        self._object_support[:-1, 1:] += self._weight01 * object_update
        self._object_support[1:, :-1] += self._weight10 * object_update
        self._object_support[1:, 1:] += self._weight11 * object_update


class IterativeAlgorithm(ABC):

    @abstractmethod
    def iterate(self, plan: CorrectionPlan) -> Sequence[float]:
        pass

    @abstractmethod
    def get_product(self) -> DataProduct:
        pass
