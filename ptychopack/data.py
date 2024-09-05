from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import torch

from .typing import BooleanTensor, ComplexTensor, RealTensor
from .propagate import FourierPropagator, WavefieldPropagator


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
            # TODO torch.fft.ifftshift(diffraction_patterns, dim=(1, 2)),
            diffraction_patterns,
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
