from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch

from .typing import BooleanTensor, ComplexTensor, RealTensor
from .propagate import WavefieldPropagator


@dataclass(frozen=True)
class DetectorData:
    diffraction_patterns: RealTensor  # [N, H, W]
    """ifftshifted diffraction patterns"""
    bad_pixels: BooleanTensor  # [H, W]
    """bad pixel mask to exclude detector dead regions or saturated pixels"""


@dataclass(frozen=True)
class DataProduct:
    positions_px: RealTensor  # [N, 2]
    """scan positions are zero-centered in pixel units"""
    probe: ComplexTensor  # [C, I, H, W]
    """probe shape is the number of coherent (OPR) modes, number of incoherent
    (mixed-states) modes, height in pixels, and width in pixels"""
    object_: ComplexTensor  # [L, H, W]
    """object shape is the number of (multi-slice) layers, height in pixels,
    and width in pixels"""
    propagators: Sequence[WavefieldPropagator]  # [L]
    """sequence of wavefield propagators, one for each object layer"""
