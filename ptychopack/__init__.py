from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ptychopack")
except PackageNotFoundError:
    pass

from .api import CorrectionPlan, DataProduct, DetectorData, IterativeAlgorithm
from .device import Device
from .dm import DifferenceMap
from .pie import PtychographicIterativeEngine
from .raar import RelaxedAveragedAlternatingReflections

__all__ = [
    "CorrectionPlan",
    "DataProduct",
    "DetectorData",
    "Device",
    "DifferenceMap",
    "IterativeAlgorithm",
    "PtychographicIterativeEngine",
    "RelaxedAveragedAlternatingReflections",
]
