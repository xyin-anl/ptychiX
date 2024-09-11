from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ptychopack")
except PackageNotFoundError:
    pass

from .core import CorrectionPlan, DataProduct, DetectorData, IterativeAlgorithm
from .dm import DifferenceMap
from .pie import PtychographicIterativeEngine
from .raar import RelaxedAveragedAlternatingReflections
from .utilities import Device

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
