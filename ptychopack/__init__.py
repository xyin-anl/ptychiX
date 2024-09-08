from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ptychopack")
except PackageNotFoundError:
    pass

from .core import CorrectionPlan, DataProduct, DetectorData, IterativeAlgorithm
from .pie import PtychographicIterativeEngine
from .utilities import Device

__all__ = [
    "CorrectionPlan",
    "DataProduct",
    "DetectorData",
    "Device",
    "IterativeAlgorithm",
    "PtychographicIterativeEngine",
]
