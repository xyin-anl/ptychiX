from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ptychopack")
except PackageNotFoundError:
    pass

from .algorithm import CorrectionPlan, IterativeAlgorithm
from .data import DataProduct, DetectorData
from .pie import PtychographicIterativeEngine

__all__ = [
    "CorrectionPlan",
    "DataProduct",
    "DetectorData",
    "IterativeAlgorithm",
    "PtychographicIterativeEngine",
]
