from types import TracebackType
from typing import overload

from .common import CorrectionPlan, IterativeAlgorithm
from .data import DataProduct, DetectorData


class RAAR(IterativeAlgorithm):

    def __init__(self) -> None:
        self._tuning_parameter: float  # beta, 0.6 - 0.95, 1.0 = DM
