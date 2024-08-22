from typing import Sequence

import torch

from .algorithm import squared_modulus, CorrectionPlan, IterativeAlgorithm
from .data import DataProduct, DetectorData
from .position import correct_positions_serial_cross_correlation


class RAAR(IterativeAlgorithm):

    def __init__(self, detector_data: DetectorData, product: DataProduct,
                 plan: CorrectionPlan) -> None:
        self._detector_data = detector_data
        self._product = product
        self._plan = plan

        self._iteration = 0
        self._tuning_parameter: float  # FIXME beta, 0.6 - 0.95, 1.0 = DM

        self._pc_probe_threshold = 0.1
        self._pc_feedback_parameter = 0.  # FIXME

    def iterate(self, plan: CorrectionPlan) -> Sequence[float]:
        return list()  # FIXME

    def get_product(self) -> DataProduct:
        return self._product
