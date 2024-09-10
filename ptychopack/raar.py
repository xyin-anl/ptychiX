from typing import Sequence
import logging

import torch

from .core import CorrectionPlan, DataProduct, DetectorData, IterativeAlgorithm
from .utilities import Device

logger = logging.getLogger(__name__)


class RelaxedAveragedAlternatingReflections(IterativeAlgorithm):

    def __init__(self, device: Device, detector_data: DetectorData, product: DataProduct) -> None:
        self._good_pixels = torch.logical_not(detector_data.bad_pixels).to(device.torch_device)
        self._diffraction_patterns = detector_data.diffraction_patterns.to(device.torch_device)
        self._positions_px = product.positions_px.to(device.torch_device)
        self._probe = product.probe[0].to(device.torch_device)  # TODO support OPR modes
        self._object = product.object_[0].to(device.torch_device)  # TODO support multislice
        self._propagators = [propagator.to(device) for propagator in product.propagators]

        self._iteration = 0
        self._beta = 0.8  # FIXME 0.6 - 0.95, 1.0 = DM

        self._pc_probe_threshold = 0.1
        self._pc_feedback = 50.

    def iterate(self, plan: CorrectionPlan) -> Sequence[float]:
        return []  # FIXME

    def get_product(self) -> DataProduct:
        return DataProduct(
            self._positions_px.cpu(),
            torch.unsqueeze(self._probe.cpu(), 0),
            torch.unsqueeze(self._object.cpu(), 0),
            [propagator.cpu() for propagator in self._propagators],
        )
