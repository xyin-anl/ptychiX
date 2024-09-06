from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch

from .data import ComplexTensor, DataProduct, RealTensor


def squared_modulus(values: ComplexTensor) -> RealTensor:
    # FIXME return torch.real(values * values.conj())
    return torch.square(torch.abs(values))


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


class IterativeAlgorithm(ABC):

    @abstractmethod
    def iterate(self, plan: CorrectionPlan) -> Sequence[float]:
        pass

    @abstractmethod
    def get_product(self) -> DataProduct:
        pass
