from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch

from .data import BooleanTensor, ComplexTensor, DataProduct, RealTensor


def squared_modulus(wavefield: ComplexTensor) -> RealTensor:
    return torch.real(torch.multiply(wavefield, torch.conj(wavefield)))


@dataclass(frozen=True)
class CorrectionPlan:
    correct_object_start: int
    correct_object_stop: int
    correct_object_step: int

    correct_probe_start: int
    correct_probe_stop: int
    correct_probe_step: int

    correct_positions_start: int
    correct_positions_stop: int
    correct_positions_step: int

    @classmethod
    def create_simple(
        cls,
        num_iterations: int,
        *,
        correct_object: bool = False,
        correct_probe: bool = False,
        correct_positions: bool = False,
    ) -> CorrectionPlan:
        return cls(
            correct_object_start=0,
            correct_object_stop=num_iterations if correct_object else 0,
            correct_object_step=1,
            correct_probe_start=0,
            correct_probe_stop=num_iterations if correct_probe else 0,
            correct_probe_step=1,
            correct_positions_start=0,
            correct_positions_stop=num_iterations if correct_positions else 0,
            correct_positions_step=1,
        )

    @property
    def number_of_iterations(self) -> int:
        return max(
            self.correct_object_stop,
            self.correct_probe_stop,
            self.correct_positions_stop,
        )

    @staticmethod
    def _is_correction_enabled(x: int, start: int, stop: int, step: int) -> bool:
        return (start <= x and x < stop and (x - start) % step == 0)

    def is_object_correction_enabled(self, iteration: int) -> bool:
        return self._is_correction_enabled(
            iteration,
            self.correct_object_start,
            self.correct_object_stop,
            self.correct_object_step,
        )

    def is_probe_correction_enabled(self, iteration: int) -> bool:
        return self._is_correction_enabled(
            iteration,
            self.correct_probe_start,
            self.correct_probe_stop,
            self.correct_probe_step,
        )

    def is_position_correction_enabled(self, iteration: int) -> bool:
        return self._is_correction_enabled(
            iteration,
            self.correct_positions_start,
            self.correct_positions_stop,
            self.correct_positions_step,
        )


class IterativeAlgorithm(ABC):

    @abstractmethod
    def iterate(self, plan: CorrectionPlan) -> Sequence[float]:
        pass

    @abstractmethod
    def get_product(self) -> DataProduct:
        pass
