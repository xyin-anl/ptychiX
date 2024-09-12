from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Device:
    backend: str
    ordinal: int
    name: str

    @property
    def torch_device(self) -> str:
        return f"{self.backend.lower()}:{self.ordinal}"

    @classmethod
    def CPU(cls) -> Device:
        return cls("cpu", 0, "CPU:0")


def list_available_devices(self) -> Iterator[Device]:
    if torch.cpu.is_available():
        for ordinal in range(torch.cpu.device_count()):
            name = f"CPU:{ordinal}"
            yield Device("cpu", ordinal, name)

    if torch.cuda.is_available():
        for ordinal in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(ordinal)
            yield Device("cuda", ordinal, name)

    if torch.xpu.is_available():
        for ordinal in range(torch.xpu.device_count()):
            name = torch.xpu.get_device_name(ordinal)
            yield Device("xpu", ordinal, name)
