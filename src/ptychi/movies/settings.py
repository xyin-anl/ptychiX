import dataclasses
from enum import StrEnum, auto
from typing import Sequence


class MovieFileTypes(StrEnum):
    GIF = auto()
    MP4 = auto()


class MovieSubjectTypes(StrEnum):
    OBJECT = auto()
    PROBE = auto()


class ProcessFunctionType(StrEnum):
    PHASE = auto()
    MAGNITUDE = auto()


class ProbePlotTypes(StrEnum):
    INCOHERENT_SUM = auto()
    SEPERATE_MODES = auto()


@dataclasses.dataclass
class SnapshotSettings:
    stride: int = 1

    scale: int = 4


@dataclasses.dataclass
class ObjectMovieSettings:
    process_function: ProcessFunctionType = ProcessFunctionType.PHASE

    slice_index: int = 0
    "Index of the object to select"

    snapshot: SnapshotSettings = dataclasses.field(default_factory=SnapshotSettings)


@dataclasses.dataclass
class ProbeMovieSettings:
    process_function: ProcessFunctionType = ProcessFunctionType.MAGNITUDE

    plot_type: ProbePlotTypes = ProbePlotTypes.INCOHERENT_SUM

    mode_indices: Sequence[int] = (0,)
    """
    Indices of modes to be selected for plotting. This is only used 
    when `plot_type` is `SEPERATE_MODES`.
    """

    snapshot: SnapshotSettings = dataclasses.field(default_factory=SnapshotSettings)
