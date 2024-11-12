import dataclasses
from dataclasses import field

import ptychi.api.options.base as base
import ptychi.api.options.task as task_options
import ptychi.api.enums as enums


@dataclasses.dataclass
class DMReconstructorOptions(base.ReconstructorOptions):
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.DM

    exit_wave_update_relaxation: float = 1
    "Relaxation multiplier for the exit wave update."


class DMObjectOptions(base.ObjectOptions):
    amplitude_clamp_limit: float = 1000
    """Maximum allowed amplitude for the object reconstruction. Values above this will be clamped 
    to this value."""


@dataclasses.dataclass
class DMProbeOptions(base.ProbeOptions):
    pass


@dataclasses.dataclass
class DMProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclasses.dataclass
class DMOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class DMOptions(task_options.PtychographyTaskOptions):
    reconstructor_options: DMReconstructorOptions = field(default_factory=DMReconstructorOptions)

    object_options: DMObjectOptions = field(default_factory=DMObjectOptions)

    probe_options: DMProbeOptions = field(default_factory=DMProbeOptions)

    probe_position_options: DMProbePositionOptions = field(default_factory=DMProbePositionOptions)

    opr_mode_weight_options: DMOPRModeWeightsOptions = field(
        default_factory=DMOPRModeWeightsOptions
    )
