from typing import Optional, Literal, Union
import dataclasses
from dataclasses import field
import json

from torch import Tensor
from numpy import ndarray

import ptychointerim.api.options.base as base
import ptychointerim.api.options.task as task_options
import ptychointerim.api.enums as enums


@dataclasses.dataclass
class PIEPtychographyReconstructorOptions(base.ReconstructorOptions):
        
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.PIE
    
    probe_alpha: float = 0.1
    """Multiplier for the update to the object."""

    object_alpha: float = 0.1
    """Multiplier for the update to the object."""


@dataclasses.dataclass
class PIEPtychographyObjectOptions(base.ObjectOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyProbeOptions(base.ProbeOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyOptions(task_options.PtychographyTaskOptions):
    
    reconstructor_options: PIEPtychographyReconstructorOptions = field(default_factory=PIEPtychographyReconstructorOptions)
    
    object_options: PIEPtychographyObjectOptions = field(default_factory=PIEPtychographyObjectOptions)
    
    probe_options: PIEPtychographyProbeOptions = field(default_factory=PIEPtychographyProbeOptions)
    
    probe_position_options: PIEPtychographyProbePositionOptions = field(default_factory=PIEPtychographyProbePositionOptions)
    
    opr_mode_weight_options: PIEPtychographyOPRModeWeightsOptions = field(default_factory=PIEPtychographyOPRModeWeightsOptions)


@dataclasses.dataclass
class EPIEPtychographyReconstructorOptions(PIEPtychographyReconstructorOptions):
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.EPIE


@dataclasses.dataclass
class EPIEPtychographyOptions(PIEPtychographyOptions):

    reconstructor_options: EPIEPtychographyReconstructorOptions = field(default_factory=EPIEPtychographyReconstructorOptions)


@dataclasses.dataclass
class RPIEPtychographyReconstructorOptions(PIEPtychographyReconstructorOptions):
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.RPIE


@dataclasses.dataclass
class RPIEPtychographyOptions(PIEPtychographyOptions):

    reconstructor_options: RPIEPtychographyReconstructorOptions = field(default_factory=RPIEPtychographyReconstructorOptions)