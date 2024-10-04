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
class PIEReconstructorOptions(base.ReconstructorOptions):
        
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.PIE
    

@dataclasses.dataclass
class PIEObjectOptions(base.ObjectOptions):
    pass


@dataclasses.dataclass
class PIEProbeOptions(base.ProbeOptions):
    pass


@dataclasses.dataclass
class PIEProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclasses.dataclass
class PIEOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class PIEOptions(task_options.PtychographyTaskOptions):
    
    reconstructor_options: PIEReconstructorOptions = field(default_factory=PIEReconstructorOptions)
    
    object_options: PIEObjectOptions = field(default_factory=PIEObjectOptions)
    
    probe_options: PIEProbeOptions = field(default_factory=PIEProbeOptions)
    
    probe_position_options: PIEProbePositionOptions = field(default_factory=PIEProbePositionOptions)
    
    opr_mode_weight_options: PIEOPRModeWeightsOptions = field(default_factory=PIEOPRModeWeightsOptions)
