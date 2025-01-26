import dataclasses
from dataclasses import field

import ptychi.api.options.base as base
import ptychi.api.options.task as task_options
import ptychi.api.enums as enums


@dataclasses.dataclass
class BHReconstructorOptions(base.ReconstructorOptions):
    
    method: str = 'GD'

    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.BH


@dataclasses.dataclass
class BHObjectOptions(base.ObjectOptions):
    pass
    

@dataclasses.dataclass
class BHProbeOptions(base.ProbeOptions):
    rho: float = 1
    pass
    

@dataclasses.dataclass
class BHProbePositionOptions(base.ProbePositionOptions):
    rho: float = 0.1
    pass


@dataclasses.dataclass
class BHOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class BHOptions(task_options.PtychographyTaskOptions):
    
    reconstructor_options: BHReconstructorOptions = field(default_factory=BHReconstructorOptions)
    
    object_options: BHObjectOptions = field(default_factory=BHObjectOptions)
    
    probe_options: BHProbeOptions = field(default_factory=BHProbeOptions)
    
    probe_position_options: BHProbePositionOptions = field(default_factory=BHProbePositionOptions)
    
    opr_mode_weight_options: BHOPRModeWeightsOptions = field(default_factory=BHOPRModeWeightsOptions)
