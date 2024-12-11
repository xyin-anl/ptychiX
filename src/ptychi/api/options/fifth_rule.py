import dataclasses
from dataclasses import field

import ptychi.api.options.base as base
import ptychi.api.options.task as task_options
import ptychi.api.enums as enums


@dataclasses.dataclass
class FifthRuleReconstructorOptions(base.ReconstructorOptions):
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.FIFTH_RULE


@dataclasses.dataclass
class FifthRuleObjectOptions(base.ObjectOptions):
    pass
    

@dataclasses.dataclass
class FifthRuleProbeOptions(base.ProbeOptions):
    rho: float = 10
    pass
    

@dataclasses.dataclass
class FifthRuleProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclasses.dataclass
class FifthRuleOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class FifthRuleOptions(task_options.PtychographyTaskOptions):
    
    reconstructor_options: FifthRuleReconstructorOptions = field(default_factory=FifthRuleReconstructorOptions)
    
    object_options: FifthRuleObjectOptions = field(default_factory=FifthRuleObjectOptions)
    
    probe_options: FifthRuleProbeOptions = field(default_factory=FifthRuleProbeOptions)
    
    probe_position_options: FifthRuleProbePositionOptions = field(default_factory=FifthRuleProbePositionOptions)
    
    opr_mode_weight_options: FifthRuleOPRModeWeightsOptions = field(default_factory=FifthRuleOPRModeWeightsOptions)
