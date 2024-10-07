from typing import Type, Optional

import torch
import tqdm
from torch.utils.data import Dataset

from ptychointerim.ptychotorch.data_structures import PtychographyVariableGroup, MultisliceObject
from ptychointerim.forward_models import ForwardModel, MultislicePtychographyForwardModel
from ptychointerim.ptychotorch.reconstructors.ad_general import AutodiffReconstructor
from ptychointerim.ptychotorch.reconstructors.base import IterativePtychographyReconstructor


class AutodiffPtychographyReconstructor(AutodiffReconstructor, IterativePtychographyReconstructor):
    def __init__(self,
                 variable_group: PtychographyVariableGroup,
                 dataset: Dataset,
                 forward_model_class: Type[ForwardModel],
                 forward_model_params: Optional[dict] = None,
                 batch_size: int = 1,
                 loss_function: torch.nn.Module = None,
                 n_epochs: int = 100,
                 metric_function: Optional[torch.nn.Module] = None,
                 *args, **kwargs
    ) -> None:
        super().__init__(
            variable_group=variable_group,
            dataset=dataset,
            forward_model_class=forward_model_class,
            forward_model_params=forward_model_params,
            batch_size=batch_size,
            loss_function=loss_function,
            n_epochs=n_epochs,
            metric_function=metric_function,
            *args, **kwargs
        )
        
    def check_inputs(self, *args, **kwargs):
        super().check_inputs(*args, **kwargs)
        
        if isinstance(self.variable_group.object, MultisliceObject):
            if self.forward_model_class != MultislicePtychographyForwardModel:
                raise ValueError(
                    "If the object is multislice, the forward model must be MultislicePtychographyForwardModel."
                )
                
    def run_post_differentiation_hooks(self, input_data, y_true):
        super().run_post_differentiation_hooks(input_data, y_true)
        
        # If OPRModeWeights is optimized in the current epoch (i.e., it has gradient) but intensity variation
        # optimization is not enabled, set the gradient of the principal mode's weights to 0. Similar is done
        # for the gradient of the eigenmode weights.
        if self.variable_group.opr_mode_weights.optimization_enabled(self.current_epoch):
            if not self.variable_group.opr_mode_weights.intensity_variation_optimization_enabled(self.current_epoch):
                self.variable_group.opr_mode_weights.tensor.grad[:, 0] = 0
            if not self.variable_group.opr_mode_weights.eigenmode_weight_optimization_enabled(self.current_epoch):
                self.variable_group.opr_mode_weights.tensor.grad[:, 1:] = 0

    def run_post_update_hooks(self) -> None:
        with torch.no_grad():
            if self.variable_group.object.optimization_enabled(self.current_epoch):
                self.variable_group.object.post_update_hook()

            if self.variable_group.probe.optimization_enabled(self.current_epoch) \
                    and self.variable_group.opr_mode_weights.optimization_enabled(self.current_epoch):
                weights = self.variable_group.probe.post_update_hook(self.variable_group.opr_mode_weights)
                self.variable_group.opr_mode_weights.set_data(weights)

            if self.variable_group.probe_positions.optimization_enabled(self.current_epoch):
                self.variable_group.probe_positions.post_update_hook()
                
    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        y_pred = self.forward_model(*input_data)
        batch_loss = self.loss_function(
            y_pred[:, self.dataset.valid_pixel_mask], y_true[:, self.dataset.valid_pixel_mask]
        )
        batch_loss.backward()
        self.run_post_differentiation_hooks(input_data, y_true)
        self.step_all_optimizers()
        self.forward_model.zero_grad()
        self.run_post_update_hooks()
        self.loss_tracker.update_batch_loss(y_pred=y_pred, y_true=y_true, loss=batch_loss.item())
