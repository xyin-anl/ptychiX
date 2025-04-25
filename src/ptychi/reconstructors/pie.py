# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from torch import Tensor

from ptychi.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
)
from ptychi.metrics import MSELossOfSqrt
from ptychi.timing.timer_utils import timer
import ptychi.image_proc as ip

if TYPE_CHECKING:
    import ptychi.api as api
    import ptychi.data_structures.parameter_group as pg


class PIEReconstructor(AnalyticalIterativePtychographyReconstructor):
    """
    The ptychographic iterative engine (PIE), as described in:

    Andrew Maiden, Daniel Johnson, and Peng Li, "Further improvements to the
    ptychographical iterative engine," Optica 4, 736-745 (2017)

    Object and probe updates are calculated using the formulas in table 1 of
    Maiden (2017).

    The `step_size` parameter is equivalent to gamma in Eq. 22 of Maiden (2017)
    when `optimizer == SGD`.
    """

    parameter_group: "pg.PlanarPtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PlanarPtychographyParameterGroup",
        dataset: Dataset,
        options: Optional["api.options.pie.PIEReconstructorOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            parameter_group=parameter_group,
            dataset=dataset,
            options=options,
            *args,
            **kwargs,
        )

    def build_loss_tracker(self):
        if self.displayed_loss_function is None:
            self.displayed_loss_function = MSELossOfSqrt()
        return super().build_loss_tracker()

    def check_inputs(self, *args, **kwargs):
        if self.parameter_group.object.is_multislice:
            raise NotImplementedError("EPIEReconstructor only supports 2D objects.")
        for var in self.parameter_group.get_optimizable_parameters():
            if "lr" not in var.optimizer_params.keys():
                raise ValueError(
                    "Optimizable parameter {} must have 'lr' in optimizer_params.".format(var.name)
                )

    @timer()
    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        self.parameter_group.probe.initialize_grad()
        (delta_o, delta_p_i, delta_pos), y_pred = self.compute_updates(
            *input_data, y_true, self.dataset.valid_pixel_mask
        )
        self.apply_updates(delta_o, delta_p_i, delta_pos)
        self.loss_tracker.update_batch_loss_with_metric_function(y_pred, y_true)

    @timer()
    def compute_updates(
        self, indices: torch.Tensor, y_true: torch.Tensor, valid_pixel_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Calculates the updates of the whole object, the probe, and other parameters.
        This function is called in self.update_step_module.forward.
        """
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions
        opr_mode_weights = self.parameter_group.opr_mode_weights

        indices = indices.cpu()
        positions = probe_positions.tensor[indices]

        y = self.forward_model.forward(indices)
        obj_patches = self.forward_model.intermediate_variables["obj_patches"]
        psi = self.forward_model.intermediate_variables["psi"]
        psi_far = self.forward_model.intermediate_variables["psi_far"]
        unique_probes = self.forward_model.intermediate_variables.shifted_unique_probes[0]

        psi_prime = self.replace_propagated_exit_wave_magnitude(psi_far, y_true)
        # Do not swap magnitude for bad pixels.
        psi_prime = torch.where(
            valid_pixel_mask.repeat(psi_prime.shape[0], probe.n_modes, 1, 1), psi_prime, psi_far
        )
        psi_prime = self.forward_model.free_space_propagator.propagate_backward(psi_prime)

        delta_o = None
        if object_.optimization_enabled(self.current_epoch):
            step_weight = self.calculate_object_step_weight(unique_probes)
            delta_o_patches = step_weight * (psi_prime - psi)
            delta_o_patches = delta_o_patches.sum(1, keepdim=True)
            delta_o = ip.place_patches_integer(
                torch.zeros_like(object_.get_slice(0)),
                positions.round().int() + object_.pos_origin_coords,
                delta_o_patches[:, 0],
                op="add",
            )
            # Add slice dimension.
            delta_o = delta_o.unsqueeze(0)

        delta_pos = None
        if probe_positions.optimization_enabled(self.current_epoch) and object_.optimizable:
            delta_pos = torch.zeros_like(probe_positions.data)
            delta_pos[indices] = probe_positions.position_correction.get_update(
                psi_prime - psi,
                obj_patches,
                delta_o_patches,
                self.forward_model.intermediate_variables.shifted_unique_probes[0],
                object_.optimizer_params["lr"],
            )

        delta_p_i = None
        if probe.optimization_enabled(self.current_epoch):
            step_weight = self.calculate_probe_step_weight(obj_patches)
            delta_p_i = step_weight * (psi_prime - psi)  # get delta p at each position
            delta_p_i = self.adjoint_shift_probe_update_direction(indices, delta_p_i, first_mode_only=True)

        # Calculate and apply opr mode updates
        if self.parameter_group.opr_mode_weights.optimization_enabled(self.current_epoch):
            opr_mode_weights.update_variable_probe(
                probe,
                indices,
                psi_prime - psi,
                delta_p_i,
                delta_p_i.mean(0),
                obj_patches,
                self.current_epoch,
                probe_mode_index=0,
            )

        return (delta_o, delta_p_i, delta_pos), y

    @timer()
    def calculate_object_step_weight(self, p: Tensor):
        """
        Calculate the weight for the object update step.

        Parameters
        ----------
        p : Tensor
            A (batch_size, n_modes, h, w) tensor giving the first OPR mode of the probe.

        Returns
        -------
        Tensor
            A (batch_size, n_modes, h, w) tensor giving the weight for the object update step.
        """
        numerator = p.abs() * p.conj()
        denominator = p.abs().sum(1, keepdim=True).max() * (
            p.abs() ** 2 + self.parameter_group.object.options.alpha * (p.abs() ** 2).sum(1, keepdim=True).max()
        )
        step_weight = numerator / denominator
        return step_weight

    @timer()
    def calculate_probe_step_weight(self, obj_patches: Tensor):
        """
        Calculate the weight for the probe update step.

        Parameters
        ----------
        obj_patches : Tensor
            A (batch_size, n_slices, h, w) tensor giving the object patches.

        Returns
        -------
        Tensor
        """
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        obj_max = (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
        numerator = obj_patches.abs() * obj_patches.conj()
        denominator = obj_max * (
            obj_patches.abs() ** 2 + self.parameter_group.probe.options.alpha * obj_max
        )
        step_weight = numerator / denominator
        return step_weight

    @timer()
    def apply_updates(self, delta_o, delta_p_i, delta_pos, *args, **kwargs):
        """
        Apply updates to optimizable parameters given the updates calculated by self.compute_updates.

        Parameters
        ----------
        delta_o : Tensor
            A (h, w, 2) tensor of object update vector.
        delta_p_i : Tensor
            A (n_patches, n_opr_modes, n_modes, h, w, 2) tensor of probe update vector.
        delta_pos : Tensor
            A (n_positions, 2) tensor of probe position vectors.
        """
        object_ = self.parameter_group.object
        probe_positions = self.parameter_group.probe_positions

        if delta_o is not None:
            object_.set_grad(-delta_o)
            object_.optimizer.step()

        if delta_p_i is not None:
            mode_slicer = self.parameter_group.probe._get_probe_mode_slicer(None)
            self.parameter_group.probe.set_grad(-delta_p_i.mean(0), slicer=(0, mode_slicer))
            self.parameter_group.probe.optimizer.step()

        if delta_pos is not None:
            probe_positions.set_grad(-delta_pos)
            probe_positions.step_optimizer(
                limit=probe_positions.options.correction_options.update_magnitude_limit
            )


class EPIEReconstructor(PIEReconstructor):
    """
    The extended ptychographic iterative engine (ePIE), as described in:

    Andrew Maiden, Daniel Johnson, and Peng Li, "Further improvements to the
    ptychographical iterative engine," Optica 4, 736-745 (2017)

    Object and probe updates are calculated using the formulas in table 1 of
    Maiden (2017).

    The `step_size` parameter is equivalent to gamma in Eq. 22 of Maiden (2017)
    when `optimizer == SGD`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @timer()
    def calculate_object_step_weight(self, p: Tensor):
        p_max = (torch.abs(p) ** 2).sum(1).max()
        step_weight = self.parameter_group.object.options.alpha * p.conj() / p_max
        return step_weight

    @timer()
    def calculate_probe_step_weight(self, obj_patches: Tensor):
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        obj_max = (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
        step_weight = self.parameter_group.probe.options.alpha * obj_patches.conj() / obj_max
        step_weight = step_weight[:, None]
        return step_weight


class RPIEReconstructor(PIEReconstructor):
    """
    The regularized ptychographic iterative engine (rPIE), as described in:

    Andrew Maiden, Daniel Johnson, and Peng Li, "Further improvements to the
    ptychographical iterative engine," Optica 4, 736-745 (2017)

    Object and probe updates are calculated using the formulas in table 1 of
    Maiden (2017).

    The `step_size` parameter is equivalent to gamma in Eq. 22 of Maiden (2017)
    when `optimizer == SGD`.

    To get the momentum-accelerated PIE (mPIE), use `optimizer == SGD` and use
    the optimizer settings `{'momentum': eta, 'nesterov': True}` where `eta` is
    the constant used in  Eq. 19 of Maiden (2017).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @timer()
    def calculate_object_step_weight(self, p: Tensor):
        p_max = (torch.abs(p) ** 2).sum(1).max()
        step_weight = p.conj() / (
            (1 - self.parameter_group.object.options.alpha) * (torch.abs(p) ** 2)
            + self.parameter_group.object.options.alpha * p_max
        )
        return step_weight

    @timer()
    def calculate_probe_step_weight(self, obj_patches: Tensor):
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        obj_max = (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
        step_weight = obj_patches.conj() / (
            (1 - self.parameter_group.probe.options.alpha) * (torch.abs(obj_patches) ** 2)
            + self.parameter_group.probe.options.alpha * obj_max
        )
        step_weight = step_weight[:, None]
        return step_weight
