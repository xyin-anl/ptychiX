from typing import Optional, TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from torch import Tensor

from ptychi.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
)
from ptychi.metrics import MSELossOfSqrt
import ptychi.maths as pmath

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

    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        (delta_o, delta_p_i, delta_pos), y_pred = self.compute_updates(
            *input_data, y_true, self.dataset.valid_pixel_mask
        )
        self.apply_updates(delta_o, delta_p_i, delta_pos)
        self.loss_tracker.update_batch_loss_with_metric_function(y_pred, y_true)

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

        indices = indices.cpu()
        positions = probe_positions.tensor[indices]

        y = self.forward_model.forward(indices)
        obj_patches = self.forward_model.intermediate_variables["obj_patches"]
        psi = self.forward_model.intermediate_variables["psi"]
        psi_far = self.forward_model.intermediate_variables["psi_far"]

        p = probe.get_opr_mode(0)

        psi_prime = self.replace_propagated_exit_wave_magnitude(psi_far, y_true)
        # Do not swap magnitude for bad pixels.
        psi_prime = torch.where(
            valid_pixel_mask.repeat(psi_prime.shape[0], probe.n_modes, 1, 1), psi_prime, psi_far
        )
        psi_prime = self.forward_model.far_field_propagator.propagate_backward(psi_prime)

        delta_o = None
        if object_.optimization_enabled(self.current_epoch):
            step_weight = self.calculate_object_step_weight(p)
            delta_o_patches = step_weight * (psi_prime - psi)
            delta_o_patches = delta_o_patches.sum(1)
            delta_o = object_.place_patches_function(
                torch.zeros_like(object_.get_slice(0)),
                positions + object_.center_pixel,
                delta_o_patches,
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
                probe,
                self.parameter_group.opr_mode_weights,
                indices,
                object_.optimizer_params["lr"],
            )

        delta_p_i = None
        if probe.optimization_enabled(self.current_epoch):
            step_weight = self.calculate_probe_step_weight(obj_patches)
            delta_p_i = step_weight * (psi_prime - psi)  # patches, inc modes, shape
            # delta_p = step_weight * (psi_prime - psi) # get delta p for each position
            # delta_p = delta_p.mean(0) # average over al positions
            # delta_p_i = delta_p[None, :, :] # add axis for opr modes

        #
        self.update_variable_probe(indices, psi_prime - psi, delta_p_i, obj_patches)

        return (delta_o, delta_p_i, delta_pos), y

    def calculate_object_step_weight(self, p: Tensor):
        """
        Calculate the weight for the object update step.

        Parameters
        ----------
        p : Tensor
            A (n_modes, h, w) tensor giving the first OPR mode of the probe.

        Returns
        -------
        Tensor
            A (batch_size, h, w) tensor giving the weight for the object update step.
        """
        numerator = p.abs() * p.conj()
        denominator = p.abs().sum(0).max() * (
            p.abs() ** 2 + self.parameter_group.object.options.alpha * (p.abs() ** 2).sum(0).max()
        )
        step_weight = numerator / denominator
        return step_weight

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
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        if delta_o is not None:
            object_.set_grad(-delta_o)
            object_.optimizer.step()

        if delta_p_i is not None:
            self.apply_probe_update(delta_p_i)
            # probe.set_grad(-delta_p_i.mean(0, keepdims=True))  # average over all positions
            # probe.optimizer.step()

        if delta_pos is not None:
            probe_positions.set_grad(-delta_pos)
            probe_positions.optimizer.step()

    def apply_probe_update(self, delta_p_i: Tensor):
        probe = self.parameter_group.probe
        delta_p = delta_p_i.mean(0, keepdims=True)
        if probe.has_multiple_opr_modes:
            delta_p = torch.nn.functional.pad(
                delta_p,
                pad=(0, 0, 0, 0, 0, 0, 0, probe.n_opr_modes - 1),
                mode="constant",
                value=0.0,
            )
        probe.set_grad(-delta_p)  # average over all positions
        probe.optimizer.step()

    def update_variable_probe(self, indices, chi, delta_p_i, obj_patches):
        if self.parameter_group.probe.has_multiple_opr_modes and (
            self.parameter_group.probe.optimization_enabled(self.current_epoch)
            or (
                not self.parameter_group.opr_mode_weights.is_dummy
                and self.parameter_group.opr_mode_weights.eigenmode_weight_optimization_enabled(
                    self.current_epoch
                )
            )
        ):
            self.update_opr_probe_modes_and_weights(indices, chi, delta_p_i, obj_patches)

        if (
            not self.parameter_group.opr_mode_weights.is_dummy
            and self.parameter_group.opr_mode_weights.intensity_variation_optimization_enabled(
                self.current_epoch
            )
        ):
            delta_weights_int = self._calculate_intensity_variation_update_direction(
                indices, chi, obj_patches
            )
            self._apply_variable_intensity_updates(delta_weights_int)

    # def update_opr_probe_modes_and_weights(self, delta_p_i, indices, obj_patches, chi):
    def update_opr_probe_modes_and_weights(self, indices, chi, delta_p_i, obj_patches):
        probe = self.parameter_group.probe.data
        weights = self.parameter_group.opr_mode_weights.data

        batch_size = len(delta_p_i)
        n_points_total = self.parameter_group.probe_positions.shape[0]
        # FIXME: reduced relax_u/v by a factor of 10 for stability, but PtychoShelves works without this.
        relax_u = (
            min(0.1, batch_size / n_points_total)
            * self.parameter_group.probe.options.eigenmode_update_relaxation
        )
        relax_v = self.parameter_group.opr_mode_weights.options.update_relaxation
        # Shape of delta_p_i:       (batch_size, n_probe_modes, h, w)
        # Use only the first incoherent mode
        delta_p_i = delta_p_i[:, 0, :, :]
        residue_update = delta_p_i - delta_p_i.mean(0)

        for i_opr_mode in range(1, self.parameter_group.probe.n_opr_modes):
            # Just take the first incoherent mode.
            eigenmode_i = self.parameter_group.probe.get_mode_and_opr_mode(
                mode=0, opr_mode=i_opr_mode
            )
            weights_i = self.parameter_group.opr_mode_weights.get_weights(indices)[:, i_opr_mode]
            eigenmode_i, weights_i = self._update_first_eigenmode_and_weight(
                residue_update,
                eigenmode_i,
                weights_i,
                relax_u,
                relax_v,
                obj_patches,
                chi,
                update_eigenmode=self.parameter_group.probe.optimization_enabled(
                    self.current_epoch
                ),
                update_weights=self.parameter_group.opr_mode_weights.eigenmode_weight_optimization_enabled(
                    self.current_epoch
                ),
            )

            # Project residue on this eigenmode, then subtract it.
            residue_update = (
                residue_update - pmath.project(residue_update, eigenmode_i) * eigenmode_i
            )

            probe[i_opr_mode, 0, :, :] = eigenmode_i
            weights[indices, i_opr_mode] = weights_i

        if self.parameter_group.probe.optimization_enabled(self.current_epoch):
            self.parameter_group.probe.set_data(probe)
        if self.parameter_group.opr_mode_weights.eigenmode_weight_optimization_enabled(
            self.current_epoch
        ):
            self.parameter_group.opr_mode_weights.set_data(weights)

    def _update_first_eigenmode_and_weight(
        self,
        residue_update,
        eigenmode_i,
        weights_i,
        relax_u,
        relax_v,
        obj_patches,
        chi,
        eps=1e-5,
        update_eigenmode=True,
        update_weights=True,
    ):
        # Shape of residue_update:          (batch_size, h, w)
        # Shape of eigenmode_i:             (h, w)
        # Shape of weights_i:               (batch_size,)

        obj_patches = obj_patches[:, 0]

        # Update eigenmode.
        # Shape of proj:                    (batch_size, h, w)
        # FIXME: What happens when weights is zero!?
        proj = ((residue_update.conj() * eigenmode_i).real + weights_i[:, None, None]) / pmath.norm(
            weights_i
        ) ** 2

        if update_eigenmode:
            # Shape of eigenmode_update:        (h, w)
            eigenmode_update = torch.mean(
                residue_update * torch.mean(proj, dim=(-2, -1), keepdim=True), dim=0
            )
            eigenmode_i = eigenmode_i + relax_u * eigenmode_update / (
                pmath.mnorm(eigenmode_update.view(-1)) + eps
            )
            eigenmode_i = eigenmode_i / pmath.mnorm(eigenmode_i.view(-1) + eps)

        if update_weights:
            # Update weights using Eq. 23a.
            # Shape of psi:                     (batch_size, h, w)
            psi = eigenmode_i * obj_patches
            # The denominator can get smaller and smaller as eigenmode_i goes down.
            # Weight update needs to be clamped.
            denom = torch.mean((torch.abs(psi) ** 2), dim=(-2, -1))
            num = torch.mean((chi[:, 0, :, :] * psi.conj()).real, dim=(-2, -1))
            weight_update = num / (denom + 0.1 * torch.mean(denom))
            weight_update = weight_update.clamp(max=10)
            weights_i = weights_i + relax_v * weight_update

        return eigenmode_i, weights_i

    def _apply_variable_intensity_updates(self, delta_weights_int):
        weights = self.parameter_group.opr_mode_weights
        weights.set_data(weights.data + 0.1 * delta_weights_int)

    def _calculate_intensity_variation_update_direction(self, indcies, chi, obj_patches):
        """
        Update variable intensity scaler - i.e., the OPR mode weight corresponding to the main mode.

        This implementation is adapted from PtychoShelves code (update_variable_probe.m) and has some
        differences from Eq. 31 of Odstrcil (2018).
        """
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        mean_probe = self.parameter_group.probe.get_mode_and_opr_mode(mode=0, opr_mode=0)
        op = obj_patches * mean_probe
        num = torch.real(op.conj() * chi[:, 0, ...])
        denom = op.abs() ** 2
        delta_weights_int_i = torch.sum(num, dim=(-2, -1)) / torch.sum(denom, dim=(-2, -1))
        # Pad it to the same shape as opr_mode_weights.
        delta_weights_int = torch.zeros_like(self.parameter_group.opr_mode_weights.data)
        delta_weights_int[indcies, 0] = delta_weights_int_i
        return delta_weights_int


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

    def calculate_object_step_weight(self, p: Tensor):
        p_max = (torch.abs(p) ** 2).sum(0).max()
        step_weight = self.parameter_group.object.options.alpha * p.conj() / p_max
        return step_weight

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

    def calculate_object_step_weight(self, p: Tensor):
        p_max = (torch.abs(p) ** 2).sum(0).max()
        step_weight = p.conj() / (
            (1 - self.parameter_group.object.options.alpha) * (torch.abs(p) ** 2)
            + self.parameter_group.object.options.alpha * p_max
        )
        return step_weight

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
