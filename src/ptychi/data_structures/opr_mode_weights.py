from typing import Union, TYPE_CHECKING

import torch
from torch import Tensor

import ptychi.data_structures.base as ds
from ptychi.reconstructors.base import AnalyticalIterativePtychographyReconstructor
import ptychi.maths as pmath
if TYPE_CHECKING:
    import ptychi.api as api


class OPRModeWeights(ds.ReconstructParameter):
    # TODO: update_relaxation is only used for LSQML. We should create dataclasses
    # to contain additional options for ReconstructParameter classes, and subclass them for specific
    # reconstruction algorithms - for example, OPRModeWeightsOptions -> LSQMLOPRModeWeightsOptions.
    def __init__(
        self,
        *args,
        name="opr_weights",
        options: "api.options.base.OPRModeWeightsOptions" = None,
        **kwargs,
    ):
        """
        Weights of OPR modes for each scan point.

        Parameters
        ----------
        name : str
            Name of the parameter.
        update_relaxation : float
            Relaxation factor, or effectively the step size, for the update step in LSQML.
        """
        super().__init__(*args, name=name, options=options, is_complex=False, **kwargs)
        if len(self.shape) != 2:
            raise ValueError("OPR weights must be of shape (n_scan_points, n_opr_modes).")

        if self.optimizable:
            if not (options.optimize_eigenmode_weights or options.optimize_intensity_variation):
                raise ValueError(
                    "When OPRModeWeights is optimizable, at least 1 of "
                    "optimize_eigenmode_weights and optimize_intensity_variation "
                    "should be set to True."
                )

        self.optimize_eigenmode_weights = options.optimize_eigenmode_weights
        self.optimize_intensity_variation = options.optimize_intensity_variation

        self.n_opr_modes = self.tensor.shape[1]

    def build_optimizer(self):
        if self.optimizer_class is None:
            return
        if self.optimizable:
            if isinstance(self.tensor, ds.ComplexTensor):
                self.optimizer = self.optimizer_class([self.tensor.data], **self.optimizer_params)
            else:
                self.optimizer = self.optimizer_class([self.tensor], **self.optimizer_params)

    def get_weights(self, indices: Union[tuple[int, ...], slice]) -> Tensor:
        return self.data[indices]

    def optimization_enabled(self, epoch: int):
        enabled = super().optimization_enabled(epoch)
        return enabled and (self.optimize_eigenmode_weights or self.optimize_intensity_variation)

    def eigenmode_weight_optimization_enabled(self, epoch: int):
        enabled = super().optimization_enabled(epoch)
        return enabled and self.optimize_eigenmode_weights

    def intensity_variation_optimization_enabled(self, epoch: int):
        enabled = super().optimization_enabled(epoch)
        return enabled and self.optimize_intensity_variation


    def update_variable_probe(
        self,
        reconstructor: AnalyticalIterativePtychographyReconstructor,
        indices: Tensor,
        chi: Tensor,
        delta_p_i: Tensor,
        obj_patches: Tensor,
    ):
        if reconstructor.parameter_group.probe.has_multiple_opr_modes and (
            reconstructor.parameter_group.probe.optimization_enabled(reconstructor.current_epoch)
            or (
                not self.is_dummy
                and self.eigenmode_weight_optimization_enabled(
                    reconstructor.current_epoch
                )
            )
        ):
            self.update_opr_probe_modes_and_weights(
                reconstructor, indices, chi, delta_p_i, obj_patches
            )

        if (
            not self.is_dummy
            and self.intensity_variation_optimization_enabled(
                reconstructor.current_epoch
            )
        ):
            delta_weights_int = self._calculate_intensity_variation_update_direction(
                reconstructor, indices, chi, obj_patches,
            )
            self._apply_variable_intensity_updates(delta_weights_int)

    def update_opr_probe_modes_and_weights(
        self,
        reconstructor: AnalyticalIterativePtychographyReconstructor,
        indices: Tensor,
        chi: Tensor,
        delta_p_i: Tensor,
        obj_patches: Tensor,
    ):
        probe = reconstructor.parameter_group.probe.data
        weights = reconstructor.parameter_group.opr_mode_weights.data

        batch_size = len(delta_p_i)
        n_points_total = reconstructor.parameter_group.probe_positions.shape[0]
        # FIXME: reduced relax_u/v by a factor of 10 for stability, but PtychoShelves works without this.
        relax_u = (
            min(0.1, batch_size / n_points_total)
            * reconstructor.parameter_group.probe.options.eigenmode_update_relaxation
        )
        relax_v = self.options.update_relaxation
        # Shape of delta_p_i:       (batch_size, n_probe_modes, h, w)
        # Use only the first incoherent mode
        delta_p_i = delta_p_i[:, 0, :, :]
        residue_update = delta_p_i - delta_p_i.mean(0)

        for i_opr_mode in range(1, reconstructor.parameter_group.probe.n_opr_modes):
            # Just take the first incoherent mode.
            eigenmode_i = reconstructor.parameter_group.probe.get_mode_and_opr_mode(
                mode=0, opr_mode=i_opr_mode
            )
            weights_i = self.get_weights(indices)[:, i_opr_mode]
            eigenmode_i, weights_i = self._update_first_eigenmode_and_weight(
                residue_update,
                eigenmode_i,
                weights_i,
                relax_u,
                relax_v,
                obj_patches,
                chi,
                update_eigenmode=reconstructor.parameter_group.probe.optimization_enabled(
                    reconstructor.current_epoch
                ),
                update_weights=self.eigenmode_weight_optimization_enabled(
                    reconstructor.current_epoch
                ),
            )

            # Project residue on this eigenmode, then subtract it.
            residue_update = (
                residue_update - pmath.project(residue_update, eigenmode_i) * eigenmode_i
            )

            # Is the view of probe data unintentionally being updated here? Check later
            probe[i_opr_mode, 0, :, :] = eigenmode_i
            weights[indices, i_opr_mode] = weights_i

        if reconstructor.parameter_group.probe.optimization_enabled(reconstructor.current_epoch):
            reconstructor.parameter_group.probe.set_data(probe)
        if self.eigenmode_weight_optimization_enabled(
            reconstructor.current_epoch
        ):
            self.set_data(weights)

    def _update_first_eigenmode_and_weight(
        self,
        residue_update: Tensor,
        eigenmode_i: Tensor,
        weights_i: Tensor,
        relax_u: Tensor,
        relax_v: Tensor,
        obj_patches: Tensor,
        chi: Tensor,
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

    def _calculate_intensity_variation_update_direction(
        self,
        reconstructor: AnalyticalIterativePtychographyReconstructor,
        indices: Tensor,
        chi: Tensor,
        obj_patches: Tensor,
    ):
        """
        Update variable intensity scaler - i.e., the OPR mode weight corresponding to the main mode.

        This implementation is adapted from PtychoShelves code (update_variable_probe.m) and has some
        differences from Eq. 31 of Odstrcil (2018).
        """
        # Just take the first slice.
        obj_patches = obj_patches[:, 0]

        mean_probe = reconstructor.parameter_group.probe.get_mode_and_opr_mode(mode=0, opr_mode=0)
        op = obj_patches * mean_probe
        num = torch.real(op.conj() * chi[:, 0, ...])
        denom = op.abs() ** 2
        delta_weights_int_i = torch.sum(num, dim=(-2, -1)) / torch.sum(denom, dim=(-2, -1))
        # Pad it to the same shape as opr_mode_weights.
        delta_weights_int = torch.zeros_like(self.data)
        delta_weights_int[indices, 0] = delta_weights_int_i
        return delta_weights_int

    def _apply_variable_intensity_updates(self, delta_weights_int: Tensor):
        self.set_data(self.data + 0.1 * delta_weights_int)