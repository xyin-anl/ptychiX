from typing import Optional, TYPE_CHECKING, Tuple
import warnings

import torch
from torch.utils.data import Dataset
from torch import Tensor
from torch.utils.data import DataLoader

from ptychi.api import enums
import ptychi.maps as maps
from ptychi.ptychotorch.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
)
from ptychi.metrics import MSELossOfSqrt
import ptychi.forward_models as fm

if TYPE_CHECKING:
    import ptychi.api as api
    import ptychi.data_structures.parameter_group as pg


class DMReconstructor(AnalyticalIterativePtychographyReconstructor):
    parameter_group: "pg.PlanarPtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PlanarPtychographyParameterGroup",
        dataset: Dataset,
        options: Optional["api.options.dm.DMReconstructorOptions"] = None,
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
        self.patch_placer = maps.get_patch_interp_function_by_enum(
            self.parameter_group.object.options.patch_interpolation_method, "placer"
        )

    def check_inputs(self, *args, **kwargs):
        if self.parameter_group.object.is_multislice:
            raise NotImplementedError("DMReconstructor only supports 2D objects.")
        if self.parameter_group.probe.has_multiple_opr_modes:
            raise NotImplementedError(
                "DMReconstructor does not support multiple OPR modes yet."
            )
        if (
            self.parameter_group.probe_positions.options.correction_options.correction_type
            != enums.PositionCorrectionTypes.GRADIENT
            and self.parameter_group.probe_positions.options.optimizable
        ):
            raise NotImplementedError(
                "DMReconstructor only supports gradient position correction at the moment."
            )
        if (
            self.parameter_group.object.options.patch_interpolation_method
            == enums.PatchInterpolationMethods.FOURIER
        ):
            warnings.warn(
                """
                Using fourier interpolation for extraction and placement typically leads to bad results!
                Use bilinear or nearest interpolation instead.
                """,
                category=UserWarning,
            )

    def build_dataloader(self):
        """
        Build the dataloader for the DM reconstructor. There is no
        batching in this reconstructor. Each batch is always a full,
        sequential sampling of the dataset.
        """
        data_loader_kwargs = {
            "dataset": self.dataset,
            "generator": torch.Generator(device=torch.get_default_device()),
            "batch_size": self.parameter_group.probe_positions.n_scan_points,
            "shuffle": False,
        }
        self.dataloader = DataLoader(**data_loader_kwargs)
        self.dataset.move_attributes_to_device(torch.get_default_device())

    def build_loss_tracker(self):
        if self.displayed_loss_function is None:
            self.displayed_loss_function = MSELossOfSqrt()
        return super().build_loss_tracker()

    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        y_pred = self.compute_updates(
            *input_data, y_true, self.dataset.valid_pixel_mask
        )
        self.loss_tracker.update_batch_loss_with_metric_function(y_pred, y_true)

    def compute_updates(
        self,
        indices: torch.Tensor,
        y_true: torch.Tensor,
        valid_pixel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Calculates the updates of the object and probe according to
        https://doi.org/10.1016/j.ultramic.2008.12.011.
        """

        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        indices = indices.cpu()
        positions = probe_positions.tensor[indices]

        obj_patches, y, psi_update = self.calculate_updated_exit_wave(
            indices, y_true, valid_pixel_mask
        )

        # The probe update must come before the object update
        if probe.optimization_enabled(self.current_epoch):
            self.update_probe(obj_patches)

        if object_.optimization_enabled(self.current_epoch):
            self.update_object(positions)

        if probe_positions.optimization_enabled(self.current_epoch):
            self.update_positions(indices, obj_patches, psi_update)

        return y

    def calculate_updated_exit_wave(
        self, indices: Tensor, y_true: Tensor, valid_pixel_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the updated exit wave according to equation (9) in
        this paper: https://doi.org/10.1016/j.ultramic.2008.12.011
        """
        # Mappings between this code and the paper:
        # - self.psi --> psi_n
        # - revised_psi --> 2 * Pi_o(psi_n)- psi_n
        # - new_psi --> Pi_o(psi_n)

        if self.current_epoch == 0:
            y = self.forward_model.forward(indices)
            self.psi = self.forward_model.intermediate_variables["psi"]

        # Calculate the new exit wave
        if self.current_epoch != 0:
            y = self.forward_model.forward(indices)
        new_psi = self.forward_model.intermediate_variables["psi"]
        obj_patches = self.forward_model.intermediate_variables["obj_patches"]

        revised_psi = self.forward_model.far_field_propagator.propagate_forward(
            2 * new_psi - self.psi
        )
        revised_psi = torch.where(
            valid_pixel_mask.repeat(
                revised_psi.shape[0], self.parameter_group.probe.n_modes, 1, 1
            ),
            self.replace_propagated_exit_wave_magnitude(revised_psi, y_true),
            revised_psi,
        )
        revised_psi = self.forward_model.far_field_propagator.propagate_backward(
            revised_psi
        )
        psi_update = (revised_psi - new_psi) * self.options.exit_wave_update_relaxation
        self.psi += psi_update

        # Save the difference map error
        dm_error = (psi_update.abs() ** 2).sum().sqrt()
        if self.current_epoch == 0:
            self.difference_map_error = torch.tensor([dm_error])
        else:
            self.difference_map_error = torch.cat(
                (self.difference_map_error, torch.tensor([dm_error]))
            )

        return obj_patches, y, psi_update

    def update_object(self, positions: Tensor):
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        p = probe.get_opr_mode(0)

        object_numerator = torch.zeros_like(object_.get_slice(0))
        object_numerator = self.patch_placer(
            object_numerator,
            positions + object_.center_pixel,
            (p.conj() * self.psi).sum(1),
            "add",
        )

        object_denominator = torch.zeros_like(
            object_.get_slice(0), dtype=object_.data.real.dtype
        )
        self.update_preconditioners()
        object_denominator = self.parameter_group.object.preconditioner

        updated_object = object_numerator / torch.sqrt(
            object_denominator**2 + (1e-3 * object_denominator.max()) ** 2
        )

        # Clamp the object amplitude
        idx = updated_object.abs() > object_.options.amplitude_clamp_limit
        updated_object[idx] = (
            updated_object[idx] / (updated_object[idx].abs())
        ) * object_.options.amplitude_clamp_limit

        object_.set_data(updated_object)

    def update_probe(self, obj_patches: Tensor):
        probe_numerator = (obj_patches.conj() * self.psi).sum(0)
        probe_denominator = (obj_patches.abs() ** 2).sum(0)
        self.parameter_group.probe.set_data(
            probe_numerator / (probe_denominator + 1e-10)
        )

    def update_positions(
        self, indices: Tensor, obj_patches: Tensor, exit_wave_update: Tensor
    ):
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        delta_pos = torch.zeros_like(probe_positions.data)
        delta_pos[indices] = probe_positions.position_correction.get_update(
            exit_wave_update,
            obj_patches,
            None,
            probe,
            self.parameter_group.opr_mode_weights,
            indices,
            None,
        )
        probe_positions.set_grad(-delta_pos)
        probe_positions.optimizer.step()
