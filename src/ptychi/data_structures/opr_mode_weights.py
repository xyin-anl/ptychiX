from typing import Union, TYPE_CHECKING
import logging

from torch import Tensor
import torch

import ptychi.data_structures.base as ds
import ptychi.image_proc as ip
import ptychi.maths as pmath
if TYPE_CHECKING:
    import ptychi.api as api
    
logger = logging.getLogger(__name__)


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
        
    @property
    def n_opr_modes(self):
        return self.tensor.shape[1]
    
    @property
    def n_scan_points(self):
        return self.tensor.shape[0]
    
    @property
    def n_eigenmodes(self):
        return self.tensor.shape[1] - 1

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
    
    def weight_smoothing_enabled(self, epoch: int):
        return self.optimization_enabled(epoch) and self.options.smoothing_method is not None
    
    def smooth_weights(self):
        """
        Smooth the weights with a median filter.
        """
        weights = self.data
        if self.options.smoothing_method == "median":
            if self.n_scan_points < 81:
                logger.warning("OPR weight smoothing with median filter could "
                               "not run because the number of scan points is less than 81.")
                return
            weights = ip.median_filter_1d(weights.T, window_size=81).T
        elif self.options.smoothing_method == "polynomial":
            if self.n_scan_points < self.options.polynomial_smoothing_degree:
                logger.warning("OPR weight smoothing with polynomial filter could "
                               "not run because the number of scan points is less than the "
                               "polynomial smoothing degree ({}).".format(
                                   self.options.polynomial_smoothing_degree))
                return
            inds = torch.arange(self.n_scan_points, device=weights.device, dtype=weights.dtype)
            for i_opr_mode in range(1, self.n_opr_modes):
                weights_current_mode = weights[:, i_opr_mode]
                fit_coeffs = pmath.polyfit(inds, weights_current_mode, deg=self.options.polynomial_smoothing_degree)
                weights_smoothed = pmath.polyval(inds, fit_coeffs)
                weights[:, i_opr_mode] = 0.5 * weights_current_mode + 0.5 * weights_smoothed
        self.set_data(weights)
    
    def remove_outliers(self):
        aevol = torch.abs(self.data)
        weights = torch.min(aevol, 1.5 * torch.quantile(aevol, 0.95)) * torch.sign(self.data)
        self.set_data(weights)
