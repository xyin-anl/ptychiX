from typing import TYPE_CHECKING

import ptychi.position_correction as position_correction
import ptychi.data_structures.base as dsbase
if TYPE_CHECKING:
    import ptychi.api as api


class ProbePositions(dsbase.ReconstructParameter):
    options: "api.options.base.ProbePositionOptions"

    def __init__(
        self,
        *args,
        name: str = "probe_positions",
        options: "api.options.base.ProbePositionOptions" = None,
        **kwargs,
    ):
        """
        Probe positions.

        :param data: a tensor of shape (N, 2) giving the probe positions in pixels.
            Input positions should be in row-major order, i.e., y-positions come first.
        """
        super().__init__(*args, name=name, options=options, is_complex=False, **kwargs)
        self.position_correction = position_correction.PositionCorrection(
            options=options.correction_options
        )

    @property
    def n_scan_points(self):
        return len(self.data)
    
    def get_slice_for_correction(self, n_slices: int = None):
        i_slice = self.options.correction_options.slice_for_correction
        if i_slice is None:
            if n_slices is None:
                raise ValueError(
                    "When `slice_for_correction` is not set, `n_slices` must "
                    "be provided to determine the middle slice."
                )
            i_slice = n_slices // 2
        return i_slice

    def get_positions_in_pixel(self):
        return self.data
    
    def position_mean_constraint_enabled(self, current_epoch: int):
        return self.options.constrain_position_mean and self.optimization_enabled(current_epoch)
    
    def constrain_position_mean(self):
        data = self.data
        data = data - data.mean(0)
        self.set_data(data)
