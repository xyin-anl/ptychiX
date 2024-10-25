from typing import TYPE_CHECKING

import ptychi.position_correction as position_correction
import ptychi.data_structures.base as ds
if TYPE_CHECKING:
    import ptychi.api as api


class ProbePositions(ds.ReconstructParameter):
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
        self.update_magnitude_limit = options.update_magnitude_limit
        self.position_correction = position_correction.PositionCorrection(
            options=options.correction_options
        )

    def get_positions_in_pixel(self):
        return self.data