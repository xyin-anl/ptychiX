import argparse
import os

import torch
import numpy as np

from ptychointerim.ptychotorch.data_structures import *
from ptychointerim.ptychotorch.io_handles import PtychographyDataset
from ptychointerim.forward_models import Ptychography2DForwardModel
from ptychointerim.ptychotorch.utils import (get_suggested_object_size, set_default_complex_dtype, get_default_complex_dtype, 
                            rescale_probe)
from ptychointerim.ptychotorch.reconstructors import *
from ptychointerim.metrics import MSELossOfSqrt
from ptychointerim.maths import orthogonalize_gs

import test_utils as tutils


def test_2d_ptycho_epie_mixed_states(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")

    name = 'test_2d_ptycho_epie_mixed_states'
    
    tutils.setup(name, cpu_only=True)
    
    dataset, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(additional_opr_modes=0)

    probe = orthogonalize_gs(torch.tensor(probe), (-1, -2), 1)
    probe = rescale_probe(probe[0], dataset.patterns)[None]
        
    object = Object2D(
        data=torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype()), 
        pixel_size_m=pixel_size_m,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1e-1}
    )

    probe = Probe(
        data=probe,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1e-1}
    )
    
    probe_positions = ProbePositions(
        data=positions_px,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1000}
    )

    reconstructor = EPIEReconstructor(
        variable_group=Ptychography2DVariableGroup(object=object, probe=probe, probe_positions=probe_positions),
        dataset=dataset,
        batch_size=96,
        n_epochs=32,
    )
    reconstructor.build()
    reconstructor.run()

    recon = reconstructor.variable_group.object.tensor.complex().detach().cpu().numpy()
    
    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.abs(recon))
        ax[1].imshow(np.angle(recon))
        plt.show()
    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=high_tol)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    parser.add_argument('--high-tol', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_epie_mixed_states(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    