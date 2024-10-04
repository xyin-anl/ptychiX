import argparse
import os

import torch
import numpy as np

from ptychointerim.ptychotorch.data_structures import *
from ptychointerim.ptychotorch.io_handles import PtychographyDataset
from ptychointerim.forward_models import Ptychography2DForwardModel
from ptychointerim.ptychotorch.utils import (get_suggested_object_size, set_default_complex_dtype, get_default_complex_dtype, 
                                             rescale_probe, generate_initial_opr_mode_weights)
from ptychointerim.ptychotorch.reconstructors import *

import test_utils as tutils


def test_2d_ptycho_lsqml(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_2d_ptycho_lsqml'
    
    tutils.setup(name, cpu_only=True)

    dataset, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true')
    
    object = Object2D(
        data=torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype()), 
        pixel_size_m=pixel_size_m,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1}
    )
    
    probe = Probe(
        data=probe,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1}
    )
    
    probe_positions = ProbePositions(
        data=positions_px,
        optimizable=False,
    )

    reconstructor = LSQMLReconstructor(
        variable_group=Ptychography2DVariableGroup(object=object, probe=probe, probe_positions=probe_positions),
        dataset=dataset,
        batch_size=96,
        noise_model='gaussian',
        n_epochs=8
    )
    reconstructor.build()
    reconstructor.run()

    recon = reconstructor.variable_group.object.tensor.complex().detach().cpu().numpy()

    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=high_tol)
    
    
def test_2d_ptycho_lsqml_poscorr(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_2d_ptycho_lsqml_poscorr'
    
    tutils.setup(name, cpu_only=True)

    dataset, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='nominal')
    
    object = Object2D(
        data=torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype()), 
        pixel_size_m=pixel_size_m,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1}
    )

    probe = Probe(
        data=probe,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1}
    )
    
    probe_positions = ProbePositions(
        data=positions_px,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 0.1},
        update_magnitude_limit=2.0
    )

    reconstructor = LSQMLReconstructor(
        variable_group=Ptychography2DVariableGroup(object=object, probe=probe, probe_positions=probe_positions),
        dataset=dataset,
        batch_size=96,
        noise_model='gaussian',
        n_epochs=8
    )
    reconstructor.build()
    reconstructor.run()

    recon = reconstructor.variable_group.object.tensor.complex().detach().cpu().numpy()
    
    if debug and not generate_gold:
        import matplotlib.pyplot as plt
        pos = reconstructor.variable_group.probe_positions.tensor.detach().cpu().numpy()
        pos_true = tutils.load_tungsten_data(pos_type='true')[3]
        plt.figure()
        plt.plot(pos[:, 1], pos[:, 0], label='corrected')
        plt.plot(pos_true[:, 1], pos_true[:, 0], label='true')
        plt.show()
    
    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=high_tol)
    

def test_2d_ptycho_lsqml_opr(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_2d_ptycho_lsqml_opr'
    
    tutils.setup(name, cpu_only=True)

    dataset, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true', additional_opr_modes=3)
    
    object = Object2D(
        data=torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype()), 
        pixel_size_m=pixel_size_m,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1}
    )

    probe = Probe(
        data=probe,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1},
        eigenmode_update_relaxation=0.1
    )
    
    probe_positions = ProbePositions(
        data=positions_px,
        optimizable=False,
    )
    
    opr_mode_weights = OPRModeWeights(
        data=generate_initial_opr_mode_weights(len(positions_px), probe.shape[0], eigenmode_weight=0.1),
        optimizable=True,
        update_relaxation=0.1
    )

    reconstructor = LSQMLReconstructor(
        variable_group=Ptychography2DVariableGroup(object=object, probe=probe, probe_positions=probe_positions, opr_mode_weights=opr_mode_weights),
        dataset=dataset,
        batch_size=44,
        noise_model='gaussian',
        n_epochs=8,
    )
    reconstructor.build()
    reconstructor.run()

    recon = reconstructor.variable_group.object.tensor.complex().detach().cpu().numpy()

    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=high_tol)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    parser.add_argument('--high-tol', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_lsqml(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    test_2d_ptycho_lsqml_poscorr(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    test_2d_ptycho_lsqml_opr(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)