import argparse

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.ptychotorch.utils import get_suggested_object_size, get_default_complex_dtype

import test_utils as tutils


def test_2d_ptycho_dm(generate_gold=False, debug=False):
    name = 'test_2d_ptycho_dm'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])
    
    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(additional_opr_modes=0)
    
    options = api.DMOptions()
    
    options.data_options.data = data
    
    options.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    
    options.probe_options.initial_guess = probe
    options.probe_options.probe_power = np.sum(np.max(data, axis=-3), axis=(-2, -1))
    options.probe_options.optimizable = True

    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.optimizable = False
    
    options.reconstructor_options.num_epochs = 8
    
    task = PtychographyTask(options)
    task.run()

    recon = task.get_data_to_cpu('object', as_numpy=True)[0]
    
    if debug and not generate_gold:
        tutils.plot_complex_image(recon)
    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_dm(generate_gold=args.generate_gold, debug=True)
    