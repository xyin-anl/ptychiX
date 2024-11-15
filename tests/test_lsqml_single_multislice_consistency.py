"""
Tests the consistency of `update_object_and_probe` and `update_object_and_probe_multislice`
in `LSQMLReconstructor`. They should behave the same on a single-slice object when the
probe and the first object slice's step sizes are solved joinly.
"""

import argparse
import random

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import (
    get_suggested_object_size,
    get_default_complex_dtype,
    generate_initial_opr_mode_weights,
)

import test_utils as tutils


def run_iterations(reconstructor, n_iters=3, use_multislice_function=True):
    torch.random.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    reconstructor.run_pre_run_hooks()
    reconstructor.run_pre_epoch_hooks()
    count = 0
    for batch_data in reconstructor.dataloader:
        input_data = [x.to(torch.get_default_device()) for x in batch_data[:-1]]
        y_true = batch_data[-1].to(torch.get_default_device())
        reconstructor.run_pre_update_hooks()
        indices = input_data[0]
        y_pred = reconstructor.forward_model(*input_data)

        psi_opt = reconstructor.run_reciprocal_space_step(y_pred, y_true, indices)
        reconstructor.run_real_space_step(psi_opt, indices)

        positions = reconstructor.forward_model.intermediate_variables["positions"]
        psi_0 = reconstructor.forward_model.intermediate_variables["psi"]
        chi = psi_opt - psi_0  # Eq, 19
        obj_patches = reconstructor.forward_model.intermediate_variables["obj_patches"]

        if use_multislice_function:
            _ = reconstructor.update_object_and_probe_multislice(
                indices, chi, obj_patches, positions
            )
        else:
            _ = reconstructor.update_object_and_probe(indices, chi, obj_patches, positions)
        count += 1
        if count >= n_iters:
            break
    return (
        reconstructor.parameter_group.object.data.detach().cpu().numpy(),
        reconstructor.parameter_group.probe.data.detach().cpu().numpy(),
    )


def test_lsqml_single_multislice_consistency(pytestconfig, **kwargs):
    name = "test_lsqml_single_multislice_consistency"

    tutils.setup(name, cpu_only=False, gpu_indices=[0])

    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type="true")

    options = api.LSQMLOptions()
    options.data_options.data = data

    options.object_options.initial_guess = torch.ones(
        [1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)],
        dtype=get_default_complex_dtype(),
    )
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1

    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1

    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.optimizable = False

    options.reconstructor_options.batch_size = 96
    options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    options.reconstructor_options.num_epochs = 8

    options.reconstructor_options.solve_obj_prb_step_size_jointly_for_first_slice_in_multislice = True
    task = PtychographyTask(options)
    reconstructor = task.reconstructor
    obj_ms, probe_ms = run_iterations(reconstructor, use_multislice_function=True)
    
    task = PtychographyTask(options)
    reconstructor = task.reconstructor
    obj_ss, probe_ss = run_iterations(reconstructor, use_multislice_function=False)

    tutils.compare_data(obj_ms, obj_ss, name=name)
    tutils.compare_data(probe_ms, probe_ss, name=name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-gold", action="store_true")
    parser.add_argument("--high-tol", action="store_true")
    args = parser.parse_args()

    test_lsqml_single_multislice_consistency(
        None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol
    )
