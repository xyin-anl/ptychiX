import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import (set_default_complex_dtype,
                          generate_initial_opr_mode_weights)
import os
os.environ['HDF5_PLUGIN_PATH'] = '/mnt/micdata3/ptycho_tools/DectrisFileReader/HDF5Plugin'
from .pear_io import (initialize_recon,
                      save_reconstructions,
                      create_reconstruction_path,
                      save_initial_conditions)
import numpy as np

import logging
logging.basicConfig(level=logging.ERROR)

def ptycho_recon(**params):

    # Set up computing device
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [params['gpu_id'] ]))
    import torch
    
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
 
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)
    set_default_complex_dtype(torch.complex64)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU device.")

    # Load data + preprocessing
    (dp, init_positions_px, init_probe, init_object, params) = initialize_recon(params)

    #recon parameters 
    options = api.LSQMLOptions()
    options.data_options.data = dp
    options.data_options.save_data_on_device = True
    options.data_options.wavelength_m = params['wavelength_m']
    #options.data_options.detector_pixel_size_m = det_pixel_size_m # Only useful for near-field ptycho
    
    options.object_options.initial_guess = init_object
    options.object_options.pixel_size_m = params['obj_pixel_size_m']
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.build_preconditioner_with_all_modes = False

    # multislice parameters
    if params['total_thickness_m'] > 0 and params['number_of_slices'] > 1:
        params['slice_distance_m'] = params['total_thickness_m'] / params['number_of_slices']
        options.object_options.slice_spacings_m = [params['slice_distance_m']] * (params['number_of_slices'] - 1)
        options.object_options.optimal_step_size_scaler = 0.9
        options.object_options.multislice_regularization.enabled = params['layer_regularization'] > 0
        options.object_options.multislice_regularization.weight = params['layer_regularization']
        options.object_options.multislice_regularization.unwrap_phase = True
        options.object_options.multislice_regularization.unwrap_image_grad_method = api.enums.ImageGradientMethods.FOURIER_DIFFERENTIATION
        options.object_options.multislice_regularization.unwrap_image_integration_method = api.enums.ImageIntegrationMethods.FOURIER
        if params['position_correction_layer'] and params['position_correction']:
            options.probe_position_options.correction_options.slice_for_correction = params['position_correction_layer']
    
    options.object_options.step_size = 1
    options.object_options.multimodal_update = params['update_object_w_higher_probe_modes']
    options.object_options.patch_interpolation_method = api.PatchInterpolationMethods.FOURIER
    options.object_options.remove_object_probe_ambiguity = api.options.base.RemoveObjectProbeAmbiguityOptions(enabled=True)
    
    options.probe_options.initial_guess = init_probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1
  
    options.probe_options.orthogonalize_incoherent_modes.enabled = True
    options.probe_options.orthogonalize_incoherent_modes.method = api.OrthogonalizationMethods.SVD
    options.probe_options.orthogonalize_opr_modes.enabled = True
    
    options.probe_options.center_constraint.enabled = params['center_probe']

    # position correction
    options.probe_position_options.position_x_px = init_positions_px[:, 1]
    options.probe_position_options.position_y_px = init_positions_px[:, 0]
    options.probe_position_options.optimizable = params['position_correction']
    options.probe_position_options.optimizer = api.Optimizers.SGD
    options.probe_position_options.step_size = 0.1
    options.probe_position_options.correction_options.correction_type = api.PositionCorrectionTypes.GRADIENT
    options.probe_position_options.correction_options.update_magnitude_limit = params['position_correction_update_limit']
    options.probe_position_options.affine_transform_constraint.enabled = True # alwayscalculate the affine matrix
    options.probe_position_options.affine_transform_constraint.apply_constraint = params['position_correction_affine_constraint']
    options.probe_position_options.affine_transform_constraint.position_weight_update_interval = np.inf # TODO: add to params
        
    # variable probe correction
    #options.probe_position_options.correction_options.gradient_method = api.PositionCorrectionGradientMethods.FOURIER
    if params['number_opr_modes'] > 0:
        options.opr_mode_weight_options.initial_weights = generate_initial_opr_mode_weights(len(init_positions_px), init_probe.shape[0])
        options.opr_mode_weight_options.optimizable = True
        options.opr_mode_weight_options.update_relaxation = 0.1
        options.opr_mode_weight_options.smoothing.enabled = False
        options.opr_mode_weight_options.smoothing.method = api.OPRWeightSmoothingMethods.MEDIAN
        options.opr_mode_weight_options.smoothing.polynomial_degree = 4

    options.opr_mode_weight_options.optimize_intensity_variation = params['intensity_correction']

    # convergence parameters
    options.reconstructor_options.batch_size = params['update_batch_size']
    #options.reconstructor_options.forward_model_options.pad_for_shift = 16
    #options.reconstructor_options.use_low_memory_forward_model = True
    if params['batch_selection_scheme'] == 'random':
        options.reconstructor_options.batching_mode = api.BatchingModes.RANDOM
    elif params['batch_selection_scheme'] == 'uniform':
        options.reconstructor_options.batching_mode = api.BatchingModes.UNIFORM
    elif params['batch_selection_scheme'] == 'compact':
        options.reconstructor_options.batching_mode = api.BatchingModes.COMPACT
        options.reconstructor_options.compact_mode_update_clustering = False
    if params['momentum_acceleration']:
        options.reconstructor_options.momentum_acceleration_gain = 0.5
        options.reconstructor_options.momentum_acceleration_gradient_mixing_factor = 1

    options.reconstructor_options.solve_step_sizes_only_using_first_probe_mode = True
    options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    options.reconstructor_options.num_epochs = params['number_of_iterations']
    options.reconstructor_options.use_double_precision_for_fft = False
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT32

    options.reconstructor_options.allow_nondeterministic_algorithms = True # a bit faster

    recon_path = create_reconstruction_path(params, options)
    
    save_initial_conditions(recon_path, params, options)

    task = PtychographyTask(options)
    
    for i in range(params['number_of_iterations'] // params['save_freq_iterations']):
        task.run(params['save_freq_iterations'])
        save_reconstructions(task, recon_path, params['save_freq_iterations']*(i+1))

