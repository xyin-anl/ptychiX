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
from .pear_utils import select_gpu
import numpy as np

import logging
logging.basicConfig(level=logging.ERROR)
import traceback
import time
from datetime import datetime  # Correct import for datetime.now()

def ptycho_recon(**params):

    if params['gpu_id'] is None:
        params['gpu_id'] = select_gpu(params)
        print(f"Auto-selected GPU: {params['gpu_id']}")
    else:
        print(f"Using GPU: {params['gpu_id']}")

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
    if params['object_thickness_m'] > 0 and params['number_of_slices'] > 1:
        params['slice_distance_m'] = params['object_thickness_m'] / params['number_of_slices']
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
    # Set batch size based on parameters
    if params['update_batch_size'] is not None:
        options.reconstructor_options.batch_size = params['update_batch_size']
    else:
        # For compact scheme, use all data points in a single batch
        # For other schemes, divide data into multiple batches
        num_of_batches = 1 if params['batch_selection_scheme'] == 'compact' else 10
        
        # Ensure batch size is at least 1 to prevent division by zero errors
        total_points = dp.shape[0]
        options.reconstructor_options.batch_size = max(1, total_points // num_of_batches)
        
        print(f"Auto-configured batch size: {options.reconstructor_options.batch_size} " 
              f"({num_of_batches} batches for {total_points} data points)")

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
        save_reconstructions(task, recon_path, params['save_freq_iterations']*(i+1), params)


def ptycho_batch_recon(start_scan, end_scan, base_params, log_dir_suffix='', scan_order='ascending'):
    """
    Process a range of scans with automatic error handling.
    
    Args:
        start_scan: First scan number to process
        end_scan: Last scan number to process (inclusive)
        base_params: Dictionary of parameters to use as a template
        log_dir_suffix: Suffix for the log directory
        scan_order: Order to process the scans
    """
    
    log_dir = os.path.join(base_params['data_directory'], 'ptychi_recons', 
                          base_params['recon_parent_dir'], 
                          f'recon_logs_{log_dir_suffix}' if log_dir_suffix else 'recon_logs')
    os.makedirs(log_dir, exist_ok=True)
    num_repeats = np.inf
    repeat_count = 0

    while repeat_count < num_repeats:
        successful_scans = []
        failed_scans = []
        ongoing_scans = []
        if scan_order == 'ascending':
            scan_list = list(range(start_scan, end_scan + 1))
        elif scan_order == 'descending':
            scan_list = list(range(end_scan, start_scan - 1, -1))
        elif scan_order == 'random':
            import random
            scan_list = list(range(start_scan, end_scan + 1))
            random.shuffle(scan_list)
        for scan_num in scan_list:
            # Create a copy of the parameters for this scan
            scan_params = base_params.copy()
            scan_params['scan_num'] = scan_num
            
            # Check if scan has already been processed
            log_files = {
                'done': os.path.join(log_dir, f'S{scan_num:04d}_done.txt'),
                'ongoing': os.path.join(log_dir, f'S{scan_num:04d}_ongoing.txt'),
                'failed': os.path.join(log_dir, f'S{scan_num:04d}_failed.txt')
            }
            
            # Delete the failed file if it exists to start fresh
            if os.path.exists(log_files['failed']):
                os.remove(log_files['failed'])
                
            if os.path.exists(log_files['done']):
                print(f"Scan {scan_num} already completed, skipping reconstruction")
                successful_scans.append(scan_num)
                continue
            if os.path.exists(log_files['ongoing']):
                print(f"Scan {scan_num} already ongoing, skipping reconstruction")
                ongoing_scans.append(scan_num)
                continue

            # Helper function to write to log files
            def write_log(file_path, content):
                with open(file_path, 'w' if 'ongoing' in file_path else 'a') as f:
                    f.write(content)
            
            # Create log file for ongoing reconstruction
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_content = [
                f"Starting reconstruction for scan {scan_num}",
                f"Timestamp: {timestamp}",
                f"Parameters:"
            ]
            log_content.extend([f"  {key}: {value}" for key, value in scan_params.items()])
            write_log(log_files['ongoing'], '\n'.join(log_content))
            
            print(f"Starting reconstruction for scan {scan_num}")
            start_time = time.time()
            
            try:
                # Run reconstruction
                ptycho_recon(**scan_params)
                
                # Handle successful completion
                elapsed_time = time.time() - start_time
                print(f"Scan {scan_num} completed successfully in {elapsed_time:.2f} seconds")
                successful_scans.append(scan_num)
                
                # Update log file
                os.rename(log_files['ongoing'], log_files['done'])
                completion_log = [
                    f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Elapsed time: {elapsed_time:.2f} seconds"
                ]
                write_log(log_files['done'], '\n'.join(completion_log))
                print(f"Waiting for 3 seconds before next scan...")
                time.sleep(3)
                
            except Exception as e:
                # Handle failure
                elapsed_time = time.time() - start_time
                print(f"Scan {scan_num} failed after {elapsed_time:.2f} seconds with error: {str(e)}")
                failed_scans.append((scan_num, str(e)))
                
                # Update log file
                os.rename(log_files['ongoing'], log_files['failed'])
                failure_log = [
                    f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Elapsed time: {elapsed_time:.2f} seconds",
                    f"Error: {str(e)}"
                ]
                write_log(log_files['failed'], '\n'.join(failure_log))

         # Print summary of processing
        #print(f"Batch processing complete. Summary:")
        print(f"Successfully processed scans: {successful_scans}")
        print(f"Failed scans: {[f[0] for f in failed_scans]}")
        print(f"Ongoing scans: {ongoing_scans}")
        if len(successful_scans) == end_scan - start_scan + 1:
            print(f"All scans completed successfully")
            break
        else:
            repeat_count += 1
            print(f"Waiting for 10 seconds...")
            time.sleep(10)
   
    print(f"Batch processing complete.")
