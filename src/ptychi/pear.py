import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import (set_default_complex_dtype,
                          generate_initial_opr_mode_weights)
import os
os.environ['HDF5_PLUGIN_PATH'] = '/mnt/micdata3/ptycho_tools/DectrisFileReader/HDF5Plugin'

from .pear_utils import select_gpu
import numpy as np

import logging
logging.basicConfig(level=logging.ERROR)
import traceback
import time
from datetime import datetime  # Correct import for datetime.now()
import uuid
import json
import tempfile
import shutil
import fcntl

def ptycho_recon(run_recon=True, **params):

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

    if params.get('beam_type', 'xray') == 'electron':
        from .pear_io_ele import (initialize_recon,
                              save_reconstructions,
                              create_reconstruction_path,
                              save_initial_conditions)
    else:
        from .pear_io import (initialize_recon,
                              save_reconstructions,
                              create_reconstruction_path,
                              save_initial_conditions)
        
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
    options.probe_options.support_constraint.enabled = params['probe_support']

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
    elif params['number_of_batches'] is not None:
        # Calculate batch size from number of batches
        total_data_points = dp.shape[0]
        options.reconstructor_options.batch_size = max(1, total_data_points // params['number_of_batches'])
    else:
        # Auto-configure based on batch selection scheme
        total_data_points = dp.shape[0]
        # Use smaller number of batches for 'compact' scheme
        num_of_batches = 1 if params['batch_selection_scheme'] == 'compact' else 10
        options.reconstructor_options.batch_size = max(1, total_data_points // num_of_batches)
        
        # Log the auto-configuration for transparency
        print(f"Auto-configured batch size: {options.reconstructor_options.batch_size} " 
              f"({num_of_batches} batches for {total_data_points} data points)")

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
    
    if run_recon:
        for i in range(params['number_of_iterations'] // params['save_freq_iterations']):
            task.run(params['save_freq_iterations'])
            save_reconstructions(task, recon_path, params['save_freq_iterations']*(i+1), params)
    
    torch.cuda.empty_cache()
    return task

class FileBasedTracker:
    def __init__(self, base_dir):
        """Initialize the tracker with a base directory for status files."""
        self.base_dir = base_dir
        self.status_dir = os.path.join(base_dir, 'status')
        self.lock_dir = os.path.join(base_dir, 'locks')
        
        # Create directories if they don't exist
        os.makedirs(self.status_dir, exist_ok=True)
        os.makedirs(self.lock_dir, exist_ok=True)
    
    def _get_status_file(self, scan_id):
        """Get the path to the status file for a scan."""
        return os.path.join(self.status_dir, f"scan_{scan_id:04d}.json")
    
    def _get_lock_file(self, scan_id):
        """Get the path to the lock file for a scan."""
        return os.path.join(self.lock_dir, f"scan_{scan_id:04d}.lock")
    
    def get_status(self, scan_id):
        """Get the current status of a scan."""
        status_file = self._get_status_file(scan_id)
        
        if not os.path.exists(status_file):
            return None
        
        try:
            with open(status_file, 'r') as f:
                data = json.load(f)
                return data.get('status')
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def start_recon(self, scan_id, worker_id, params):
        """
        Try to start a reconstruction for a scan.
        Returns True if successful, False if already in progress or completed.
        """
        lock_file = self._get_lock_file(scan_id)
        status_file = self._get_status_file(scan_id)
        
        # Create or open the lock file
        try:
            lock_fd = open(lock_file, 'w')
            # Try to acquire an exclusive, non-blocking lock
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, BlockingIOError):
            # Another process has the lock
            return False
        
        try:
            # Check if status file exists and scan is already done or in progress
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        data = json.load(f)
                        if data.get('status') in ['ongoing', 'done']:
                            return False
                except (json.JSONDecodeError, FileNotFoundError):
                    # Corrupted or missing status file, we can proceed
                    pass
            
            # Create a temporary file first to avoid partial writes
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                dir=self.status_dir,
                delete=False
            )
            
            # Prepare status data
            status_data = {
                'status': 'ongoing',
                'scan_id': scan_id,
                'worker_id': worker_id,
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                # No params included as requested
            }
            
            # Write to temporary file
            json.dump(status_data, temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            
            # Atomically move the temporary file to the final location
            shutil.move(temp_file.name, status_file)
            
            return True
            
        finally:
            # Release the lock
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
    
    def complete_recon(self, scan_id, success=True, error=None):
        """Mark a reconstruction as completed or failed."""
        lock_file = self._get_lock_file(scan_id)
        status_file = self._get_status_file(scan_id)
        
        # Acquire lock
        with open(lock_file, 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            
            try:
                # Read current status
                if os.path.exists(status_file):
                    with open(status_file, 'r') as f:
                        status_data = json.load(f)
                else:
                    # Create new status data if file doesn't exist
                    status_data = {
                        'scan_id': scan_id,
                        'start_time': 'unknown'
                    }
                
                # Update status
                status_data['status'] = 'done' if success else 'failed'
                status_data['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if error:
                    status_data['error'] = str(error)
                
                # Write to temporary file first
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w', 
                    dir=self.status_dir,
                    delete=False
                )
                
                # Write status data line by line
                temp_file.write("{\n")
                for i, (key, value) in enumerate(status_data.items()):
                    if isinstance(value, str):
                        temp_file.write(f'    "{key}": "{value}"')
                    else:
                        # Convert the value to JSON format
                        json_value = json.dumps(value)
                        # Write the key-value pair with proper formatting
                        temp_file.write(f'    "{key}": {json_value}')
                    
                    if i < len(status_data) - 1:
                        temp_file.write(",\n")
                    else:
                        temp_file.write("\n")
                temp_file.write("}\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file.close()
                
                # Atomically move the temporary file to the final location
                shutil.move(temp_file.name, status_file)
                
            except Exception as e:
                print(f"Error updating status for scan {scan_id}: {str(e)}")

def ptycho_batch_recon(start_scan, end_scan, base_params, log_dir_suffix='', scan_order='ascending', exclude_scans=[]):
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
    
    # Create tracker
    tracker = FileBasedTracker(log_dir)
    
    # Generate a unique worker ID
    worker_id = f"worker_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    
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

        scan_list = [scan for scan in scan_list if scan not in exclude_scans]
        
        for scan_num in scan_list:
            # Create a copy of the parameters for this scan
            scan_params = base_params.copy()
            
            scan_params['scan_num'] = scan_num
            
            # Check status using tracker
            status = tracker.get_status(scan_num)
            
            if status == 'done':
                print(f"Scan {scan_num} already completed, skipping reconstruction")
                successful_scans.append(scan_num)
                continue
                
            if status == 'ongoing':
                print(f"Scan {scan_num} already ongoing, skipping reconstruction")
                ongoing_scans.append(scan_num)
                continue
            
            # Try to start reconstruction
            if not tracker.start_recon(scan_num, worker_id, scan_params):
                print(f"Could not acquire lock for scan {scan_num}, skipping")
                continue
                
            print(f"\033[91mStarting reconstruction for scan {scan_num}\033[0m")
            start_time = time.time()
            
            try:
                # Run reconstruction
                ptycho_recon(run_recon=True, **scan_params)
                
                # Handle successful completion
                elapsed_time = time.time() - start_time
                print(f"Scan {scan_num} completed successfully in {elapsed_time:.2f} seconds")
                successful_scans.append(scan_num)
                
                # Update status
                tracker.complete_recon(scan_num, success=True)
                
                print(f"Waiting for 5 seconds before next scan...")
                time.sleep(5)

                if scan_order == 'descending':
                    # Break the for loop after processing the current scan
                    break
                    
            except Exception as e:
                # Handle failure
                elapsed_time = time.time() - start_time
                print(f"Scan {scan_num} failed after {elapsed_time:.2f} seconds with error: {str(e)}")
                failed_scans.append((scan_num, str(e)))
                
                # Update status
                tracker.complete_recon(scan_num, success=False, error=str(e))
        
        # Print summary of processing
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
