import os, glob, time
import h5py
import hdf5plugin # for reading raw hdf5 files
import numpy as np
import scipy.ndimage
from tifffile import imwrite
from ptychi.image_proc import unwrap_phase_2d
import ptychi.api as api
from ptychi.utils import (add_additional_opr_probe_modes_to_probe, 
                          to_tensor,
                          orthogonalize_initial_probe,
                          get_suggested_object_size,
                          get_default_complex_dtype)
from ptychi.maths import decompose_2x2_affine_transform_matrix
import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle
from ptychi.pear_utils import make_fzp_probe, resize_complex_array
import sys
import torch
import json

def save_reconstructions(task, recon_path, iter, params):
    pixel_size_um = task.object_options.pixel_size_m * 1e6
    
    # Object
    recon_object = task.get_data_to_cpu('object', as_numpy=True)
    recon_object_roi = task.object.get_object_in_roi().cpu().detach()
    if recon_object_roi.shape[0] > 1:  # multislice recon
        object_ph_stack = [normalize_by_bit_depth(unwrap_phase_2d(slice.cuda(), 
                                                                 image_grad_method='fourier_differentiation',
                                                                 image_integration_method='fourier').cpu(), '16')
                          for slice in recon_object_roi]
        imwrite(f'{recon_path}/object_ph_layers/object_ph_layers_Niter{iter}.tiff',
                np.array(object_ph_stack),
                photometric='minisblack',
                resolution=(1 / pixel_size_um, 1 / pixel_size_um),
                metadata={'unit': 'um', 'pixel_size': pixel_size_um},
                imagej=True)

        object_ph_sum = normalize_by_bit_depth(unwrap_phase_2d(torch.prod(recon_object_roi, dim=0).cuda(),
                                                              image_grad_method='fourier_differentiation',
                                                              image_integration_method='fourier').cpu(), '16')
        imwrite(f'{recon_path}/object_ph_total/object_ph_total_Niter{iter}.tiff',
                np.array(object_ph_sum),
                photometric='minisblack',
                resolution=(1 / pixel_size_um, 1 / pixel_size_um),
                metadata={'unit': 'um', 'pixel_size': pixel_size_um},
                imagej=True)
        
        object_mag_stack = [normalize_by_bit_depth(np.abs(slice), '16')
                           for slice in recon_object_roi]
        imwrite(f'{recon_path}/object_mag_layers/object_mag_layers_Niter{iter}.tiff',
                np.array(object_mag_stack),
                photometric='minisblack',
                resolution=(1 / pixel_size_um, 1 / pixel_size_um),
                metadata={'unit': 'um', 'pixel_size': pixel_size_um},
                imagej=True)
        
        object_mag_sum = normalize_by_bit_depth(np.abs(torch.prod(recon_object_roi, dim=0)).cpu(), '16')
        imwrite(f'{recon_path}/object_mag_total/object_mag_total_Niter{iter}.tiff',
                np.array(object_mag_sum),
                photometric='minisblack',
                resolution=(1 / pixel_size_um, 1 / pixel_size_um),
                metadata={'unit': 'um', 'pixel_size': pixel_size_um},
                imagej=True)

    else:
        #imwrite(f'{recon_path}/object_ph/object_ph_roi_Niter{iter}.tiff', normalize_by_bit_depth(np.angle(recon_object_roi[0,]), '16'))
        object_ph_unwrapped = unwrap_phase_2d(recon_object_roi[0,].cuda(), image_grad_method='fourier_differentiation', image_integration_method='fourier')
        imwrite(f'{recon_path}/object_ph/object_ph_Niter{iter}.tiff', 
                normalize_by_bit_depth(object_ph_unwrapped.cpu(), '16'),
                photometric='minisblack',
                resolution=(1 / pixel_size_um, 1 / pixel_size_um),
                metadata={'unit': 'um', 'pixel_size': pixel_size_um},
                imagej=True)
        imwrite(f'{recon_path}/object_mag/object_mag_Niter{iter}.tiff',
                normalize_by_bit_depth(np.abs(recon_object_roi[0,]), '16'),
                photometric='minisblack',
                resolution=(1 / pixel_size_um, 1 / pixel_size_um),
                metadata={'unit': 'um', 'pixel_size': pixel_size_um},
                imagej=True)
    
    # Calculate the phase 
    #  recon_object_roi_ph = unwrap_phase_2d(recon_object_roi[0,].cuda(), image_grad_method='fourier_differentiation', image_integration_method='fourier')
    # imwrite(f'{recon_path}/object_ph/object_ph_unwrap_roi_Niter{iter}.tiff', normalize_by_bit_depth(recon_object_roi_ph.cpu(), '16'))
    
    # Probe
    recon_probe = task.get_data_to_cpu('probe', as_numpy=True)
    probe_mag = np.hstack(np.abs(recon_probe[0,]))
    N_probe = probe_mag.shape[0]
    #plt.imsave(f'{recon_path}/probe_mag/probe_mag_Niter{iter}.png', probe_mag, cmap='plasma')

    norm = plt.Normalize(vmin=probe_mag.min(), vmax=probe_mag.max())
    cmap = plt.cm.plasma
    colored_probe_mag = cmap(norm(probe_mag))  # This creates an RGBA array
    colored_probe_mag = (colored_probe_mag[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB uint8

    # Save with ImageJ-compatible resolution information
    imwrite(f'{recon_path}/probe_mag/probe_mag_Niter{iter}.tiff', 
            colored_probe_mag,
            photometric='rgb',
            resolution=(1 / pixel_size_um, 1 / pixel_size_um),
            metadata={'unit': 'um', 'pixel_size': pixel_size_um},
            imagej=True)
    
    # Save probe propagation in multislice reconstruction
    if recon_object_roi.shape[0] > 1:
        from ptychi.pear_utils import near_field_evolution
        
        # Get the primary probe mode
        # Extract the primary probe mode based on dimensionality
        probe = recon_probe[0, 0] if recon_probe.ndim == 4 else (
                recon_probe[0] if recon_probe.ndim == 3 else recon_probe)
        
        # Check if slice_spacings attribute exists in the correct location
        # Get slice spacings directly from object_options
        slice_spacings = task.object_options.slice_spacings_m
        n_layers = len(slice_spacings) + 1
        
        # Initialize array to store propagated probes
        probe_propagation = np.zeros((n_layers, probe.shape[0], probe.shape[1]), dtype=np.complex64)
        probe_propagation[0] = probe  # First layer is the original probe
        
        # Physical size of the array (extent)
        extent = probe.shape[0] * task.object_options.pixel_size_m
        
        # Get wavelength
        if hasattr(task.reconstructor.parameter_group, 'wavelength'):
            wavelength = task.reconstructor.parameter_group.wavelength.cpu().numpy()
        else:
            wavelength = task.data_options.wavelength_m
        
        # Propagate probe through each layer
        for i in range(1, n_layers):
            # Propagate from previous layer to current layer
            z_distance = slice_spacings[i-1]
            try:
                u_1, _, _, _ = near_field_evolution(
                    probe_propagation[i-1], 
                    z_distance, 
                    wavelength, 
                    extent,
                    use_ASM_only=True
                )
                probe_propagation[i] = u_1
            except Exception as e:
                print(f"Warning: Error during probe propagation at layer {i}: {str(e)}")
                # Copy previous layer as fallback
                probe_propagation[i] = probe_propagation[i-1]
        
        # Create colored versions of the propagated probes
        colored_probes_mag_stack = []
        for i in range(n_layers):
            probe_mag = np.abs(probe_propagation[i])
            # Normalize the probe magnitude
            norm = plt.Normalize(vmin=probe_mag.min(), vmax=probe_mag.max())
            # Apply colormap
            colored_probe = cmap(norm(probe_mag))  # This creates an RGBA array
            # Convert to RGB uint8
            colored_probe_rgb = (colored_probe[:, :, :3] * 255).astype(np.uint8)
            colored_probes_mag_stack.append(colored_probe_rgb)
        
        # Save as tiff stack
        imwrite(f'{recon_path}/probe_propagation_mag/probe_propagation_mag_Niter{iter}.tiff',
                np.array(colored_probes_mag_stack),
                photometric='rgb',
                resolution=(1 / pixel_size_um, 1 / pixel_size_um),
                metadata={'unit': 'um', 'pixel_size': pixel_size_um},
                imagej=True)
    
    # # Create figure and axis
    # fig, ax = plt.subplots()

    # # Display the image
    # im = ax.imshow(probe_mag, cmap='plasma')

    # # Add scale bar
    # bar_length_pixels = 1e-6/task.object_options.pixel_size_m  # Length of scale bar in pixels
    # bar_width_pixels = N_probe/30    # Width of scale bar in pixels
    # bar_position_x = N_probe/20     # from left edge
    # bar_position_y = N_probe*0.9  # from bottom

    # # Create and add the scale bar
    # scale_bar = Rectangle((bar_position_x, bar_position_y), 
    #                     bar_length_pixels, bar_width_pixels, 
    #                     fc='white', ec='none')
    # ax.add_patch(scale_bar)

    # # Optional: Add text above scale bar
    # plt.text(bar_position_x + bar_length_pixels/2, 
    #         bar_position_y - 3, 
    #         '1 um', 
    #         color='white', 
    #         ha='center', 
    #         va='bottom',
    #         fontsize=5)

    # # Remove axes
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_frame_on(False)

    # # Save the figure
    # plt.savefig(f'{recon_path}/probe_mag/probe_mag_Niter{iter}.png', 
    #             dpi=500, 
    #             bbox_inches='tight', 
    #             pad_inches=0)
    # plt.close()

    #plt.imsave(f'{recon_path}/probe_mag/probe_mag_Niter{iter}.tiff', normalize_by_bit_depth(probe_mag, '16'), cmap='plasma')
    #imwrite(f'{recon_path}/probe_mag/pxrobe_mag_Niter{iter}.tiff', normalize_by_bit_depth(probe_mag, '16'))
    # Save probe at each scan position
    if recon_probe.shape[0] > 1 and params.get('save_probe_at_each_scan_position', False):
        opr_mode_weights = task.reconstructor.parameter_group.opr_mode_weights.data.cpu().detach().numpy()
        probes = task.reconstructor.parameter_group.probe.get_unique_probes(task.reconstructor.parameter_group.opr_mode_weights.data, mode_to_apply=0)
        probes = probes[:,0,:,:].cpu().detach().numpy() # only keep the primary mode
        
        # Create a colored version of each probe magnitude
        colored_probes_mag_stack = []
        for i in range(min(probes.shape[0], 500)):
            probe_mag = np.abs(probes[i])
            # Normalize the probe magnitude
            norm = plt.Normalize(vmin=probe_mag.min(), vmax=probe_mag.max())
            # Apply colormap
            colored_probe = cmap(norm(probe_mag))  # This creates an RGBA array
            # Convert to RGB uint8
            colored_probe_rgb = (colored_probe[:, :, :3] * 255).astype(np.uint8)
            colored_probes_mag_stack.append(colored_probe_rgb)
        
        # Save as tiff stack
        imwrite(f'{recon_path}/probe_mag_opr/probes_mag_opr_Niter{iter}.tiff',
                np.array(colored_probes_mag_stack),
                photometric='rgb',
                resolution=(1 / pixel_size_um, 1 / pixel_size_um),
                metadata={'unit': 'um', 'pixel_size': pixel_size_um},
                imagej=True)
        
    # Save scan positions
    if params['position_correction']:
        scan_positions = task.get_data_to_cpu('probe_positions', as_numpy=True)
        plt.figure()
        plt.scatter(-init_positions_x_um, init_positions_y_um, s=1, edgecolors='blue')
        plt.scatter(-scan_positions[:, 1]*task.object_options.pixel_size_m/1e-6, scan_positions[:, 0]*task.object_options.pixel_size_m/1e-6, s=10, edgecolors='red', facecolors='none')
        # Calculate average position differences
        x_diff = np.mean(np.abs(-scan_positions[:, 1]*task.object_options.pixel_size_m/1e-6 - (-init_positions_x_um)))
        y_diff = np.mean(np.abs(scan_positions[:, 0]*task.object_options.pixel_size_m/1e-6 - init_positions_y_um))
        plt.xlabel(f'X [um] (average error: {x_diff*1e3:.2f} nm)')
        plt.ylabel(f'Y [um] (average error: {y_diff*1e3:.2f} nm)')
        plt.legend(['Initial positions', 'Refined positions'], loc='upper center', bbox_to_anchor=(0.5, 1.15))
        plt.grid(True)
        plt.xlim(pos_x_min*range_factor, pos_x_max*range_factor)
        plt.ylim(pos_y_min*range_factor, pos_y_max*range_factor)
        plt.savefig(f'{recon_path}/positions/positions_Niter{iter}.png', dpi=300)
        plt.close()

        # Plot affine transformation parameters from probe position correction
        mat = task.reconstructor.parameter_group.probe_positions.affine_transform_matrix
        
        # Extract transformation parameters
        scale, asymmetry, rotation, shear = decompose_2x2_affine_transform_matrix(mat[:,:-1])
        
        # Store parameters for plotting
        pos_scale.append(scale.cpu().item())
        pos_assymetry.append(asymmetry.cpu().item())
        pos_rotation.append(rotation.cpu().item())
        pos_shear.append(shear.cpu().item())
        iterations.append(iter)
        
        # Create figure with subplots for each transformation parameter
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        
        # Plot scale factor
        axs[0, 0].plot(iterations, pos_scale, 'o-', color='blue')
        axs[0, 0].set_ylim(0.97, 1.03)
        axs[0, 0].set_xlabel('Iterations')
        axs[0, 0].set_ylabel('Scale Factor')
        axs[0, 0].set_title('Scale')
        axs[0, 0].grid(True)
        
        # Plot asymmetry
        axs[0, 1].plot(iterations, pos_assymetry, 'o-', color='blue')
        axs[0, 1].set_xlabel('Iterations')
        axs[0, 1].set_ylabel('Asymmetry')
        axs[0, 1].set_title('Asymmetry')
        axs[0, 1].grid(True)
        
        # Plot rotation
        axs[1, 0].plot(iterations, pos_rotation, 'o-', color='blue')
        axs[1, 0].set_xlabel('Iterations')
        axs[1, 0].set_ylabel('Rotation (rad)')
        axs[1, 0].set_title('Rotation')
        axs[1, 0].grid(True)
        
        # Plot shear
        axs[1, 1].plot(iterations, pos_shear, 'o-', color='blue')
        axs[1, 1].set_xlabel('Iterations')
        axs[1, 1].set_ylabel('Shear')
        axs[1, 1].set_title('Shear')
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f'{recon_path}/positions_affine/positions_affine_Niter{iter}.png', dpi=300)
        plt.close(fig)
        
    # Plot loss vs iterations
    loss = task.reconstructor.loss_tracker.table['loss']

    plt.figure()
    plt.plot(loss, label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    loss_array = loss.values
    min_loss = min(loss_array)
    final_loss = loss_array[-1] if len(loss_array) > 0 else 0
    plt.title(f'Min Loss: {min_loss:.2e}. Final Loss: {final_loss:.2e}')
    plt.savefig(f'{recon_path}/loss/loss_Niter{iter}.png', dpi=300)
    plt.close()

    # Save results in hdf5 format
    with h5py.File(f'{recon_path}/recon_Niter{iter}.h5', 'w') as hdf_file:
        hdf_file.create_dataset('probe', data=recon_probe)
        hdf_file.create_dataset('object', data=recon_object)
        hdf_file.create_dataset('loss', data=loss)
        hdf_file.create_dataset('positions_px', data=scan_positions)
        hdf_file.create_dataset('obj_pixel_size_m', data=task.object_options.pixel_size_m)
        if recon_probe.shape[0] > 1:
            hdf_file.create_dataset('opr_mode_weights', data=opr_mode_weights)
    
    if params['number_of_iterations'] == iter:
        if params.get('collect_object_phase', False):  # copy final recon to a collection folder
            recon_object = task.get_data_to_cpu('object', as_numpy=False)
            obj_ph_collection_dir = os.path.join(params['data_directory'], 'ptychi_recons', params['recon_parent_dir'], 'object_ph_collection')
            os.makedirs(obj_ph_collection_dir, exist_ok=True)
            object_full_ph_unwrapped = unwrap_phase_2d(recon_object[0,].cuda(), image_grad_method='fourier_differentiation', image_integration_method='fourier')
            print(f"Saving final object phase to {obj_ph_collection_dir}/S{params['scan_num']:04d}.tiff")
            imwrite(f'{obj_ph_collection_dir}/S{params["scan_num"]:04d}.tiff', 
                    normalize_by_bit_depth(object_full_ph_unwrapped.cpu(), '16'),
                    photometric='minisblack',
                    resolution=(1 / pixel_size_um, 1 / pixel_size_um),
                    metadata={'unit': 'um', 'pixel_size': pixel_size_um},
                    imagej=True)

def create_reconstruction_path(params, options):
    # Construct the base reconstruction path
    recon_path_base = os.path.join(params['data_directory'], 'ptychi_recons', params['recon_parent_dir'], f'S{params['scan_num']:04d}')
   
    # Append batching mode to the path
    batching_mode_suffix = {
        api.BatchingModes.RANDOM: 'r',
        api.BatchingModes.UNIFORM: 's',
        api.BatchingModes.COMPACT: 'c'
    }.get(options.reconstructor_options.batching_mode, '')

    recon_path = recon_path_base + f'/Ndp{options.data_options.data.shape[1]}_LSQML_{batching_mode_suffix}{options.reconstructor_options.batch_size}'
    if options.reconstructor_options.momentum_acceleration_gain > 0:
        recon_path += f'_m{options.reconstructor_options.momentum_acceleration_gain}'

    recon_path += f'_p{options.probe_options.initial_guess.shape[1]}'

    # Append optional parameters to the path
    if options.probe_options.center_constraint.enabled:
        recon_path += '_cp'
    if options.object_options.multimodal_update:
        recon_path += '_mm'

    if options.opr_mode_weight_options.optimizable:
        recon_path += f'_opr{options.probe_options.initial_guess.shape[0] - 1}'
    if options.opr_mode_weight_options.optimize_intensity_variation:
        recon_path += '_ic'

    if params['object_thickness_m'] > 0 and params['number_of_slices'] > 1:
        recon_path += f'_Ns{params['number_of_slices']}_T{params['object_thickness_m']/1e-6}um'
        if options.object_options.multislice_regularization.enabled and options.object_options.multislice_regularization.weight > 0:
            recon_path += f'_reg{options.object_options.multislice_regularization.weight}'
 
    if options.probe_position_options.optimizable:
        recon_path += '_pc'
        if options.probe_position_options.correction_options.update_magnitude_limit > 0:
            recon_path += f'_ul{options.probe_position_options.correction_options.update_magnitude_limit}'
        if options.probe_position_options.correction_options.slice_for_correction:
            recon_path += f'_layer{options.probe_position_options.correction_options.slice_for_correction}'
        if options.probe_position_options.affine_transform_constraint.apply_constraint:
            recon_path += '_affine'

    if params.get('init_probe_propagation_distance_mm', 0) != 0:
        recon_path += f'_pd{params['init_probe_propagation_distance_mm']}'

    # Append any additional suffix
    if params['recon_dir_suffix']:
        recon_path += f'_{params['recon_dir_suffix']}'

    # Ensure the directory structure exists
    if options.object_options.slice_spacings_m: # multislice recon
        os.makedirs(os.path.join(recon_path, 'object_ph_layers'), exist_ok=True)
        os.makedirs(os.path.join(recon_path, 'object_ph_total'), exist_ok=True)
        os.makedirs(os.path.join(recon_path, 'object_mag_layers'), exist_ok=True)
        os.makedirs(os.path.join(recon_path, 'object_mag_total'), exist_ok=True)
        os.makedirs(os.path.join(recon_path, 'probe_propagation_mag'), exist_ok=True)
    else:
        os.makedirs(os.path.join(recon_path, 'object_ph'), exist_ok=True)
        os.makedirs(os.path.join(recon_path, 'object_mag'), exist_ok=True)
    
    os.makedirs(os.path.join(recon_path, 'probe_mag'), exist_ok=True)
    if options.opr_mode_weight_options.optimizable:
        os.makedirs(os.path.join(recon_path, 'probe_mag_opr'), exist_ok=True)
    os.makedirs(os.path.join(recon_path, 'loss'), exist_ok=True)
    if params['position_correction']:
        os.makedirs(os.path.join(recon_path, 'positions'), exist_ok=True)
        os.makedirs(os.path.join(recon_path, 'positions_affine'), exist_ok=True)
    print(f'Reconstruction results will be saved in: {recon_path}')

    return recon_path

def save_initial_conditions(recon_path, params, options):
    det_pixel_size_m = params['det_pixel_size_m']
    # save sum of all diffraction patterns
    dp_sum = np.sum(options.data_options.data, axis=0)

    #plt.imsave(f'{recon_path}/dp_sum.png', dp_sum, cmap='jet', metadata={'unit': 'pixel', 'pixel_size': 1})  
    #imwrite(f'{recon_path}/dp_sum.tiff', normalize_by_bit_depth(np.sum(options.data_options.data, axis=0), '16'))
    
    # Apply the jet colormap to convert data to RGB
    norm = plt.Normalize(vmin=dp_sum.min(), vmax=dp_sum.max())
    cmap = plt.cm.jet
    colored_dp_sum = cmap(norm(dp_sum))  # This creates an RGBA array
    colored_dp_sum = (colored_dp_sum[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB uint8

    # Save with ImageJ-compatible resolution information
    imwrite(f'{recon_path}/dp_sum.tiff', 
            colored_dp_sum,
            photometric='rgb',
            resolution=(1 / (det_pixel_size_m/1e-6), 1 / (det_pixel_size_m/1e-6)),
            metadata={'unit': 'um', 'pixel_size': det_pixel_size_m/1e-6},
            imagej=True)

    # Save options to a file
    with open(f'{recon_path}/ptychi_options.pkl', 'wb') as f:
        import pickle
        options_dict_temp = options.__dict__.copy()
        if not params['save_diffraction_patterns']:
            # Store the data temporarily
            data_backup = options_dict_temp['data_options'].data
            # Remove data from the copy that will be saved
            options_dict_temp['data_options'].data = None
            # Save the options without the data
            pickle.dump(options_dict_temp, f)
            # Restore the data
            options_dict_temp['data_options'].data = data_backup
        else:
            pickle.dump(options_dict_temp, f)

    # save initial probe
    init_probe_mag = np.abs(options.probe_options.initial_guess[0].cpu().detach().numpy())
    probe_temp = np.hstack(init_probe_mag)
    N_probe = probe_temp.shape[0]
    # plt.imsave(f'{recon_path}/init_probe_mag.png', probe_temp, cmap='plasma')
    # #imwrite(f'{recon_path}/init_probe_mag.tiff', normalize_by_bit_depth(probe_temp, '16'))

    norm = plt.Normalize(vmin=probe_temp.min(), vmax=probe_temp.max())
    cmap = plt.cm.plasma
    colored_probe_temp = cmap(norm(probe_temp))  # This creates an RGBA array
    colored_probe_temp = (colored_probe_temp[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB uint8

    # Save with ImageJ-compatible resolution information
    imwrite(f'{recon_path}/init_probe_mag.tiff', 
            colored_probe_temp,
            photometric='rgb',
            resolution=(1 / (options.object_options.pixel_size_m/1e-6), 1 / (options.object_options.pixel_size_m/1e-6)),
            metadata={'unit': 'um', 'pixel_size': options.object_options.pixel_size_m /1e-6},
            imagej=True)
    
    # Create figure and axis
    # fig, ax = plt.subplots()

    # # Display the image
    # im = ax.imshow(probe_temp, cmap='plasma')

    # # Add scale bar
    # bar_length_pixels = 1e-6/options.object_options.pixel_size_m  # Length of scale bar in pixels
    # bar_width_pixels = N_probe/30    # Width of scale bar in pixels
    # bar_position_x = N_probe/20     # from left edge
    # bar_position_y = N_probe*0.9  # from bottom

    # # Create and add the scale bar
    # scale_bar = Rectangle((bar_position_x, bar_position_y), 
    #                     bar_length_pixels, bar_width_pixels, 
    #                     fc='white', ec='none')
    # ax.add_patch(scale_bar)

    # # Optional: Add text above scale bar
    # plt.text(bar_position_x + bar_length_pixels/2, 
    #         bar_position_y - 3, 
    #         '1 um', 
    #         color='white', 
    #         ha='center', 
    #         va='bottom',
    #         fontsize=5)

    # # Remove axes
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_frame_on(False)

    # # Save the figure
    # plt.savefig(f'{recon_path}/init_probe_mag.png', 
    #             dpi=500, 
    #             bbox_inches='tight', 
    #             pad_inches=0)
    # plt.close()

    # plot initial positions
    global pos_x_min, pos_x_max, pos_y_min, pos_y_max, init_positions_y_um, init_positions_x_um, range_factor
    init_positions_y_um = options.probe_position_options.position_y_px*options.object_options.pixel_size_m/1e-6
    init_positions_x_um = options.probe_position_options.position_x_px*options.object_options.pixel_size_m/1e-6
    plt.figure()
    plt.scatter(-init_positions_x_um, init_positions_y_um, s=1, edgecolors='blue')
    plt.xlabel('X [um]')
    plt.ylabel('Y [um]')
    plt.legend(['Initial positions'], loc='upper center', bbox_to_anchor=(0.5, 1.15))
    plt.grid(True)
    pos_x_min, pos_x_max = plt.xlim()
    pos_y_min, pos_y_max = plt.ylim()
    
    range_factor = 1.1
    plt.xlim(pos_x_min*range_factor, pos_x_max*range_factor)
    plt.ylim(pos_y_min*range_factor, pos_y_max*range_factor)
    plt.savefig(f'{recon_path}/init_positions.png', dpi=300)
    plt.close()

    if params['position_correction']:
        global pos_scale, pos_assymetry, pos_rotation, pos_shear, iterations
        pos_scale = []
        pos_assymetry = []
        pos_rotation = []
        pos_shear = []
        iterations = []
    # Save parameters to a JSON file in the reconstruction path
    params_file_path = f'{recon_path}/pear_params.json'
    with open(params_file_path, 'w') as params_file:
        json.dump(params, params_file, indent=4)

def initialize_recon(params):
    instrument = params['instrument'].lower()
    dp_Npix = params['diff_pattern_size_pix']

    # Load diffraction patterns and positions
    if params.get('load_processed_hdf5') or instrument == 'simu':
        dp, positions_m = _load_data_hdf5(
            params.get('path_to_processed_hdf5_dp'),
            params.get('path_to_processed_hdf5_pos'),
            dp_Npix
        )
    else:
        dp, positions_m = _load_data_raw(
            instrument,
            params.get('data_directory'),
            params.get('scan_num'),
            dp_Npix,
            params.get('diff_pattern_center_x'),
            params.get('diff_pattern_center_y')
        )
    print("Shape of diffraction patterns (dp):", dp.shape)

    # Load external positions
    if params['path_to_init_positions']:
        positions_m = _prepare_initial_positions(params)
    
    params['det_pixel_size_m'] = 75e-6 if instrument in ['velo', 'velociprobe', 'bnp', 'bionanoprobe', '2ide', '2xfm', 'lynx', 'simu'] else 172e-6
    
    params['wavelength_m'] = 1.23984193e-9 / params.get('beam_energy_kev')
    print("Wavelength (nm):", f"{params['wavelength_m'] * 1e9:.3f}")

    params['obj_pixel_size_m'] = params['wavelength_m'] * params['det_sample_dist_m'] / params['det_pixel_size_m'] / dp_Npix #pixel size
    print("Pixel size of reconstructed object (nm):", f"{params['obj_pixel_size_m'] * 1e9:.3f}")

    init_positions_px = positions_m / params['obj_pixel_size_m']

    # Load initial probe
    init_probe = _prepare_initial_probe(dp, params)

    # Load initial object
    init_object = _prepare_initial_object(params, init_positions_px, init_probe.shape[-2:], round(1e-6/params['obj_pixel_size_m']))

    return dp, init_positions_px, init_probe, init_object, params

def _load_data_raw(instrument, base_path, scan_num, dp_Npix, dp_cen_x, dp_cen_y):
    instrument_loaders = {
        'velo': _load_data_velo,
        'velociprobe': _load_data_velo,
        'bnp': _load_data_bnp,
        'bionanoprobe': _load_data_bnp,
        '12idc': _load_data_12idc,
        '2xfm': _load_data_2xfm,
        '2ide': _load_data_2xfm,
        'lynx': _load_data_lynx
    }
    instrument = instrument.lower()
    if instrument not in instrument_loaders:
        raise ValueError(f"Unsupported instrument: {instrument}")
    
    dp, positions = instrument_loaders[instrument](base_path, scan_num, dp_Npix, dp_cen_x, dp_cen_y)

    return dp, positions

def _prepare_initial_positions(params):
    print("Loading initial positions from a ptychi reconstruction.")
    positions_px = _load_ptychi_recon(params['path_to_init_positions'], 'positions_px')
    input_obj_pixel_size = _load_ptychi_recon(params['path_to_init_positions'], 'obj_pixel_size_m')
    
    return positions_px*input_obj_pixel_size

def _prepare_initial_object(params, positions_px, probe_size, extra_size):
    if params['path_to_init_object']:
        print("Loading initial object from a ptychi reconstruction.")
        init_object = _load_ptychi_recon(params['path_to_init_object'], 'object')
        input_obj_pixel_size = _load_ptychi_recon(params['path_to_init_object'], 'obj_pixel_size_m')
        # TODO: more options for multislice recon
        if input_obj_pixel_size != params['obj_pixel_size_m']:
            print(f"Input object's pixel size ({input_obj_pixel_size*1e9:.3f} nm) does not match the expected pixel size ({params['obj_pixel_size_m']*1e9:.3f} nm).")
            print(f"Resizing input object to match the current reconstruction.")
            
            # Calculate zoom factor based on pixel size ratio
            zoom_factor = input_obj_pixel_size / params['obj_pixel_size_m']
            
            # Get target shape after zooming first slice
            target_shape = (int(init_object.shape[-2] * zoom_factor), 
                           int(init_object.shape[-1] * zoom_factor))
            
            # Use resize_complex_array to resize the object
            init_object = resize_complex_array(init_object, new_shape=target_shape)
            
            # Convert to tensor
            init_object = to_tensor(init_object)
        
    else:
        print("Generating a random initial object.")

        init_object = torch.ones([params['number_of_slices'], *get_suggested_object_size(positions_px, probe_size, extra=extra_size)], dtype=get_default_complex_dtype())
        init_object = init_object + 1j*torch.rand(*init_object.shape) * 1e-3
 
    print("Shape of initial object:", init_object.shape)

    return init_object

def _prepare_initial_probe(dp, params):
    path_to_init_probe = params.get('path_to_init_probe')
    num_probe_modes = params.get('number_probe_modes')
    num_opr_modes = params.get('number_opr_modes')

    if params.get('use_model_FZP_probe', False):
        print("Generating a model FZP probe.")
        if params['instrument'].lower() == 'velo' or params['instrument'].lower() == 'velociprobe':
            dRn = 50e-9
            Rn = 90e-6
            D_H = 60e-6
            D_FZP = 250e-6
        elif params['instrument'].lower() == 'ptycho_probe':
            dRn = 15e-9
            Rn = 90e-6
            D_H = 15e-6
            D_FZP = 250e-6
        else:
            dRn = 50e-9
            Rn = 90e-6
            D_H = 60e-6
            D_FZP = 250e-6
        N_probe_orig = dp.shape[-2]*params['obj_pixel_size_m']/4e-9
        # Round N_probe_orig up to the nearest power of 2 for faster FFT
        N_probe_orig = int(2 ** np.ceil(np.log2(N_probe_orig)))

        probe_orig = make_fzp_probe(N_probe_orig, params['wavelength_m'], 4e-9, 0, Rn, dRn, D_FZP, D_H)
        probe = resize_complex_array(probe_orig, zoom_factor=(4e-9/params['obj_pixel_size_m'], 4e-9/params['obj_pixel_size_m']))
        # Crop probe to match the diffraction pattern size
        if probe.shape[-1] > dp.shape[1]:
            # Calculate the center of the probe
            center_y, center_x = probe.shape[0] // 2, probe.shape[1] // 2
            # Calculate the half-width of the target size
            half_height, half_width = dp.shape[1] // 2, dp.shape[1] // 2
            # Crop the probe around its center
            probe = probe[center_y - half_height:center_y + half_height,
                          center_x - half_width:center_x + half_width]
            # print(f"Probe cropped from {probe.shape[0]}x{probe.shape[1]} to {dp.shape[1]}x{dp.shape[1]}")
    else:
        if path_to_init_probe.endswith('.mat'):
            print("Loading initial probe from a foldslice reconstruction.")
            probe = _load_probe_foldslice(path_to_init_probe)
        elif params.get('path_to_init_probe').endswith('.h5'):
            print("Loading initial probe from a ptychi reconstruction.")
            probe = _load_ptychi_recon(path_to_init_probe, 'probe')
        else:
            raise ValueError("Unsupported file format for initial probe. Only .mat and .h5 files are supported.")

    print("Shape of input probe:", probe.shape)
    
    # TODO: load opr weights too
    if probe.ndim == 4:
        probe = probe[0]
    if probe.ndim == 2:
        probe = probe[None,:,:] # add incoherent mode dimension

    #   p = options.probe_options.initial_guess[0:1]
    # probe = orthogonalize_initial_probe(to_tensor(probe))
    # p = add_additional_opr_probe_modes_to_probe(to_tensor(p), 2)

    # Assuming probe is a [n_incoherent_modes, h, w]
    if probe.shape[0] >= num_probe_modes:
        probe = probe[:num_probe_modes,:,:]
    else:
        probe_temp = np.zeros((num_probe_modes, probe.shape[-2], probe.shape[-1]), dtype=np.complex64)
        probe_temp[:probe.shape[0],:,:] = probe
        probe_temp[probe.shape[0]:,:,:] = probe[-1,:,:]
        probe = probe_temp

    #probe = probe.transpose(0, 2, 1)
    # TODO: determine zoom factor based on pixel size ratio
    if probe.shape[-1:] != dp.shape[-1:]:
        print(f"Resizing probe ({probe.shape[-1]}) to match the diffraction pattern size ({dp.shape[-1]}).")
        probe = resize_complex_array(probe, new_shape=(dp.shape[-2], dp.shape[-1]))

    # Propagate probe if a propagation distance is specified
    propagation_distance_mm = params.get('init_probe_propagation_distance_mm', 0)
    if propagation_distance_mm != 0:
        from ptychi.pear_utils import near_field_evolution
        extent = probe.shape[-1] * params['obj_pixel_size_m']
        # Convert mm to meters for propagation
        propagation_distance_m = propagation_distance_mm * 1e-3
        
        # Log the propagation operation
        print(f"Propagating initialprobe by {propagation_distance_mm} mm")
        
        # Propagate each probe mode
        for i in range(probe.shape[0]):
                probe[i], _, _, _ = near_field_evolution(
                    probe[i],
                    propagation_distance_m,
                    params['wavelength_m'],
                    extent,
                    use_ASM_only=True
                )

    # Add OPR mode dimension
    probe = probe[None, ...]
    probe = orthogonalize_initial_probe(to_tensor(probe))
    # Add n_opr_modes - 1 eigenmodes which are randomly initialized
    probe = add_additional_opr_probe_modes_to_probe(to_tensor(probe), num_opr_modes)

    print("Shape of probe after preprocessing:", probe.shape)

    return probe

def _load_probe_foldslice(recon_file):
    #print(f"Attempting to load probe from: {recon_file}")

    try:
        with h5py.File(recon_file, "r") as hdf_file:
            probes = hdf_file['probe'][:]
    except:
        import scipy.io
        print(f"Attempting to load probe using scipy.io.loadmat")
        mat_contents = scipy.io.loadmat(recon_file)
        if 'probe' in mat_contents:
            probes = mat_contents['probe']
            if probes.ndim == 4:
                probes = probes[...,0]
                probes = probes.transpose(2,0,1)

            if probes.ndim == 3:
                print("Transposing probe to (n_probe_modes, h, w)")
                probes = probes.transpose(2,0,1)

    #print("Shape of probes:", probes.shape)
    if probes.dtype == [('real', '<f8'), ('imag', '<f8')]: # For mat v7.3, the complex128 is read as this complicated datatype via h5py
        #print(f"Loaded object.dtype = {object.dtype}, cast it to 'complex128'")
        probes = probes.view('complex128')
    return probes

def _load_ptychi_recon(recon_file, variable_name):
    with h5py.File(recon_file, "r") as hdf_file:
        # Check if the dataset is a scalar or not
        dataset = hdf_file[variable_name]
        if dataset.shape == ():  # It's a scalar
            array = dataset[()]  # Use [()] for scalar datasets
        else:
            array = dataset[:]  # Use [:] for non-scalar datasets
    
    return array

def _normalize_from_zero_to_one(arr):
    norm_arr = (arr - arr.min())/(arr.max()-arr.min())
    return norm_arr

def normalize_by_bit_depth(arr, bit_depth):

    if bit_depth == '8':
        norm_arr_in_bit_depth = np.uint8(255*_normalize_from_zero_to_one(arr))
    elif bit_depth == '16':
        norm_arr_in_bit_depth = np.uint16(65535*_normalize_from_zero_to_one(arr))
    elif bit_depth == '32':
        norm_arr_in_bit_depth = np.float32(_normalize_from_zero_to_one(arr))
    elif bit_depth == 'raw':
        norm_arr_in_bit_depth = np.float32(arr)
    else:
        print(f'Unsuported bit_depth :{bit_depth} was passed into `result_modes`, `raw` is used instead')
        norm_arr_in_bit_depth = np.float32(arr)
    
    return norm_arr_in_bit_depth

def _load_data_hdf5(h5_dp_path, h5_position_path, dp_Npix):
    # if h5_dp_path == h5_position_path: # assume it's a ptychodus product
    #     print("Loading processed scan positions and diffraction patterns in ptychodus convention.")
    # positions = np.stack([f_meta['probe_position_y_m'][...], f_meta['probe_position_x_m'][...]], axis=1)
    #     pixel_size_m = f_meta['object'].attrs['pixel_height_m']
    #     positions_px = positions / pixel_size_m
    #     if subtract_position_mean:
    #         positions_px -= positions_px.mean(axis=0)

    #else: # assume foldslice convention
    print("Loading processed scan positions and diffraction patterns in foldslice convention.")
    dp = h5py.File(h5_dp_path, 'r')['dp'][...]
    det_xwidth = int(dp_Npix/2)
    cen = int(dp.shape[1] / 2)
    dp = dp[:, cen - det_xwidth:cen + det_xwidth, cen - det_xwidth:cen + det_xwidth]

    ppY = h5py.File(h5_position_path, 'r')['ppY'][:].flatten()
    ppX = h5py.File(h5_position_path, 'r')['ppX'][:].flatten()
    ppX = ppX - (np.max(ppX) + np.min(ppX)) / 2
    ppY = ppY - (np.max(ppY) + np.min(ppY)) / 2
    positions = np.stack((ppY, ppX), axis=1)

    return dp, positions

def _load_data_2xfm(base_path, scan_num, det_Npixel, cen_x, cen_y):
    print("Loading scan positions and diffraction patterns measured by the XFM instrument at 2IDE.")
    sys.path.append("/mnt/micdata3/ptycho_tools/utility")
    from readMDA import readMDA
    dp_dir = f'{base_path}/ptycho/'
    filePath = '/entry/data/data'

    # Load scan positions from original file
    MDAfile_path = f'{base_path}/mda/2xfm_{scan_num:04d}.mda'

    #if not os.path.exists(XRFfile_path):
    #    raise FileNotFoundError(f"The XRF file path does not exist: {XRFfile_path}")

    if not os.path.exists(MDAfile_path):
        raise FileNotFoundError(f"The MDA file path does not exist: {MDAfile_path}")
    
    mda_data = readMDA(MDAfile_path)

    x_pos=np.array(mda_data[2].p[0].data)[1]
    y_pos=np.array(mda_data[1].d[5].data)
    STXM=np.array(mda_data[2].d[1].data)
    Ny,Nx=STXM.shape[0],STXM.shape[1]
    x_pos-=x_pos.mean()
    y_pos-=y_pos.mean()
    x_pos*=1e-3
    y_pos*=1e-3

    N_scan_x = x_pos.shape[0]    
    N_scan_y = y_pos.shape[0]
    print(f'N_scan_y={N_scan_y}, N_scan_x={N_scan_x}, N_scan_dp={N_scan_x * N_scan_y}')

    # Load diffraction patterns
    index_x_lb, index_x_ub = int(cen_x - det_Npixel // 2), int(cen_x + (det_Npixel + 1) // 2)
    index_y_lb, index_y_ub = int(cen_y - det_Npixel // 2), int(cen_y + (det_Npixel + 1) // 2)

    dp, scan_posx, scan_posy = [], [], []

    for i in range(N_scan_y):
        print(f'Loading scan line No.{i+1}...')
        #fileName = data_dir+'fly{:03d}_data_{:03d}.h5'.format(scanNo,i+1+N_scan_y_lb)

        fileName = os.path.join(dp_dir, f'fly{scan_num:03d}_data_{i+1:03d}.h5')
        with h5py.File(fileName, 'r') as h5_data:
            # h5_data = h5py.File(fileName,'r')
            dp_temp = h5_data[filePath][...]
            dp_temp[dp_temp<0] = 0
            dp_temp[dp_temp>1e6] = 0
            # print(fileName, dp_temp.shape)

            if dp_temp.shape[0] < 5:
                print(f'A lot of pixels are missed on this line: {dp_temp.shape[0]} pixels, Skip!')
                continue
            
            dp_crop = dp_temp[:, index_y_lb:index_y_ub, index_x_lb:index_x_ub]
            dp.append(dp_crop)
            scan_posx.extend(x_pos[:dp_crop.shape[0]])
            scan_posy.extend([y_pos[i]] * dp_crop.shape[0])

    positions = np.column_stack((scan_posy, scan_posx))
    dp = np.concatenate(dp, axis=0) if dp else np.array([])  # Concatenate if dp is not empty

    return dp, positions

def _load_data_12idc(base_path, scan_num, det_Npixel, cen_x, cen_y):
    print("Loading scan positions and diffraction patterns measured by the Ptycho-SAXS instrument at 12IDC.")
    det_xwidth=int(det_Npixel/2)
# 
    files = glob.glob(f'{base_path}/ptycho/{scan_num:03d}/*{scan_num:03d}_*.h5')
    N_lines = max(int(name.split('_')[-2]) for name in files)# number of scan lines
    N_pts = max(int(name.split('_')[-1][:-3]) for name in files) # number of scan points per line
    print(f'Number of scan lines: {N_lines}, Number of scan points per line: {N_pts}')

    pos=[]
    # Preallocate dp array
    #dp = np.zeros((N_lines * N_pts, det_Npixel, det_Npixel))  # Adjust shape as necessary
    dp = []
    for line in range(N_lines):
        print (f'Loading scan line No.{line+1}...')

        start_time = time.time()
        for point in range(N_pts):
            pos_file = glob.glob(f'{base_path}/positions/{scan_num:03d}/*{scan_num:03d}_{line+1:05d}_{point:d}.dat')[0]
            pos_arr = np.genfromtxt(pos_file, delimiter='')
            avg_pos = np.mean(pos_arr, axis=0)
            pos.append(avg_pos)

            file_path = glob.glob(f'{base_path}/ptycho/{scan_num:03d}/*{scan_num:03d}_{line+1:05d}_{point+1:05d}.h5')[0]
            #print(file_path)
            #with h5py.File(file_path, 'r', libver='latest', swmr=True) as h5f:
            with h5py.File(file_path, 'r') as h5f:
                data = h5f['/entry/data/data'][cen_y-det_xwidth:cen_y+det_xwidth, cen_x-det_xwidth:cen_x+det_xwidth]
            #with h5py.File(file_path, 'r') as h5f:
            #    data = h5f['/entry/data/data'][cen_y-det_xwidth:cen_y+det_xwidth, cen_x-det_xwidth:cen_x+det_xwidth]
            data[data<0] = 0
            #data[data>1e6] = 0
            data[data>70000] = 0
            #dp[line*N_pts + point] = data  # Directly assign to preallocated array
            #print(data.shape)

            dp.append(data)
            #print(dp.shape)
        end_time = time.time()
        #print(f'Time cost for loading data: {end_time - start_time:.2f} seconds')

    #print(f'Scan {scan_num:03d}: Total diffraction patterns: {dp.shape[0]:d}; Total scan points: {len(pos):d}')

    pos = np.array(pos)
    dp=np.array(dp)    
    dp_sum = np.sum(dp, axis=(1, 2))
    pos_index = np.argwhere(dp_sum > (dp_sum.mean() / 2))
    #pos_index = np.argwhere(dp_sum > (dp_sum.mean() / 3))

    dp=dp[pos_index[:,0]]
    pos=pos[pos_index[:,0]]
    print(f'Number of selected scan points={pos.shape[0]}')
    #print(f'Shape of dp: {dp.shape}')

    ppY = pos[:,1]*1e-9
    ppX = -pos[:,2]*1e-9

    # Shift positions to center around (0,0)
    ppX = ppX - (np.max(ppX) + np.min(ppX)) / 2
    ppY = ppY - (np.max(ppY) + np.min(ppY)) / 2
    positions = np.column_stack((ppY, ppX))

    return dp, positions

def _load_data_bnp(base_path, scan_num, det_Npixel, cen_x, cen_y):
    print("Loading scan positions and diffraction patterns measured by the Bionanoprobe instrument.")

    dp_dir = f'{base_path}/ptycho/'
    filePath = '/entry/data/data'

    # Load scan positions from original file
    XRFfile_path = f'{base_path}/img.dat/bnp_fly{scan_num:04d}.mda.h5'
    if not os.path.exists(XRFfile_path):
        raise FileNotFoundError(f"The XRF file path does not exist: {XRFfile_path}")
    
    XRFfile = h5py.File(XRFfile_path)
    x_pos = XRFfile['MAPS/x_axis'][:]
    y_pos = XRFfile['MAPS/y_axis'][:]

    x_pos-=x_pos.mean()
    y_pos-=y_pos.mean()
    x_pos*=1e-6
    y_pos*=1e-6

    # Load diffraction patterns
    index_x_lb, index_x_ub = int(cen_x - det_Npixel // 2), int(cen_x + (det_Npixel + 1) // 2)
    index_y_lb, index_y_ub = int(cen_y - det_Npixel // 2), int(cen_y + (det_Npixel + 1) // 2)

    N_scan_y, N_scan_x = y_pos.size, x_pos.size
    print(f'N_scan_y={N_scan_y}, N_scan_x={N_scan_x}, N_scan_dp={N_scan_x * N_scan_y}')

    dp, scan_posx, scan_posy = [], [], []

    for i in range(N_scan_y):
        print(f'Loading scan line No.{i+1}...')
        fileName = os.path.join(dp_dir, f'bnp_fly{scan_num:04d}_{i:06d}.h5')
        with h5py.File(fileName, 'r') as h5_data:
            # h5_data = h5py.File(fileName,'r')
            dp_temp = h5_data[filePath][...]
            dp_temp[dp_temp<0] = 0
            dp_temp[dp_temp>1e7] = 0
            # print(fileName, dp_temp.shape)
            #dp_temp = np.clip(h5_data[filePath][...], 0, 1e7)
            #dp_temp = np.clip(h5_data[filePath][...], 0, 1e7)
            if dp_temp.shape[0] < 5:
                print(f'A lot of pixels are missed on this line: {dp_temp.shape[0]} pixels, Skip!')
                continue
            
            dp_crop = dp_temp[:, index_y_lb:index_y_ub, index_x_lb:index_x_ub]
            dp.append(dp_crop)
            scan_posx.extend(x_pos[:dp_crop.shape[0]])
            scan_posy.extend([y_pos[i]] * dp_crop.shape[0])

    positions = np.column_stack((scan_posy, scan_posx))
    dp = np.concatenate(dp, axis=0) if dp else np.array([])  # Concatenate if dp is not empty

    return dp, positions

def _read_position_file(posfile):
    """
    Read a position file and return a dictionary with header info and data columns.
    
    Expected file format:
      - The first line contains: <string> <integer>, <string> <float>
      - The second line contains column names separated by whitespace.
      - The remaining lines contain numeric data corresponding to the columns.
    
    Parameters:
        posfile (str): Path to the position file.
        
    Returns:
        dict: Dictionary with header keys and data arrays.
    """
    if not os.path.exists(posfile):
        raise FileNotFoundError(f"Position file {posfile} not found")
    
    struct_out = {}
    with open(posfile, 'r') as f:
        # Read and parse the header line
        header_line = f.readline().strip()
        parts = header_line.split(',')
        if len(parts) != 2:
            raise ValueError("Header line format is incorrect")
        
        # Process first part (key and integer value)
        key1_val = parts[0].strip().split()
        if len(key1_val) != 2:
            raise ValueError("First header part format is incorrect")
        key1, val1 = key1_val[0], key1_val[1]
        
        # Process second part (key and float value)
        key2_val = parts[1].strip().split()
        if len(key2_val) != 2:
            raise ValueError("Second header part format is incorrect")
        key2, val2 = key2_val[0], key2_val[1]
        
        struct_out[key1] = int(val1)
        struct_out[key2] = float(val2)
        
        # Read column names (second line)
        names_line = f.readline().strip()
        names = names_line.split()
        
        # Read the remaining numerical data using numpy
        data = np.loadtxt(f)
        if data.ndim == 1:
            data = data.reshape(-1, len(names))
        
        # Assign each column to the dictionary using the column names
        for i, name in enumerate(names):
            struct_out[name] = data[:, i]
            
    return struct_out

def _load_data_lynx(base_path, scan_num, det_Npixel, cen_x, cen_y):
    print("Loading scan positions and diffraction patterns measured by the LYNX instrument.")
    # Load positions from .dat file
    pos_file = f"{base_path}/scan_positions/scan_{scan_num:05d}.dat"
    out_orch = _read_position_file(pos_file)
    
    x_positions = -out_orch['Average_x_st_fzp'] 
    y_positions = -out_orch['Average_y_st_fzp'] 

    # Center positions around (0,0)
    x_positions = x_positions - np.mean(x_positions)
    y_positions = y_positions - np.mean(y_positions)
    
    # Convert to meters if needed (adjust multiplier as needed)
    x_positions = x_positions * 1e-6  # Adjust this factor based on your data units
    y_positions = y_positions * 1e-6
    
    # Stack positions
    positions = np.column_stack((y_positions, x_positions))

    # Determine subfolder based on scan number
    subfolder_start = (scan_num // 1000) * 1000
    subfolder_end = subfolder_start + 999
    data_dir = os.path.join(base_path, 'eiger_4', f'S{subfolder_start:05d}-{subfolder_end:05d}', f'S{scan_num:05d}')
    file_path = os.path.join(data_dir, f'run_{scan_num:05d}_000000000000.h5')
    # Validate inputs
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    # Calculate detector ROI indices
    N_dp_x_input = min(942, det_Npixel)
    N_dp_y_input = min(942, det_Npixel) 
    
    index_x_lb = int(cen_x - N_dp_x_input // 2)
    index_x_ub = int(cen_x + (N_dp_x_input + 1) // 2)
    index_y_lb = int(cen_y - N_dp_y_input // 2)
    index_y_ub = int(cen_y + (N_dp_y_input + 1) // 2)

    # Load diffraction patterns
    with h5py.File(file_path, 'r') as h5_data:
        dp_temp = h5_data['entry/data/eiger_4'][:]
        N_scan_dp = dp_temp.shape[0]
        print(f'Number of diffraction patterns: {N_scan_dp}')

        # Initialize output array
        dp = np.zeros((N_scan_dp, N_dp_y_input, N_dp_x_input))
        
        # Process each diffraction pattern
        for j in range(N_scan_dp):
            roi = dp_temp[j, index_y_lb:index_y_ub, index_x_lb:index_x_ub]
            scipy.ndimage.zoom(roi, [1, 1], output=dp[j], order=1)

    # Clean up data
    dp[dp < 0] = 0
    dp[dp > 1e7] = 0

    return dp, positions

def _load_data_velo(base_path, scan_num, det_Npixel, cen_x, cen_y):
    print("Loading scan positions and diffraction patterns measured by the Velociprobe instrument.")

    dp_dir = f'{base_path}/ptycho/fly{scan_num:03d}/'

    # Load scan positions from original file
    getpos_path = os.path.join(base_path, 'positions')
    s = glob.glob(getpos_path + '/fly{:03d}_0.txt'.format(scan_num))
    pos = np.genfromtxt(s[0], delimiter=',')

    x, y = [], []
    for trigger in range(1, int(pos[:, 7].max() + 1)):
        st = np.argwhere(pos[:, 7] == trigger)
        x.append((pos[st[0], 1] + pos[st[-1], 1]) / 2.)  # LI
        y.append(-(pos[st[0], 5] + pos[st[-1], 5]) / 2.)  # no LI
    ppX = np.asarray(x) * 1e-9
    ppY = np.asarray(y) * 1e-9
    rot_ang = 0
    ppX *= np.cos(rot_ang / 180 * np.pi)
    N_scan_pos = ppX.size
    print('Number of scan positions: ' + str(N_scan_pos))

    # Load diffraction patterns
    N_dp_x_input = det_Npixel
    N_dp_y_input = det_Npixel
    index_x_lb = (cen_x - np.floor(N_dp_x_input / 2.0)).astype(int)
    index_x_ub = (cen_x + np.ceil(N_dp_x_input / 2.0)).astype(int)
    index_y_lb = (cen_y - np.floor(N_dp_y_input / 2.0)).astype(int)
    index_y_ub = (cen_y + np.ceil(N_dp_y_input / 2.0)).astype(int)

    # Determine N_scan_y
    list = os.listdir(dp_dir)
    N_scan_y = len(list) - 1 - 1 - 1
    print('N_scan_y=' + str(N_scan_y))

    # Determine N_scan_x
    filePath = 'entry/data/data'
    fileName = f'{dp_dir}fly{scan_num:03d}_data_{1:06d}.h5'
    h5_data = h5py.File(fileName, 'r')
    dp_temp = h5_data[filePath][...]
    N_scan_x = dp_temp.shape[0]
    print('N_scan_x=' + str(N_scan_x))

    N_scan_x_lb = 0
    N_scan_y_lb = 0
    N_scan_dp = N_scan_x * N_scan_y
    print('N_scan_dp=' + str(N_scan_dp))

    resampleFactor = 1
    resizeFactor = 1

    dp = np.zeros((N_scan_dp, int(N_dp_y_input * resizeFactor), int(N_dp_x_input * resizeFactor)))
    print(dp.shape)

    for i in range(N_scan_y):
        fileName = dp_dir + 'fly' + '%03d' % (scan_num) + '_data_' + '%06d' % (i + 1 + N_scan_y_lb) + '.h5'

        h5_data = h5py.File(fileName, 'r', libver='latest')
        dp_temp = h5_data[filePath][...]
        for j in range(N_scan_x):
            index = i * N_scan_x + j
            scipy.ndimage.interpolation.zoom(dp_temp[j + N_scan_x_lb, index_y_lb:index_y_ub, index_x_lb:index_x_ub], [resizeFactor, resizeFactor], dp[index, :, :], 1)

    dp[dp < 0] = 0
    dp[dp > 1e7] = 0

    dp = dp[::resampleFactor, :, :]

    if N_scan_pos > N_scan_dp:  # if there are more positions than dp
        ppX = ppX[0:N_scan_dp]
        ppY = ppY[0:N_scan_dp]
    else:
        dp = dp[0:N_scan_pos, :, :]

    # Shift positions to center around (0,0)
    ppX = ppX - (np.max(ppX) + np.min(ppX)) / 2
    ppY = ppY - (np.max(ppY) + np.min(ppY)) / 2
    positions = np.column_stack((ppY, ppX))

    return dp, positions
