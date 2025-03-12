import subprocess
import re
import numpy as np
import glob
import os

def select_gpu(params):
    # Get GPU stats
    gpu_stats = get_gpu_usage()

    if gpu_stats is None or len(gpu_stats) == 0:
        # Default to GPU 0 if can't get stats
        return 0
    else:
        # Find GPU with 0% utilization and lowest memory usage
        zero_util_gpus = [gpu for gpu in gpu_stats if gpu['utilization'] == 0]
        
        if zero_util_gpus:
            # Among GPUs with 0% utilization, pick the one with lowest memory usage
            selected_gpu = min(zero_util_gpus, key=lambda x: x['memory_percent'])
        else:
            print(f"No GPU has 0% utilization. Selecting GPU with lowest combined score.")
            # If no GPU has 0% utilization, pick the one with lowest combined score
            selected_gpu = min(gpu_stats, 
                                key=lambda x: x['utilization'] + x['memory_percent'])
        
        return selected_gpu['index']

# Check GPU usage and select least utilized GPU
def get_gpu_usage():
    try:
        # Run nvidia-smi to get GPU utilization and memory usage
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
        output = output.decode('utf-8').strip().split('\n')
        
        gpu_stats = []
        for line in output:
            index, util, mem_used, mem_total = line.split(', ')
            gpu_stats.append({
                'index': int(index),
                'utilization': int(util),
                'memory_used': int(mem_used),
                'memory_total': int(mem_total),
                'memory_percent': (int(mem_used) / int(mem_total)) * 100
            })
        return gpu_stats
    except:
        return None

def make_fzp_probe(N, lambda_, dx, Ls, Rn, dRn, D_FZP, D_H):

    """
    Generates a Fresnel zone plate probe.
    Parameters:
        N (int): Number of pixels.
        lambda_ (float): Wavelength.
        dx (float): Pixel size (in meters) in sample plane.
        Ls (float): Distance (in meters) from focal plane to sample.
        Rn (float): Radius of outermost zone (in meters).
        dRn (float): Width of outermost zone (in meters).
        D_FZP (float): Diameter of pinhole.
        D_H (float): Diameter of the central beamstop (in meters).
    Returns:
        ndarray: Calculated probe field in the sample plane.
    """

    fl = 2 * Rn * dRn / lambda_  # focal length corresponding to central wavelength
    dx_fzp = lambda_ * fl / N / dx  # pixel size in the FZP plane
    # Coordinate in the FZP plane
    lx_fzp = np.linspace(-dx_fzp * N / 2, dx_fzp * N / 2, N)
    x_fzp, y_fzp = np.meshgrid(lx_fzp, lx_fzp)
    # Transmission function of the FZP
    T = np.exp(-1j * 2 * np.pi / lambda_ * (x_fzp**2 + y_fzp**2) / (2 * fl))
    C = (np.sqrt(x_fzp**2 + y_fzp**2) <= (D_FZP / 2)).astype(np.float64)  # circular function of FZP
    H = (np.sqrt(x_fzp**2 + y_fzp**2) >= (D_H / 2)).astype(np.float64)  # central block
    # Probe on sample plane using the Fresnel propagation function defined previously
    probe = fresnel_propagation(C * T * H, dx_fzp, fl+Ls, lambda_)
    #print(f"dx_fzp: {dx_fzp}, fl: {fl}, Ls: {Ls}, lambda_: {lambda_}")
    #print(f"Center pixel value of probe: {probe[probe.shape[0]//2, probe.shape[1]//2]}")

    return probe

def fresnel_propagation(IN, dxy, z, lambda_):
    """
    Performs Fresnel propagation for a given input wavefield.
    
    Parameters:
        IN (ndarray): Input object or wavefield.
        dxy (float): Pixel pitch of the object.
        z (float): Distance of the propagation.
        lambda_ (float): Wavelength of the wave.
    
    Returns:
        ndarray: Output wavefield after Fresnel propagation.
    """
    M, N = IN.shape
    k = 2 * np.pi / lambda_

    # Coordinate grid for input plane
    lx = np.linspace(-dxy * M / 2, dxy * M / 2, M)
    x, y = np.meshgrid(lx, lx)

    # Coordinate grid for output plane
    fc = 1 / dxy
    fu = lambda_ * z * fc
    lu = np.fft.ifftshift(np.linspace(-fu / 2, fu / 2, M))
    u, v = np.meshgrid(lu, lu)

    if z > 0:
        # Propagation in the positive z direction
        pf = np.exp(1j * k * z) * np.exp(1j * k * (u**2 + v**2) / (2 * z))
        kern = IN * np.exp(1j * k * (x**2 + y**2) / (2 * z))
        
        kerntemp = np.fft.fftshift(kern)
        cgh = np.fft.fft2(kerntemp)
        OUT = np.fft.fftshift(cgh * pf)
    else:
        # Propagation in the negative z direction (or backward propagation)
        z = abs(z)
        pf = np.exp(1j * k * z) * np.exp(1j * k * (x**2 + y**2) / (2 * z))
        cgh = np.fft.ifft2(np.fft.ifftshift(IN) / np.exp(1j * k * (u**2 + v**2) / (2 * z)))
        OUT = np.fft.fftshift(cgh) / pf
    
    return OUT

def resize_complex_array(complex_array, new_shape=None, zoom_factor=None):
    """
    Resize a complex array using scipy.ndimage.zoom.
    
    Parameters:
    -----------
    complex_array : ndarray
        The complex array to resize
    new_shape : tuple, optional
        The target shape (height, width). Required if zoom_factor is None.
    zoom_factor : float or tuple, optional
        The zoom factor to apply. Can be a single float or a tuple for different factors per dimension.
        Required if new_shape is None.
    
    Returns:
    --------
    ndarray
        The resized complex array
    """
    import numpy as np
    import scipy.ndimage
    
    # Input validation
    if new_shape is None and zoom_factor is None:
        raise ValueError("Either new_shape or zoom_factor must be provided")
    
    # Handle different array dimensions
    if complex_array.ndim == 2:
        # Calculate zoom_factor from new_shape if needed
        if zoom_factor is None:
            zoom_factor = (new_shape[0] / complex_array.shape[0], 
                          new_shape[1] / complex_array.shape[1])
        
        # Resize 2D array
        real_part = scipy.ndimage.zoom(complex_array.real, zoom_factor, order=1)
        imag_part = scipy.ndimage.zoom(complex_array.imag, zoom_factor, order=1)
        return real_part + 1j * imag_part
    
    elif complex_array.ndim == 3:
        # For 3D arrays, resize each 2D slice separately
        if new_shape is None:
            if isinstance(zoom_factor, (int, float)):
                new_shape = (int(complex_array.shape[1] * zoom_factor),
                            int(complex_array.shape[2] * zoom_factor))
            else:  # zoom_factor is a tuple
                if len(zoom_factor) == 2:
                    new_shape = (int(complex_array.shape[1] * zoom_factor[0]),
                                int(complex_array.shape[2] * zoom_factor[1]))
                else:
                    new_shape = (int(complex_array.shape[1] * zoom_factor[1]),
                                int(complex_array.shape[2] * zoom_factor[2]))
        
        # Create output array
        result = np.zeros((complex_array.shape[0], new_shape[0], new_shape[1]), dtype=complex)
        
        # Calculate 2D zoom factor for each slice
        slice_zoom_factor = (new_shape[0] / complex_array.shape[1], 
                            new_shape[1] / complex_array.shape[2])
        
        # Process each slice
        for i in range(complex_array.shape[0]):
            real_part = scipy.ndimage.zoom(complex_array[i].real, slice_zoom_factor, order=1)
            imag_part = scipy.ndimage.zoom(complex_array[i].imag, slice_zoom_factor, order=1)
            result[i] = real_part + 1j * imag_part
        
        return result
    
    else:
        raise ValueError(f"Unsupported array dimension: {complex_array.ndim}")

# potential alternative to resize_complex_array
def resize_complex_fourier(complex_array, new_shape):
    # Get dimensions
    old_shape = complex_array.shape
    
    # Compute FFT of the input array
    fft_array = np.fft.fftshift(np.fft.fft2(complex_array))
    
    # Create a new array for the resized FFT
    new_fft = np.zeros(new_shape, dtype=complex)
    
    # Calculate padding or cropping dimensions
    pad_x = (new_shape[1] - old_shape[1]) // 2
    pad_y = (new_shape[0] - old_shape[0]) // 2
    
    # Copy the central portion of the FFT
    if pad_x >= 0 and pad_y >= 0:  # Upsampling
        new_fft[pad_y:pad_y+old_shape[0], pad_x:pad_x+old_shape[1]] = fft_array
    else:  # Downsampling
        new_fft = fft_array[-pad_y:old_shape[0]+pad_y, -pad_x:old_shape[1]+pad_x]
    
    # Compute inverse FFT
    return np.fft.ifft2(np.fft.ifftshift(new_fft))

def near_field_evolution(u_0, z, wavelength, extent, use_ASM_only=False):
    """
    Near-field evolution function for wave propagation.
    Automatically switches between ASM and Fraunhofer propagation.
    
    Parameters:
    -----------
    u_0 : ndarray
        Input wavefield
    z : float
        Propagation distance
    wavelength : float
        Wavelength of the wave
    extent : float or tuple
        Physical size of the array (in meters)
    use_ASM_only : bool, optional
        Force using Angular Spectrum Method only
        
    Returns:
    --------
    u_1 : ndarray
        Propagated wavefield
    H : ndarray
        Transfer function
    h : ndarray
        Impulse response
    dH : ndarray
        Derivative of transfer function
    """
    import numpy as np
    
    # Initialize return values
    H = None
    h = None
    u_1 = None
    dH = None
    
    # Handle extent as a scalar or tuple
    if np.isscalar(extent):
        extent = np.array([extent, extent])
    else:
        extent = np.array(extent)
    
    # If z is 0, return input field unchanged
    if z == 0:
        H = 1
        u_1 = u_0
        return u_1, H, h, dH
    
    # If z is infinity, return empty arrays
    if np.isinf(z):
        return u_1, H, h, dH
    
    # Get dimensions of input field
    Npix = u_0.shape
    
    # Create normalized coordinate grids
    xgrid = (0.5 + np.arange(-Npix[0]//2, Npix[0]//2)) / Npix[0]
    ygrid = (0.5 + np.arange(-Npix[1]//2, Npix[1]//2)) / Npix[1]
    
    # Wave number
    k = 2 * np.pi / wavelength
    
    # Undersampling parameter
    F = np.mean(extent**2 / (wavelength * z * np.array(Npix)))
    
    if abs(F) < 1 and not use_ASM_only:
        # Farfield propagation
        print(f'Farfield regime, F/Npix={F}')
        Xrange = xgrid * extent[0]
        Yrange = ygrid * extent[1]
        X, Y = np.meshgrid(Xrange, Yrange)
        h = np.exp(1j * k * z + 1j * k / (2 * z) * (X.T**2 + Y.T**2))
        
        # This serves as low pass filter for the far nearfield
        H = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(h)))
        # Renormalize to conserve flux in image
        H = H / np.abs(H[Npix[0]//2, Npix[1]//2])
    else:
        # Standard Angular Spectrum Method (ASM)
        kx = 2 * np.pi * xgrid / extent[0] * Npix[0]
        ky = 2 * np.pi * ygrid / extent[1] * Npix[1]
        Kx, Ky = np.meshgrid(kx, ky)
        
        dH = (-1j * (Kx.T**2 + Ky.T**2) / (2 * k))
        
        # Make it a bit more sensitive to z distance
        H = np.exp(1j * z * np.sqrt(k**2 - Kx.T**2 - Ky.T**2))
        h = None
    
    # Apply transfer function to input field
    u_1 = np.fft.ifft2(np.fft.ifftshift(H) * np.fft.fft2(u_0))
    
    return u_1, H, h, dH

def find_matching_recon(path, scan_num):
    if 'scan_num' in path: # if path is a string pattern
        formatted_path = path.format(scan_num=scan_num)
        matching_files = glob.glob(formatted_path)

        if matching_files:
            #Sort files by modification time (newest first)
            matching_files.sort(key=os.path.getmtime, reverse=True)
            return matching_files[0]
        else:
            raise FileNotFoundError(f"No matching reconstruction file found for pattern: {formatted_path}")
    else:
        return path
