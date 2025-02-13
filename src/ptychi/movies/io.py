import numpy as np
import h5py
from PIL import Image
import cv2
import os
from .settings import MovieFileTypes


dataset_name = "frames"


def append_array_to_h5(array: np.ndarray, file_path: str, create_new_file: bool = False):
    if not os.path.exists(file_path) or create_new_file:
        # Create new file and 3D dataset
        with h5py.File(file_path, "w") as f:
            max_shape = (
                None,
                array.shape[0],
                array.shape[1],
            )  # Unlimited depth, fixed height & width
            dset = f.create_dataset(
                dataset_name,
                shape=(1, array.shape[0], array.shape[1]),  # Initial shape (depth=1)
                maxshape=max_shape,
                dtype=array.dtype,
            )
            dset[0] = array  # Store the first array
    else:
        # Append data to the existing 3D dataset
        with h5py.File(file_path, "a") as f:
            dset = f[dataset_name]
            new_depth = dset.shape[0] + 1  # Increase depth by 1
            dset.resize((new_depth, array.shape[0], array.shape[1]))  # Expand along depth
            dset[-1] = array  # Append new 2D array at the last depth


def save_movie_to_file(
    array: np.ndarray,
    file_type: MovieFileTypes,
    output_path: str,
    fps: int = 30,
    colormap: int = cv2.COLORMAP_BONE,
    enhance_contrast: bool = False,
):
    if file_type == MovieFileTypes.GIF:
        numpy_to_gif(array, output_path, fps, colormap, enhance_contrast)
    elif file_type == MovieFileTypes.MP4:
        numpy_to_mp4(array, output_path, fps, colormap, enhance_contrast)


def numpy_to_gif(
    array: np.ndarray,
    output_path: str,
    fps: int = 30,
    colormap: int = cv2.COLORMAP_BONE,
    enhance_contrast: bool = False,
):
    """
    Convert a 3D NumPy array (frames along the first axis) to a GIF file.

    Parameters:
        array (np.ndarray): 3D NumPy array of shape (frames, height, width) with values between 0 and 1.
        output_path (str): Path to save the GIF file.
        fps (int): Frames per second for the output video.
        colormap (int): OpenCV colormap to apply if input is grayscale.
        enhance_contrast (bool): Whether to apply contrast enhancement (default: True).
    """
    frames, height, width = array.shape[:3]
    is_grayscale = len(array.shape) == 3
    gif_frames = []

    for i in range(frames):
        frame = array[i]

        # Apply contrast enhancement if needed
        if enhance_contrast:
            frame = np.power(frame, 0.5)  # Apply gamma correction to boost small values

        # Normalize to [0, 255] and convert to uint8
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if is_grayscale:
            frame = cv2.applyColorMap(frame, colormap)  # Apply colormap to grayscale frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL
        gif_frames.append(Image.fromarray(frame))

    # Save frames as GIF
    gif_frames[0].save(
        output_path, save_all=True, append_images=gif_frames[1:], duration=int(1000 / fps), loop=0
    )
    print(f"GIF saved to {output_path}")


def numpy_to_mp4(
    array: np.ndarray,
    output_path: str,
    fps: int = 30,
    colormap: int = cv2.COLORMAP_BONE,
    enhance_contrast: bool = False,
):
    """
    Convert a 3D NumPy array (frames along the first axis) to an MP4 file.

    Parameters:
        array (np.ndarray): 3D NumPy array of shape (frames, height, width) with values between 0 and 1.
        output_path (str): Path to save the MP4 file.
        fps (int): Frames per second for the output video.
        colormap (int): OpenCV colormap to apply if input is grayscale.
        enhance_contrast (bool): Whether to apply contrast enhancement (default: True).
    """
    frames, height, width = array.shape[:3]
    is_grayscale = len(array.shape) == 3

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(frames):
        frame = array[i]

        # Apply contrast enhancement if needed
        if enhance_contrast:
            frame = np.power(frame, 0.5)  # Apply gamma correction to boost small values

        # Normalize to [0, 255] and convert to uint8
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if is_grayscale:
            frame = cv2.applyColorMap(frame, colormap)  # Apply colormap to grayscale frame

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")
