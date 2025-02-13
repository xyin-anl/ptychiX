from typing import List, Optional
import numpy as np

from .settings import MovieFileSettings, MovieFileTypes

from .mappings import prepare_movie_subject, movie_setting_types
from .settings import ProbeMovieSettings
from .io import append_array_to_h5, save_movie_to_file
import os
import h5py
from collections import defaultdict

import ptychi.reconstructors.base as base

ENABLE_MOVIES = False
dataset_name = "frames"


class MovieSubject:
    current_frame: int = 0
    frames: np.ndarray

    def __init__(
        self,
        settings: movie_setting_types,
        folder: str,
        movie_name: str,
        save_intermediate_data_to_hdf5: bool = False,
    ):
        self.settings = settings
        self.folder = folder
        self.movie_name = movie_name
        self.save_intermediate_data_to_hdf5 = save_intermediate_data_to_hdf5
        self.hdf5_file_path = os.path.join(self.folder, self.movie_name + ".hdf5")

    def save_frame_enabled_this_epoch(self, current_epoch: int):
        if current_epoch % self.settings.snapshot.stride == 0:
            return True
        else:
            return False

    def record_frame(self, movie_frame_array: np.ndarray):
        if self.save_intermediate_data_to_hdf5:
            append_array_to_h5(
                array=movie_frame_array,
                file_path=self.hdf5_file_path,
                create_new_file=self.current_frame == 0,
            )
        else:
            if self.current_frame == 0:
                self.frames = movie_frame_array[None]
            else:
                self.frames = np.append(self.frames, movie_frame_array[None], axis=0)
        self.current_frame += 1

    def create_movie(
        self, file_type: MovieFileTypes, movie_file_settings: Optional[MovieFileSettings] = None
    ):
        if movie_file_settings is None:
            movie_file_settings = self.settings.movie_file
        output_path = os.path.join(self.folder, self.movie_name + "." + file_type)
        if self.save_intermediate_data_to_hdf5:
            with h5py.File(self.hdf5_file_path) as F:
                frames = F[dataset_name][:]
        else:
            frames = self.frames
        save_movie_to_file(
            array=frames,
            file_type=file_type,
            output_path=output_path,
            fps=self.settings.movie_file.fps,
            enhance_contrast=self.settings.movie_file.high_contrast,
            colormap=self.settings.movie_file.colormap,
        )

    def reset(self):
        self.current_frame = 0


class SubjectList:
    subject_list: dict[str, MovieSubject] = {}

    def add_subject(
        self,
        settings: movie_setting_types,
        folder: str,
        movie_name: str,
    ):
        if self.is_duplicate(settings):
            return
        else:
            self.subject_list[movie_name] = MovieSubject(settings, folder, movie_name)

    def record_all_frames(self, reconstructor: "base.Reconstructor"):
        for subject in self.subject_list.values():
            if subject.save_frame_enabled_this_epoch(reconstructor.current_epoch):
                movie_frame_array = prepare_movie_subject(reconstructor, subject.settings)
                subject.record_frame(movie_frame_array)

    def create_all_movies(self, file_type: MovieFileTypes):
        for subject in self.subject_list.values():
            subject.create_movie(file_type)

    def reset_all(self):
        for subject in self.subject_list.values():
            subject.reset()

    def is_duplicate(self, settings: movie_setting_types):
        for subject in self.subject_list.values():
            if subject.settings == settings:
                return True
        return False
