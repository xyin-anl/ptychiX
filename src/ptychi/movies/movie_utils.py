from typing import List, TypeVar, Union
import numpy as np

from .settings import MovieFileTypes, MovieSubjectTypes, ObjectMovieSettings

from .mappings import prepare_movie_subject, movie_setting_types
from .settings import ProbeMovieSettings
from .io import numpy_to_mp4, numpy_to_gif, append_array_to_h5, save_movie_to_file
import os
import h5py

import ptychi.reconstructors.base as base

ENABLE_MOVIES = False
dataset_name = "frames"

# T = Union[ObjectMovieSettings, ProbeMovieSettings]


class MovieSubject:
    current_frame: int = 0
    frames: np.ndarray

    def __init__(
        self,
        # movie_subject: MovieSubjectTypes,
        settings: movie_setting_types,
        folder: str,
        file_name: str,
        save_intermediate_data_to_hdf5: bool = False,
    ):
        self.settings = settings
        # self.movie_subject_type = movie_subject
        # self.scale = scale
        # self.stride = stride
        self.folder = folder
        self.file_name = file_name
        self.save_intermediate_data_to_hdf5 = save_intermediate_data_to_hdf5
        self.hdf5_file_path = os.path.join(self.folder, self.file_name + ".hdf5")

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

    def create_movie(self, file_type: MovieFileTypes, fps: int = 5):
        output_path = os.path.join(self.folder, self.file_name + "." + file_type)
        if self.save_intermediate_data_to_hdf5:
            with h5py.File(self.hdf5_file_path) as F:
                frames = F[dataset_name][:]
        else:
            frames = self.frames
        save_movie_to_file(
            array=frames,
            file_type=file_type,
            output_path=output_path,
            fps=fps,
        )

    def reset(self):
        self.current_frame = 0


class SubjectList:
    subject_list: List[MovieSubject] = []

    def add_subject(
        self,
        settings: movie_setting_types,
        folder: str,
        file_name: str,
    ):
        if self.is_duplicate(settings):
            return
        else:
            self.subject_list += [MovieSubject(settings, folder, file_name)]

    def record_all_frames(self, reconstructor: "base.Reconstructor"):
        for subject in self.subject_list:
            if subject.save_frame_enabled_this_epoch(reconstructor.current_epoch):
                movie_frame_array = prepare_movie_subject(reconstructor, subject.settings)
                subject.record_frame(movie_frame_array)

    def create_all_movies(self, file_type: MovieFileTypes):
        for subject in self.subject_list:
            subject.create_movie(file_type)

    def reset_all(self):
        for subject in self.subject_list:
            subject.reset()

    def is_duplicate(self, settings: movie_setting_types):
        for subject in self.subject_list:
            if subject.settings == settings:
                return True
        return False


if __name__ == "__main__":
    subject_list = SubjectList()
    # subject_list.add_subject(MovieSubjectTypes.)
