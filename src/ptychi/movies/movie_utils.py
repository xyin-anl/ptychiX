from typing import List
import numpy as np
from enum import StrEnum, auto
from .io import numpy_to_mp4, numpy_to_gif, append_array_to_h5
import os
import h5py

import ptychi.reconstructors.base as base

ENABLE_MOVIES = False
dataset_name = "frames"


class MovieSubjectTypes(StrEnum):
    OBJECT_MAGNITUDE = auto()
    OBJECT_PHASE = auto()
    PROBE_INTENSITY = auto()
    # LOSS = auto()


class MovieSubject:
    current_frame: int = 0
    frames: np.ndarray

    def __init__(
        self,
        movie_subject: MovieSubjectTypes,
        scale: int,
        stride: int,
        folder: str,
        file_name: str,
        save_intermediate_data_to_hdf5: bool = False,
    ):
        self.movie_subject_type = movie_subject
        self.scale = scale
        self.stride = stride
        self.folder = folder
        self.file_name = file_name
        self.save_intermediate_data_to_hdf5 = save_intermediate_data_to_hdf5
        self.hdf5_file_path = os.path.join(self.folder, self.file_name + ".hdf5")

    def save_frame_enabled_this_epoch(self, current_epoch: int):
        if current_epoch % self.stride == 0:
            return True
        else:
            return False

    def save_movie_frame(self, movie_frame_array: np.ndarray):
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

    def create_gif(self, fps: int = 5):
        if self.save_intermediate_data_to_hdf5:
            with h5py.File(self.hdf5_file_path) as F:
                numpy_to_gif(
                    array=F[dataset_name][:],
                    output_path=os.path.join(self.folder, self.file_name + ".gif"),
                    fps=fps,
                )
        else:
            numpy_to_gif(
                array=self.frames,
                output_path=os.path.join(self.folder, self.file_name + ".gif"),
                fps=fps,
            )

    def create_mp4(self, fps: int = 5):
        if self.save_intermediate_data_to_hdf5:
            with h5py.File(self.hdf5_file_path) as F:
                numpy_to_mp4(
                    array=F[dataset_name][:],
                    output_path=os.path.join(self.folder, self.file_name + ".mp4"),
                    fps=fps,
                )
        else:
            numpy_to_mp4(
                array=self.frames,
                output_path=os.path.join(self.folder, self.file_name + ".mp4"),
                fps=fps,
            )

    def reset(self):
        self.current_frame = 0


class SubjectList:
    subject_list: List[MovieSubject] = []

    def add_subject(
        self,
        movie_subject: MovieSubjectTypes,
        scale: int,
        stride: int,
        folder: str,
        file_name: str,
    ):
        self.subject_list += [MovieSubject(movie_subject, scale, stride, folder, file_name)]

    def save_movie_frame_to_file(self, reconstructor: "base.Reconstructor"):
        for subject in self.subject_list:
            if subject.save_frame_enabled_this_epoch(reconstructor.current_epoch):
                movie_frame_array = prepare_movie_subject(
                    reconstructor, subject.movie_subject_type, subject.scale
                )
                subject.save_movie_frame(movie_frame_array)

    def reset_all(self):
        for subject in self.subject_list:
            subject.reset()


def prepare_movie_subject(
    reconstructor: "base.Reconstructor",
    movie_subject: MovieSubject,
    scale: int,
) -> np.ndarray:
    if movie_subject is MovieSubjectTypes.OBJECT_MAGNITUDE:
        array_out = reconstructor.parameter_group.object.data[0].abs().cpu().detach().numpy()
    elif movie_subject is MovieSubjectTypes.OBJECT_PHASE:
        array_out = reconstructor.parameter_group.object.data[0].angle().cpu().detach().numpy()
    elif movie_subject is MovieSubjectTypes.PROBE_INTENSITY:
        array_out = (
            (reconstructor.parameter_group.probe.data[0].abs() ** 2).sum(0).cpu().detach().numpy()
        )
    # elif movie_subject is MovieSubjectTypes.LOSS:
    #     return reconstructor.loss_tracker.table["loss"]

    return array_out[::scale, ::scale]


MOVIE_LIST = SubjectList()


def toggle_movies(enable: bool):
    global ENABLE_MOVIES
    ENABLE_MOVIES = enable


def clear_movie_globals():
    global MOVIE_LIST
    MOVIE_LIST = SubjectList()


def add_to_movie_list(
    movie_subject: MovieSubjectTypes, scale: int, stride: int, folder: str, file_name: str
):
    global MOVIE_LIST
    if MOVIE_LIST is None:
        MOVIE_LIST = SubjectList()
    MOVIE_LIST.add_subject(movie_subject, scale, stride, folder, file_name)


def update_movies(reconstructor: "base.Reconstructor"):
    global MOVIE_LIST
    MOVIE_LIST.save_movie_frame_to_file(reconstructor)


def reset_movies():
    global MOVIE_LIST
    MOVIE_LIST.reset_all()


if __name__ == "__main__":
    subject_list = SubjectList()
    # subject_list.add_subject(MovieSubjectTypes.)
