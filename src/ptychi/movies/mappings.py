from typing import Union
from ptychi.movies.settings import (
    ObjectMovieSettings,
    ProbeMovieSettings,
    ProbePlotTypes,
    ProcessFunctionType,
)
import ptychi.reconstructors.base as base
import torch
import numpy as np

movie_setting_types = Union[ObjectMovieSettings, ProbeMovieSettings]


def prepare_movie_subject(
    reconstructor: "base.Reconstructor",
    movie_settings: movie_setting_types,
) -> np.ndarray:
    if isinstance(movie_settings, ObjectMovieSettings):
        array_out = prepare_object(reconstructor, movie_settings)
    elif isinstance(movie_settings, ProbeMovieSettings):
        array_out = prepare_probe(reconstructor, movie_settings)

    scale = movie_settings.snapshot.scale
    array_out = array_out.cpu().detach().numpy()[::scale, ::scale]
    return array_out


def process_function(array: torch.Tensor, process_function_type: ProcessFunctionType):
    if process_function_type is ProcessFunctionType.MAGNITUDE:
        return array.abs()
    elif process_function_type is ProcessFunctionType.PHASE:
        return array.angle()


def prepare_object(reconstructor: "base.Reconstructor", movie_settings: ObjectMovieSettings):
    array_out = reconstructor.parameter_group.object.data[movie_settings.slice_index]
    array_out = process_function(array_out, movie_settings.process_function)
    return array_out


def prepare_probe(reconstructor: "base.Reconstructor", movie_settings: ProbeMovieSettings):
    array_out = reconstructor.parameter_group.probe.data
    if movie_settings.plot_type is ProbePlotTypes.INCOHERENT_SUM:
        array_out = (array_out[0].abs() ** 2).sum(0)
    elif movie_settings.plot_type is ProbePlotTypes.SEPERATE_MODES:
        array_out = array_out[0, movie_settings.mode_indices]
        array_out = process_function(array_out, movie_settings.process_function)
        spatial_shape = array_out.shape[2:]
        n_modes = array_out.shape[1]
        new_dims = (spatial_shape[0], spatial_shape[1] * n_modes)
        array_out = array_out[0].reshape(new_dims)
    return array_out