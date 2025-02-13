from typing import Optional
from .movie_utils import SubjectList
from .mappings import movie_setting_types
from .settings import MovieFileTypes, MovieFileSettings
from ptychi.timing.timer_utils import timer
import ptychi.reconstructors.base as base

MOVIE_LIST = SubjectList()


def toggle_movies(enable: bool):
    global ENABLE_MOVIES
    ENABLE_MOVIES = enable


def clear_movie_globals():
    global MOVIE_LIST
    MOVIE_LIST = SubjectList()


def add_to_movie_list(settings: movie_setting_types, folder: str, movie_name: str):
    global MOVIE_LIST
    if MOVIE_LIST is None:
        MOVIE_LIST = SubjectList()
    MOVIE_LIST.add_subject(settings, folder, movie_name)


@timer()
def update_movies(reconstructor: "base.Reconstructor"):
    global MOVIE_LIST
    MOVIE_LIST.record_all_frames(reconstructor)


def create_all_movies(file_type: MovieFileTypes):
    global MOVIE_LIST
    MOVIE_LIST.create_all_movies(file_type)


def create_movie(
    movie_name: str,
    file_type: MovieFileTypes,
    movie_file_settings: Optional[MovieFileSettings] = None,
):
    if not check_if_movie_exists(movie_name):
        return
    global MOVIE_LIST
    MOVIE_LIST.subject_list[movie_name].create_movie(file_type, movie_file_settings)


def get_movie_settings(movie_name: str) -> movie_setting_types:
    if not check_if_movie_exists(movie_name):
        return
    global MOVIE_LIST
    return MOVIE_LIST.subject_list[movie_name].settings


def reset_movies():
    global MOVIE_LIST
    MOVIE_LIST.reset_all()


def check_if_movie_exists(movie_name: SystemError):
    global MOVIE_LIST
    if movie_name not in MOVIE_LIST.subject_list.keys():
        print(f"{movie_name} is not found")
        return False
    else:
        return True
