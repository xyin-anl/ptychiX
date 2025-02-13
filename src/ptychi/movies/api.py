from .movie_utils import SubjectList
import ptychi.reconstructors.base as base
from .mappings import movie_setting_types

MOVIE_LIST = SubjectList()


def toggle_movies(enable: bool):
    global ENABLE_MOVIES
    ENABLE_MOVIES = enable


def clear_movie_globals():
    global MOVIE_LIST
    MOVIE_LIST = SubjectList()


def add_to_movie_list(settings: movie_setting_types, folder: str, file_name: str):
    global MOVIE_LIST
    if MOVIE_LIST is None:
        MOVIE_LIST = SubjectList()
    MOVIE_LIST.add_subject(settings, folder, file_name)


def update_movies(reconstructor: "base.Reconstructor"):
    global MOVIE_LIST
    MOVIE_LIST.record_all_frames(reconstructor)


def create_all_movies(file_type: str):
    global MOVIE_LIST
    MOVIE_LIST.create_all_movies(file_type)


def reset_movies():
    global MOVIE_LIST
    MOVIE_LIST.reset_all()
