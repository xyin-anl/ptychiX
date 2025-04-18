# ruff: noqa: F401

try:
    from . import api
    from . import settings
    MOVIES_INSTALLED = True
except ImportError:
    MOVIES_INSTALLED = False