from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('ptychopack')
except PackageNotFoundError:
    pass
