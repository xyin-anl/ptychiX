from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('ptychointerim')
except PackageNotFoundError:
    pass

from .device import Device, list_available_devices
