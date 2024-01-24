from importlib import metadata

from . import data, environments, threads, trainers

__version__ = metadata.version(__name__)
