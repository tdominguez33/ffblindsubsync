# -*- coding: utf-8 -*-
from importlib import metadata

_DEFAULT_VERSION = "0.5.1"


def _resolve_version() -> str:
    try:
        from ffsubsync._frozen_version import FFSUBSYNC_VERSION

        return FFSUBSYNC_VERSION
    except ImportError:
        pass
    try:
        return metadata.version("ffsubsync")
    except metadata.PackageNotFoundError:
        return _DEFAULT_VERSION


__version__ = _resolve_version()


def get_version() -> str:
    return __version__


def make_version_tuple(vstr=None):
    if vstr is None:
        vstr = __version__
    if vstr.startswith("v"):
        vstr = vstr[1:]
    components = []
    for component in vstr.split("+")[0].split("."):
        try:
            components.append(int(component))
        except ValueError:
            break
    return tuple(components)
