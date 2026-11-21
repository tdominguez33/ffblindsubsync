# -*- coding: utf-8 -*-
from importlib import metadata
from pathlib import Path
import re
from typing import Optional


def _read_pyproject_version() -> Optional[str]:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    in_project = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            in_project = line == "[project]"
            continue
        if in_project:
            m = re.match(r'^version\s*=\s*["\']([^"\']+)["\']', line)
            if m:
                return m.group(1).strip()
    return None


def _resolve_version() -> str:
    try:
        from ffsubsync._frozen_version import FFSUBSYNC_VERSION

        return FFSUBSYNC_VERSION
    except ImportError:
        pass
    pyproject_ver = _read_pyproject_version()
    if pyproject_ver:
        return pyproject_ver
    try:
        return metadata.version("ffsubsync")
    except metadata.PackageNotFoundError:
        return "0.0.0"


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
