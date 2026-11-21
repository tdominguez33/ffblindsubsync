# -*- coding: utf-8 -*-
import pytest
from ffsubsync.version import _read_pyproject_version, get_version, make_version_tuple


@pytest.mark.parametrize(
    "vstr, expected",
    [("v0.1.1", (0, 1, 1)), ("v1.2.3", (1, 2, 3)), ("4.5.6.1", (4, 5, 6, 1))],
)
def test_version_tuple_from_string(vstr, expected):
    assert make_version_tuple(vstr) == expected


def test_version_reads_pyproject_single_source_of_truth():
    pyproject_ver = _read_pyproject_version()
    assert pyproject_ver
    assert get_version() == pyproject_ver
