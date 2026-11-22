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


def test_ffmpeg_probe_patched_with_no_window(monkeypatch):
    import subprocess
    import ffmpeg
    import ffsubsync.ffmpeg_utils as fu
    import ffsubsync.speech_transformers  # noqa: F401 - triggers install_no_window_ffmpeg_probe

    assert ffmpeg.probe is fu.probe_no_window

    captured = {}

    class _FakeProc:
        returncode = 0

        def communicate(self):
            return (b'{"format": {"duration": "12.5"}}', b"")

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(fu.subprocess, "Popen", fake_popen)
    data = ffmpeg.probe("movie.mkv", cmd="ffprobe.exe", select_streams="s:0")
    assert data == {"format": {"duration": "12.5"}}
    assert captured["args"][0] == "ffprobe.exe"
    assert "-select_streams" in captured["args"]
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        assert captured["kwargs"].get("creationflags") == subprocess.CREATE_NO_WINDOW
    if hasattr(subprocess, "STARTUPINFO"):
        si = captured["kwargs"].get("startupinfo")
        assert si is not None
        assert si.dwFlags & subprocess.STARTF_USESHOWWINDOW


def test_probe_no_window_raises_ffmpeg_error_on_nonzero(monkeypatch):
    import ffmpeg
    import ffsubsync.ffmpeg_utils as fu

    class _FailProc:
        returncode = 1

        def communicate(self):
            return (b"", b"bad file")

    monkeypatch.setattr(fu.subprocess, "Popen", lambda *a, **k: _FailProc())
    with pytest.raises(ffmpeg.Error):
        fu.probe_no_window("missing.mkv")
