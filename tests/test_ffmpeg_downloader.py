# -*- coding: utf-8 -*-
import io
import zipfile

import ffsubsync.ffmpeg_downloader as fd


def _make_fake_ffmpeg_zip(include_ffprobe=True):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ffmpeg-9.0-essentials/README.txt", "readme")
        zf.writestr("ffmpeg-9.0-essentials/bin/ffmpeg.exe", b"FAKE_FFMPEG")
        if include_ffprobe:
            zf.writestr("ffmpeg-9.0-essentials/bin/ffprobe.exe", b"FAKE_FFPROBE")
    return buf.getvalue()


def test_find_ffmpeg_prefers_portable_over_system(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "_probe_ffmpeg_version", lambda _p: "9.0.2")
    monkeypatch.setattr(fd.shutil, "which", lambda name: "C:\\sys\\" + name)

    # Without portable files, falls back to system
    sys_info = fd.find_ffmpeg_binaries(tmp_path)
    assert sys_info["available"] is True
    assert sys_info["source"] == "system"

    # Once placed in portable folder, portable wins
    (tmp_path / "ffmpeg.exe").write_bytes(b"x")
    (tmp_path / "ffprobe.exe").write_bytes(b"y")
    port_info = fd.find_ffmpeg_binaries(tmp_path)
    assert port_info["available"] is True
    assert port_info["source"] == "portable"
    assert port_info["resource_dir"] == str(tmp_path)


def test_download_latest_ffmpeg_extracts_binaries_and_falls_back(tmp_path, monkeypatch):
    zip_bytes = _make_fake_ffmpeg_zip()

    class _FakeResp:
        def __init__(self, data):
            self._stream = io.BytesIO(data)
            self.headers = {"Content-Length": str(len(data))}

        def read(self, n=-1):
            return self._stream.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen_urls = []

    def fake_urlopen(req, timeout=30):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        seen_urls.append(url)
        if "bad-mirror" in url:
            raise OSError("network down")
        return _FakeResp(zip_bytes)

    monkeypatch.setattr(fd.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fd, "_probe_ffmpeg_version", lambda _p: "9.0.2")
    monkeypatch.setattr(fd, "_REQUIRED_BINS", ("ffmpeg.exe", "ffprobe.exe"))

    events = []
    info = fd.download_latest_ffmpeg(
        target_dir=tmp_path,
        progress_cb=events.append,
        urls=("https://bad-mirror.example/ff.zip", "https://good.example/ff.zip"),
    )
    assert seen_urls == [
        "https://bad-mirror.example/ff.zip",
        "https://good.example/ff.zip",
    ]
    assert info["available"] is True
    assert info["source"] == "portable"
    assert (tmp_path / "ffmpeg.exe").read_bytes() == b"FAKE_FFMPEG"
    assert (tmp_path / "ffprobe.exe").read_bytes() == b"FAKE_FFPROBE"
    assert not (tmp_path / "README.txt").exists()
    phases = [e["phase"] for e in events]
    assert "downloading" in phases and "extracting" in phases and phases[-1] == "ready"
    # No leftover temporary zip files
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".zip"] == []


def test_get_portable_dir_falls_back_when_preferred_readonly(tmp_path, monkeypatch):
    readonly_dir = tmp_path / "program_files"
    fallback_dir = tmp_path / "local_appdata"
    fallback_dir.mkdir()
    (fallback_dir / "ffmpeg.exe").write_bytes(b"ff")
    (fallback_dir / "ffprobe.exe").write_bytes(b"fp")

    monkeypatch.setattr(fd, "_preferred_portable_dir", lambda: readonly_dir)
    monkeypatch.setattr(fd, "_get_fallback_dir", lambda: fallback_dir)
    monkeypatch.setattr(fd, "_is_dir_writable", lambda p: p == fallback_dir)
    monkeypatch.setattr(fd, "_probe_ffmpeg_version", lambda _p: "9.0.2")
    monkeypatch.setattr(fd, "_REQUIRED_BINS", ("ffmpeg.exe", "ffprobe.exe"))
    monkeypatch.setattr(fd.shutil, "which", lambda _name: None)

    assert fd.get_portable_dir() == fallback_dir
    info = fd.find_ffmpeg_binaries()
    assert info["available"] is True
    assert info["source"] == "portable"
    assert info["resource_dir"] == str(fallback_dir)
