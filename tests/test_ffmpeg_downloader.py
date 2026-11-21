# -*- coding: utf-8 -*-
import io
import os
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
        if not seen_urls or seen_urls[-1] != url:
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


def test_find_ffmpeg_custom_path(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "_probe_ffmpeg_version", lambda _p: "8.1")
    monkeypatch.setattr(fd, "_REQUIRED_BINS", ("ffmpeg.exe", "ffprobe.exe"))
    monkeypatch.setattr(fd.shutil, "which", lambda _name: None)

    custom_dir = tmp_path / "custom_ff"
    bin_dir = custom_dir / "bin"
    bin_dir.mkdir(parents=True)

    # Missing binaries in custom dir reports unavailable with custom_error
    bad = fd.find_ffmpeg_binaries(portable_dir=tmp_path, custom_path=str(custom_dir))
    assert bad["available"] is False
    assert bad["custom_error"] is not None

    # Valid binaries inside custom_dir/bin succeed
    (bin_dir / "ffmpeg.exe").write_bytes(b"ff")
    (bin_dir / "ffprobe.exe").write_bytes(b"fp")
    ok = fd.find_ffmpeg_binaries(portable_dir=tmp_path, custom_path=str(custom_dir))
    assert ok["available"] is True
    assert ok["source"] == "custom"
    assert ok["resource_dir"] == str(bin_dir)

    # Passing ffmpeg.exe file directly resolves its parent folder
    ok_file = fd.find_ffmpeg_binaries(
        portable_dir=tmp_path, custom_path=str(bin_dir / "ffmpeg.exe")
    )
    assert ok_file["available"] is True
    assert ok_file["source"] == "custom"
    assert ok_file["resource_dir"] == str(bin_dir)


def test_download_latest_ffmpeg_uses_range_skip_ffplay(tmp_path, monkeypatch):
    # Build a real deflated ZIP with ffmpeg.exe, a large incompressible ffplay.exe, and ffprobe.exe.
    ffmpeg_payload = b"FFMPEG_BINARY_DATA_" * 200
    ffplay_payload = os.urandom(300_000)  # ~300 KB incompressible dummy ffplay
    ffprobe_payload = b"FFPROBE_BINARY_DATA_" * 200

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ffmpeg-9.1-essentials/bin/ffmpeg.exe", ffmpeg_payload)
        zf.writestr("ffmpeg-9.1-essentials/bin/ffplay.exe", ffplay_payload)
        zf.writestr("ffmpeg-9.1-essentials/bin/ffprobe.exe", ffprobe_payload)
    archive = buf.getvalue()

    bytes_transferred = 0

    class _RangeResp:
        def __init__(self, data, status, headers, url):
            self._stream = io.BytesIO(data)
            self.status = status
            self.headers = headers
            self._url = url

        def geturl(self):
            return self._url

        def read(self, n=-1):
            return self._stream.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=30):
        nonlocal bytes_transferred
        url = req.full_url if hasattr(req, "full_url") else str(req)
        method = req.get_method() if hasattr(req, "get_method") else "GET"
        range_hdr = req.headers.get("Range") if hasattr(req, "headers") else None
        if method == "HEAD":
            return _RangeResp(
                b"",
                200,
                {"Content-Length": str(len(archive)), "Accept-Ranges": "bytes"},
                url,
            )
        if range_hdr and range_hdr.startswith("bytes="):
            s_str, e_str = range_hdr[len("bytes=") :].split("-", 1)
            s = int(s_str)
            e = min(len(archive) - 1, int(e_str))
            chunk = archive[s : e + 1]
            bytes_transferred += len(chunk)
            return _RangeResp(
                chunk,
                206,
                {"Content-Length": str(len(chunk))},
                url,
            )
        raise AssertionError("Should not fall back to full GET when Range works")

    monkeypatch.setattr(fd.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fd, "_probe_ffmpeg_version", lambda _p: "9.1")
    monkeypatch.setattr(fd, "_REQUIRED_BINS", ("ffmpeg.exe", "ffprobe.exe"))

    events = []
    info = fd.download_latest_ffmpeg(
        target_dir=tmp_path,
        progress_cb=events.append,
        urls=("https://github.example/ffmpeg-9.1-essentials_build.zip",),
    )
    assert info["available"] is True
    assert (tmp_path / "ffmpeg.exe").read_bytes() == ffmpeg_payload
    assert (tmp_path / "ffprobe.exe").read_bytes() == ffprobe_payload
    assert not (tmp_path / "ffplay.exe").exists()
    # We must have transferred far less than the full archive (ffplay.exe was skipped!)
    assert bytes_transferred < len(archive) * 0.5
    phases = [e["phase"] for e in events]
    assert "downloading" in phases and phases[-1] == "ready"
