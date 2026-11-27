# -*- coding: utf-8 -*-
"""Tests for compressed archive subtitle extraction (.zip, .rar, .7z, .tar.gz, .gz)."""
from datetime import timedelta
import gzip
import importlib.util
import io
import os
from pathlib import Path
import struct
import sys
import tarfile
import types
import zipfile
import zlib

import numpy as np
import pytest
import srt

from ffsubsync.archive_utils import (
    _extract_with_bsdtar,
    _find_cli_tool,
    extract_subtitles_from_archive,
    is_archive_filename,
)
from ffsubsync.ffsubsync import discover_subtitle_files, make_parser, run, validate_args


def _irregular_intervals(n: int = 38, seed: int = 13):
    rng = np.random.default_rng(seed)
    intervals = []
    t = 2.0
    for _ in range(n):
        dur = float(rng.uniform(1.2, 2.5))
        gap = float(rng.uniform(1.0, 3.6))
        intervals.append((t, t + dur))
        t += dur + gap
    return intervals


def _srt_bytes(intervals, shift: float = 0.0, prefix: str = "Line") -> bytes:
    subs = []
    for i, (a, b) in enumerate(intervals, start=1):
        start_s = max(0.0, a + shift)
        end_s = max(start_s + 0.1, b + shift)
        subs.append(
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=start_s),
                end=timedelta(seconds=end_s),
                content="%s %d — readable dialogue line." % (prefix, i),
            )
        )
    return srt.compose(subs).encode("utf-8")


def _build_minimal_rar4(files):
    """Build a valid uncompressed (STORE) RAR4 archive from ``[(name, bytes), ...]``."""
    # RAR 4.x marker block
    out = bytearray(b"Rar!\x1a\x07\x00")
    # Main archive header (type 0x73, flags 0, size 13)
    main_body = struct.pack("<BHHHI", 0x73, 0x0000, 13, 0, 0)
    main_crc = zlib.crc32(main_body) & 0xFFFF
    out += struct.pack("<H", main_crc) + main_body

    for name, data in files:
        name_bytes = name.encode("utf-8")
        # File header (type 0x74): flags 0x8000 = LONG_BLOCK (ADD_SIZE present)
        # Base file header = 32 bytes + len(name_bytes)
        hdr_size = 32 + len(name_bytes)
        file_crc = zlib.crc32(data) & 0xFFFFFFFF
        body = (
            struct.pack(
                "<BHHIIBI I B B H I",
                0x74,  # HEAD_TYPE = FILE_HEAD
                0x8000,  # HEAD_FLAGS = LONG_BLOCK
                hdr_size,  # HEAD_SIZE
                len(data),  # PACK_SIZE
                len(data),  # UNP_SIZE
                2,  # HOST_OS = Windows
                file_crc,  # FILE_CRC
                0,  # FTIME
                20,  # UNP_VER = 2.0
                0x30,  # METHOD = STORE
                len(name_bytes),  # NAME_SIZE
                0x20,  # ATTR
            )
            + name_bytes
        )
        hdr_crc = zlib.crc32(body) & 0xFFFF
        out += struct.pack("<H", hdr_crc) + body + data

    # End of archive block (type 0x7B, flags 0x4000, size 7)
    end_body = struct.pack("<BHH", 0x7B, 0x4000, 7)
    end_crc = zlib.crc32(end_body) & 0xFFFF
    out += struct.pack("<H", end_crc) + end_body
    return bytes(out)


def test_is_archive_filename():
    for name in (
        "subs.zip",
        "MOVIE.RAR",
        "pack.7z",
        "pack.tar.gz",
        "pack.tgz",
        "pack.tar.xz",
        "movie.srt.gz",
    ):
        assert is_archive_filename(name) is True
    for name in ("movie.srt", "movie.ass", "video.mkv", "readme.txt"):
        assert is_archive_filename(name) is False


def test_zip_extraction_filters_junk_handles_nested_and_zip_slip(tmp_path):
    inner_zip_buf = io.BytesIO()
    with zipfile.ZipFile(inner_zip_buf, "w") as izf:
        izf.writestr("nested_sub.srt", b"1\n00:00:01,000 --> 00:00:02,000\nHi\n")

    zip_path = tmp_path / "subs_pack.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Spanish/movie.srt", b"1\n00:00:01,000 --> 00:00:02,000\nHola\n")
        zf.writestr("English/movie.srt", b"1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        zf.writestr("movie.synced.srt", b"should be skipped")
        zf.writestr("__MACOSX/._movie.srt", b"junk")
        zf.writestr("README.txt", b"not a subtitle")
        zf.writestr("../../evil.vtt", b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nSafe\n")
        zf.writestr("extra.zip", inner_zip_buf.getvalue())

    records, tmp_dir = extract_subtitles_from_archive(str(zip_path))
    try:
        names = [r["display_name"] for r in records]
        assert any("English/movie.srt" in n for n in names)
        assert any("Spanish/movie.srt" in n for n in names)
        assert any("evil.vtt" in n for n in names)
        assert any("extra.zip › nested_sub.srt" in n for n in names)
        assert not any("synced" in n for n in names)
        assert not any("__MACOSX" in n for n in names)
        # Zip-slip safety: every extracted file lives directly inside tmp_dir
        for rec in records:
            assert Path(rec["path"]).parent.resolve() == Path(tmp_dir).resolve()
            assert Path(rec["default_output_path"]).parent.resolve() == tmp_path.resolve()
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_tar_gz_and_single_srt_gz_extraction(tmp_path):
    tgz = tmp_path / "subs.tar.gz"
    content = b"1\n00:00:01,000 --> 00:00:02,000\nFrom tar\n"
    with tarfile.open(tgz, "w:gz") as tf:
        info = tarfile.TarInfo("sub_a.srt")
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))

    recs_tar, tmp_tar = extract_subtitles_from_archive(str(tgz))
    try:
        assert len(recs_tar) == 1
        assert recs_tar[0]["display_name"] == "subs.tar.gz › sub_a.srt"
        assert Path(recs_tar[0]["path"]).read_bytes() == content
    finally:
        import shutil

        shutil.rmtree(tmp_tar, ignore_errors=True)

    gz_sub = tmp_path / "movie.es.srt.gz"
    with gzip.open(gz_sub, "wb") as gf:
        gf.write(content)

    recs_gz, tmp_gz = extract_subtitles_from_archive(str(gz_sub))
    try:
        assert len(recs_gz) == 1
        assert recs_gz[0]["output_name"] == "movie.es.srt"
        assert recs_gz[0]["default_output_path"] == str(tmp_path / "movie.es.synced.srt")
    finally:
        import shutil

        shutil.rmtree(tmp_gz, ignore_errors=True)


def test_7z_extraction_with_py7zr(tmp_path):
    py7zr = pytest.importorskip("py7zr")
    src = tmp_path / "orig.srt"
    src.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello 7z\n", encoding="utf-8")
    seven_path = tmp_path / "pack.7z"
    with py7zr.SevenZipFile(seven_path, "w") as sz:
        sz.write(src, arcname="folder/inside.srt")

    recs, tmp_dir = extract_subtitles_from_archive(str(seven_path))
    try:
        assert len(recs) == 1
        assert recs[0]["display_name"] == "pack.7z › inside.srt"
        assert "Hello 7z" in Path(recs[0]["path"]).read_text(encoding="utf-8")
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_rar_extraction_with_unrar2_cffi(tmp_path):
    pytest.importorskip("unrar.cffi.rarfile")
    sub_data = b"1\n00:00:01,000 --> 00:00:02,000\nHello RAR\n"
    rar_bytes = _build_minimal_rar4(
        [
            ("release/movie.es.srt", sub_data),
            ("readme.nfo", b"ignored"),
        ]
    )
    rar_path = tmp_path / "movie_subs.rar"
    rar_path.write_bytes(rar_bytes)

    recs, tmp_dir = extract_subtitles_from_archive(str(rar_path))
    try:
        assert len(recs) == 1
        assert recs[0]["display_name"] == "movie_subs.rar › movie.es.srt"
        assert Path(recs[0]["path"]).read_bytes() == sub_data
        assert recs[0]["default_output_path"] == str(tmp_path / "movie.es.synced.srt")
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_bsdtar_cli_fallback_extracts_zip(tmp_path):
    if not _find_cli_tool(("bsdtar", "tar")):
        pytest.skip("bsdtar/tar not available")
    zp = tmp_path / "fallback.zip"
    content = b"1\n00:00:01,000 --> 00:00:02,000\nVia bsdtar\n"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("sub.srt", content)
    res = _extract_with_bsdtar(str(zp))
    assert res is not None
    assert len(res) == 1
    assert res[0][0] == "sub.srt"
    assert res[0][1] == content


def test_cli_archive_ranking_saves_winner_next_to_archive_and_cleans_temp(tmp_path):
    ref_intervals = _irregular_intervals(n=38, seed=17)
    wrong_intervals = _irregular_intervals(n=38, seed=83)

    ref_path = tmp_path / "ref.srt"
    ref_path.write_bytes(_srt_bytes(ref_intervals, shift=0.0))

    zip_path = tmp_path / "downloads.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("bad.srt", _srt_bytes(wrong_intervals, shift=0.0, prefix="Bad"))
        zf.writestr("good.srt", _srt_bytes(ref_intervals, shift=3.0, prefix="Good"))

    args = make_parser().parse_args(
        [str(ref_path), "-i", str(zip_path), "--no-fix-framerate"]
    )
    res = run(args)
    assert res["sync_was_successful"] is True
    assert res["multi_mode"] is True
    cands = res["candidates"]
    assert len(cands) == 2
    assert cands[0]["input_name"] == "downloads.zip › good.srt"
    assert abs(cands[0]["offset_seconds"] - (-3.0)) < 0.25

    # Winner saved in archive's parent folder with clean inner filename
    expected_out = tmp_path / "good.synced.srt"
    assert expected_out.is_file()
    assert not (tmp_path / "bad.synced.srt").exists()
    assert cands[0]["output_path"] == str(expected_out)

    # Temp extraction directory was cleaned up after run()
    assert not os.path.exists(cands[0]["input_path"])


def test_cli_rejects_overwrite_input_with_archive(tmp_path):
    ref = tmp_path / "ref.srt"
    ref.write_text("", encoding="utf-8")
    zp = tmp_path / "one.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("a.srt", b"1\n00:00:01,000 --> 00:00:02,000\nHi\n")

    args = make_parser().parse_args([str(ref), "-i", str(zp), "--overwrite-input"])
    with pytest.raises(ValueError, match="overwrite-input"):
        validate_args(args)


def test_desktop_api_resolves_archive_with_display_names_and_output_paths(
    tmp_path, monkeypatch
):
    fake_webview = types.ModuleType("webview")
    fake_webview.Window = object
    fake_webview.screens = []
    fake_dom = types.ModuleType("webview.dom")
    fake_dom.DOMEventHandler = object
    fake_webview.dom = fake_dom

    prev_w = sys.modules.get("webview")
    prev_d = sys.modules.get("webview.dom")
    sys.modules["webview"] = fake_webview
    sys.modules["webview.dom"] = fake_dom
    try:
        mod_path = Path(__file__).resolve().parent.parent / "gui" / "desktop_app.py"
        spec = importlib.util.spec_from_file_location("_test_desktop_arch", str(mod_path))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
    finally:
        if prev_w is not None:
            sys.modules["webview"] = prev_w
        else:
            sys.modules.pop("webview", None)
        if prev_d is not None:
            sys.modules["webview.dom"] = prev_d
        else:
            sys.modules.pop("webview.dom", None)

    monkeypatch.setattr(mod, "_config_path", lambda: tmp_path / "config.json")
    monkeypatch.setattr(mod, "get_portable_dir", lambda: tmp_path)
    api = mod.DesktopApi()

    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("es/pelicula.srt", b"1\n00:00:01,000 --> 00:00:02,000\nHola\n")
        zf.writestr("en/movie.srt", b"1\n00:00:01,000 --> 00:00:02,000\nHello\n")

    res = api.resolve_subtitle_inputs([str(zip_path)])
    assert res["error"] is None
    assert len(res["entries"]) == 2
    displays = [e["display"] for e in res["entries"]]
    assert "bundle.zip › movie.srt" in displays
    assert "bundle.zip › pelicula.srt" in displays
    for e in res["entries"]:
        assert Path(e["default_output_path"]).parent == tmp_path
    api._cleanup_archive_temps()


def test_archive_utils_imports_without_lzma_and_bz2(tmp_path, monkeypatch):
    """Simulate Pyodide where lzma and bz2 are unvendored from Python stdlib."""
    monkeypatch.setitem(sys.modules, "lzma", None)
    monkeypatch.setitem(sys.modules, "bz2", None)
    mod_path = Path(__file__).resolve().parent.parent / "ffsubsync" / "archive_utils.py"
    spec = importlib.util.spec_from_file_location("_test_archive_utils_pyodide", str(mod_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.lzma is None
    assert mod.bz2 is None

    zp = tmp_path / "ok.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("sub.srt", b"1\n00:00:01,000 --> 00:00:02,000\nOk\n")
    recs, tmp_dir = mod.extract_subtitles_from_archive(str(zp))
    try:
        assert len(recs) == 1
        assert recs[0]["output_name"] == "sub.srt"
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

    xz_path = tmp_path / "movie.srt.xz"
    xz_path.write_bytes(b"not-real-xz")
    with pytest.raises(ValueError, match="compression module"):
        mod.extract_subtitles_from_archive(str(xz_path))

