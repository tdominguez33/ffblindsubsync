# -*- coding: utf-8 -*-
"""Regression tests for strictly monotonic overall progress across Desktop and Web."""
import importlib.util
import logging
from pathlib import Path
import sys
import types

import numpy as np


def _load_module(name: str, rel_path: str):
    mod_path = Path(__file__).resolve().parent.parent / rel_path
    spec = importlib.util.spec_from_file_location(name, str(mod_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_web_bridge_progress_handler_is_monotonic_across_alt_pass_and_multi():
    bridge = _load_module("_test_web_bridge_prog", "web/src/ffsubsync_bridge.py")

    events = []

    def cb(frac, msg):
        events.append((float(frac), str(msg)))

    handler = bridge._ProgressLogHandler(cb, start_frac=0.0, end_frac=1.0)
    rec = lambda text: logging.LogRecord("ffsubsync", logging.INFO, "", 0, text, (), None)

    # Primary alignment pass + surround FC alt pass + split + write
    for line in (
        "extracting speech segments from subtitles...",
        "computing alignment 1/4...",
        "computing alignment 2/4...",
        "computing alignment 3/4...",
        "computing alignment 4/4...",
        "comparing alignment with surround center channel (FC)...",
        "computing alignment 1/4...",
        "computing alignment 2/4...",
        "computing alignment 3/4...",
        "computing alignment 4/4...",
        "split alignment produced 2 segments",
        "writing output to /work/output.srt",
    ):
        handler.emit(rec(line))

    fracs = [f for f, _ in events]
    assert len(fracs) == 12
    assert fracs == sorted(fracs), f"Progress went backwards: {fracs}"
    assert fracs[0] > 0.0
    assert fracs[-1] >= 0.95

    # Multi-candidate range switching stays strictly monotonic and preserves candidate label
    multi_events = []
    mh = bridge._ProgressLogHandler(
        lambda f, m: multi_events.append((float(f), str(m))),
        start_frac=0.0,
        end_frac=0.5,
    )
    mh.set_range(0.0, 0.5, message_override="Analyzing subtitle 1/2 (a.srt)…")
    mh.emit(rec("computing alignment 1/2..."))
    mh.emit(rec("comparing alignment with surround center channel (FC)..."))
    mh.emit(rec("computing alignment 1/2..."))
    mh.emit(rec("computing alignment 2/2..."))

    mh.set_range(0.5, 1.0, message_override="Analyzing subtitle 2/2 (b.srt)…")
    mh.emit(rec("computing alignment 1/2..."))
    mh.emit(rec("computing alignment 2/2..."))

    m_fracs = [f for f, _ in multi_events]
    assert m_fracs == sorted(m_fracs), f"Multi progress went backwards: {m_fracs}"
    assert all("1/2 (a.srt)" in m for _, m in multi_events[:4])
    assert all("2/2 (b.srt)" in m for _, m in multi_events[4:])
    assert m_fracs[3] <= 0.5 <= m_fracs[4]


def test_web_bridge_prepare_reference_audio_reaches_full_range_with_step_pct(
    monkeypatch, tmp_path
):
    bridge = _load_module("_test_web_bridge_audio_prog", "web/src/ffsubsync_bridge.py")
    monkeypatch.setattr(bridge, "_WORK_DIR", str(tmp_path))
    monkeypatch.setattr(
        bridge,
        "_make_browser_detector",
        lambda vad, fr, nsl: (lambda chunk: np.ones(max(1, len(chunk) // 320), dtype=np.float32)),
    )

    events = []
    pcm = bytes(32000)  # 1 second of 16 kHz s16le mono
    res = bridge.prepare_reference_audio(
        "cache-k",
        pcm,
        16000,
        progress_cb=lambda f, m: events.append((float(f), str(m))),
    )
    assert res["ok"] is True
    fracs = [f for f, _ in events]
    assert fracs == sorted(fracs)
    assert fracs[-1] == 1.0
    assert any("(100%)" in msg for _, msg in events)


def test_pgs_streaming_progress_emits_monotonic_updates(monkeypatch):
    import io
    from ffsubsync import speech_transformers as st

    lines = [
        b"pts_time=10.000000|duration_time=N/A|size=4000\n",
        b"pts_time=12.500000|duration_time=N/A|size=28\n",
        b"pts_time=50.000000|duration_time=3.000000|size=4500\n",
        b"pts_time=95.000000|duration_time=2.500000|size=4200\n",
    ]

    class FakeProc:
        def __init__(self):
            self.stdout = io.BytesIO(b"".join(lines))
            self.returncode = 0

        def wait(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(st, "_probe_media_duration", lambda *a, **kw: 100.0)
    monkeypatch.setattr(st.subprocess, "Popen", lambda *a, **kw: FakeProc())

    prog = []
    timings = st._get_pgs_timings_via_ffprobe(
        "movie.mkv", "0:s:0", progress_handler=lambda info: prog.append(info)
    )
    assert timings == [(10.0, 12.5), (50.0, 53.0), (95.0, 97.5)]
    assert len(prog) >= 2
    fracs = [p.fraction for p in prog]
    assert fracs[0] == 0.0
    assert fracs[-1] == 1.0
    assert fracs == sorted(fracs)


def test_desktop_progress_is_monotonic_with_pgs_step_pct_and_clean_archive_names(
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
        desktop = _load_module("_test_desktop_prog", "gui/desktop_app.py")
    finally:
        if prev_w is not None:
            sys.modules["webview"] = prev_w
        else:
            sys.modules.pop("webview", None)
        if prev_d is not None:
            sys.modules["webview.dom"] = prev_d
        else:
            sys.modules.pop("webview.dom", None)

    monkeypatch.setattr(desktop, "_config_path", lambda: tmp_path / "config.json")
    monkeypatch.setattr(desktop, "get_portable_dir", lambda: tmp_path)

    ref_vid = tmp_path / "movie.mkv"
    ref_vid.write_bytes(b"fake-video")
    sub_a = tmp_path / "0001_sub_a.srt"
    sub_b = tmp_path / "0002_sub_b.srt"
    sub_a.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi A\n", encoding="utf-8")
    sub_b.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi B\n", encoding="utf-8")

    api = desktop.DesktopApi()
    api._archive_meta = {
        str(sub_a): {
            "display_name": "super_long_release_archive.rar › movie_es.srt",
            "output_name": "movie_es.srt",
            "default_output_path": str(tmp_path / "movie_es.synced.srt"),
        },
        str(sub_b): {
            "display_name": "super_long_release_archive.rar › movie_en.srt",
            "output_name": "movie_en.srt",
            "default_output_path": str(tmp_path / "movie_en.synced.srt"),
        },
    }
    monkeypatch.setattr(
        api,
        "_find_ffmpeg",
        lambda: {"available": True, "resource_dir": str(tmp_path)},
    )

    observed = []

    def fake_run(
        args,
        progress_handler=None,
        reference_pipe=None,
        candidate_progress_cb=None,
        return_reference_pipe=False,
        in_memory_only=False,
    ):
        ff_log = logging.getLogger("ffsubsync")
        prev_level = ff_log.level
        ff_log.setLevel(logging.INFO)
        try:
            def record():
                st = api.get_poll_state()["sync"]
                observed.append(
                    (
                        st.get("fraction"),
                        st.get("message"),
                        st.get("message_key"),
                        st.get("message_params"),
                    )
                )

            # Simulate PGS timing streaming 0% -> 100%
            ff_log.info("reading hdmv_pgs_subtitle timings for stream 0:s:0 from movie.mkv...")
            record()
            for step in (0.1, 0.5, 1.0):
                progress_handler(
                    desktop.ProgressInfo(
                        processed_seconds=step * 100.0,
                        total_seconds=100.0,
                    )
                )
                record()

            # Simulate multi-candidate alignment + surround FC alt pass
            for idx, p in enumerate((str(sub_a), str(sub_b))):
                candidate_progress_cb(idx, 2, p)
                record()
                for line in (
                    "extracting speech segments from subtitles...",
                    "computing alignments...",
                    "computing alignment 1/4...",
                    "computing alignment 4/4...",
                    "comparing alignment with surround center channel (FC)...",
                    "computing alignment 1/4...",
                    "computing alignment 4/4...",
                    "split alignment: 1 segment",
                ):
                    ff_log.info(line)
                    record()
        finally:
            ff_log.setLevel(prev_level)

        return {
            "sync_was_successful": True,
            "multi_mode": True,
            "candidates": [],
        }

    monkeypatch.setattr(desktop, "run", fake_run)

    started = api.start_sync(
        {
            "ref_path": str(ref_vid),
            "in_paths": [str(sub_a), str(sub_b)],
        }
    )
    assert started["started"] is True

    import time

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if api.get_poll_state()["sync"].get("done"):
            break
        time.sleep(0.02)

    final_st = api.get_poll_state()["sync"]
    assert final_st["done"] is True

    fracs = [f for f, _, _, _ in observed]
    assert all(f is not None for f in fracs), f"Unexpected None fraction: {observed}"
    assert fracs == sorted(fracs), f"Desktop progress went backwards: {observed}"
    assert fracs[3] < 0.70  # PGS finished around ref_end (0.55)
    assert fracs[-1] > 0.90

    # PGS emitted per-step percentage keys in button text
    pgs_keys = [k for _, _, k, _ in observed if k == "progress.readingPgsPct"]
    assert len(pgs_keys) == 3

    # Analyzing subtitle progress shows ONLY inner filename, not .rar prefix
    cand_msgs = [
        (m, params)
        for _, m, k, params in observed
        if k == "progress.analyzingSubtitle"
    ]
    assert len(cand_msgs) >= 2
    assert any(params and params.get("name") == "movie_es.srt" for _, params in cand_msgs)
    assert any(params and params.get("name") == "movie_en.srt" for _, params in cand_msgs)
    for msg, _ in cand_msgs:
        assert ".rar" not in (msg or "")

