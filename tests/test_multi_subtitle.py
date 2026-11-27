# -*- coding: utf-8 -*-
"""Tests for multi-subtitle mode, folder discovery, ranking, and reference caching."""
from datetime import timedelta
import importlib.util
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
import pytest
import srt

from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.ffsubsync import (
    discover_subtitle_files,
    format_ranking_table,
    make_parser,
    run,
    validate_args,
)
from ffsubsync.quality import compute_overall_score, rank_sync_candidates
from ffsubsync.speech_transformers import VideoSpeechTransformer


def _write_srt(path: Path, intervals, shift: float = 0.0, text_prefix: str = "Line") -> str:
    subs = []
    for i, (a, b) in enumerate(intervals, start=1):
        start_s = max(0.0, a + shift)
        end_s = max(start_s + 0.1, b + shift)
        subs.append(
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=start_s),
                end=timedelta(seconds=end_s),
                content="%s %d — clear readable dialogue." % (text_prefix, i),
            )
        )
    path.write_text(srt.compose(subs), encoding="utf-8")
    return str(path)


def _irregular_intervals(n: int = 45, seed: int = 7):
    rng = np.random.default_rng(seed)
    intervals = []
    t = 2.0
    for _ in range(n):
        dur = float(rng.uniform(1.2, 2.5))
        gap = float(rng.uniform(1.0, 3.8))
        intervals.append((t, t + dur))
        t += dur + gap
    return intervals


# ---------------------------------------------------------------------------
# Ranking formula tests
# ---------------------------------------------------------------------------


def test_overall_score_caps_poor_sync_even_when_subtitle_is_clean():
    # Wrong episode: terrible sync (22), pristine subtitle formatting (96).
    wrong_ep_quality = {
        "reference_kind": "audio",
        "sync": {
            "score": 22,
            "metrics": {
                "temporal_consistency": 15,
                "overlap_lift": 18,
                "peak_confidence": 20,
                "sync_integrity": 35,
            },
        },
        "subtitle": {
            "score": 96,
            "metrics": {
                "timeline_coverage": 95,
                "reading_speed": 98,
                "timing_hygiene": 96,
                "text_integrity": 100,
            },
        },
    }
    # Right episode: solid sync (78), decent subtitle quality (68).
    right_ep_quality = {
        "reference_kind": "audio",
        "sync": {
            "score": 78,
            "metrics": {
                "temporal_consistency": 80,
                "overlap_lift": 76,
                "peak_confidence": 79,
                "sync_integrity": 77,
            },
        },
        "subtitle": {
            "score": 68,
            "metrics": {
                "timeline_coverage": 85,
                "reading_speed": 65,
                "timing_hygiene": 62,
                "text_integrity": 70,
            },
        },
    }

    wrong_overall = compute_overall_score(wrong_ep_quality, sync_was_successful=True)
    right_overall = compute_overall_score(right_ep_quality, sync_was_successful=True)
    assert wrong_overall <= 30
    assert right_overall >= 70
    assert right_overall > wrong_overall

    ranked = rank_sync_candidates(
        [
            {"input_name": "wrong.srt", "sync_was_successful": True, "quality": wrong_ep_quality},
            {"input_name": "broken.srt", "sync_was_successful": False, "error": "parse error"},
            {"input_name": "right.srt", "sync_was_successful": True, "quality": right_ep_quality},
        ]
    )
    assert [c["input_name"] for c in ranked] == ["right.srt", "wrong.srt", "broken.srt"]
    assert [c["rank"] for c in ranked] == [1, 2, 3]
    assert ranked[-1]["overall_score"] == 0


def test_overall_score_caps_sparse_forced_track():
    forced_quality = {
        "reference_kind": "audio",
        "sync": {"score": 82, "metrics": {}},
        "subtitle": {
            "score": 55,
            "metrics": {"timeline_coverage": 12},
        },
    }
    score = compute_overall_score(forced_quality, sync_was_successful=True)
    assert score <= 32


# ---------------------------------------------------------------------------
# Folder & file discovery tests
# ---------------------------------------------------------------------------


def test_discover_subtitle_files_top_level_and_recursive(tmp_path):
    sub_dir = tmp_path / "subs"
    nested = sub_dir / "nested"
    nested.mkdir(parents=True)

    ref_sub = sub_dir / "ref.srt"
    a_srt = sub_dir / "a.srt"
    b_ass = sub_dir / "b.ass"
    synced = sub_dir / "a.synced.srt"
    notes = sub_dir / "readme.txt"
    deep = nested / "deep.vtt"

    for p in (ref_sub, a_srt, b_ass, synced, notes, deep):
        p.write_text("dummy", encoding="utf-8")

    # Top-level only by default; skips *.synced.*, non-subtitles, and reference itself.
    found, had_dir = discover_subtitle_files(
        [str(sub_dir)], recursive=False, exclude_path=str(ref_sub)
    )
    assert had_dir is True
    assert [os.path.basename(p) for p in found] == ["a.srt", "b.ass"]

    # Recursive picks up nested/*.vtt as well.
    found_rec, _ = discover_subtitle_files(
        [str(sub_dir)], recursive=True, exclude_path=str(ref_sub)
    )
    assert [os.path.basename(p) for p in found_rec] == ["a.srt", "b.ass", "deep.vtt"]

    # Deduplicates explicit + directory overlap.
    deduped, _ = discover_subtitle_files(
        [str(a_srt), str(sub_dir)], recursive=False, exclude_path=str(ref_sub)
    )
    assert [os.path.basename(p) for p in deduped] == ["a.srt", "b.ass"]


def test_discover_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty_subs"
    empty.mkdir()
    (empty / "notes.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="no subtitle files found"):
        discover_subtitle_files([str(empty)])


def test_cli_single_file_inside_folder_writes_synced_file_not_stdout(tmp_path):
    intervals = _irregular_intervals(n=35, seed=19)
    ref_path = _write_srt(tmp_path / "ref.srt", intervals, shift=0.0)
    sub_dir = tmp_path / "only_one"
    sub_dir.mkdir()
    _write_srt(sub_dir / "solo.srt", intervals, shift=2.0)

    args = make_parser().parse_args([ref_path, "-i", str(sub_dir), "--no-fix-framerate"])
    res = run(args)
    assert res["sync_was_successful"] is True
    assert res["multi_mode"] is True
    assert (sub_dir / "solo.synced.srt").is_file()


# ---------------------------------------------------------------------------
# Core & CLI multi-subtitle ranking, single reference fit, and output flags
# ---------------------------------------------------------------------------


def test_cli_multi_subtitle_default_saves_only_winner_and_fits_ref_once(
    tmp_path, monkeypatch
):
    ref_intervals = _irregular_intervals(n=45, seed=11)
    wrong_intervals = _irregular_intervals(n=45, seed=99)

    ref_path = _write_srt(tmp_path / "reference.srt", ref_intervals, shift=0.0)
    sub_dir = tmp_path / "candidates"
    sub_dir.mkdir()
    good_path = _write_srt(sub_dir / "good.srt", ref_intervals, shift=4.5)
    wrong_path = _write_srt(sub_dir / "wrong.srt", wrong_intervals, shift=0.0)
    corrupt_path = sub_dir / "corrupt.srt"
    corrupt_path.write_text("not valid subtitle content at all", encoding="utf-8")

    # Verify reference pipeline is fit only ONCE across all 3 candidates.
    from ffsubsync import ffsubsync as ffs_mod

    orig_make_ref = ffs_mod.make_reference_pipe
    fit_calls = []

    def counting_make_ref(*a, **kw):
        pipe = orig_make_ref(*a, **kw)
        orig_fit = pipe.fit

        def tracked_fit(*fa, **fkw):
            fit_calls.append(fa[0] if fa else None)
            return orig_fit(*fa, **fkw)

        pipe.fit = tracked_fit
        return pipe

    monkeypatch.setattr(ffs_mod, "make_reference_pipe", counting_make_ref)

    args = make_parser().parse_args(
        [ref_path, "-i", str(sub_dir), "--no-fix-framerate"]
    )
    res = run(args)

    assert len(fit_calls) == 1, "reference should be extracted only once for the batch"
    assert res["sync_was_successful"] is True
    assert res["multi_mode"] is True
    cands = res["candidates"]
    assert len(cands) == 3
    assert cands[0]["rank"] == 1
    assert os.path.basename(cands[0]["input_path"]) == "good.srt"
    assert abs(cands[0]["offset_seconds"] - (-4.5)) < 0.25
    assert cands[0]["overall_score"] > cands[1]["overall_score"]

    # Default multi-subtitle CLI writes ONLY the #1 winner as <winner>.synced.srt
    winner_out = sub_dir / "good.synced.srt"
    wrong_out = sub_dir / "wrong.synced.srt"
    assert winner_out.is_file()
    assert not wrong_out.exists()
    assert cands[0]["output_path"] == str(winner_out)
    assert cands[1]["output_path"] is None

    table = format_ranking_table(cands)
    assert "good.srt" in table
    assert "wrong.srt" in table


def test_cli_multi_subtitle_explicit_output_rank_only_and_save_all(tmp_path):
    ref_intervals = _irregular_intervals(n=40, seed=21)
    wrong_intervals = _irregular_intervals(n=40, seed=77)

    ref_path = _write_srt(tmp_path / "ref.srt", ref_intervals)
    good_path = _write_srt(tmp_path / "good.srt", ref_intervals, shift=3.0)
    wrong_path = _write_srt(tmp_path / "wrong.srt", wrong_intervals, shift=0.0)

    # 1. --rank-only writes zero files
    args_rank = make_parser().parse_args(
        [ref_path, "-i", good_path, wrong_path, "--rank-only", "--no-fix-framerate"]
    )
    res_rank = run(args_rank)
    assert res_rank["sync_was_successful"] is True
    assert not (tmp_path / "good.synced.srt").exists()
    assert not (tmp_path / "wrong.synced.srt").exists()

    # 2. Explicit -o writes #1 winner to that exact file
    best_out = tmp_path / "winner_custom.srt"
    args_best = make_parser().parse_args(
        [ref_path, "-i", wrong_path, good_path, "-o", str(best_out), "--no-fix-framerate"]
    )
    res_best = run(args_best)
    assert res_best["sync_was_successful"] is True
    assert best_out.is_file()
    assert res_best["best_candidate"]["input_name"] == "good.srt"
    assert res_best["best_candidate"]["output_path"] == str(best_out)

    # 3. --save-all with -o <dir> writes every candidate into that directory
    out_dir = tmp_path / "all_synced"
    args_all = make_parser().parse_args(
        [
            ref_path,
            "-i",
            good_path,
            wrong_path,
            "--save-all",
            "-o",
            str(out_dir),
            "--no-fix-framerate",
        ]
    )
    run(args_all)
    assert (out_dir / "good.synced.srt").is_file()
    assert (out_dir / "wrong.synced.srt").is_file()


def test_cli_flag_validation_conflicts(tmp_path):
    ref = tmp_path / "ref.srt"
    a = tmp_path / "a.srt"
    b = tmp_path / "b.srt"
    for p in (ref, a, b):
        p.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="--rank-only"):
        validate_args(
            make_parser().parse_args([str(ref), "-i", str(a), str(b), "--rank-only", "--save-all"])
        )
    with pytest.raises(ValueError, match="--save-all"):
        validate_args(
            make_parser().parse_args(
                [str(ref), "-i", str(a), str(b), "--save-all", "-o", str(tmp_path / "single.srt")]
            )
        )


def test_video_transformer_fallback_to_audio_is_cached_and_preserves_embedded(monkeypatch):
    vst = VideoSpeechTransformer(
        vad="subs_then_webrtc",
        sample_rate=SAMPLE_RATE,
        frame_rate=48000,
        non_speech_label=0.0,
    )
    embedded_arr = np.array([1.0, 0.0, 1.0])
    audio_arr = np.array([0.0, 1.0, 1.0])
    vst.video_speech_results_ = embedded_arr
    vst.used_embedded_subs_ = True
    vst.reference_source_ = "embedded_text"

    calls = []

    def fake_run_audio(fname):
        calls.append(fname)
        vst.video_speech_results_ = audio_arr
        vst.alt_video_speech_results_ = [("center channel (FC)", audio_arr)]
        vst.used_embedded_subs_ = False
        vst.reference_source_ = "audio"

    monkeypatch.setattr(vst, "_run_audio_pipeline", fake_run_audio)

    sp1, alts1 = vst.fallback_to_audio("movie.mkv")
    sp2, alts2 = vst.fallback_to_audio("movie.mkv")
    assert len(calls) == 1, "audio fallback must be computed once and cached"
    assert np.array_equal(sp1, audio_arr)
    assert np.array_equal(sp2, audio_arr)
    assert len(alts1) == 1 and len(alts2) == 1
    # Original embedded subtitle signal stays intact for subsequent candidates
    assert vst.used_embedded_subs_ is True
    assert np.array_equal(vst.video_speech_results_, embedded_arr)


# ---------------------------------------------------------------------------
# Desktop API multi-subtitle & reference cache test
# ---------------------------------------------------------------------------


def _load_desktop_app_module():
    fake_webview = types.ModuleType("webview")
    fake_webview.Window = object
    fake_webview.screens = []

    class _FD:
        OPEN = 1
        FOLDER = 2
        SAVE = 3

    fake_webview.FileDialog = _FD
    fake_dom = types.ModuleType("webview.dom")
    fake_dom.DOMEventHandler = object
    fake_webview.dom = fake_dom

    prev_webview = sys.modules.get("webview")
    prev_dom = sys.modules.get("webview.dom")
    sys.modules["webview"] = fake_webview
    sys.modules["webview.dom"] = fake_dom
    try:
        mod_path = Path(__file__).resolve().parent.parent / "gui" / "desktop_app.py"
        spec = importlib.util.spec_from_file_location("_test_desktop_multi", str(mod_path))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod
    finally:
        if prev_webview is not None:
            sys.modules["webview"] = prev_webview
        else:
            sys.modules.pop("webview", None)
        if prev_dom is not None:
            sys.modules["webview.dom"] = prev_dom
        else:
            sys.modules.pop("webview.dom", None)


def test_desktop_api_multi_sync_caches_reference_and_saves_chosen(tmp_path, monkeypatch):
    mod = _load_desktop_app_module()
    monkeypatch.setattr(mod, "_config_path", lambda: tmp_path / "config.json")
    monkeypatch.setattr(mod, "get_portable_dir", lambda: tmp_path)
    monkeypatch.setattr(
        mod,
        "find_ffmpeg_binaries",
        lambda custom_path=None: {"available": False, "source": None},
    )

    ref_intervals = _irregular_intervals(n=40, seed=31)
    wrong_intervals = _irregular_intervals(n=40, seed=53)
    ref_path = _write_srt(tmp_path / "ref.srt", ref_intervals)
    good_path = _write_srt(tmp_path / "good.srt", ref_intervals, shift=2.5)
    wrong_path = _write_srt(tmp_path / "wrong.srt", wrong_intervals, shift=0.0)

    api = mod.DesktopApi()
    chosen_save = str(tmp_path / "saved_by_user.srt")

    class _FakeWin:
        def create_file_dialog(self, *a, **kw):
            return [chosen_save]

    api.attach_window(_FakeWin())

    # First run: analyzes both in memory (no files written) and caches reference pipe.
    res = api.start_sync(
        {
            "ref_path": ref_path,
            "in_paths": [wrong_path, good_path],
            "vad": "subs_then_webrtc",
            "frame_rate": 48000,
            "split_mode": "off",
            "no_fix_framerate": True,
            "gss": False,
        }
    )
    assert res["started"] is True
    deadline = time.time() + 15.0
    while time.time() < deadline:
        st = api.get_poll_state()["sync"]
        if st.get("done"):
            break
        time.sleep(0.03)
    st = api.get_poll_state()["sync"]
    assert st["done"] is True
    assert st["ok"] is True
    assert st["multi_mode"] is True
    assert st["ref_cached"] is False
    assert not (tmp_path / "good.synced.srt").exists()
    assert not (tmp_path / "wrong.synced.srt").exists()
    assert st["candidates"][0]["input_name"] == "good.srt"

    # Save chosen candidate via Windows Save As dialog mock.
    save_res = api.save_candidate(st["candidates"][0]["id"])
    assert save_res["saved"] is True
    assert save_res["path"] == chosen_save
    assert os.path.isfile(chosen_save)

    # Second run against the same reference: reuses cached reference!
    api.start_sync(
        {
            "ref_path": ref_path,
            "in_paths": [good_path, wrong_path],
            "vad": "subs_then_webrtc",
            "frame_rate": 48000,
            "split_mode": "off",
            "no_fix_framerate": True,
            "gss": False,
        }
    )
    deadline = time.time() + 15.0
    while time.time() < deadline:
        st2 = api.get_poll_state()["sync"]
        if st2.get("done") and st2.get("seq") != st.get("seq"):
            break
        time.sleep(0.03)
    st2 = api.get_poll_state()["sync"]
    assert st2["ok"] is True
    assert st2["ref_cached"] is True


# ---------------------------------------------------------------------------
# Web bridge multi-subtitle & reference cache test
# ---------------------------------------------------------------------------


def test_web_bridge_cached_multi_subtitle_ranking(tmp_path):
    bridge_dir = str(Path(__file__).resolve().parent.parent / "web" / "src")
    if bridge_dir not in sys.path:
        sys.path.insert(0, bridge_dir)
    import ffsubsync_bridge as bridge

    orig_work = bridge._WORK_DIR
    bridge._WORK_DIR = str(tmp_path / "web_work")
    try:
        ref_intervals = _irregular_intervals(n=40, seed=41)
        wrong_intervals = _irregular_intervals(n=40, seed=67)
        ref_bytes = Path(_write_srt(tmp_path / "ref.srt", ref_intervals)).read_bytes()
        good_bytes = Path(_write_srt(tmp_path / "good.srt", ref_intervals, shift=3.5)).read_bytes()
        wrong_bytes = Path(_write_srt(tmp_path / "wrong.srt", wrong_intervals, shift=0.0)).read_bytes()

        prep = bridge.prepare_reference_subtitles("cache-k1", "ref.srt", ref_bytes)
        assert prep["ok"] is True
        assert bridge.has_cached_reference("cache-k1") is True

        res = bridge.sync_with_cached_reference(
            "cache-k1",
            [
                {"name": "wrong.srt", "bytes": wrong_bytes},
                {"name": "good.srt", "bytes": good_bytes},
            ],
            no_fix_framerate=True,
        )
        assert res["ok"] is True
        assert res["multi_mode"] is True
        cands = res["candidates"]
        assert len(cands) == 2
        assert cands[0]["rank"] == 1
        assert cands[0]["input_name"] == "good.srt"
        assert cands[0]["output_name"] == "good.synced.srt"
        assert "-->" in cands[0]["output_text"]
        assert cands[0]["overall_score"] > cands[1]["overall_score"]
        assert abs(cands[0]["offset_seconds"] - (-3.5)) < 0.25
    finally:
        bridge._WORK_DIR = orig_work
