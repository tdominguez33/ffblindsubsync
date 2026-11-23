# -*- coding: utf-8 -*-
"""Tests for embedded-subtitle extraction in ``VideoSpeechTransformer``.

These exercise the single-pass extraction path (one ffmpeg invocation for all
streams, enumerated via ffprobe) and its fallbacks, while guaranteeing the
extraction stays in memory -- nothing is written next to the source media.
``subprocess.Popen`` is stubbed to stand in for ffprobe/ffmpeg, but the real
subtitle-speech pipeline parses the SRT bytes, so selection logic is tested
end to end.
"""
import io
import os

import numpy as np
import pytest

import ffsubsync.speech_transformers as st
from ffsubsync.constants import DEFAULT_FRAME_RATE, SAMPLE_RATE
from ffsubsync.speech_transformers import VideoSpeechTransformer


def _srt(end_sec):
    """A valid SRT string with one short cue per second up to ``end_sec``."""
    lines = []
    for i in range(1, end_sec + 1):
        lines.append(str(i))
        lines.append(
            "00:00:{:02d},000 --> 00:00:{:02d},500".format(i - 1, i - 1)
        )
        lines.append("line {}".format(i))
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def _parse_speech_results(srt_bytes, start_seconds=0):
    """Run the real pipeline on SRT bytes to get the expected speech results."""
    pipe = st.make_subtitle_speech_pipeline(start_seconds=start_seconds).fit(
        io.BytesIO(srt_bytes)
    )
    return pipe.steps[-1][1].subtitle_speech_results_


def _transformer(**overrides):
    kwargs = dict(
        vad="subs_then_webrtc",
        sample_rate=SAMPLE_RATE,
        frame_rate=DEFAULT_FRAME_RATE,
        non_speech_label=0.0,
        # the fixtures here are a few cues long; the sparse-stream threshold
        # has its own dedicated tests below
        min_embedded_subtitle_cues=0,
    )
    kwargs.update(overrides)
    return VideoSpeechTransformer(**kwargs)


class _FakeProc:
    def __init__(self, returncode=0, stdout_bytes=b""):
        self.returncode = returncode
        self._stdout = stdout_bytes

    def communicate(self, *a, **k):
        return (self._stdout, b"")


def _install_fake_popen(
    monkeypatch,
    *,
    probe_lines,
    srt_by_stream,
    probe_unavailable=False,
    probe_rc=0,
    single_pass_rc=0,
):
    """Stub ``subprocess.Popen`` to emulate ffprobe + ffmpeg.

    ``probe_lines`` is the CSV ffprobe emits (``index,codec_name`` per line).
    ``srt_by_stream`` maps an ffmpeg ``-map`` specifier (e.g. ``"0:2"``) to the
    SRT bytes that stream yields. Records every invocation in the returned
    ``calls`` list.
    """
    calls = []
    real_popen = st.subprocess.Popen

    def fake_popen(args, **kwargs):
        binary = os.path.basename(str(args[0])).lower()
        if "ffmpeg" not in binary and "ffprobe" not in binary:
            # not one of our calls (e.g. platform.system() shelling out to
            # ``ver`` on Windows) -- let it run for real
            return real_popen(args, **kwargs)
        calls.append(list(args))
        if "ffprobe" in binary:
            if probe_unavailable:
                raise OSError("ffprobe not found")
            payload = "\n".join(probe_lines).encode("utf-8")
            return _FakeProc(returncode=probe_rc, stdout_bytes=payload)
        # ffmpeg
        if args[-1] == "-":
            # per-stream extraction to stdout
            stream = args[args.index("-map") + 1]
            data = srt_by_stream.get(stream, b"")
            # ffmpeg returns nonzero when the mapped stream does not exist
            return _FakeProc(returncode=0 if data else 1, stdout_bytes=data)
        # single-pass extraction: write each stream's SRT to its output path
        if single_pass_rc == 0:
            i = 0
            while i < len(args):
                if args[i] == "-map":
                    stream = args[i + 1]
                    out_path = args[i + 4]  # -map S -f srt PATH
                    data = srt_by_stream.get(stream, b"")
                    if data:
                        with open(out_path, "wb") as f:
                            f.write(data)
                i += 1
        return _FakeProc(returncode=single_pass_rc)

    monkeypatch.setattr(st.subprocess, "Popen", fake_popen)
    return calls


def _count_ffmpeg_invocations(calls):
    return sum(1 for c in calls if "ffmpeg" in os.path.basename(c[0]).lower())


def test_single_pass_extracts_all_streams_in_one_invocation(monkeypatch):
    srt_by_stream = {"0:2": _srt(10), "0:3": _srt(30), "0:4": _srt(20)}
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=["2,subrip", "3,ass", "4,mov_text"],
        srt_by_stream=srt_by_stream,
    )
    t = _transformer()
    t.try_fit_using_embedded_subs("movie.mkv")

    # exactly one ffmpeg pass (the perf win) plus the single ffprobe call
    assert _count_ffmpeg_invocations(calls) == 1
    assert sum(1 for c in calls if "ffprobe" in os.path.basename(c[0]).lower()) == 1
    # the longest stream (0:3, 30s) is the one whose results are kept
    expected = _parse_speech_results(srt_by_stream["0:3"])
    assert t.video_speech_results_ is not None
    assert np.array_equal(t.video_speech_results_, expected)


def test_bitmap_subtitle_streams_are_skipped(monkeypatch):
    # a PGS (bitmap) stream must not be mapped -- it cannot become SRT and would
    # otherwise abort the whole single-pass extraction
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=["0,hdmv_pgs_subtitle", "1,subrip"],
        srt_by_stream={"0:1": _srt(12)},
    )
    t = _transformer()
    t.try_fit_using_embedded_subs("movie.mkv")

    ffmpeg_call = next(
        c for c in calls if "ffmpeg" in os.path.basename(c[0]).lower()
    )
    mapped = [ffmpeg_call[i + 1] for i, a in enumerate(ffmpeg_call) if a == "-map"]
    assert mapped == ["0:1"]  # only the text stream
    assert t.video_speech_results_ is not None


def test_falls_back_to_per_stream_when_ffprobe_unavailable(monkeypatch):
    # without ffprobe we cannot enumerate, so fall back to probing 0:s:0.. one
    # at a time (original behavior), stopping at the first missing stream
    srt_by_stream = {"0:s:0": _srt(8), "0:s:1": _srt(25)}
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=[],
        srt_by_stream=srt_by_stream,
        probe_unavailable=True,
    )
    t = _transformer()
    t.try_fit_using_embedded_subs("movie.mkv")

    # per-stream means multiple ffmpeg invocations, all to stdout ("-")
    ffmpeg_calls = [c for c in calls if "ffmpeg" in os.path.basename(c[0]).lower()]
    assert len(ffmpeg_calls) >= 2
    assert all(c[-1] == "-" for c in ffmpeg_calls)
    expected = _parse_speech_results(srt_by_stream["0:s:1"])
    assert np.array_equal(t.video_speech_results_, expected)


def test_single_pass_failure_falls_back_to_per_stream(monkeypatch):
    # if the combined ffmpeg pass fails wholesale, degrade to per-stream over
    # the same enumerated streams rather than giving up
    srt_by_stream = {"0:2": _srt(15)}
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=["2,subrip"],
        srt_by_stream=srt_by_stream,
        single_pass_rc=1,
    )
    t = _transformer()
    t.try_fit_using_embedded_subs("movie.mkv")

    ffmpeg_calls = [c for c in calls if "ffmpeg" in os.path.basename(c[0]).lower()]
    # one failed single pass + at least one per-stream retry
    assert len(ffmpeg_calls) >= 2
    assert t.video_speech_results_ is not None


def test_explicit_ref_stream_uses_per_stream(monkeypatch):
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=["irrelevant"],
        srt_by_stream={"0:s:3": _srt(18)},
    )
    t = _transformer(ref_stream="0:s:3")
    t.try_fit_using_embedded_subs("movie.mkv")

    # ref_stream short-circuits enumeration: no ffprobe, single mapped stream
    assert all("ffprobe" not in os.path.basename(c[0]).lower() for c in calls)
    ffmpeg_calls = [c for c in calls if "ffmpeg" in os.path.basename(c[0]).lower()]
    assert len(ffmpeg_calls) == 1
    assert ffmpeg_calls[0][ffmpeg_calls[0].index("-map") + 1] == "0:s:3"


def test_raises_when_no_subtitle_streams(monkeypatch):
    _install_fake_popen(
        monkeypatch,
        probe_lines=[],  # ffprobe finds nothing -> no usable stream, no ffmpeg
        srt_by_stream={},
    )
    t = _transformer()
    with pytest.raises(ValueError, match="lack a usable text subtitle stream"):
        t.try_fit_using_embedded_subs("movie.mkv")


def test_raises_when_explicit_ref_stream_missing(monkeypatch):
    _install_fake_popen(
        monkeypatch,
        probe_lines=[],
        srt_by_stream={},
    )
    t = _transformer(ref_stream="0:s:9")
    with pytest.raises(ValueError, match="Stream 0:s:9 not found"):
        t.try_fit_using_embedded_subs("movie.mkv")


def test_extraction_writes_nothing_next_to_video(monkeypatch, tmp_path):
    # the key regression guard: extraction must not create a Subs/ folder or any
    # files alongside the source media
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"not really a video")
    _install_fake_popen(
        monkeypatch,
        probe_lines=["2,subrip", "3,ass"],
        srt_by_stream={"0:2": _srt(10), "0:3": _srt(20)},
    )
    t = _transformer()
    t.try_fit_using_embedded_subs(str(video))

    leftovers = sorted(p.name for p in tmp_path.iterdir())
    assert leftovers == ["movie.mkv"]  # nothing else created
    assert not (tmp_path / "Subs").exists()


def test_empty_extracted_stream_is_skipped(monkeypatch):
    # a stream that yields no data (empty temp file) must be skipped, not crash
    srt_by_stream = {"0:2": b"", "0:3": _srt(14)}
    _install_fake_popen(
        monkeypatch,
        probe_lines=["2,subrip", "3,subrip"],
        srt_by_stream=srt_by_stream,
    )
    t = _transformer()
    t.try_fit_using_embedded_subs("movie.mkv")

    expected = _parse_speech_results(srt_by_stream["0:3"])
    assert np.array_equal(t.video_speech_results_, expected)


def test_build_ffmpeg_args_keeps_aresample_filter():
    # regression guard: the audio-extraction path must keep the async resample
    # filter (deliberately added upstream to fix audio timestamp drift)
    t = _transformer(vad="webrtc")
    args = t._build_ffmpeg_args("movie.mkv")
    assert "-af" in args
    assert args[args.index("-af") + 1] == "aresample=async=1"


def _srt_at(timestamps):
    lines = []
    for idx, ts in enumerate(timestamps, 1):
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = ts % 60
        lines.append(str(idx))
        lines.append("{:02d}:{:02d}:{:06.3f}".format(h, m, s).replace(".", ",") + " --> " +
                     "{:02d}:{:02d}:{:06.3f}".format(h, m, s + 0.8).replace(".", ","))
        lines.append("line {}".format(idx))
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def test_forced_and_commentary_streams_are_deprioritized(monkeypatch):
    # Tagged forced and commentary tracks should not even be extracted when a normal
    # track exists; and untagged sparse forced tracks with late timestamps must lose
    # to full dialogue tracks.
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=[
            "2,subrip,1,0,Forced",
            "3,subrip,0,1,Director Commentary",
            "4,subrip,0,0,English",
        ],
        srt_by_stream={"0:4": _srt(15)},
    )
    t = _transformer()
    t.try_fit_using_embedded_subs("movie.mkv")
    ffmpeg_call = next(c for c in calls if "ffmpeg" in os.path.basename(c[0]).lower())
    mapped = [ffmpeg_call[i + 1] for i, a in enumerate(ffmpeg_call) if a == "-map"]
    assert mapped == ["0:4"]

    # Untagged tracks: 0:2 has only 2 cues ending at 3600s, 0:3 has 40 cues ending at 120s
    sparse_forced = _srt_at([10.0, 3600.0])
    full_dialogue = _srt_at([2.0 + i * 3.0 for i in range(40)])
    _install_fake_popen(
        monkeypatch,
        probe_lines=["2,subrip", "3,subrip"],
        srt_by_stream={"0:2": sparse_forced, "0:3": full_dialogue},
    )
    t2 = _transformer()
    t2.try_fit_using_embedded_subs("movie.mkv")
    assert np.array_equal(t2.video_speech_results_, _parse_speech_results(full_dialogue))


def test_find_pgs_stream_skips_forced_when_full_exists(monkeypatch):
    monkeypatch.setattr(st, "ffmpeg_bin_path", lambda *a, **k: "ffprobe")
    monkeypatch.setattr(
        st.ffmpeg,
        "probe",
        lambda *a, **k: {
            "streams": [
                {
                    "index": 2,
                    "codec_type": "subtitle",
                    "codec_name": "hdmv_pgs_subtitle",
                    "disposition": {"forced": 1, "comment": 0},
                    "tags": {"title": "Forced"},
                },
                {
                    "index": 3,
                    "codec_type": "subtitle",
                    "codec_name": "hdmv_pgs_subtitle",
                    "disposition": {"forced": 0, "comment": 0},
                    "tags": {"title": "English"},
                },
            ]
        },
    )
    assert st.find_pgs_stream("movie.mkv") == "0:s:1"


def test_find_vobsub_stream_skips_commentary_and_forced(monkeypatch):
    monkeypatch.setattr(st, "ffmpeg_bin_path", lambda *a, **k: "ffprobe")
    monkeypatch.setattr(
        st.ffmpeg,
        "probe",
        lambda *a, **k: {
            "streams": [
                {
                    "index": 2,
                    "codec_type": "subtitle",
                    "codec_name": "dvd_subtitle",
                    "disposition": {"forced": 0, "comment": 1},
                    "tags": {"title": "Commentary"},
                },
                {
                    "index": 3,
                    "codec_type": "subtitle",
                    "codec_name": "dvd_subtitle",
                    "disposition": {"forced": 1, "comment": 0},
                    "tags": {"title": "Forced"},
                },
                {
                    "index": 4,
                    "codec_type": "subtitle",
                    "codec_name": "dvd_subtitle",
                    "disposition": {"forced": 0, "comment": 0},
                    "tags": {"title": "English"},
                },
            ]
        },
    )
    assert st.find_pgs_stream("movie.mkv") == "0:s:2"
    assert st._find_bitmap_timing_streams("movie.mkv", include_deprioritized=False) == [
        ("0:s:2", 4, "dvd_subtitle")
    ]


def test_find_bitmap_timing_streams_excludes_only_commentary_when_requested(monkeypatch):
    monkeypatch.setattr(st, "ffmpeg_bin_path", lambda *a, **k: "ffprobe")
    monkeypatch.setattr(
        st.ffmpeg,
        "probe",
        lambda *a, **k: {
            "streams": [
                {
                    "index": 2,
                    "codec_type": "subtitle",
                    "codec_name": "dvd_subtitle",
                    "disposition": {"forced": 0, "comment": 1},
                    "tags": {"title": "Director Commentary"},
                },
            ]
        },
    )
    assert st._find_bitmap_timing_streams("movie.mkv", include_deprioritized=False) == []
    assert st.find_pgs_stream("movie.mkv") == "0:s:0"


def test_sparse_streams_fall_back_to_audio(monkeypatch):
    _install_fake_popen(
        monkeypatch,
        probe_lines=["2,subrip,0,0,English"],
        srt_by_stream={"0:2": _srt(40)},
    )
    t = _transformer(min_embedded_subtitle_cues=st.DEFAULT_MIN_EMBEDDED_SUBTITLE_CUES)
    with pytest.raises(ValueError, match="only 40 cues"):
        t.try_fit_using_embedded_subs("movie.mkv")
    assert t.video_speech_results_ is None


def test_explicit_ref_stream_bypasses_sparse_threshold(monkeypatch):
    _install_fake_popen(
        monkeypatch,
        probe_lines=[],
        srt_by_stream={"0:s:1": _srt(5)},
    )
    t = _transformer(
        ref_stream="0:s:1",
        min_embedded_subtitle_cues=st.DEFAULT_MIN_EMBEDDED_SUBTITLE_CUES,
    )
    t.try_fit_using_embedded_subs("movie.mkv")
    assert np.array_equal(t.video_speech_results_, _parse_speech_results(_srt(5)))


def test_only_forced_streams_means_no_usable_stream(monkeypatch):
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=["2,subrip,1,0,Forced", "3,hdmv_pgs_subtitle,0,0,English"],
        srt_by_stream={"0:2": _srt(5), "0:s:0": _srt(5)},
    )
    t = _transformer()
    with pytest.raises(ValueError, match="lack a usable text subtitle stream"):
        t.try_fit_using_embedded_subs("movie.mkv")
    assert _count_ffmpeg_invocations(calls) == 0


def test_try_sync_falls_back_to_audio_when_embedded_alignment_unreliable(tmp_path):
    from ffsubsync import ffsubsync

    # Build 60-cue input subtitle with irregular spacing so only the true offset aligns.
    timestamps = []
    pos = 5.0
    for i in range(60):
        timestamps.append(pos)
        pos += 2.0 + ((i * 7) % 5) * 0.7
    srtin = tmp_path / "in.srt"
    srtin.write_bytes(_srt_at(timestamps))
    srtout = tmp_path / "out.srt"

    total_len = int((pos + 40.0) * SAMPLE_RATE)
    # Embedded reference is uncorrelated/anti-correlated noise -> negative score & low SNR.
    rng = np.random.default_rng(123)
    bad_embedded = (rng.random(total_len) < 0.03).astype(float)

    # Audio fallback reference matches the input shifted by +8.0s.
    true_offset_s = 8.0
    good_audio = np.zeros(total_len, dtype=float)
    for ts in timestamps:
        a = int(round((ts + true_offset_s) * SAMPLE_RATE))
        b = int(round((ts + true_offset_s + 0.8) * SAMPLE_RATE))
        good_audio[a:b] = 1.0

    fallback_called = []

    class _FakeEmbeddedStep:
        can_fallback_to_audio = True
        alt_video_speech_results_ = []

        def transform(self, _x):
            return bad_embedded

        def fallback_to_audio(self, fname):
            fallback_called.append(fname)
            return good_audio, []

    step = _FakeEmbeddedStep()

    class _FakePipe:
        def __getitem__(self, idx):
            return [step][idx]

        def transform(self, x):
            return step.transform(x)

    args = ffsubsync.make_parser().parse_args(
        ["movie.mkv", "-i", str(srtin), "-o", str(srtout), "--no-fix-framerate"]
    )
    result = {"retval": 0}
    ok = ffsubsync.try_sync(args, _FakePipe(), result)
    assert fallback_called == ["movie.mkv"]
    assert ok is True
    assert result["sync_was_successful"] is True
    assert result["offset_seconds"] == pytest.approx(true_offset_s, abs=0.05)


def test_embedded_text_prefers_clean_audio_lang_over_sdh_with_more_cues(monkeypatch):
    import json

    # 0:6 = clean English (220 cues), 0:8 = English SDH (260 cues), 0:10 = French (240 cues).
    # Clean English matching primary audio language must beat SDH even though SDH has more cues,
    # using a single ffprobe JSON pass.
    clean_eng = _srt_at([2.0 + i * 2.0 for i in range(220)])
    sdh_eng = _srt_at([1.0 + i * 2.0 for i in range(260)])
    french = _srt_at([1.5 + i * 2.0 for i in range(240)])
    probe_json = json.dumps(
        {
            "streams": [
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "eac3",
                    "disposition": {"default": 1},
                    "tags": {"language": "eng"},
                },
                {
                    "index": 6,
                    "codec_type": "subtitle",
                    "codec_name": "subrip",
                    "disposition": {"hearing_impaired": 0},
                    "tags": {"language": "eng", "title": "English"},
                },
                {
                    "index": 8,
                    "codec_type": "subtitle",
                    "codec_name": "subrip",
                    "disposition": {"hearing_impaired": 1},
                    "tags": {"language": "eng", "title": "English (SDH)"},
                },
                {
                    "index": 10,
                    "codec_type": "subtitle",
                    "codec_name": "subrip",
                    "disposition": {"hearing_impaired": 0},
                    "tags": {"language": "fre", "title": "French"},
                },
            ]
        }
    )
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=[probe_json],
        srt_by_stream={"0:6": clean_eng, "0:8": sdh_eng, "0:10": french},
    )
    t = _transformer(min_embedded_subtitle_cues=50)
    t.try_fit_using_embedded_subs("dogma.mkv")
    assert sum(1 for c in calls if "ffprobe" in os.path.basename(c[0]).lower()) == 1
    assert np.array_equal(t.video_speech_results_, _parse_speech_results(clean_eng))


def test_audio_decode_error_includes_ffmpeg_stderr_and_rc(monkeypatch):
    class _ErrProc:
        def __init__(self):
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"No such filter: 'aresample'\n")

        def wait(self):
            return 1

    monkeypatch.setattr(st.ffmpeg, "probe", lambda *a, **k: {"format": {"duration": "10"}})
    monkeypatch.setattr(st.subprocess, "Popen", lambda *a, **k: _ErrProc())

    t = _transformer(vad="webrtc")
    with pytest.raises(ValueError) as excinfo:
        t._fit_using_audio("movie.mkv")
    msg = str(excinfo.value)
    assert "ffmpeg exited with code 1" in msg
    assert "No such filter" in msg
    assert "too old" in msg
