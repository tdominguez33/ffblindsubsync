# -*- coding: utf-8 -*-
"""Tests for surround dual-channel (mix + FC) extraction, peak SNR, and density-mismatch sync."""
from datetime import timedelta
import types

import numpy as np
import pytest
import srt

from ffsubsync import ffsubsync
from ffsubsync.aligners import FFTAligner, MaxScoreAligner
from ffsubsync.constants import DEFAULT_FRAME_RATE, SAMPLE_RATE
from ffsubsync.generic_subtitles import GenericSubtitle
import ffsubsync.speech_transformers as st
from ffsubsync.speech_transformers import (
    VideoSpeechTransformer,
    _layout_has_center_channel,
    _probe_audio_stream_has_center,
)
from ffsubsync.split_aligner import alignment_has_positive_signal


# ---- layout / stream probing -----------------------------------------------


@pytest.mark.parametrize(
    "layout, expected",
    [
        ("5.1", True),
        ("5.1(side)", True),
        ("5.1.2", True),
        ("7.1", True),
        ("7.1(wide)", True),
        ("7.1.4", True),
        ("3.0", True),
        ("3.1", True),
        ("5.0", True),
        ("hexagonal", True),
        ("octagonal", True),
        ("stereo", False),
        ("mono", False),
        ("2.1", False),
        ("2.2", False),
        ("quad", False),
        ("quad(side)", False),
        ("2.1.2", False),
        ("unknown", False),
        ("", False),
    ],
)
def test_layout_has_center_channel(layout, expected):
    assert _layout_has_center_channel(layout) is expected


def test_probe_audio_stream_has_center_selects_requested_stream():
    probe = {
        "streams": [
            {"index": 0, "codec_type": "video"},
            {
                "index": 1,
                "codec_type": "audio",
                "channels": 8,
                "channel_layout": "5.1.2",
                "disposition": {"default": 1},
            },
            {
                "index": 2,
                "codec_type": "audio",
                "channels": 2,
                "channel_layout": "stereo",
                "disposition": {"default": 0},
            },
        ]
    }
    assert _probe_audio_stream_has_center(probe, None) is True
    assert _probe_audio_stream_has_center(probe, "0:a:0") is True
    assert _probe_audio_stream_has_center(probe, "a:0") is True
    assert _probe_audio_stream_has_center(probe, "0:a:1") is False
    assert _probe_audio_stream_has_center(probe, "a:1") is False
    assert _probe_audio_stream_has_center(None, None) is False
    assert _probe_audio_stream_has_center({"format": {"duration": "10"}}, None) is False


# ---- dual-channel FFmpeg extraction & fallback -----------------------------


def test_build_ffmpeg_args_dual_center_uses_two_channel_filtergraph():
    t = VideoSpeechTransformer(
        vad="webrtc",
        sample_rate=SAMPLE_RATE,
        frame_rate=DEFAULT_FRAME_RATE,
        non_speech_label=0.0,
    )
    args = t._build_ffmpeg_args("movie.mkv", dual_center=True)
    assert args[args.index("-ac") + 1] == "2"
    af = args[args.index("-af") + 1]
    assert "asplit=2" in af
    assert "pan=mono|c0=FC" in af
    assert "amerge=inputs=2" in af


class _ChunkedStdout:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, _n):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _FakeProc:
    def __init__(self, chunks, rc=0):
        self.stdout = _ChunkedStdout(chunks)
        self._rc = rc

    def wait(self):
        return self._rc


def test_dual_center_deinterleaves_channels_across_unaligned_reads(monkeypatch):
    # 4 windows of 10ms at 48kHz = 4 * 480 = 1920 stereo sample pairs.
    # Put all zeros in ch0 (mix) and all 10000 in ch1 (FC), split across odd byte chunks.
    win_samples = DEFAULT_FRAME_RATE // SAMPLE_RATE  # 480
    n_windows = 4
    total_samples = win_samples * n_windows
    stereo = np.zeros((total_samples, 2), dtype=np.int16)
    stereo[:, 0] = 0
    stereo[:, 1] = 10000
    raw = stereo.tobytes()
    # Intentionally split at non-window-aligned byte offsets
    chunks = [raw[:500], raw[500:3100], raw[3100:]]

    probe = {
        "format": {"duration": "0.04"},
        "streams": [
            {
                "index": 1,
                "codec_type": "audio",
                "channels": 8,
                "channel_layout": "5.1.2",
                "disposition": {"default": 1},
            }
        ],
    }
    monkeypatch.setattr(st.ffmpeg, "probe", lambda *a, **k: probe)
    monkeypatch.setattr(st.subprocess, "Popen", lambda *a, **k: _FakeProc(chunks, 0))

    # Fake detector: returns 1.0 per window if mean abs int16 > 500 else 0.0
    def _fake_make_detector(sample_rate, frame_rate, non_speech_label):
        ws = frame_rate // sample_rate

        def _detect(u8):
            s16 = np.frombuffer(u8, dtype=np.int16)
            nw = len(s16) // ws
            return np.array(
                [
                    1.0 if np.mean(np.abs(s16[i * ws : (i + 1) * ws])) > 500 else non_speech_label
                    for i in range(nw)
                ],
                dtype=float,
            )

        return _detect

    monkeypatch.setattr(st, "_make_webrtcvad_detector", _fake_make_detector)

    t = VideoSpeechTransformer(
        vad="webrtc",
        sample_rate=SAMPLE_RATE,
        frame_rate=DEFAULT_FRAME_RATE,
        non_speech_label=0.0,
    ).fit("movie.mkv")

    assert len(t.video_speech_results_) == n_windows
    assert np.all(t.video_speech_results_ == 0.0)
    assert len(t.alt_video_speech_results_) == 1
    label, fc_arr = t.alt_video_speech_results_[0]
    assert "FC" in label
    assert len(fc_arr) == n_windows
    assert np.all(fc_arr == 1.0)


def test_dual_center_falls_back_to_mono_if_filtergraph_fails(monkeypatch):
    probe = {
        "format": {"duration": "1.0"},
        "streams": [
            {
                "index": 1,
                "codec_type": "audio",
                "channels": 6,
                "channel_layout": "5.1",
                "disposition": {"default": 1},
            }
        ],
    }
    monkeypatch.setattr(st.ffmpeg, "probe", lambda *a, **k: probe)
    monkeypatch.setattr(
        st, "_make_webrtcvad_detector", lambda *a, **k: (lambda _b: np.ones(10))
    )

    calls = []

    def fake_popen(args, **kwargs):
        calls.append(list(args))
        if len(calls) == 1:
            # First (dual_center) call fails with empty stdout
            return _FakeProc([], rc=1)
        # Fallback mono call succeeds
        return _FakeProc([b"\x00" * (DEFAULT_FRAME_RATE * 2)], rc=0)

    monkeypatch.setattr(st.subprocess, "Popen", fake_popen)

    t = VideoSpeechTransformer(
        vad="webrtc",
        sample_rate=SAMPLE_RATE,
        frame_rate=DEFAULT_FRAME_RATE,
        non_speech_label=0.0,
    ).fit("movie.mkv")

    assert len(calls) == 2
    assert calls[0][calls[0].index("-ac") + 1] == "2"
    assert calls[1][calls[1].index("-ac") + 1] == "1"
    assert len(t.video_speech_results_) == 10
    assert t.alt_video_speech_results_ == []


# ---- peak SNR & symmetric signal check under density mismatch --------------


def _make_sparse_vad_and_dense_subs(seed=42, true_offset=1200):
    """Build dense subtitles (~68% active) and sparse VAD (~11% active) shifted by true_offset."""
    rng = np.random.default_rng(seed)
    n = 60000  # 600 seconds at 100 Hz
    sub = np.zeros(n, dtype=float)
    pos = 50
    cues = []
    while pos < n - 300:
        dur = int(rng.integers(150, 350))
        gap = int(rng.integers(30, 150))
        sub[pos : pos + dur] = 1.0
        inner = srt.Subtitle(
            index=len(cues) + 1,
            start=timedelta(seconds=pos / SAMPLE_RATE),
            end=timedelta(seconds=(pos + dur) / SAMPLE_RATE),
            content="dialogue line",
        )
        cues.append(GenericSubtitle(inner.start, inner.end, inner))
        pos += dur + gap

    ref = np.zeros(n + true_offset + 500, dtype=float)
    # Only ~16% of subtitle frames fire in sparse VAD (overall ~11% speech),
    # plus a tiny bit of background noise outside subtitles.
    for c in cues:
        a = int(round(c.start.total_seconds() * SAMPLE_RATE)) + true_offset
        b = int(round(c.end.total_seconds() * SAMPLE_RATE)) + true_offset
        burst_len = max(10, int((b - a) * 0.16))
        burst_start = a + int(rng.integers(0, max(1, (b - a) - burst_len)))
        ref[burst_start : burst_start + burst_len] = 1.0
    return ref, sub, cues


def test_peak_snr_is_strong_even_when_raw_score_is_negative():
    ref, sub, cues = _make_sparse_vad_and_dense_subs(true_offset=1200)
    aligner = FFTAligner(max_offset_samples=5000).fit(ref, sub, get_score=True)
    score, offset = aligner.transform()
    assert abs(offset - 1200) <= 5
    # Density mismatch (~11% vs ~68%) forces raw dot-product score negative
    assert score < 0
    # Yet the cross-correlation peak stands far above background noise
    assert aligner.best_peak_snr_ is not None
    assert aligner.best_peak_snr_ > 5.0
    # And symmetric alignment_has_positive_signal also recognizes it
    assert alignment_has_positive_signal(
        ref,
        cues,
        [float(offset)] * len(cues),
        sample_rate=SAMPLE_RATE,
        start_seconds=0.0,
    )


def test_peak_snr_is_low_on_uncorrelated_noise():
    rng = np.random.default_rng(7)
    ref = (rng.random(40000) < 0.3).astype(float)
    sub = (rng.random(35000) < 0.5).astype(float)
    aligner = FFTAligner(max_offset_samples=3000).fit(ref, sub, get_score=True)
    assert aligner.best_peak_snr_ is not None
    assert aligner.best_peak_snr_ < 5.0


# ---- try_sync candidate comparison (downmix vs FC) -------------------------


def test_try_sync_prefers_higher_snr_center_channel(tmp_path):
    true_offset_s = 15.0
    true_offset = int(true_offset_s * SAMPLE_RATE)
    _, _, cues = _make_sparse_vad_and_dense_subs(seed=11, true_offset=true_offset)
    srt_text = srt.compose([c.inner for c in cues])
    srtin = tmp_path / "in.srt"
    srtin.write_text(srt_text, encoding="utf-8")
    srtout = tmp_path / "out.srt"

    # Primary (downmix) is mostly noise; center channel (FC) matches the true offset cleanly.
    rng = np.random.default_rng(99)
    total_len = int(cues[-1].end.total_seconds() * SAMPLE_RATE) + true_offset + 500
    noisy_mix = (rng.random(total_len) < 0.1).astype(float)
    clean_fc = np.zeros(total_len, dtype=float)
    for c in cues:
        a = int(round(c.start.total_seconds() * SAMPLE_RATE)) + true_offset
        b = int(round(c.end.total_seconds() * SAMPLE_RATE)) + true_offset
        clean_fc[a:b] = 1.0

    fake_step = types.SimpleNamespace(
        transform=lambda _x: noisy_mix,
        alt_video_speech_results_=[("center channel (FC)", clean_fc)],
    )
    fake_ref = [fake_step]

    class _FakePipe:
        def __getitem__(self, idx):
            return fake_ref[idx]

        def transform(self, x):
            return fake_step.transform(x)

    args = ffsubsync.make_parser().parse_args(
        ["ref.mkv", "-i", str(srtin), "-o", str(srtout), "--no-fix-framerate"]
    )
    result = {"retval": 0}
    ok = ffsubsync.try_sync(args, _FakePipe(), result)
    assert ok is True
    assert result["sync_was_successful"] is True
    assert result["offset_seconds"] == pytest.approx(true_offset_s, abs=0.05)


def test_try_sync_accepts_negative_raw_score_with_high_peak_snr(tmp_path):
    true_offset_s = 12.0
    true_offset = int(true_offset_s * SAMPLE_RATE)
    ref, _, cues = _make_sparse_vad_and_dense_subs(seed=42, true_offset=true_offset)
    srtin = tmp_path / "in.srt"
    srtin.write_text(srt.compose([c.inner for c in cues]), encoding="utf-8")
    srtout = tmp_path / "out.srt"

    fake_step = types.SimpleNamespace(
        transform=lambda _x: ref,
        alt_video_speech_results_=[],
    )

    class _FakePipe:
        def __getitem__(self, idx):
            return [fake_step][idx]

        def transform(self, x):
            return fake_step.transform(x)

    args = ffsubsync.make_parser().parse_args(
        ["ref.mkv", "-i", str(srtin), "-o", str(srtout), "--no-fix-framerate"]
    )
    result = {"retval": 0}
    ok = ffsubsync.try_sync(args, _FakePipe(), result)
    assert ok is True
    assert result["sync_was_successful"] is True
    assert result["offset_seconds"] == pytest.approx(true_offset_s, abs=0.1)


def test_deserialize_speech_transformer_loads_alt_fc_channel(tmp_path):
    true_offset_s = 14.0
    true_offset = int(true_offset_s * SAMPLE_RATE)
    _, _, cues = _make_sparse_vad_and_dense_subs(seed=17, true_offset=true_offset)
    srtin = tmp_path / "in.srt"
    srtin.write_text(srt.compose([c.inner for c in cues]), encoding="utf-8")
    srtout = tmp_path / "out.srt"

    rng = np.random.default_rng(123)
    total_len = int(cues[-1].end.total_seconds() * SAMPLE_RATE) + true_offset + 500
    noisy_mix = (rng.random(total_len) < 0.1).astype(float)
    clean_fc = np.zeros(total_len, dtype=float)
    for c in cues:
        a = int(round(c.start.total_seconds() * SAMPLE_RATE)) + true_offset
        b = int(round(c.end.total_seconds() * SAMPLE_RATE)) + true_offset
        clean_fc[a:b] = 1.0

    npz_path = tmp_path / "ref.npz"
    np.savez_compressed(npz_path, speech=noisy_mix, alt_fc_speech=clean_fc)

    args = ffsubsync.make_parser().parse_args(
        [str(npz_path), "-i", str(srtin), "-o", str(srtout), "--no-fix-framerate"]
    )
    res = ffsubsync.run(args)
    assert res["sync_was_successful"] is True
    assert res["offset_seconds"] == pytest.approx(true_offset_s, abs=0.05)
