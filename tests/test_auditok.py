# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ffsubsync.constants import is_valid_vad
from ffsubsync.speech_transformers import (
    VideoSpeechTransformer,
    _make_auditok_detector,
    _parse_auditok_spec,
)


@pytest.mark.parametrize(
    "vad, expected",
    [
        ("subs_then_webrtc", True),
        ("webrtc", True),
        ("auditok", True),
        ("subs_then_auditok", True),
        ("auditok:auto", True),
        ("auditok:otsu", True),
        ("auditok:percentile", True),
        ("auditok:p20", True),
        ("auditok:p99", True),
        ("auditok:50", True),
        ("auditok:55.5", True),
        ("auditok:webrtc", True),
        ("auditok:webrtc:0", True),
        ("auditok:webrtc:3", True),
        ("subs_then_auditok:webrtc:2", True),
        ("subs_then_auditok:p75", True),
        ("auditok:p0", False),
        ("auditok:p100", False),
        ("auditok:webrtc:4", False),
        ("auditok:bogus", False),
        ("unknown", False),
    ],
)
def test_is_valid_vad(vad, expected):
    assert is_valid_vad(vad) is expected


def test_parse_auditok_spec():
    assert _parse_auditok_spec("auditok") == "auto"
    assert _parse_auditok_spec("subs_then_auditok") == "auto"
    assert _parse_auditok_spec("auditok:50") == "50"
    assert _parse_auditok_spec("subs_then_auditok:webrtc:2") == "webrtc:2"
    assert _parse_auditok_spec("auditok:p30") == "p30"


def _make_speech_and_silence_pcm(frame_rate=16000):
    """3s audio: 1s quiet noise floor + 1s loud 440Hz tone + 1s quiet noise floor."""
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 1.0, frame_rate, endpoint=False)
    quiet = rng.integers(-50, 51, size=frame_rate, dtype=np.int16)
    tone = (np.sin(2.0 * np.pi * 440.0 * t) * 12000.0).astype(np.int16)
    return np.concatenate([quiet, tone, quiet]).tobytes()


@pytest.mark.parametrize("spec", ["auto", "otsu", "p50", "50"])
def test_auditok_energy_modes_detect_active_region(spec):
    pcm = _make_speech_and_silence_pcm(frame_rate=16000)
    detect = _make_auditok_detector(100, 16000, 0.0, vad_spec=spec)
    res = detect(pcm)
    assert len(res) == 300
    assert res[:80].sum() == 0.0
    assert res[100:200].sum() >= 70.0
    # auditok keeps up to max_silence (0.25s = 25 frames) at the end of an event
    assert res[230:].sum() == 0.0


def test_auditok_near_silent_guard_returns_non_speech():
    # Very low dither noise (~amplitude 3 -> ~9.5 dB < 25 dB guard)
    rng = np.random.default_rng(0)
    dither = rng.integers(-3, 4, size=16000 * 2, dtype=np.int16).tobytes()
    detect = _make_auditok_detector(100, 16000, 0.0, vad_spec="auto")
    res = detect(dither)
    assert len(res) == 200
    assert np.all(res == 0.0)


def test_auditok_webrtc_hybrid_runs_and_dispatches_before_plain_webrtc():
    pcm = _make_speech_and_silence_pcm(frame_rate=16000)
    detect = _make_auditok_detector(100, 16000, 0.0, vad_spec="webrtc:2")
    res = detect(pcm)
    assert len(res) == 300

    # Ensure VideoSpeechTransformer._make_detector routes "auditok:webrtc:2" to auditok
    t = VideoSpeechTransformer(
        vad="auditok:webrtc:2", sample_rate=100, frame_rate=16000, non_speech_label=0.0
    )
    det = t._make_detector()
    out = det(np.frombuffer(pcm, np.uint8))
    assert len(out) == 300
