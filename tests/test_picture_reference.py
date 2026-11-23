# -*- coding: utf-8 -*-
from unittest.mock import patch

from ffsubsync.speech_transformers import (
    PGSSpeechTransformer,
    _get_dvd_timings_via_ffprobe,
    find_pgs_stream,
)


def _stream(
    index,
    codec,
    forced=0,
    comment=0,
    hearing_impaired=0,
    default=0,
    title=None,
    lang=None,
    frames=None,
):
    tags = {}
    if title is not None:
        tags["title"] = title
    if lang is not None:
        tags["language"] = lang
    if frames is not None:
        tags["NUMBER_OF_FRAMES-eng"] = str(frames)
    return {
        "index": index,
        "codec_type": "subtitle",
        "codec_name": codec,
        "disposition": {
            "forced": forced,
            "comment": comment,
            "hearing_impaired": hearing_impaired,
            "default": default,
        },
        "tags": tags,
    }


def _frame(pts_time, end_ms, num_rects=1, start_ms=0):
    return {
        "pts_time": "%.6f" % pts_time,
        "start_display_time": start_ms,
        "end_display_time": end_ms,
        "num_rects": num_rects,
    }


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_dvd_and_dvb(mock_probe, mock_bin):
    for codec in ("dvd_subtitle", "dvb_subtitle"):
        mock_probe.return_value = {"streams": [_stream(0, "subrip"), _stream(1, codec)]}
        assert find_pgs_stream("test.mkv") == "0:s:1"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_prefers_most_cues_by_number_of_frames(mock_probe, mock_bin):
    mock_probe.return_value = {
        "streams": [
            _stream(2, "hdmv_pgs_subtitle", frames=40),
            _stream(3, "hdmv_pgs_subtitle", frames=1500),
            _stream(4, "hdmv_pgs_subtitle", frames=1200),
        ]
    }
    assert find_pgs_stream("test.mkv") == "0:s:1"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_passes_over_commentary_even_with_more_frames(mock_probe, mock_bin):
    mock_probe.return_value = {
        "streams": [
            _stream(2, "hdmv_pgs_subtitle", title="Director's Commentary", frames=900),
            _stream(3, "hdmv_pgs_subtitle", comment=1, frames=900),
            _stream(4, "hdmv_pgs_subtitle", frames=500),
        ]
    }
    assert find_pgs_stream("test.mkv") == "0:s:2"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_dvd_timings_from_end_display_time(mock_probe, mock_bin):
    mock_probe.return_value = {"frames": [_frame(7.0, 2992), _frame(14.0, 1500)]}
    assert _get_dvd_timings_via_ffprobe("test.mkv", "0:s:0") == [
        (7.0, 9.992),
        (14.0, 15.5),
    ]
    _, kwargs = mock_probe.call_args
    assert kwargs["select_streams"] == "s:0"
    assert "show_frames" in kwargs


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_dvd_timings_honour_start_display_time(mock_probe, mock_bin):
    mock_probe.return_value = {"frames": [_frame(7.0, 3000, start_ms=500)]}
    assert _get_dvd_timings_via_ffprobe("test.mkv", "0:s:0") == [(7.5, 10.0)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_dvd_timings_end_at_next_caption(mock_probe, mock_bin):
    mock_probe.return_value = {"frames": [_frame(7.0, 5000), _frame(9.0, 1000)]}
    assert _get_dvd_timings_via_ffprobe("test.mkv", "0:s:0") == [
        (7.0, 9.0),
        (9.0, 10.0),
    ]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_dvd_timings_empty_frame_clears(mock_probe, mock_bin):
    mock_probe.return_value = {
        "frames": [_frame(7.0, 30000), _frame(10.0, 30000, num_rects=0)]
    }
    assert _get_dvd_timings_via_ffprobe("test.mkv", "0:s:0") == [(7.0, 10.0)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_prefers_audio_language_and_non_sdh_over_more_frames(mock_probe, mock_bin):
    # English audio + full English VobSub (1500) + Chinese VobSub (1861) + English SDH (1900):
    # Full clean English matching the audio language must beat both SDH and Chinese.
    mock_probe.return_value = {
        "streams": [
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "disposition": {"default": 1},
                "tags": {"language": "eng"},
            },
            _stream(2, "dvd_subtitle", lang="chi", frames=1861),
            _stream(3, "dvd_subtitle", lang="eng", hearing_impaired=1, title="English (SDH)", frames=1900),
            _stream(4, "dvd_subtitle", lang="eng", frames=1500),
        ]
    }
    assert find_pgs_stream("test.mkv") == "0:s:2"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_pgs_transformer_dispatches_dvd_to_show_frames(mock_probe, mock_bin):
    def _fake_probe(fname, **kwargs):
        if kwargs.get("show_entries") == "stream=codec_name":
            return {"streams": [{"codec_name": "dvd_subtitle"}]}
        if "show_frames" in kwargs:
            return {"frames": [_frame(2.0, 1500), _frame(5.0, 2000)]}
        raise AssertionError("unexpected probe call: %r" % (kwargs,))

    mock_probe.side_effect = _fake_probe
    t = PGSSpeechTransformer(sample_rate=100, ref_stream="0:s:0").fit("movie.mkv")
    assert t.num_timings_ == 2
    assert t.transform().sum() == 350.0
