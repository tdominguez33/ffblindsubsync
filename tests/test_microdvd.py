# -*- coding: utf-8 -*-
"""Tests for robust MicroDVD (.sub) parsing, FPS inference, writing, and fail-fast output validation."""
import pysubs2
import pytest

from ffsubsync import ffsubsync
from ffsubsync.subtitle_parser import GenericSubtitleParser, _detect_microdvd_fps


def _microdvd_lines(fps: float, offset_s: float = 0.0, include_header: bool = True, ad_prefix: bool = False) -> str:
    lines = []
    if ad_prefix:
        lines.append("{1}{24}Advertise your product or brand here")
    if include_header:
        lines.append("{1}{1}%.3f" % fps)
    for i in range(25):
        s_sec = i * 3.0 + 1.0 + offset_s
        e_sec = s_sec + 1.5
        lines.append("{%d}{%d}Dialogue line %d" % (int(round(s_sec * fps)), int(round(e_sec * fps)), i + 1))
    return "\n".join(lines) + "\n"


def test_detect_microdvd_fps_finds_header_after_ad():
    txt = "{1}{24}OpenSubtitles ad\n{1}{1}25.000\n{50}{100}Hello\n"
    assert _detect_microdvd_fps(txt) == pytest.approx(25.0)
    assert _detect_microdvd_fps("{25}{100}Hello\n") is None


def test_microdvd_parser_handles_ad_before_fps_header(tmp_path):
    sub_path = tmp_path / "ad_first.sub"
    sub_path.write_text(_microdvd_lines(25.0, include_header=True, ad_prefix=True), encoding="utf-8")
    parser = GenericSubtitleParser(fmt="sub", encoding="utf-8").fit(str(sub_path))
    assert parser.subs_._fps == pytest.approx(25.0)
    assert len(parser.subs_) >= 25


def test_microdvd_parser_requires_sub_fps_with_clear_message(tmp_path):
    sub_path = tmp_path / "no_header.sub"
    sub_path.write_text(_microdvd_lines(23.976, include_header=False), encoding="utf-8")
    with pytest.raises(pysubs2.exceptions.UnknownFPSError) as excinfo:
        GenericSubtitleParser(fmt="sub", encoding="utf-8").fit(str(sub_path))
    assert "--sub-fps" in str(excinfo.value)


def test_microdvd_parser_uses_explicit_or_fallback_fps(tmp_path):
    sub_path = tmp_path / "no_header.sub"
    sub_path.write_text(_microdvd_lines(25.0, include_header=False), encoding="utf-8")

    p_explicit = GenericSubtitleParser(fmt="sub", encoding="utf-8", fps=25.0).fit(str(sub_path))
    assert p_explicit.subs_._fps == pytest.approx(25.0)
    assert p_explicit.subs_[0].start.total_seconds() == pytest.approx(1.0, abs=0.05)

    p_fallback = GenericSubtitleParser(fmt="sub", encoding="utf-8", fallback_fps=25.0).fit(str(sub_path))
    assert p_fallback.subs_._fps == pytest.approx(25.0)
    assert p_fallback.subs_[0].start.total_seconds() == pytest.approx(1.0, abs=0.05)


def test_microdvd_sub_sync_and_write_roundtrip(tmp_path):
    ref_path = tmp_path / "ref.sub"
    in_path = tmp_path / "in.sub"
    out_path = tmp_path / "out.sub"
    ref_path.write_text(_microdvd_lines(25.0, offset_s=0.0, include_header=True), encoding="utf-8")
    in_path.write_text(_microdvd_lines(25.0, offset_s=4.0, include_header=False), encoding="utf-8")

    args = ffsubsync.make_parser().parse_args(
        [str(ref_path), "-i", str(in_path), "-o", str(out_path), "--sub-fps", "25", "--no-fix-framerate"]
    )
    res = ffsubsync.run(args)
    assert res["sync_was_successful"] is True
    assert res["offset_seconds"] == pytest.approx(-4.0, abs=0.1)
    assert out_path.exists()
    written = pysubs2.SSAFile.from_string(out_path.read_text(encoding="utf-8"))
    assert written.fps == pytest.approx(25.0)
    assert written[0].start / 1000.0 == pytest.approx(1.0, abs=0.1)


def test_microdvd_infers_fps_from_video_probe(tmp_path, monkeypatch):
    video_path = tmp_path / "movie.mkv"
    video_path.write_bytes(b"fake")
    in_path = tmp_path / "no_header.sub"
    in_path.write_text(_microdvd_lines(23.976, include_header=False), encoding="utf-8")

    monkeypatch.setattr(
        ffsubsync.ffmpeg,
        "probe",
        lambda *a, **k: {
            "streams": [
                {"codec_type": "video", "avg_frame_rate": "24000/1001", "r_frame_rate": "24000/1001"}
            ]
        },
    )
    args = ffsubsync.make_parser().parse_args([str(video_path), "-i", str(in_path), "-o", str(tmp_path / "out.srt")])
    pipe = ffsubsync.get_srt_pipe_maker(args, str(in_path))(1.0).fit(str(in_path))
    subs = pipe.named_steps["parse"].subs_
    assert subs._fps == pytest.approx(24000 / 1001, rel=1e-4)
    assert subs[0].start.total_seconds() == pytest.approx(1.0, abs=0.05)


def test_validate_args_fails_fast_on_unsupported_output_format(tmp_path):
    ref_path = tmp_path / "movie.mkv"
    in_path = tmp_path / "in.srt"
    out_path = tmp_path / "out.xyz"
    ref_path.write_bytes(b"fake")
    in_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")

    args = ffsubsync.make_parser().parse_args([str(ref_path), "-i", str(in_path), "-o", str(out_path)])
    with pytest.raises(ValueError, match="unsupported output subtitle format"):
        ffsubsync.validate_args(args)
