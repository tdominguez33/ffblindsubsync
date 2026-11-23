# -*- coding: utf-8 -*-
from datetime import timedelta
import os

import pytest
import srt

from ffsubsync import ffsubsync, generic_subtitles
from ffsubsync.generic_subtitles import GenericSubtitle, GenericSubtitlesFile


def subtitle_file(content="Hello", encoding="utf-8"):
    sub = srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), content)
    return GenericSubtitlesFile(
        [GenericSubtitle.wrap_inner_subtitle(sub)], sub_format="srt", encoding=encoding
    )


def test_failed_encoding_preserves_original(tmp_path):
    output = tmp_path / "movie.srt"
    output.write_bytes(b"original subtitles")

    with pytest.raises(UnicodeEncodeError):
        subtitle_file("Bonjour à tous", encoding="ascii").write_file(str(output))

    assert output.read_bytes() == b"original subtitles"
    assert list(tmp_path.iterdir()) == [output]


def test_failed_replace_preserves_original_and_cleans_temporary_file(
    tmp_path, monkeypatch
):
    output = tmp_path / "movie.srt"
    output.write_bytes(b"original subtitles")

    def fail_replace(source, destination):
        raise PermissionError("cannot replace output")

    monkeypatch.setattr(generic_subtitles.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="cannot replace output"):
        subtitle_file().write_file(str(output))

    assert output.read_bytes() == b"original subtitles"
    assert list(tmp_path.iterdir()) == [output]


def test_new_output_file(tmp_path):
    output = tmp_path / "new.srt"
    subtitle_file().write_file(str(output))
    assert "Hello" in output.read_text(encoding="utf-8")


def test_sanitize_negative_timestamps_clamps_partial_and_counts_dropped():
    subs = GenericSubtitlesFile(
        [
            GenericSubtitle.wrap_inner_subtitle(
                srt.Subtitle(1, timedelta(seconds=-5), timedelta(seconds=-1), "gone")
            ),
            GenericSubtitle.wrap_inner_subtitle(
                srt.Subtitle(2, timedelta(seconds=-1), timedelta(seconds=2), "partial")
            ),
            GenericSubtitle.wrap_inner_subtitle(
                srt.Subtitle(3, timedelta(seconds=3), timedelta(seconds=5), "normal")
            ),
        ],
        sub_format="srt",
        encoding="utf-8",
    )
    cleaned, clipped = subs.sanitize_negative_timestamps()
    assert clipped == 1
    assert len(cleaned) == 2
    assert cleaned[0].start == timedelta(0)
    assert cleaned[0].end == timedelta(seconds=2)
    assert cleaned[1].start == timedelta(seconds=3)


def test_try_sync_reports_clipped_cues_and_preserves_partial(tmp_path):
    srtin = tmp_path / "in.srt"
    srtin.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nfirst\n\n"
        "2\n00:00:04,000 --> 00:00:07,000\nsecond\n\n"
        "3\n00:00:10,000 --> 00:00:12,000\nthird\n",
        encoding="utf-8",
    )
    srtout = tmp_path / "out.srt"
    args = ffsubsync.make_parser().parse_args(
        ["-i", str(srtin), "-o", str(srtout), "--apply-offset-seconds", "-5"]
    )
    res = ffsubsync.run(args)
    assert res["retval"] == 0
    assert res["clipped_cues"] == 1
    parsed = list(srt.parse(srtout.read_text(encoding="utf-8")))
    assert [c.content for c in parsed] == ["second", "third"]
    assert parsed[0].start == timedelta(0)
    assert parsed[0].end == timedelta(seconds=2)


def test_make_test_case_handles_missing_files_after_early_exception(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    srtin = tmp_path / "in.srt"
    srtin.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")
    args = ffsubsync.make_parser().parse_args(
        [
            "ref.mkv",
            "-i",
            str(srtin),
            "-o",
            str(tmp_path / "out.srt"),
            "--make-test-case",
        ]
    )
    # Neither ffsubsync.log, out.srt, nor ref.npz exists: should not raise FileNotFoundError
    rc = ffsubsync.make_test_case(args, str(tmp_path / "ref.npz"), sync_was_successful=False)
    assert rc == 0
    archives = [p for p in tmp_path.iterdir() if p.name.startswith("ref.mkv.")]
    assert len(archives) == 1
