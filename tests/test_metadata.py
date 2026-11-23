# -*- coding: utf-8 -*-
from datetime import timedelta

import numpy as np
import pytest
import srt

from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.speech_transformers import (
    SubtitleSpeechTransformer,
    _classify_speech_cues,
    _is_metadata,
)
from ffsubsync.split_aligner import _cue_sample_bounds


@pytest.mark.parametrize(
    "content",
    [
        "[music]",
        "(applause)",
        "{laughter}",
        "（音乐）",  # full-width brackets
        "【掌声】",
        "「効果音」",
        "♪",
        "♪♪",
        "♪ ♫ ♬",
        "<i>[music]</i>",  # cue wrapped in markup
        "<font color='#fff'>(applause)</font>",
        "<i></i>",  # markup with no text
        "   ",  # whitespace only
        "",
        r"{\an8}[music]",  # ASS override tag + SDH
        r"{\i1}(sighs){\i0}",
        r"{\pos(320,40)}",  # ASS tag with no visible text
        "[MUSIC PLAYING]\n(DOOR SLAMS)",  # multi-line mixed bracket styles
        "- [FOOTSTEPS]\n- [DOOR CREAKS]",  # dialogue dashes before SDH
        "[dramatic music\nplaying]",  # bracket spanning multiple lines
        r"[SIGHS]\N(GROANS)",  # ASS newline separator
        "WOMAN: [SCREAMING]",  # speaker label + SDH only
        "CROWD: (CHEERING) (APPLAUSE)",
        "- [SIGHS]...",
        "Advertise your product or brand here\ncontact www.OpenSubtitles.org today",
    ],
)
def test_is_metadata_detects_non_dialogue(content):
    assert _is_metadata(content, False) is True


@pytest.mark.parametrize(
    "content",
    [
        "John: hello there",
        "JOHN: hello there",
        "<i>Hello?</i>",  # markup-wrapped dialogue must NOT be metadata
        r"{\an8}Hello there",  # ASS positioned dialogue
        "♪ We are the champions ♪",  # lyrics with words are kept
        "[door creaks] and he walks in",  # bracketed aside + real dialogue
        "[door creaks]\nHello there",  # mixed multi-line SDH + dialogue
        "(5 > 3) is true",
    ],
)
def test_is_metadata_keeps_dialogue(content):
    assert _is_metadata(content, False) is False


@pytest.mark.parametrize(
    "content",
    [
        "this is in english",
        "Bob - Alice",
        "Synced and corrected by SomeUser",
        "Subtitles by CoolTeam",
        "https://example.com/subs",
    ],
)
def test_is_metadata_boundary_only_heuristics(content):
    # Credits/URLs and "english" / " - " are only treated as metadata at start/end
    assert _is_metadata(content, True) is True
    assert _is_metadata(content, False) is False


def _generic_sub(start, end, content):
    inner = srt.Subtitle(
        index=1,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        content=content,
    )
    return GenericSubtitle(timedelta(seconds=start), timedelta(seconds=end), inner)


def test_non_dialogue_line_contributes_no_speech_but_is_not_dropped():
    subs = [
        _generic_sub(0, 1, "♪ ♫"),  # cue: should be zeroed in the signal
        _generic_sub(1, 2, "Hello there"),  # dialogue: should be speech
    ]
    transformer = SubtitleSpeechTransformer(sample_rate=SAMPLE_RATE).fit(subs)
    speech = transformer.subtitle_speech_results_
    # the cue's frames are non-speech; the dialogue's frames are speech
    assert not np.any(speech[: SAMPLE_RATE])
    assert np.all(speech[SAMPLE_RATE : 2 * SAMPLE_RATE] > 0)
    # the transformer never drops subtitles; it only affects the speech signal
    assert len(subs) == 2


def test_youtube_rolling_caption_transition_excluded_from_speech_signal():
    subs = [
        _generic_sub(1.0, 3.0, "Hello everyone"),
        _generic_sub(3.0, 3.01, "Hello everyone"),  # 10ms YouTube scroll frame
        _generic_sub(3.01, 5.0, "Hello everyone\nwelcome to the video"),
        _generic_sub(5.0, 5.02, "welcome to the video"),  # 20ms scroll frame
        _generic_sub(5.02, 7.0, "welcome to the video\ntoday we build stuff"),
    ]
    mask = _classify_speech_cues(subs)
    assert list(mask) == [True, False, True, False, True]
    _, _, split_mask = _cue_sample_bounds(subs, SAMPLE_RATE, 0.0)
    assert list(split_mask) == [True, False, True, False, True]


def test_metadata_filter_safety_fallback_for_all_bracketed_file():
    subs = [_generic_sub(i * 2.0, i * 2.0 + 1.0, "[line %d]" % i) for i in range(12)]
    mask = _classify_speech_cues(subs)
    assert np.all(mask)
    transformer = SubtitleSpeechTransformer(sample_rate=SAMPLE_RATE).fit(subs)
    assert np.sum(transformer.subtitle_speech_results_ > 0) > 0
