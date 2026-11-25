# -*- coding: utf-8 -*-
"""Tests for global block-level alignment profiles (balanced / onset / early)."""
from datetime import timedelta
import logging
import os

import numpy as np
import pytest
import srt

from ffsubsync.alignment_profile import (
    compute_profile_offset_adjustment,
    estimate_onset_offset_samples,
    normalize_alignment_profile,
)
from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.ffsubsync import make_parser, run, validate_args
from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.quality import compute_quality_metrics


def _make_vad_and_wide_subtitles(
    n: int = 48,
    *,
    extra_reading_s: float = 0.80,
    shift_s: float = 0.0,
    jitter_s: float = 0.0,
    start_jitter_s: float = 0.0,
    seed: int = 42,
):
    """Build irregular VAD speech bursts + wider subtitle cues (reading tail).

    Each speech burst is ``[t, t + speech_dur]`` while its subtitle cue starts
    at ``t + start_jitter + shift_s`` and extends ``extra_reading_s`` past
    speech end (modeling realistic reading linger and VAD/subtitler jitter).
    """
    rng = np.random.default_rng(seed)
    speech_intervals = []
    sub_cues = []
    t = 3.0
    for i in range(n):
        speech_dur = float(rng.uniform(1.0, 2.0))
        extra = max(0.1, extra_reading_s + float(rng.uniform(-jitter_s, jitter_s)))
        s_jit = float(rng.uniform(-start_jitter_s, start_jitter_s)) if start_jitter_s else 0.0
        gap = float(rng.uniform(1.2, 3.5))
        speech_intervals.append((t, t + speech_dur))
        sub_cues.append(
            srt.Subtitle(
                index=i + 1,
                start=timedelta(seconds=t + s_jit + shift_s),
                end=timedelta(seconds=t + speech_dur + extra + shift_s),
                content="Dialogue line number %d spoken here." % (i + 1),
            )
        )
        t += speech_dur + extra + gap

    total_samples = int((t + 10.0) * SAMPLE_RATE)
    ref = np.zeros(total_samples, dtype=np.float64)
    for a, b in speech_intervals:
        ref[int(round(a * SAMPLE_RATE)) : int(round(b * SAMPLE_RATE))] = 1.0
    return ref, speech_intervals, sub_cues


def _wrap_srt(sub_cues):
    return [GenericSubtitle.wrap_inner_subtitle(s) for s in sub_cues]


def test_normalize_alignment_profile():
    assert normalize_alignment_profile(None) == "balanced"
    assert normalize_alignment_profile("") == "balanced"
    assert normalize_alignment_profile("unknown") == "balanced"
    assert normalize_alignment_profile("BALANCED") == "balanced"
    assert normalize_alignment_profile("  Onset ") == "onset"
    assert normalize_alignment_profile("early") == "early"


def test_balanced_profile_is_exact_noop():
    ref, _, raw_cues = _make_vad_and_wide_subtitles(n=40, extra_reading_s=0.80)
    # Simulate centered FFT placement (-0.40s before speech onset).
    delta = compute_profile_offset_adjustment(
        ref, _wrap_srt(raw_cues), -0.40 * SAMPLE_RATE, profile="balanced"
    )
    assert delta == 0.0


def test_onset_and_early_profiles_on_wider_vad_subtitles():
    ref, _, raw_cues = _make_vad_and_wide_subtitles(
        n=48, extra_reading_s=0.80, jitter_s=0.12, seed=7
    )
    cues = _wrap_srt(raw_cues)
    # Symmetric FFT centers the 0.80s extra reading duration -> cues start 0.40s early.
    base_off = -0.40 * SAMPLE_RATE

    tau_samples, contrast = estimate_onset_offset_samples(ref, cues, base_off)
    assert tau_samples is not None
    assert contrast > 0.5
    assert abs((tau_samples / SAMPLE_RATE) - 0.40) < 0.04

    delta_balanced = compute_profile_offset_adjustment(
        ref, cues, base_off, profile="balanced"
    )
    delta_onset = compute_profile_offset_adjustment(
        ref, cues, base_off, profile="onset"
    )
    delta_early = compute_profile_offset_adjustment(
        ref, cues, base_off, profile="early"
    )

    assert delta_balanced == 0.0
    # Onset moves cue starts right onto speech start (+0.40s).
    assert abs((delta_onset / SAMPLE_RATE) - 0.40) < 0.04
    # Early stays strictly earlier than balanced (and capped sensibly).
    assert delta_early < delta_balanced < delta_onset
    assert -0.25 <= (delta_early / SAMPLE_RATE) <= -0.05


def test_onset_profile_is_noop_when_subtitle_reference_matches_duration():
    ref, speech_intervals, _ = _make_vad_and_wide_subtitles(
        n=40, extra_reading_s=0.0, seed=11
    )
    # Subtitle cues already match the reference intervals identically (sub-vs-sub).
    matching_cues = _wrap_srt(
        [
            srt.Subtitle(
                index=i + 1,
                start=timedelta(seconds=a),
                end=timedelta(seconds=b),
                content="Matched cue %d" % (i + 1),
            )
            for i, (a, b) in enumerate(speech_intervals)
        ]
    )

    delta_onset = compute_profile_offset_adjustment(
        ref, matching_cues, 0.0, profile="onset"
    )
    delta_early = compute_profile_offset_adjustment(
        ref, matching_cues, 0.0, profile="early"
    )

    assert abs(delta_onset / SAMPLE_RATE) <= 0.02
    assert -0.28 <= (delta_early / SAMPLE_RATE) <= -0.18


def test_onset_profile_handles_narrower_subtitles_than_speech():
    rng = np.random.default_rng(21)
    speech_intervals = []
    sub_cues = []
    t = 3.0
    for i in range(40):
        speech_dur = float(rng.uniform(2.0, 2.8))
        sub_dur = speech_dur - 0.60
        gap = float(rng.uniform(1.2, 3.0))
        speech_intervals.append((t, t + speech_dur))
        # Centered inside speech: starts 0.30s *after* speech onset.
        sub_cues.append(
            srt.Subtitle(
                index=i + 1,
                start=timedelta(seconds=t + 0.30),
                end=timedelta(seconds=t + 0.30 + sub_dur),
                content="Short subtitle %d" % (i + 1),
            )
        )
        t += speech_dur + gap

    ref = np.zeros(int((t + 10.0) * SAMPLE_RATE), dtype=np.float64)
    for a, b in speech_intervals:
        ref[int(round(a * SAMPLE_RATE)) : int(round(b * SAMPLE_RATE))] = 1.0

    delta_onset = compute_profile_offset_adjustment(
        ref, _wrap_srt(sub_cues), 0.0, profile="onset"
    )
    assert abs((delta_onset / SAMPLE_RATE) - (-0.30)) < 0.04


def test_profile_safe_on_noisy_flat_signal_and_tiny_inputs():
    # Flat all-ones reference has zero onset contrast -> onset must stay 0.0.
    _, _, raw_cues = _make_vad_and_wide_subtitles(n=20, extra_reading_s=0.60)
    flat_ref = np.ones(20000, dtype=np.float64)
    assert (
        compute_profile_offset_adjustment(
            flat_ref, _wrap_srt(raw_cues), 0.0, profile="onset"
        )
        == 0.0
    )
    # Fewer than 5 cues -> safe no-op for every profile.
    tiny = _wrap_srt(raw_cues[:3])
    assert compute_profile_offset_adjustment(flat_ref, tiny, 0.0, profile="onset") == 0.0
    assert compute_profile_offset_adjustment(flat_ref, tiny, 0.0, profile="early") == 0.0


def test_end_to_end_cli_profiles_preserve_block_rigidity_and_quality(tmp_path, caplog):
    # Realistic Audio VAD scenario: speech bursts are ~1.0-2.0s while subtitles
    # linger ~0.70s after speech with natural +/-0.15s start jitter, so symmetric
    # FFT correlation naturally centers the extra reading time (~0.30-0.35s early).
    ref, speech_intervals, in_cues = _make_vad_and_wide_subtitles(
        n=60,
        extra_reading_s=0.70,
        shift_s=5.0,
        jitter_s=0.18,
        start_jitter_s=0.15,
        seed=101,
    )
    ref_npz = tmp_path / "ref.npz"
    in_srt = tmp_path / "in.srt"
    np.savez_compressed(ref_npz, speech=ref)
    in_srt.write_text(srt.compose(in_cues), encoding="utf-8")

    outputs = {}
    results = {}
    for profile in ("balanced", "onset", "early"):
        out_srt = tmp_path / ("out_%s.srt" % profile)
        args = make_parser().parse_args(
            [
                str(ref_npz),
                "-i",
                str(in_srt),
                "-o",
                str(out_srt),
                "--no-fix-framerate",
                "--alignment-profile",
                profile,
            ]
        )
        with caplog.at_level(logging.INFO, logger="ffsubsync.ffsubsync"):
            res = run(args)
        assert res["sync_was_successful"] is True
        assert os.path.isfile(out_srt)
        assert res["alignment_profile"] == profile
        parsed = list(srt.parse(out_srt.read_text(encoding="utf-8")))
        assert len(parsed) == len(in_cues)
        outputs[profile] = parsed
        results[profile] = res

    # Verify rigid block shift: relative spacing between cues is identical to input.
    orig_diffs = [
        (in_cues[i].start - in_cues[0].start).total_seconds()
        for i in range(len(in_cues))
    ]
    for profile, parsed in outputs.items():
        new_diffs = [
            (parsed[i].start - parsed[0].start).total_seconds()
            for i in range(len(parsed))
        ]
        assert max(abs(a - b) for a, b in zip(orig_diffs, new_diffs)) < 0.003

    # Verify mean start error against actual speech onset (sub_start - speech_start).
    def _mean_start_delta(parsed_subs):
        return float(
            np.mean(
                [
                    sub.start.total_seconds() - sp_start
                    for sub, (sp_start, _) in zip(parsed_subs, speech_intervals)
                ]
            )
        )

    err_balanced = _mean_start_delta(outputs["balanced"])
    err_onset = _mean_start_delta(outputs["onset"])
    err_early = _mean_start_delta(outputs["early"])

    # Balanced centers the reading tail -> cues start noticeably before speech.
    assert -0.50 < err_balanced < -0.10
    # Onset lands right on speech onset (~0.00s).
    assert abs(err_onset) < 0.08
    # Early starts strictly earlier than balanced.
    assert err_early < err_balanced < err_onset

    # Quality badges stay high across all 3 profiles (no false consistency penalty!).
    for profile in ("balanced", "onset", "early"):
        q = results[profile]["quality"]
        assert q is not None
        assert q["sync"]["score"] >= 80, (profile, q)
        assert q["sync"]["metrics"]["temporal_consistency"] >= 80, (profile, q)


def test_end_to_end_auto_split_with_onset_profile(tmp_path):
    ref, speech_intervals, base_cues = _make_vad_and_wide_subtitles(
        n=50, extra_reading_s=0.80, shift_s=0.0, jitter_s=0.06, seed=303
    )
    # Introduce a +6.0s mid-file break in the second half.
    broken_cues = []
    for i, c in enumerate(base_cues):
        shift = 6.0 if i >= 25 else 0.0
        broken_cues.append(
            srt.Subtitle(
                index=c.index,
                start=c.start + timedelta(seconds=shift),
                end=c.end + timedelta(seconds=shift),
                content=c.content,
            )
        )

    ref_npz = tmp_path / "ref_split.npz"
    in_srt = tmp_path / "broken.srt"
    out_srt = tmp_path / "broken_onset.srt"
    np.savez_compressed(ref_npz, speech=ref)
    in_srt.write_text(srt.compose(broken_cues), encoding="utf-8")

    args = make_parser().parse_args(
        [
            str(ref_npz),
            "-i",
            str(in_srt),
            "-o",
            str(out_srt),
            "--no-fix-framerate",
            "--auto-split",
            "--alignment-profile",
            "onset",
        ]
    )
    res = run(args)
    assert res["sync_was_successful"] is True
    assert res["alignment_profile"] == "onset"
    assert res["profile_offset_seconds"] > 0.20
    assert res["split_segments"] is not None and len(res["split_segments"]) == 2

    parsed = list(srt.parse(out_srt.read_text(encoding="utf-8")))
    assert len(parsed) == len(base_cues)
    # Both halves start right at speech onset without any overlaps between cues.
    mean_err_first = float(
        np.mean(
            [
                parsed[i].start.total_seconds() - speech_intervals[i][0]
                for i in range(25)
            ]
        )
    )
    mean_err_second = float(
        np.mean(
            [
                parsed[i].start.total_seconds() - speech_intervals[i][0]
                for i in range(25, 50)
            ]
        )
    )
    assert abs(mean_err_first) < 0.08
    assert abs(mean_err_second) < 0.08
    for i in range(1, len(parsed)):
        assert parsed[i].start >= parsed[i - 1].end
    assert res["quality"]["sync"]["score"] >= 80


def test_quality_metrics_profile_offset_compensation():
    ref, speech_intervals, raw_cues = _make_vad_and_wide_subtitles(
        n=48, extra_reading_s=0.80, shift_s=0.0, seed=55
    )
    # Cues start right at speech onset (+0.40s relative to symmetric centroid).
    cues = _wrap_srt(raw_cues)
    q_comp = compute_quality_metrics(
        ref,
        cues,
        best_peak_snr=14.0,
        best_peak_margin=7.0,
        profile_offset_seconds=0.40,
    )
    assert q_comp["sync"]["metrics"]["temporal_consistency"] >= 85
    assert q_comp["sync"]["score"] >= 85


def test_validate_args_rejects_invalid_alignment_profile():
    args = make_parser().parse_args(["ref.srt", "-i", "in.srt"])
    args.alignment_profile = "invalid-mode"
    with pytest.raises(ValueError, match="alignment-profile"):
        validate_args(args)
