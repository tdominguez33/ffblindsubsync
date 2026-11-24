# -*- coding: utf-8 -*-
"""Tests for post-sync confidence and subtitle quality scoring (0-100)."""
from datetime import timedelta
import logging
import os

import numpy as np
import pysubs2
import srt

from ffsubsync.aligners import FFTAligner
from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.ffsubsync import make_parser, run
from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.quality import compute_quality_metrics


def _make_irregular_timeline(n: int = 48, seed: int = 42):
    """Build irregular dialogue (start_s, end_s) intervals + binary reference signal."""
    rng = np.random.default_rng(seed)
    intervals = []
    t = 2.0
    for _ in range(n):
        dur = float(rng.uniform(1.2, 2.8))
        gap = float(rng.uniform(1.0, 4.0))
        intervals.append((t, t + dur))
        t += dur + gap
    total_samples = int((t + 5.0) * SAMPLE_RATE)
    ref = np.zeros(total_samples, dtype=np.float64)
    for a, b in intervals:
        ref[int(round(a * SAMPLE_RATE)) : int(round(b * SAMPLE_RATE))] = 1.0
    return ref, intervals


def _cues_from_intervals(intervals, text_fn=None, shift_fn=None):
    cues = []
    for i, (a, b) in enumerate(intervals):
        shift = shift_fn(i, len(intervals)) if shift_fn else 0.0
        dur = max(0.1, b - a)
        # ~12 CPS default comfortable English dialogue
        if text_fn:
            text = text_fn(i, dur)
        else:
            n_chars = max(8, int(round(dur * 12)))
            text = ("Hello there, line %d. " % (i + 1) * 4)[:n_chars].strip()
        sub = srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=max(0.0, a + shift)),
            end=timedelta(seconds=max(0.05, b + shift)),
            content=text,
        )
        cues.append(GenericSubtitle.wrap_inner_subtitle(sub))
    return cues


def test_perfect_sync_and_clean_subtitles_score_high():
    ref, intervals = _make_irregular_timeline(n=48)
    cues = _cues_from_intervals(intervals)

    q = compute_quality_metrics(
        ref,
        cues,
        sample_rate=SAMPLE_RATE,
        best_peak_snr=18.0,
        best_peak_margin=9.0,
        sync_was_successful=True,
    )

    assert set(q.keys()) == {"reference_kind", "sync", "subtitle"}
    assert q["reference_kind"] == "audio"
    assert set(q["sync"].keys()) == {"score", "metrics"}
    assert set(q["subtitle"].keys()) == {"score", "metrics"}
    assert set(q["sync"]["metrics"].keys()) == {
        "temporal_consistency",
        "overlap_lift",
        "peak_confidence",
        "sync_integrity",
    }
    assert set(q["subtitle"]["metrics"].keys()) == {
        "timeline_coverage",
        "reading_speed",
        "timing_hygiene",
        "text_integrity",
    }
    assert q["sync"]["score"] >= 90
    assert q["sync"]["metrics"]["temporal_consistency"] >= 95
    assert q["sync"]["metrics"]["overlap_lift"] >= 85
    assert q["subtitle"]["score"] >= 90
    assert q["subtitle"]["metrics"]["timeline_coverage"] >= 95
    assert q["subtitle"]["metrics"]["reading_speed"] >= 90
    assert q["subtitle"]["metrics"]["timing_hygiene"] == 100
    assert q["subtitle"]["metrics"]["text_integrity"] == 100


def test_temporal_inconsistency_and_drift_lower_consistency_score():
    ref, intervals = _make_irregular_timeline(n=48)

    # User's scenario: starts at +0.00s, drifts to +0.20s mid-video, ends at -0.10s.
    def _wobble(i, n):
        frac = i / float(max(1, n - 1))
        if frac < 0.33:
            return 0.0
        if frac < 0.67:
            return 0.20
        return -0.10

    # Stronger linear drift: 0.00s -> +0.80s across the video.
    def _linear_drift(i, n):
        return 0.80 * (i / float(max(1, n - 1)))

    q_perfect = compute_quality_metrics(
        ref,
        _cues_from_intervals(intervals),
        best_peak_snr=15.0,
        best_peak_margin=7.0,
    )
    q_wobble = compute_quality_metrics(
        ref,
        _cues_from_intervals(intervals, shift_fn=_wobble),
        best_peak_snr=15.0,
        best_peak_margin=7.0,
    )
    q_drift = compute_quality_metrics(
        ref,
        _cues_from_intervals(intervals, shift_fn=_linear_drift),
        best_peak_snr=15.0,
        best_peak_margin=7.0,
    )

    c_perf = q_perfect["sync"]["metrics"]["temporal_consistency"]
    c_wobble = q_wobble["sync"]["metrics"]["temporal_consistency"]
    c_drift = q_drift["sync"]["metrics"]["temporal_consistency"]

    assert c_perf >= 95
    assert c_wobble < 75
    assert c_drift <= 45
    assert c_perf > c_wobble > c_drift
    # Bottleneck cap: temporal_consistency < 50 prevents a green (>= 75) sync badge.
    assert q_drift["sync"]["score"] <= 69


def test_bottleneck_cap_prevents_green_badge_when_any_submetric_below_50():
    ref, intervals = _make_irregular_timeline(n=48)

    # Sync: strong overlap & consistency, but weak peak_confidence (< 50) -> capped <= 69.
    q_sync_cap = compute_quality_metrics(
        ref,
        _cues_from_intervals(intervals),
        best_peak_snr=2.5,
        best_peak_margin=0.4,
        sync_was_successful=True,
    )
    assert q_sync_cap["sync"]["metrics"]["peak_confidence"] < 50
    assert q_sync_cap["sync"]["score"] <= 69

    # Subtitle: clean coverage/timing/text, but terrible reading speed (< 50) -> capped <= 69.
    fast_cues = _cues_from_intervals(
        intervals,
        text_fn=lambda i, dur: "A" * max(80, int(dur * 50)),
    )
    q_sub_cap = compute_quality_metrics(ref, fast_cues, best_peak_snr=15.0)
    assert q_sub_cap["subtitle"]["metrics"]["reading_speed"] < 50
    assert q_sub_cap["subtitle"]["score"] <= 69


def test_sparse_audio_vad_plateau_tolerated_while_real_drift_still_caught():
    """Wider subtitle cues over narrower VAD bursts shouldn't false-alarm, but real drift should."""
    _, intervals = _make_irregular_timeline(n=64, seed=99)
    total_samples = int((intervals[-1][1] + 10.0) * SAMPLE_RATE)
    sparse_ref = np.zeros(total_samples, dtype=np.float64)
    rng = np.random.default_rng(123)
    for a, b in intervals:
        # Spoken burst is ~45% of subtitle duration with small natural timing jitter (+/-0.18s)
        dur = b - a
        jitter = float(rng.uniform(-0.18, 0.18))
        va = max(0.0, a + 0.10 * dur + jitter)
        vb = min(total_samples / float(SAMPLE_RATE), va + 0.45 * dur)
        sparse_ref[int(round(va * SAMPLE_RATE)) : int(round(vb * SAMPLE_RATE))] = 1.0

    aligned_cues = _cues_from_intervals(intervals)
    drifting_cues = _cues_from_intervals(
        intervals, shift_fn=lambda i, n: 0.90 * (i / float(max(1, n - 1)))
    )

    q_aligned = compute_quality_metrics(
        sparse_ref, aligned_cues, best_peak_snr=10.0, best_peak_margin=5.0
    )
    q_drifting = compute_quality_metrics(
        sparse_ref, drifting_cues, best_peak_snr=10.0, best_peak_margin=5.0
    )

    assert q_aligned["sync"]["metrics"]["temporal_consistency"] >= 75
    assert q_drifting["sync"]["metrics"]["temporal_consistency"] <= 45


def test_misaligned_and_failed_sync_is_capped():
    ref, intervals = _make_irregular_timeline(n=48)
    # Shift all cues far off-target (+15s).
    bad_cues = _cues_from_intervals(intervals, shift_fn=lambda _i, _n: 15.0)

    q = compute_quality_metrics(
        ref,
        bad_cues,
        best_peak_snr=2.0,
        best_peak_margin=0.3,
        sync_was_successful=False,
    )
    assert q["sync"]["score"] <= 35
    assert q["sync"]["metrics"]["overlap_lift"] <= 20


def test_peak_margin_distinguishes_unique_vs_ambiguous_correlation():
    rng = np.random.default_rng(7)
    unique_ref = (rng.random(4000) > 0.7).astype(float)
    unique_sub = unique_ref[500:3500]
    aligner_u = FFTAligner(max_offset_samples=1000).fit(unique_ref, unique_sub)
    assert aligner_u.best_peak_snr_ is not None and aligner_u.best_peak_snr_ > 10.0
    assert aligner_u.best_peak_margin_ is not None and aligner_u.best_peak_margin_ > 5.0

    # Strictly periodic signal repeats every 400 samples -> competing identical peaks!
    unit = np.zeros(400, dtype=float)
    unit[:100] = 1.0
    periodic = np.tile(unit, 10)
    aligner_p = FFTAligner(max_offset_samples=1200).fit(periodic, periodic[400:3600])
    assert aligner_p.best_peak_margin_ is not None
    assert aligner_p.best_peak_margin_ < aligner_u.best_peak_margin_


def test_subtitle_intrinsic_defects_lower_specific_metrics():
    ref, intervals = _make_irregular_timeline(n=48)

    # 1. Partial coverage (only first 30% of the video).
    partial_cues = _cues_from_intervals(intervals[:14])
    q_partial = compute_quality_metrics(ref, partial_cues, best_peak_snr=12.0)
    assert q_partial["subtitle"]["metrics"]["timeline_coverage"] <= 40
    assert q_partial["subtitle"]["score"] <= 65

    # 2. Extremely fast reading speed (>45 CPS).
    fast_cues = _cues_from_intervals(
        intervals,
        text_fn=lambda i, dur: "A" * max(80, int(dur * 50)),
    )
    q_fast = compute_quality_metrics(ref, fast_cues, best_peak_snr=12.0)
    assert q_fast["subtitle"]["metrics"]["reading_speed"] <= 25

    # 3. Broken encoding / mojibake + empty cues.
    corrupt_cues = _cues_from_intervals(
        intervals,
        text_fn=lambda i, _d: "Caf\ufffd Ã©l â€œholaâ€" if i % 2 == 0 else "",
    )
    q_corrupt = compute_quality_metrics(ref, corrupt_cues, best_peak_snr=12.0)
    assert q_corrupt["subtitle"]["metrics"]["text_integrity"] <= 30
    assert q_corrupt["subtitle"]["score"] <= 55

    # 4. Overlapping SRT cues + flash durations.
    overlapping = []
    for i, (a, b) in enumerate(intervals):
        sub = srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=a),
            end=timedelta(seconds=b + 5.0 if i % 2 == 0 else a + 0.15),
            content="Sample subtitle line",
        )
        overlapping.append(GenericSubtitle.wrap_inner_subtitle(sub))
    q_timing = compute_quality_metrics(ref, overlapping, best_peak_snr=12.0)
    assert q_timing["subtitle"]["metrics"]["timing_hygiene"] <= 50


def test_ass_multilayer_simultaneous_events_not_penalized_as_overlaps():
    ref, intervals = _make_irregular_timeline(n=24)
    cues = []
    for i, (a, b) in enumerate(intervals):
        ev_dlg = pysubs2.SSAEvent(
            start=int(a * 1000),
            end=int(b * 1000),
            text="Dialogue line %d" % i,
            style="Default",
            layer=0,
        )
        ev_sign = pysubs2.SSAEvent(
            start=int(a * 1000),
            end=int(b * 1000),
            text="{\\an8}Sign %d" % i,
            style="Signs",
            layer=1,
        )
        cues.append(GenericSubtitle.wrap_inner_subtitle(ev_dlg))
        cues.append(GenericSubtitle.wrap_inner_subtitle(ev_sign))

    q = compute_quality_metrics(ref, cues, sub_format="ass", best_peak_snr=14.0)
    assert q["subtitle"]["metrics"]["timing_hygiene"] == 100


def test_end_to_end_cli_run_populates_quality_and_logs(tmp_path, caplog):
    _, intervals = _make_irregular_timeline(n=36)
    ref_cues = [
        srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=a),
            end=timedelta(seconds=b),
            content="Reference dialogue number %d." % (i + 1),
        )
        for i, (a, b) in enumerate(intervals)
    ]
    in_cues = [
        srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=a + 4.5),
            end=timedelta(seconds=b + 4.5),
            content="Input dialogue number %d." % (i + 1),
        )
        for i, (a, b) in enumerate(intervals)
    ]

    ref_path = tmp_path / "ref.srt"
    in_path = tmp_path / "in.srt"
    out_path = tmp_path / "out.srt"
    ref_path.write_text(srt.compose(ref_cues), encoding="utf-8")
    in_path.write_text(srt.compose(in_cues), encoding="utf-8")

    args = make_parser().parse_args(
        [str(ref_path), "-i", str(in_path), "-o", str(out_path)]
    )
    with caplog.at_level(logging.INFO, logger="ffsubsync.ffsubsync"):
        result = run(args)

    assert result["sync_was_successful"] is True
    assert os.path.isfile(out_path)
    q = result["quality"]
    assert q is not None
    assert q["reference_kind"] == "subtitles"
    assert q["sync"]["score"] >= 85
    assert q["subtitle"]["score"] >= 85
    assert any("sync confidence:" in rec.message for rec in caplog.records)
    assert any("subtitle quality:" in rec.message for rec in caplog.records)
