# -*- coding: utf-8 -*-
from datetime import timedelta

import numpy as np
import srt

from ffsubsync import ffsubsync
from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.generic_subtitles import GenericSubtitle, GenericSubtitlesFile
from ffsubsync.split_aligner import (
    _find_local_offset_fft,
    compute_auto_split_offsets,
    compute_split_offsets,
    count_dropped_cues,
)
from ffsubsync.subtitle_transformers import VariableSubtitleShifter


def _cue(start_s, end_s, content="hello world"):
    inner = srt.Subtitle(
        index=1,
        start=timedelta(seconds=start_s),
        end=timedelta(seconds=end_s),
        content=content,
    )
    return GenericSubtitle(inner.start, inner.end, inner)


def _reference(intervals, total_seconds):
    arr = np.zeros(int(total_seconds * SAMPLE_RATE) + 2, dtype=float)
    for a, b in intervals:
        arr[int(round(a * SAMPLE_RATE)) : int(round(b * SAMPLE_RATE))] = 1.0
    return arr


def _num_splits(offsets):
    return sum(1 for i in range(1, len(offsets)) if offsets[i] != offsets[i - 1])


def _align(reference, cues, split_penalty, max_offset_seconds=7):
    return compute_split_offsets(
        reference,
        cues,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=split_penalty,
        max_offset_samples=int(max_offset_seconds * SAMPLE_RATE),
    )


def test_recovers_mid_file_break():
    # First half aligned at offset 0; second half needs +6s (e.g. a break was
    # removed from the reference), so no single global offset can fix both.
    # Distinct block lengths + a 7s max-offset make each cue's target unique
    # (in particular the second-half cues cannot mirror back onto the first-half
    # reference within the offset window).
    reference = _reference(
        [(1.0, 1.4), (3.0, 3.9), (17.0, 17.6), (19.0, 20.0)], total_seconds=22
    )
    cues = [_cue(1.0, 1.4), _cue(3.0, 3.9), _cue(11.0, 11.6), _cue(13.0, 14.0)]
    offsets = _align(reference, cues, split_penalty=0.5 * SAMPLE_RATE)
    assert offsets == [0, 0, 6 * SAMPLE_RATE, 6 * SAMPLE_RATE]
    assert _num_splits(offsets) == 1


def test_collapses_to_single_offset():
    # Every cue is uniformly late by 5s; the summed rating peaks sharply at +5s,
    # so a large penalty (which forbids splitting) yields one global offset.
    reference = _reference(
        [(6.0, 6.4), (8.0, 8.9), (10.0, 10.6), (12.0, 13.0)], total_seconds=14
    )
    cues = [_cue(1.0, 1.4), _cue(3.0, 3.9), _cue(5.0, 5.6), _cue(7.0, 8.0)]
    offsets = _align(reference, cues, split_penalty=1e9)
    assert offsets == [5 * SAMPLE_RATE] * 4
    assert _num_splits(offsets) == 0


def test_already_aligned_is_noop():
    reference = _reference([(1.0, 1.4), (3.0, 3.9)], total_seconds=6)
    cues = [_cue(1.0, 1.4), _cue(3.0, 3.9)]
    offsets = _align(reference, cues, split_penalty=0.5 * SAMPLE_RATE)
    assert offsets == [0, 0]
    assert _num_splits(offsets) == 0


def test_length_penalty_prefers_matching_block():
    # One cue of length 1s. The reference has a same-sized block at +10s and a long
    # 5s block at +3s. Pure overlap is a flat 1s for any offset that drops the cue
    # inside the long block, so its argmax tie-breaks to the (wrong) low offset +3.
    # The length penalty charges reference speech just outside the cue edges, so the
    # exactly-sized block at +10 (whose neighbours are silent) wins.
    reference = _reference([(3.0, 8.0), (10.0, 11.0)], total_seconds=13)
    cues = [_cue(0.0, 1.0)]

    pure = _align(reference, cues, split_penalty=1e9, max_offset_seconds=12)
    assert pure == [3 * SAMPLE_RATE]

    penalized = compute_split_offsets(
        reference,
        cues,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=1e9,
        length_penalty=0.25,
        max_offset_samples=12 * SAMPLE_RATE,
    )
    assert penalized == [10 * SAMPLE_RATE]


def test_subsample_improves_fractional_offset():
    # The reference block is grid-aligned ([5.00, 6.00]) but the cue is offset from
    # the sample grid by fractions of a sample, so its ideal shift is non-integer.
    # Whole-sample search can only get within ~1 sample; sub-sample search recovers
    # the fractional shift and thus strictly more overlap.
    reference = _reference([(5.0, 6.0)], total_seconds=13)
    cues = [_cue(0.003, 1.007)]

    whole, whole_score = compute_split_offsets(
        reference, cues, sample_rate=SAMPLE_RATE, start_seconds=0,
        split_penalty=1e9, max_offset_samples=10 * SAMPLE_RATE, return_score=True,
    )
    fine, fine_score = compute_split_offsets(
        reference, cues, sample_rate=SAMPLE_RATE, start_seconds=0,
        split_penalty=1e9, max_offset_samples=10 * SAMPLE_RATE,
        offset_step_samples=0.1, return_score=True,
    )

    # Sub-sample recovers essentially the full 1s (100 samples) of overlap...
    assert fine_score > whole_score + 0.2
    assert fine_score > 99.9
    # ...at an offset near 4.99-5.00s but not snapped to a whole 10ms sample.
    fine_seconds = fine[0] / float(SAMPLE_RATE)
    assert 4.99 <= fine_seconds <= 5.0


def test_empty_input():
    assert (
        compute_split_offsets(
            _reference([(1, 2)], total_seconds=3),
            [],
            sample_rate=SAMPLE_RATE,
            start_seconds=0,
            split_penalty=1.0,
            max_offset_samples=100,
        )
        == []
    )


def test_monotonic_prevents_backward_scene_overlap():
    # Scene A (5..15s) has continuous dense speech; Scene B (25..35s) has
    # sparser speech matching cues in 25..35s, except with some dropped VAD frames
    # that would tempt an unconstrained DP to jump Scene B cues backwards by -20s
    # right on top of Scene A. Monotonicity forbids pulling cues backwards across
    # an earlier cue's end time.
    ref_intervals = [(5.0, 15.0)]
    cues = []
    t = 5.0
    for _ in range(5):
        cues.append(_cue(t, t + 1.5))
        t += 1.6
    # Dense cues in Scene B starting right after Scene A (small gap)
    t = 16.0
    for k in range(6):
        cues.append(_cue(t, t + 1.4))
        # Leave partial speech in Scene B so Scene A's solid block would score higher
        if k % 2 == 0:
            ref_intervals.append((t, t + 0.8))
        t += 1.5

    reference = _reference(ref_intervals, total_seconds=35)
    offsets = compute_split_offsets(
        reference,
        cues,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=0.5 * SAMPLE_RATE,
        max_offset_samples=15 * SAMPLE_RATE,
        enforce_monotonic=True,
    )
    shifted_starts = [c.start.total_seconds() + off / SAMPLE_RATE for c, off in zip(cues, offsets)]
    shifted_ends = [c.end.total_seconds() + off / SAMPLE_RATE for c, off in zip(cues, offsets)]
    for i in range(1, len(cues)):
        assert shifted_starts[i] >= shifted_ends[i - 1] - 1e-6


def test_auto_split_zero_splits_on_unbroken_noisy_file():
    # 50 irregular cues shifted uniformly late by 7s (so true offset is -7s).
    # One middle block has noisy/dropped reference speech, plus dense speech 50s
    # earlier that must NOT lure cues into a spurious split.
    rng = np.random.default_rng(42)
    ref_intervals = []
    cues = []
    t = 10.0
    for i in range(50):
        dur = float(rng.uniform(1.0, 2.2))
        gap = float(rng.uniform(0.4, 1.5))
        cues.append(_cue(t + 7.0, t + 7.0 + dur))
        if 15 <= i < 23:
            # Noisy/clipped VAD on a middle scene, still ~55% intact
            ref_intervals.append((t, t + dur * 0.55))
        else:
            ref_intervals.append((t, t + dur))
        t += dur + gap

    reference = _reference(ref_intervals, total_seconds=t + 20)
    offsets, _ = compute_auto_split_offsets(
        reference,
        cues,
        global_offset_samples=-7.0 * SAMPLE_RATE,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=12.0 * SAMPLE_RATE,
        length_penalty=0.25,
        max_search_samples=60 * SAMPLE_RATE,
        log_segments=False,
    )
    assert _num_splits(offsets) == 0
    assert all(abs(off / SAMPLE_RATE - (-7.0)) < 0.05 for off in offsets)


def test_auto_split_recovers_large_break_beyond_60s():
    # First 25 cues aligned at 0s; second 25 cues need +120s (an inserted 2-min scene
    # well outside the standard +/-60s FFT window).
    rng = np.random.default_rng(7)
    ref_intervals = []
    cues = []
    t = 2.0
    for i in range(50):
        dur = float(rng.uniform(1.0, 2.0))
        gap = float(rng.uniform(0.5, 1.4))
        cues.append(_cue(t, t + dur))
        shift = 0.0 if i < 25 else 120.0
        ref_intervals.append((t + shift, t + dur + shift))
        t += dur + gap

    reference = _reference(ref_intervals, total_seconds=t + 140)
    offsets, _ = compute_auto_split_offsets(
        reference,
        cues,
        global_offset_samples=0.0,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=12.0 * SAMPLE_RATE,
        length_penalty=0.25,
        max_search_samples=300 * SAMPLE_RATE,
        log_segments=False,
    )
    assert _num_splits(offsets) == 1
    for i in range(25):
        assert abs(offsets[i] / SAMPLE_RATE - 0.0) < 0.1
    for i in range(25, 50):
        assert abs(offsets[i] / SAMPLE_RATE - 120.0) < 0.1


def test_auto_split_recovers_negative_break_from_removed_commercial():
    # First 25 cues at 0s, then a 45s commercial break in the input cues that was
    # removed from the reference video, so last 25 cues need -35s.
    rng = np.random.default_rng(11)
    ref_intervals = []
    cues = []
    t = 2.0
    for i in range(50):
        dur = float(rng.uniform(1.0, 2.0))
        gap = float(rng.uniform(0.5, 1.4))
        cue_t = t if i < 25 else t + 35.0
        cues.append(_cue(cue_t, cue_t + dur))
        ref_intervals.append((t, t + dur))
        t += dur + gap

    reference = _reference(ref_intervals, total_seconds=t + 50)
    offsets, _ = compute_auto_split_offsets(
        reference,
        cues,
        global_offset_samples=0.0,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=12.0 * SAMPLE_RATE,
        length_penalty=0.25,
        max_search_samples=300 * SAMPLE_RATE,
        log_segments=False,
    )
    assert _num_splits(offsets) == 1
    for i in range(25):
        assert abs(offsets[i] / SAMPLE_RATE - 0.0) < 0.1
    for i in range(25, 50):
        assert abs(offsets[i] / SAMPLE_RATE - (-35.0)) < 0.1


def _add_background_noise(ref, total_seconds, density=0.45, seed=99):
    rng = np.random.default_rng(seed)
    n = len(ref)
    t = 0
    while t < n:
        burst_len = int(rng.integers(30, 120))
        gap_len = int(burst_len * (1.0 - density) / max(0.05, density))
        if rng.random() < density:
            ref[t : min(n, t + burst_len)] = 1.0
            t += burst_len + int(rng.integers(20, 80))
        else:
            t += max(20, gap_len)
    return ref


def test_auto_split_noisy_background_vad_recovers_break():
    # Real movie VAD often stays active ~50% of the timeline due to score/SFX.
    # A +45s mid-file break must still be detected and cleanly aligned.
    rng = np.random.default_rng(101)
    ref_intervals = []
    cues = []
    t = 5.0
    for i in range(60):
        dur = float(rng.uniform(1.1, 2.0))
        gap = float(rng.uniform(0.8, 1.8))
        cues.append(_cue(t, t + dur))
        shift = 0.0 if i < 30 else 45.0
        ref_intervals.append((t + shift, t + dur + shift))
        t += dur + gap

    total_s = t + 60.0
    reference = _add_background_noise(
        _reference(ref_intervals, total_seconds=total_s),
        total_seconds=total_s,
        density=0.42,
        seed=102,
    )
    # Re-apply true dialogue speech cleanly on top of background noise
    for a, b in ref_intervals:
        reference[int(round(a * SAMPLE_RATE)) : int(round(b * SAMPLE_RATE))] = 1.0

    offsets, _ = compute_auto_split_offsets(
        reference,
        cues,
        global_offset_samples=0.0,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=12.0 * SAMPLE_RATE,
        length_penalty=0.25,
        max_search_samples=300 * SAMPLE_RATE,
        log_segments=False,
    )
    assert _num_splits([o for o in offsets if o is not None]) == 1
    assert count_dropped_cues(offsets) == 0
    for i in range(30):
        assert abs(offsets[i] / SAMPLE_RATE - 0.0) < 0.25
    for i in range(30, 60):
        assert abs(offsets[i] / SAMPLE_RATE - 45.0) < 0.25


def test_auto_split_noisy_background_vad_unbroken_zero_splits():
    # Unbroken movie shifted by -7s with ~45% background VAD noise and one
    # slightly weaker middle scene must produce 0 splits.
    rng = np.random.default_rng(203)
    ref_intervals = []
    cues = []
    t = 12.0
    for i in range(60):
        dur = float(rng.uniform(1.1, 2.1))
        gap = float(rng.uniform(0.7, 1.7))
        cues.append(_cue(t + 7.0, t + 7.0 + dur))
        if 20 <= i < 28:
            ref_intervals.append((t, t + dur * 0.7))
        else:
            ref_intervals.append((t, t + dur))
        t += dur + gap

    total_s = t + 30.0
    reference = _add_background_noise(
        _reference(ref_intervals, total_seconds=total_s),
        total_seconds=total_s,
        density=0.40,
        seed=204,
    )
    for a, b in ref_intervals:
        reference[int(round(a * SAMPLE_RATE)) : int(round(b * SAMPLE_RATE))] = 1.0

    offsets, _ = compute_auto_split_offsets(
        reference,
        cues,
        global_offset_samples=-7.0 * SAMPLE_RATE,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=12.0 * SAMPLE_RATE,
        length_penalty=0.25,
        max_search_samples=300 * SAMPLE_RATE,
        log_segments=False,
    )
    assert count_dropped_cues(offsets) == 0
    assert _num_splits(offsets) == 0
    assert all(abs(o / SAMPLE_RATE - (-7.0)) < 0.1 for o in offsets)


def test_auto_split_deleted_scene_with_dialogue_option_b():
    # Extended-cut SRT on Theatrical-cut video:
    # 0..24: Scene 1 (0s offset)
    # 25..44: 2-minute deleted scene WITH 20 dialogue cues (absent from reference)
    # 45..74: Scene 2 (shifted -120s onto reference right after Scene 1)
    rng = np.random.default_rng(305)
    ref_intervals = []
    cues = []
    t = 2.0
    for _ in range(25):
        dur = float(rng.uniform(1.1, 1.8))
        gap = float(rng.uniform(0.6, 1.3))
        cues.append(_cue(t, t + dur))
        ref_intervals.append((t, t + dur))
        t += dur + gap

    scene1_end = t
    deleted_t = scene1_end + 1.0
    for idx in range(20):
        dur = float(rng.uniform(1.2, 2.0))
        gap = 4.2
        cues.append(_cue(deleted_t, deleted_t + dur, content="deleted_%d" % idx))
        deleted_t += dur + gap

    # Scene 2 in SRT starts ~120s after Scene 1 ends, while in reference it
    # plays right after Scene 1 (at scene1_end + 2.0).
    shift = 120.0
    ref_t = scene1_end + 2.0
    for idx in range(30):
        dur = float(rng.uniform(1.1, 1.9))
        gap = float(rng.uniform(0.6, 1.4))
        cues.append(_cue(ref_t + shift, ref_t + shift + dur, content="scene2_%d" % idx))
        ref_intervals.append((ref_t, ref_t + dur))
        ref_t += dur + gap

    reference = _reference(ref_intervals, total_seconds=ref_t + 30.0)
    offsets, _ = compute_auto_split_offsets(
        reference,
        cues,
        global_offset_samples=0.0,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=12.0 * SAMPLE_RATE,
        length_penalty=0.25,
        max_search_samples=300 * SAMPLE_RATE,
        log_segments=False,
    )
    dropped = count_dropped_cues(offsets)
    assert dropped >= 18
    for i in range(25):
        assert offsets[i] is not None and abs(offsets[i] / SAMPLE_RATE - 0.0) < 0.2
    for i in range(45, 75):
        assert offsets[i] is not None and abs(offsets[i] / SAMPLE_RATE - (-120.0)) < 0.2

    sub_file = GenericSubtitlesFile(cues, sub_format="srt", encoding="utf-8")
    shifter = VariableSubtitleShifter(
        [None if o is None else o / float(SAMPLE_RATE) for o in offsets]
    )
    out_subs = list(shifter.fit_transform(sub_file))
    assert all("deleted_" not in s.content for s in out_subs)
    for k in range(1, len(out_subs)):
        assert out_subs[k].start >= out_subs[k - 1].end


def test_auto_split_rejects_spurious_backward_jump_onto_dense_scene():
    # Dense Scene A followed by Scene B with a brief dropped-VAD patch must NOT
    # jump backwards onto Scene A.
    rng = np.random.default_rng(407)
    ref_intervals = []
    cues = []
    t = 5.0
    for i in range(50):
        dur = float(rng.uniform(1.2, 2.0))
        gap = 0.25 if i < 20 else float(rng.uniform(0.7, 1.5))
        cues.append(_cue(t, t + dur))
        if 30 <= i < 36:
            # 6 cues with 50% VAD drop
            ref_intervals.append((t, t + dur * 0.5))
        else:
            ref_intervals.append((t, t + dur))
        t += dur + gap

    reference = _reference(ref_intervals, total_seconds=t + 20.0)
    offsets, _ = compute_auto_split_offsets(
        reference,
        cues,
        global_offset_samples=0.0,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=12.0 * SAMPLE_RATE,
        length_penalty=0.25,
        max_search_samples=120 * SAMPLE_RATE,
        log_segments=False,
    )
    assert count_dropped_cues(offsets) == 0
    assert _num_splits(offsets) == 0


def test_auto_split_edge_negative_and_positive_offsets():
    # Local FFT must recover negative shifts near t=0 and positive shifts near end.
    rng = np.random.default_rng(509)
    # Part 1: cues start at t=10..40s, matching ref[0..15s] at -25s (partial edge overlap)
    cues_neg = []
    ref_neg_intervals = []
    t = 10.0
    while t < 38.0:
        dur = float(rng.uniform(1.0, 1.6))
        gap = float(rng.uniform(0.5, 1.1))
        cues_neg.append(_cue(t, t + dur))
        if t >= 25.0:
            ref_neg_intervals.append((t - 25.0, t - 25.0 + dur))
        t += dur + gap
    ref_neg = _reference(ref_neg_intervals, total_seconds=60.0)
    starts_neg = np.array([c.start.total_seconds() * SAMPLE_RATE for c in cues_neg])
    ends_neg = np.array([c.end.total_seconds() * SAMPLE_RATE for c in cues_neg])
    found_neg = _find_local_offset_fft(
        ref_neg, starts_neg, ends_neg, -300 * SAMPLE_RATE, 300 * SAMPLE_RATE
    )
    assert found_neg is not None
    assert abs(found_neg / SAMPLE_RATE - (-25.0)) < 0.15

    # Part 2: cues end around 40s, matching ref near end at +25s (partial tail overlap)
    cues_pos = []
    ref_pos_intervals = []
    t = 10.0
    while t < 38.0:
        dur = float(rng.uniform(1.0, 1.6))
        gap = float(rng.uniform(0.5, 1.1))
        cues_pos.append(_cue(t, t + dur))
        if t + 25.0 + dur <= 50.0:
            ref_pos_intervals.append((t + 25.0, t + 25.0 + dur))
        t += dur + gap
    ref_pos = _reference(ref_pos_intervals, total_seconds=50.0)
    starts_pos = np.array([c.start.total_seconds() * SAMPLE_RATE for c in cues_pos])
    ends_pos = np.array([c.end.total_seconds() * SAMPLE_RATE for c in cues_pos])
    found_pos = _find_local_offset_fft(
        ref_pos, starts_pos, ends_pos, -300 * SAMPLE_RATE, 300 * SAMPLE_RATE
    )
    assert found_pos is not None
    assert abs(found_pos / SAMPLE_RATE - 25.0) < 0.15


def test_auto_split_moderate_half_second_break():
    # A 0.5s cut across 30+ cues is clearly visible while watching and must be fixed.
    rng = np.random.default_rng(611)
    ref_intervals = []
    cues = []
    t = 2.0
    for i in range(64):
        dur = float(rng.uniform(1.2, 2.0))
        gap = float(rng.uniform(0.6, 1.4))
        cues.append(_cue(t, t + dur))
        shift = 0.0 if i < 32 else 0.5
        ref_intervals.append((t + shift, t + dur + shift))
        t += dur + gap

    reference = _reference(ref_intervals, total_seconds=t + 10.0)
    offsets, _ = compute_auto_split_offsets(
        reference,
        cues,
        global_offset_samples=0.0,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=12.0 * SAMPLE_RATE,
        length_penalty=0.25,
        max_search_samples=300 * SAMPLE_RATE,
        log_segments=False,
    )
    assert _num_splits(offsets) == 1
    for i in range(32):
        assert abs(offsets[i] / SAMPLE_RATE - 0.0) < 0.05
    for i in range(32, 64):
        assert abs(offsets[i] / SAMPLE_RATE - 0.5) < 0.05


def test_auto_split_short_cold_open():
    # 5-cue cold open shifted +45s before a 45-cue main show at 0s.
    rng = np.random.default_rng(713)
    ref_intervals = []
    cues = []
    t = 2.0
    for i in range(50):
        dur = float(rng.uniform(1.3, 1.9))
        gap = float(rng.uniform(0.5, 1.0))
        if i < 5:
            cues.append(_cue(t, t + dur))
            ref_intervals.append((t + 45.0, t + dur + 45.0))
            if i == 4:
                t += 55.0
        else:
            cues.append(_cue(t, t + dur))
            ref_intervals.append((t, t + dur))
        t += dur + gap

    reference = _reference(ref_intervals, total_seconds=t + 20.0)
    offsets, _ = compute_auto_split_offsets(
        reference,
        cues,
        global_offset_samples=0.0,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=12.0 * SAMPLE_RATE,
        length_penalty=0.25,
        max_search_samples=300 * SAMPLE_RATE,
        log_segments=False,
    )
    assert _num_splits(offsets) == 1
    for i in range(5):
        assert abs(offsets[i] / SAMPLE_RATE - 45.0) < 0.1
    for i in range(5, 50):
        assert abs(offsets[i] / SAMPLE_RATE - 0.0) < 0.1


def _write_srt_from_cues(path, cues):
    lines = []
    for idx, sub in enumerate(cues, 1):
        sub_copy = srt.Subtitle(
            index=idx, start=sub.start, end=sub.end, content=sub.content
        )
        lines.append(sub_copy.to_srt())
    path.write_text("".join(lines), encoding="utf-8")


def test_run_auto_split_large_global_offset_succeeds(tmp_path):
    rng = np.random.default_rng(815)
    ref_cues = []
    in_cues = []
    t = 100.0
    for i in range(40):
        dur = float(rng.uniform(1.1, 1.9))
        gap = float(rng.uniform(0.6, 1.3))
        ref_cues.append(_cue(t, t + dur, content="line %d" % i))
        in_cues.append(_cue(t - 95.0, t - 95.0 + dur, content="line %d" % i))
        t += dur + gap

    ref_path = tmp_path / "ref.srt"
    in_path = tmp_path / "in.srt"
    out_path = tmp_path / "out.srt"
    _write_srt_from_cues(ref_path, ref_cues)
    _write_srt_from_cues(in_path, in_cues)

    parser = ffsubsync.make_parser()
    args = parser.parse_args(
        [str(ref_path), "-i", str(in_path), "-o", str(out_path), "--auto-split"]
    )
    res = ffsubsync.run(args)
    assert res["sync_was_successful"] is True
    assert abs(res["offset_seconds"] - 95.0) < 0.25


def test_run_auto_split_framerate_mismatch_and_break(tmp_path):
    rng = np.random.default_rng(917)
    scale = 25.0 / 23.976
    ref_cues = []
    in_cues = []
    t = 20.0
    for i in range(60):
        dur = float(rng.uniform(1.2, 2.0))
        gap = float(rng.uniform(0.7, 1.6))
        ref_start = t if i < 30 else t + 30.0
        ref_cues.append(_cue(ref_start, ref_start + dur, content="line %d" % i))
        in_cues.append(_cue(t / scale, (t + dur) / scale, content="line %d" % i))
        t += dur + gap

    ref_path = tmp_path / "ref.srt"
    in_path = tmp_path / "in.srt"
    out_path = tmp_path / "out.srt"
    _write_srt_from_cues(ref_path, ref_cues)
    _write_srt_from_cues(in_path, in_cues)

    parser = ffsubsync.make_parser()
    args = parser.parse_args(
        [str(ref_path), "-i", str(in_path), "-o", str(out_path), "--auto-split"]
    )
    res = ffsubsync.run(args)
    assert res["sync_was_successful"] is True
    assert abs(res["framerate_scale_factor"] - scale) < 0.01
    assert res["split_segments"] is not None and len(res["split_segments"]) == 2
    synced = list(srt.parse(out_path.read_text(encoding="utf-8")))
    max_err = max(
        abs(a.start.total_seconds() - b.start.total_seconds())
        for a, b in zip(ref_cues, synced)
    )
    assert max_err < 0.25


def test_run_uppercase_srt_and_vtt_reference(tmp_path):
    vtt_lines = ["WEBVTT", ""]
    in_cues = []
    t = 10.0
    for i in range(25):
        dur = 1.4
        s_str = "%02d:%02d:%06.3f" % (int(t // 3600), int((t % 3600) // 60), t % 60)
        e_t = t + dur
        e_str = "%02d:%02d:%06.3f" % (
            int(e_t // 3600),
            int((e_t % 3600) // 60),
            e_t % 60,
        )
        vtt_lines.append("%s --> %s" % (s_str, e_str))
        vtt_lines.append("Line %d" % i)
        vtt_lines.append("")
        in_cues.append(_cue(t - 4.0, t - 4.0 + dur, content="Line %d" % i))
        t += dur + 1.0

    ref_path = tmp_path / "ref.VTT"
    in_path = tmp_path / "in.SRT"
    out_path = tmp_path / "out.SRT"
    ref_path.write_text("\n".join(vtt_lines), encoding="utf-8")
    _write_srt_from_cues(in_path, in_cues)

    parser = ffsubsync.make_parser()
    args = parser.parse_args(
        [str(ref_path), "-i", str(in_path), "-o", str(out_path), "--auto-split"]
    )
    res = ffsubsync.run(args)
    assert res["sync_was_successful"] is True
    assert out_path.is_file()
    assert abs(res["offset_seconds"] - 4.0) < 0.25


def test_auto_split_sparse_vad_unbroken_preserves_global_offset():
    # Sparse ~15% WebRTC VAD (like Lord of War): only ~25% of dialogue cues have
    # VAD hits plus ~12% random background hits, uniform +7.0s offset.
    rng = np.random.default_rng(1201)
    ref_intervals = []
    cues = []
    t = 15.0
    for _ in range(200):
        dur = float(rng.uniform(1.2, 2.2))
        gap = float(rng.uniform(1.2, 3.0))
        cues.append(_cue(t, t + dur))
        if rng.random() < 0.25:
            ref_intervals.append((t + 7.0, t + 7.0 + dur * float(rng.uniform(0.6, 1.0))))
        t += dur + gap

    total_s = t + 30.0
    reference = _add_background_noise(
        _reference(ref_intervals, total_seconds=total_s),
        total_seconds=total_s,
        density=0.12,
        seed=1202,
    )
    for a, b in ref_intervals:
        reference[int(round(a * SAMPLE_RATE)) : int(round(b * SAMPLE_RATE))] = 1.0

    offsets, _ = compute_auto_split_offsets(
        reference,
        cues,
        global_offset_samples=7.0 * SAMPLE_RATE,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=12.0 * SAMPLE_RATE,
        length_penalty=0.25,
        max_search_samples=300 * SAMPLE_RATE,
        log_segments=False,
    )
    assert count_dropped_cues(offsets) == 0
    assert _num_splits(offsets) == 0
    assert all(abs(o / SAMPLE_RATE - 7.0) < 0.1 for o in offsets)


def test_auto_split_sparse_vad_recovers_real_break():
    # Sparse ~20% VAD (dialogue hit rate ~45%, background ~12%) with a +45s mid-file
    # break: must still detect and align the break cleanly.
    rng = np.random.default_rng(1301)
    ref_intervals = []
    cues = []
    t = 15.0
    for i in range(200):
        dur = float(rng.uniform(1.2, 2.2))
        gap = float(rng.uniform(1.2, 2.8))
        cues.append(_cue(t, t + dur))
        shift = 0.0 if i < 100 else 45.0
        if rng.random() < 0.45:
            ref_intervals.append((t + shift, t + shift + dur * float(rng.uniform(0.7, 1.0))))
        t += dur + gap

    total_s = t + 65.0
    reference = _add_background_noise(
        _reference(ref_intervals, total_seconds=total_s),
        total_seconds=total_s,
        density=0.12,
        seed=1302,
    )
    for a, b in ref_intervals:
        reference[int(round(a * SAMPLE_RATE)) : int(round(b * SAMPLE_RATE))] = 1.0

    offsets, _ = compute_auto_split_offsets(
        reference,
        cues,
        global_offset_samples=0.0,
        sample_rate=SAMPLE_RATE,
        start_seconds=0,
        split_penalty=12.0 * SAMPLE_RATE,
        length_penalty=0.25,
        max_search_samples=300 * SAMPLE_RATE,
        log_segments=False,
    )
    assert count_dropped_cues(offsets) == 0
    assert _num_splits(offsets) == 1
    for i in range(95):
        assert abs(offsets[i] / SAMPLE_RATE - 0.0) < 0.5
    for i in range(105, 200):
        assert abs(offsets[i] / SAMPLE_RATE - 45.0) < 0.5
