# -*- coding: utf-8 -*-
"""Adaptive block-level alignment profile refinement.

Standard symmetric cross-correlation (FFT / split alignment) places wider
subtitle cues roughly *centered* over narrower VAD speech bursts, leaving half
of the extra reading duration before speech starts and half after it ends.

This module computes a single block-level fine offset adjustment (no per-line
warping) from the already-computed 100 Hz reference speech signal:

- ``balanced``: no adjustment (exact existing symmetric behavior).
- ``onset``:    aligns cue starts with the measured speech onset so subtitles
                appear right when dialogue begins, leaving extra reading time
                at the end of each cue.
- ``early``:    places cue starts slightly ahead of speech onset for early
                reading lead-in.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ffsubsync.constants import (
    ALIGNMENT_PROFILE_CHOICES,
    DEFAULT_ALIGNMENT_PROFILE,
    SAMPLE_RATE,
)
from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.split_aligner import (
    _build_padded_cumsum,
    _cue_sample_bounds,
    _interp_cumsum,
)

# Leading silence gap required to treat a cue as an isolated utterance start.
_MIN_GAP_SECONDS: float = 0.25
_RELAXED_GAP_SECONDS: float = 0.10
_MIN_ISOLATED_CUES: int = 8
_MIN_TOTAL_SPEECH_CUES: int = 5

# Local search and Haar step-edge window for measuring speech onset.
_SEARCH_WINDOW_SECONDS: float = 1.00
_STEP_WINDOW_SECONDS: float = 0.20
_MIN_STEP_CONTRAST: float = 0.12

# Safe bounds and perceptual targets (in seconds).
_ONSET_MIN_DELTA_SECONDS: float = -0.50
_ONSET_MAX_DELTA_SECONDS: float = 0.85

_EARLY_LEAD_SECONDS: float = 0.25
_EARLY_EXTRA_STEP_SECONDS: float = 0.15
_EARLY_MAX_LEAD_SECONDS: float = 0.50
_EARLY_MIN_DELTA_SECONDS: float = -0.50
_EARLY_FALLBACK_SHIFT_SECONDS: float = -0.15

# Minimum fraction of base speech overlap that must be preserved after shifting.
_ONSET_MIN_OVERLAP_RETENTION: float = 0.88
_EARLY_MIN_OVERLAP_RETENTION: float = 0.78


def normalize_alignment_profile(profile: Optional[str]) -> str:
    """Return a valid alignment profile token, defaulting to ``balanced``."""
    cleaned = str(profile or "").strip().lower()
    if cleaned in ALIGNMENT_PROFILE_CHOICES:
        return cleaned
    return DEFAULT_ALIGNMENT_PROFILE


def _prepare_kept_speech_bounds(
    cues: Sequence[GenericSubtitle],
    base_offsets_samples: Union[float, Sequence[Optional[float]]],
    *,
    ref_len: int,
    sample_rate: int,
    start_seconds: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return base-aligned ``(starts, ends)`` in samples for kept speech cues."""
    cue_list = list(cues)
    n = len(cue_list)
    if n == 0 or ref_len <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    starts, ends, is_speech = _cue_sample_bounds(cue_list, sample_rate, start_seconds)
    if isinstance(base_offsets_samples, (int, float, np.number)):
        offsets = np.full(n, float(base_offsets_samples), dtype=np.float64)
        kept = np.ones(n, dtype=bool)
    else:
        if len(base_offsets_samples) != n:
            raise ValueError(
                "expected one offset per cue (got %d offsets for %d cues)"
                % (len(base_offsets_samples), n)
            )
        offsets = np.zeros(n, dtype=np.float64)
        kept = np.zeros(n, dtype=bool)
        for i, off in enumerate(base_offsets_samples):
            if off is not None:
                offsets[i] = float(off)
                kept[i] = True

    aligned_starts = starts + offsets
    aligned_ends = ends + offsets
    durs = aligned_ends - aligned_starts
    mask = (
        kept
        & is_speech
        & (durs > 0.0)
        & (aligned_ends > 0.0)
        & (aligned_starts < float(ref_len))
    )
    return aligned_starts[mask], aligned_ends[mask]


def _select_isolated_starts(
    aligned_starts: np.ndarray,
    aligned_ends: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Pick cue starts preceded by a short silence gap (clean utterance starts)."""
    m = len(aligned_starts)
    if m == 0:
        return np.zeros(0, dtype=np.float64)

    order = np.argsort(aligned_starts, kind="mergesort")
    s_sorted = aligned_starts[order]
    e_sorted = aligned_ends[order]

    # Running maximum of prior cue ends handles any overlapping cues safely.
    prev_max_end = np.empty(m, dtype=np.float64)
    prev_max_end[0] = -np.inf
    if m > 1:
        prev_max_end[1:] = np.maximum.accumulate(e_sorted[:-1])
    gaps = s_sorted - prev_max_end

    for min_gap_s in (_MIN_GAP_SECONDS, _RELAXED_GAP_SECONDS):
        iso = s_sorted[gaps >= min_gap_s * float(sample_rate)]
        if len(iso) >= _MIN_ISOLATED_CUES:
            return iso

    if m >= _MIN_TOTAL_SPEECH_CUES:
        return s_sorted
    return np.zeros(0, dtype=np.float64)


def _estimate_onset_from_bounds(
    ref_bin: np.ndarray,
    aligned_starts: np.ndarray,
    aligned_ends: np.ndarray,
    sample_rate: int,
) -> Tuple[Optional[float], float]:
    """Internal step-edge onset estimator from precomputed speech bounds."""
    ref_len = len(ref_bin)
    iso_starts = _select_isolated_starts(aligned_starts, aligned_ends, sample_rate)
    if len(iso_starts) < _MIN_TOTAL_SPEECH_CUES:
        return None, 0.0

    search_r = max(1, int(round(_SEARCH_WINDOW_SECONDS * sample_rate)))
    step_w = max(1, int(round(_STEP_WINDOW_SECONDS * sample_rate)))
    eval_r = search_r + step_w

    lags = np.arange(-eval_r, eval_r + 1, dtype=np.int64)
    centers = np.rint(iso_starts).astype(np.int64)
    idx = centers[:, None] + lags[None, :]
    valid = (idx >= 0) & (idx < ref_len)
    counts = np.sum(valid, axis=0)
    if np.any(counts == 0):
        return None, 0.0

    safe_idx = np.clip(idx, 0, ref_len - 1)
    profile_curve = np.sum(np.where(valid, ref_bin[safe_idx], 0.0), axis=0) / counts.astype(
        np.float64
    )

    cs = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(profile_curve)])
    center_pos = np.arange(step_w, step_w + 2 * search_r + 1, dtype=np.int64)
    before_mean = (cs[center_pos] - cs[center_pos - step_w]) / float(step_w)
    after_mean = (cs[center_pos + step_w] - cs[center_pos]) / float(step_w)
    step_scores = after_mean - before_mean

    best_idx = int(np.argmax(step_scores))
    max_step = float(step_scores[best_idx])
    if max_step < _MIN_STEP_CONTRAST:
        return None, max_step

    # Refine within the connected near-peak cap (>= 98.5% of max_step) so broad
    # transitions don't jitter while sharp edges stay exact.
    thresh = max_step * 0.985
    lo = best_idx
    while lo > 0 and step_scores[lo - 1] >= thresh:
        lo -= 1
    hi = best_idx
    while hi + 1 < len(step_scores) and step_scores[hi + 1] >= thresh:
        hi += 1

    cand_lags = np.arange(-search_r + lo, -search_r + hi + 1, dtype=np.float64)
    weights = np.maximum(step_scores[lo : hi + 1] - thresh + 1e-9, 0.0)
    tau_samples = float(np.sum(cand_lags * weights) / np.sum(weights))
    return tau_samples, max_step


def estimate_onset_offset_samples(
    reference_speech: Optional[np.ndarray],
    cues: Sequence[GenericSubtitle],
    base_offsets_samples: Union[float, Sequence[Optional[float]]] = 0.0,
    *,
    sample_rate: int = SAMPLE_RATE,
    start_seconds: float = 0.0,
) -> Tuple[Optional[float], float]:
    """Estimate the shift (in samples) from base cue starts to speech onset.

    A positive return value means speech begins *after* the base-aligned cue
    starts (i.e. cues currently appear early and shifting later by ``+tau``
    places their starts right at speech onset).
    """
    if reference_speech is None or len(reference_speech) == 0:
        return None, 0.0

    ref_bin = (np.asarray(reference_speech, dtype=np.float64) > 0.5).astype(np.float64)
    aligned_starts, aligned_ends = _prepare_kept_speech_bounds(
        cues,
        base_offsets_samples,
        ref_len=len(ref_bin),
        sample_rate=sample_rate,
        start_seconds=start_seconds,
    )
    return _estimate_onset_from_bounds(ref_bin, aligned_starts, aligned_ends, sample_rate)


def _clamp_by_overlap_retention(
    reference_speech: np.ndarray,
    aligned_starts: np.ndarray,
    aligned_ends: np.ndarray,
    target_delta: float,
    min_retention: float,
) -> float:
    """Shrink ``target_delta`` toward 0 if needed to keep speech overlap healthy."""
    if abs(target_delta) < 1e-6 or len(aligned_starts) == 0:
        return 0.0

    pad = int(np.ceil(abs(target_delta))) + 4
    _, padded = _build_padded_cumsum(reference_speech, pad)

    def _overlap(delta: float) -> float:
        cs = _interp_cumsum(padded, aligned_starts + delta + pad)
        ce = _interp_cumsum(padded, aligned_ends + delta + pad)
        return float(np.sum(ce - cs))

    base_ov = _overlap(0.0)
    if base_ov <= 0.0:
        return 0.0

    min_ov = min_retention * base_ov
    if _overlap(target_delta) >= min_ov:
        return float(target_delta)

    lo = 0.0
    hi = 1.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if _overlap(mid * target_delta) >= min_ov:
            lo = mid
        else:
            hi = mid
    return float(lo * target_delta)


def compute_profile_offset_adjustment(
    reference_speech: Optional[np.ndarray],
    cues: Sequence[GenericSubtitle],
    base_offsets_samples: Union[float, Sequence[Optional[float]]] = 0.0,
    *,
    profile: str = DEFAULT_ALIGNMENT_PROFILE,
    sample_rate: int = SAMPLE_RATE,
    start_seconds: float = 0.0,
) -> float:
    """Compute a global fine shift (in samples) for ``profile``.

    Add the returned value uniformly to the base offset(s) of the block.
    """
    norm_profile = normalize_alignment_profile(profile)
    if norm_profile == "balanced":
        return 0.0
    if reference_speech is None or len(reference_speech) == 0:
        return 0.0

    ref_arr = np.asarray(reference_speech, dtype=np.float64)
    ref_bin = (ref_arr > 0.5).astype(np.float64)
    aligned_starts, aligned_ends = _prepare_kept_speech_bounds(
        cues,
        base_offsets_samples,
        ref_len=len(ref_bin),
        sample_rate=sample_rate,
        start_seconds=start_seconds,
    )
    if len(aligned_starts) < _MIN_TOTAL_SPEECH_CUES:
        return 0.0

    tau_samples, _ = _estimate_onset_from_bounds(
        ref_bin,
        aligned_starts,
        aligned_ends,
        sample_rate,
    )

    sr = float(sample_rate)
    if norm_profile == "onset":
        if tau_samples is None:
            return 0.0
        target_s = float(
            np.clip(
                tau_samples / sr,
                _ONSET_MIN_DELTA_SECONDS,
                _ONSET_MAX_DELTA_SECONDS,
            )
        )
        min_retention = _ONSET_MIN_OVERLAP_RETENTION
    else:  # "early"
        if tau_samples is None:
            target_s = _EARLY_FALLBACK_SHIFT_SECONDS
        else:
            tau_s = tau_samples / sr
            # Aim for ~250 ms before speech onset, while ensuring "early" is
            # never later than balanced (<= 0) and capped at ~500 ms lead-in.
            raw_s = min(
                tau_s - _EARLY_LEAD_SECONDS,
                max(tau_s - _EARLY_MAX_LEAD_SECONDS, -_EARLY_EXTRA_STEP_SECONDS),
            )
            target_s = float(np.clip(raw_s, _EARLY_MIN_DELTA_SECONDS, 0.0))
        min_retention = _EARLY_MIN_OVERLAP_RETENTION

    target_samples = target_s * sr
    return _clamp_by_overlap_retention(
        ref_arr,
        aligned_starts,
        aligned_ends,
        target_samples,
        min_retention,
    )
