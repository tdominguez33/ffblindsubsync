# -*- coding: utf-8 -*-
"""alass-style split-penalty alignment.

ffsubsync's default aligner (:class:`ffsubsync.aligners.FFTAligner`) emits a single
global offset for the whole file. That cannot correct a subtitle whose required
offset *changes* partway through -- e.g. a commercial break, an inserted/removed
scene ("director's cut"), or two discs concatenated into one file.

This module implements the core idea from `alass <https://github.com/kaegi/alass>`_:
let every cue take its own offset, but charge a *split penalty* every time two
consecutive cues are assigned different offsets. We maximize::

    sum_i rating(cue_i @ offset_i)  -  split_penalty * (number of splits)

via dynamic programming, yielding a piecewise-constant offset function subject to a
chronological non-overlapping constraint (a split can never pull a later cue backwards
across an earlier cue). With a large penalty the optimum collapses to a single global
offset (i.e. the default behavior).

It also provides :func:`compute_auto_split_offsets`, a safe two-stage ("auto-split")
strategy designed to be usable by default: it checks whether the global offset keeps
consistent speech overlap across the whole timeline, and only searches for + validates
splits when genuine misaligned regions exist.
"""
import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.speech_transformers import _is_metadata

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


# Upper bound (in samples) on the length-penalty guard band. The guard is normally
# the cue's own duration (a scale-free "is this block about my size?" probe), but a
# very long cue would otherwise reach across neighbouring dialogue and penalize a
# perfectly good placement, so we clamp it here (2s at 100Hz).
_MAX_GUARD_SAMPLES: int = 200
_EDGE_STRONG_MIN_SHIFT_SECONDS: float = 2.0


def _cue_sample_bounds(
    cues: List[GenericSubtitle], sample_rate: int, start_seconds: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-cue ``(start_sample, end_sample, is_speech)`` arrays.

    Bounds are kept as floats (no rounding to whole samples) so the sub-sample
    offset grid can exploit their true millisecond timing. The ``_is_metadata`` gate
    matches :class:`SubtitleSpeechTransformer` so non-dialogue cues (blank lines,
    bracketed sound cues, music notes, ...) are marked non-speech and contribute no
    rating -- the DP then leaves them on a neighbour's offset for free.
    """
    n = len(cues)
    starts = np.zeros(n, dtype=np.float64)
    ends = np.zeros(n, dtype=np.float64)
    is_speech = np.zeros(n, dtype=bool)
    for i, sub in enumerate(cues):
        start = (sub.start.total_seconds() - start_seconds) * sample_rate
        duration = (sub.end.total_seconds() - sub.start.total_seconds()) * sample_rate
        starts[i] = start
        ends[i] = start + duration
        is_speech[i] = not _is_metadata(sub.content, i == 0 or i + 1 == n)
    return starts, ends, is_speech


def _build_padded_cumsum(
    reference: np.ndarray, pad: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return binary ``ref`` and its zero/total-padded prefix sum for O(1) overlap."""
    ref = (np.asarray(reference, dtype=np.float64) > 0).astype(np.float64)
    cumsum = np.concatenate([np.zeros(1), np.cumsum(ref)])
    padded = np.concatenate([np.zeros(pad), cumsum, np.full(pad, cumsum[-1])])
    return ref, padded


def _interp_cumsum(padded: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Exact integral of the 0/1 reference up to fractional positions ``pos``."""
    max_pos = len(padded) - 1
    clipped = np.clip(pos, 0.0, float(max_pos))
    lo = np.floor(clipped).astype(np.int64)
    np.clip(lo, 0, max_pos - 1, out=lo)
    frac = clipped - lo
    return padded[lo] * (1.0 - frac) + padded[lo + 1] * frac


def _cue_scores_at_offset(
    starts: np.ndarray,
    ends: np.ndarray,
    guards: np.ndarray,
    is_speech: np.ndarray,
    padded: np.ndarray,
    pad: int,
    offset: float,
    length_penalty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized per-cue ``(overlap, rating)`` evaluated at a single ``offset``."""
    start_pos = starts + offset + pad
    end_pos = ends + offset + pad
    cs = _interp_cumsum(padded, start_pos)
    ce = _interp_cumsum(padded, end_pos)
    overlap = np.where(is_speech, ce - cs, 0.0)
    if length_penalty:
        gl = cs - _interp_cumsum(padded, start_pos - guards)
        gr = _interp_cumsum(padded, end_pos + guards) - ce
        rating = np.where(
            is_speech & (guards > 0), overlap - length_penalty * (gl + gr), overlap
        )
    else:
        rating = overlap
    return overlap, rating


def compute_split_offsets(
    reference: np.ndarray,
    cues: List[GenericSubtitle],
    *,
    sample_rate: int,
    start_seconds: float,
    split_penalty: float,
    max_offset_samples: int,
    length_penalty: float = 0.0,
    offset_step_samples: float = 1.0,
    candidate_offsets: Optional[np.ndarray] = None,
    enforce_monotonic: bool = True,
    return_score: bool = False,
    log_segments: bool = True,
) -> Union[List[float], Tuple[List[float], float]]:
    """Return one offset (in samples) per cue, aligned 1:1 with ``cues``.

    Parameters
    ----------
    reference:
        The reference speech signal (1.0 where speech is present), as produced by a
        reference pipeline at ``sample_rate`` Hz.
    cues:
        The (already framerate-scaled) subtitle cues to align.
    split_penalty:
        Cost, in *overlap samples*, charged each time consecutive cues take different
        offsets. Larger => fewer splits; very large => a single global offset.
    max_offset_samples:
        Half-width of the candidate offset grid when ``candidate_offsets`` is omitted;
        offsets range over ``[-max_offset_samples, +max_offset_samples]``.
    length_penalty:
        Weight of the edge/length term ("standard scoring"). 0 => pure overlap.
    offset_step_samples:
        Spacing of the candidate offset grid, in samples. 1.0 => whole samples
        (default); values < 1 give sub-sample precision via linear interpolation.
    candidate_offsets:
        Optional explicit sorted 1D array of candidate offsets (in samples). When
        provided, overrides the uniform ``[-max_offset_samples, +max_offset_samples]``
        grid.
    enforce_monotonic:
        If True (default), enforce ``o_i >= o_{i-1} - max(0, start_i - end_{i-1})``
        on split transitions so a cue can never jump backwards over earlier cues and
        interleave separate scenes.
    return_score:
        If True, also return the maximized objective value (used to compare
        alignments across candidate framerate scales).
    log_segments:
        If True, log a human-readable summary of the resulting segments.
    """
    n = len(cues)
    if n == 0:
        return ([], 0.0) if return_score else []

    starts, ends, is_speech = _cue_sample_bounds(cues, sample_rate, start_seconds)
    guards = np.minimum(np.maximum(ends - starts, 0.0), float(_MAX_GUARD_SAMPLES))

    if candidate_offsets is not None:
        offsets = np.sort(np.unique(np.asarray(candidate_offsets, dtype=np.float64)))
        if len(offsets) == 0:
            raise ValueError("candidate_offsets must not be empty")
        n_off = len(offsets)
        max_abs_off = int(np.ceil(np.max(np.abs(offsets))))
    else:
        step = float(offset_step_samples)
        if step <= 0:
            step = 1.0
        n_off = int(round(2 * max_offset_samples / step)) + 1
        offsets = -float(max_offset_samples) + step * np.arange(n_off)
        max_abs_off = int(abs(max_offset_samples))

    pad = max_abs_off + _MAX_GUARD_SAMPLES + 2
    _, padded = _build_padded_cumsum(reference, pad)

    def _rating_row(i: int) -> np.ndarray:
        if not is_speech[i]:
            return np.zeros(n_off)
        start_pos = starts[i] + offsets + pad
        end_pos = ends[i] + offsets + pad
        cs = _interp_cumsum(padded, start_pos)
        ce = _interp_cumsum(padded, end_pos)
        overlap = ce - cs
        if length_penalty and guards[i] > 0:
            gl = cs - _interp_cumsum(padded, start_pos - guards[i])
            gr = _interp_cumsum(padded, end_pos + guards[i]) - ce
            return overlap - length_penalty * (gl + gr)
        return overlap

    # Forward DP. dp[o] = best total rating for cues[0..i] with cue i at offset o.
    # back[i, o] records whether cue i kept the previous cue's offset (-1) or jumped,
    # in which case it records the predecessor offset index.
    dp = _rating_row(0)
    back = np.full((n, n_off), -1, dtype=np.int32)
    off_indices = np.arange(n_off, dtype=np.int32)
    for i in range(1, n):
        keep_value = dp
        if enforce_monotonic:
            # A split may pull cue i earlier by at most the silence gap between
            # end_{i-1} and start_i, preventing shifted cues from crossing over
            # earlier cues. Running prefix max keeps this O(n_off) per cue.
            gap_samples = max(0.0, float(starts[i] - ends[i - 1]))
            pref_val = np.maximum.accumulate(dp)
            is_new_best = np.empty(n_off, dtype=bool)
            is_new_best[0] = True
            is_new_best[1:] = pref_val[1:] > pref_val[:-1]
            pref_idx = np.maximum.accumulate(
                np.where(is_new_best, off_indices, np.int32(-1))
            )
            reach = np.searchsorted(offsets, offsets + gap_samples + 1e-9, side="right") - 1
            jump_value = pref_val[reach] - split_penalty
            best_prev_for_k = pref_idx[reach]
            jumped = jump_value > keep_value
            dp = _rating_row(i) + np.where(jumped, jump_value, keep_value)
            back[i] = np.where(jumped, best_prev_for_k, np.int32(-1))
        else:
            best_prev_idx = int(np.argmax(dp))
            jump_value = dp[best_prev_idx] - split_penalty
            jumped = jump_value > keep_value
            dp = _rating_row(i) + np.where(jumped, jump_value, keep_value)
            back[i] = np.where(jumped, np.int32(best_prev_idx), np.int32(-1))

    best_total = float(dp.max())

    # Backtrack.
    offset_idx = np.empty(n, dtype=np.int32)
    cur = int(np.argmax(dp))
    for i in range(n - 1, -1, -1):
        offset_idx[i] = cur
        if i > 0:
            b = int(back[i, cur])
            cur = b if b >= 0 else cur

    result = [float(offsets[idx]) for idx in offset_idx]
    if log_segments:
        log_split_segments(result, sample_rate)
    if return_score:
        return result, best_total
    return result


def count_dropped_cues(offsets: List[Optional[float]]) -> int:
    """Return the number of orphan cues marked as dropped (``None``)."""
    return sum(1 for off in offsets if off is None)


def summarize_split_segments(
    offsets: List[Optional[float]], sample_rate: int
) -> List[Tuple[int, float]]:
    """Collapse consecutive equal non-None offsets into ``[(cue_count, offset_seconds), ...]``."""
    if not offsets:
        return []
    segments: List[Tuple[int, float]] = []
    for off in offsets:
        if off is None:
            continue
        off_s = off / float(sample_rate)
        if segments and abs(segments[-1][1] - off_s) < 1e-9:
            segments[-1] = (segments[-1][0] + 1, segments[-1][1])
        else:
            segments.append((1, off_s))
    return segments


def log_split_segments(
    offsets: List[Optional[float]], sample_rate: int
) -> None:
    """Emit a human-readable summary of the piecewise offset function."""
    segments = summarize_split_segments(offsets, sample_rate)
    dropped = count_dropped_cues(offsets)
    if not segments and dropped == 0:
        return
    n_splits = max(0, len(segments) - 1)
    if dropped > 0:
        logger.info(
            "split alignment: %d segment(s), %d split(s), %d orphan cue(s) removed",
            len(segments),
            n_splits,
            dropped,
        )
    else:
        logger.info("split alignment: %d segment(s), %d split(s)", len(segments), n_splits)
    for count, off_s in segments:
        logger.info("  %d cue(s) offset %.3fs", count, off_s)


def _find_local_offset_fft(
    ref: np.ndarray,
    sub_starts: np.ndarray,
    sub_ends: np.ndarray,
    min_offset_samples: int,
    max_offset_samples: int,
) -> Optional[int]:
    """Find the best density-centered correlation offset for a local cue chunk via FFT."""
    if len(sub_starts) == 0 or len(ref) == 0 or max_offset_samples < min_offset_samples:
        return None
    sub_lo = int(np.floor(np.min(sub_starts)))
    sub_hi = int(np.ceil(np.max(sub_ends)))
    if sub_hi <= sub_lo:
        return None

    s = np.zeros(sub_hi - sub_lo, dtype=np.float64)
    for a, b in zip(sub_starts, sub_ends):
        ia = max(0, int(round(a)) - sub_lo)
        ib = min(len(s), int(round(b)) - sub_lo)
        if ib > ia:
            g_lo = max(0, ia - _MAX_GUARD_SAMPLES)
            g_hi = min(len(s), ib + _MAX_GUARD_SAMPLES)
            s[g_lo:g_hi] = -1.0
    for a, b in zip(sub_starts, sub_ends):
        ia = max(0, int(round(a)) - sub_lo)
        ib = min(len(s), int(round(b)) - sub_lo)
        if ib > ia:
            s[ia:ib] = 1.0
    if not np.any(s > 0):
        return None

    win_lo = sub_lo + min_offset_samples
    win_hi = sub_hi + max_offset_samples
    copy_lo = max(0, win_lo)
    copy_hi = min(len(ref), win_hi)
    if copy_hi <= copy_lo:
        return None

    local_p = float(np.clip(np.mean(ref[copy_lo:copy_hi]), 0.10, 0.90))
    r = np.zeros(win_hi - win_lo, dtype=np.float64)
    r[copy_lo - win_lo : copy_hi - win_lo] = ref[copy_lo:copy_hi] - local_p

    n_conv = 1 << int(np.ceil(np.log2(len(r) + len(s))))
    corr = np.fft.irfft(
        np.fft.rfft(r, n=n_conv) * np.conj(np.fft.rfft(s, n=n_conv)), n=n_conv
    )

    max_lag = max_offset_samples - min_offset_samples
    lags = np.arange(max_lag + 1, dtype=np.int64)
    global_offs = min_offset_samples + lags
    shifted_lo = sub_lo + global_offs
    shifted_hi = sub_hi + global_offs
    valid = (shifted_hi > 0) & (shifted_lo < len(ref))
    if not np.any(valid):
        return None
    valid_corr = np.where(valid, corr[: max_lag + 1], -np.inf)
    best_lag = int(np.argmax(valid_corr))
    if valid_corr[best_lag] <= 0.0:
        return None
    return int(min_offset_samples + best_lag)


def _build_candidate_offsets(
    centers: List[float],
    min_off: float,
    max_off: float,
    radius_samples: float,
    step: float,
) -> np.ndarray:
    """Build a sorted, fine-grained 1D array around a sparse list of center offsets."""
    # Deduplicate centers that lie within 0.5 * radius of an existing center so
    # overlapping neighborhoods don't do redundant work.
    unique_centers: List[float] = []
    for c in centers:
        c_clamped = float(np.clip(c, min_off, max_off))
        if all(abs(c_clamped - u) >= 0.4 * radius_samples for u in unique_centers):
            unique_centers.append(c_clamped)

    chunks = []
    for c in unique_centers:
        lo = max(min_off, c - radius_samples)
        hi = min(max_off, c + radius_samples)
        count = max(1, int(round((hi - lo) / step)) + 1)
        chunks.append(np.linspace(lo, hi, count))
    return np.sort(np.unique(np.concatenate(chunks)))


def _segment_runs(offsets: List[float]) -> List[Tuple[int, int, float]]:
    """Return ``[(start_idx, end_idx_excl, offset), ...]`` runs."""
    runs: List[Tuple[int, int, float]] = []
    for i, off in enumerate(offsets):
        if runs and abs(runs[-1][2] - off) < 1e-9:
            s, _, o = runs[-1]
            runs[-1] = (s, i + 1, o)
        else:
            runs.append((i, i + 1, off))
    return runs


def _ref_density(
    padded: np.ndarray, pad: int, ref_len: int, lo: float, hi: float
) -> float:
    """Return fraction of ``ref`` active in ``[max(0, lo), min(ref_len, hi))``."""
    if ref_len <= 0:
        return 0.0
    a = max(0.0, float(lo))
    b = min(float(ref_len), float(hi))
    if b <= a:
        return 0.0
    pos = np.array([a + pad, b + pad], dtype=np.float64)
    cs = _interp_cumsum(padded, pos)
    return float(np.clip((cs[1] - cs[0]) / (b - a), 0.0, 1.0))


def _overlap_lift(raw_ratio: float, density: float) -> float:
    """Excess speech overlap normalized against local background VAD density."""
    return (raw_ratio - density) / max(0.15, 1.0 - density)


def _cue_excess_ratings(
    starts: np.ndarray,
    ends: np.ndarray,
    guards: np.ndarray,
    is_speech: np.ndarray,
    padded: np.ndarray,
    pad: int,
    offset: float,
    length_penalty: float,
    density: float,
) -> np.ndarray:
    """Per-cue rating minus the expected chance score at background ``density``."""
    _, rat = _cue_scores_at_offset(
        starts, ends, guards, is_speech, padded, pad, offset, length_penalty
    )
    durs = np.maximum(ends - starts, 0.0)
    baseline = np.where(
        is_speech & (durs > 0),
        np.maximum(
            0.15 * durs,
            density * np.maximum(0.0, durs - 2.0 * length_penalty * guards),
        ),
        0.0,
    )
    return rat - baseline


def _segment_stats(
    a: int,
    b: int,
    off: float,
    *,
    starts: np.ndarray,
    ends: np.ndarray,
    guards: np.ndarray,
    is_speech: np.ndarray,
    durs: np.ndarray,
    padded: np.ndarray,
    pad: int,
    ref_len: int,
    length_penalty: float,
    sample_rate: int,
) -> Tuple[int, float, float, float, float]:
    """Return ``(n_speech, raw_ratio, overlap_lift, rating_sum, speech_dur)``."""
    if b <= a:
        return 0, 0.0, 0.0, 0.0, 0.0
    sl = slice(a, b)
    sp = is_speech[sl] & (durs[sl] > 0)
    n_sp = int(np.sum(sp))
    sp_dur = float(np.sum(durs[sl][sp]))
    if n_sp == 0 or sp_dur <= 0.0:
        return 0, 0.0, 0.0, 0.0, 0.0
    ov, rat = _cue_scores_at_offset(
        starts[sl], ends[sl], guards[sl], is_speech[sl], padded, pad, off, length_penalty
    )
    raw_ratio = float(np.sum(ov)) / sp_dur
    margin = 5.0 * sample_rate
    p_loc = _ref_density(
        padded,
        pad,
        ref_len,
        float(starts[a]) + off - margin,
        float(ends[b - 1]) + off + margin,
    )
    lift = _overlap_lift(raw_ratio, p_loc)
    return n_sp, raw_ratio, lift, float(np.sum(rat)), sp_dur


def _is_strong_edge(
    is_edge: bool,
    n_sp: int,
    r_own: float,
    lift_own: float,
    lift_nb: float,
    delta: float,
    sample_rate: int,
) -> bool:
    """True when a short boundary segment (cold open / post-credits) has clear sync."""
    return (
        is_edge
        and n_sp >= 3
        and lift_nb < 0.10
        and lift_own > 0.45
        and r_own > 0.65
        and delta >= _EDGE_STRONG_MIN_SHIFT_SECONDS * sample_rate
    )


def _coalesce_runs(
    cur_runs: List[Tuple[int, int, float]]
) -> List[Tuple[int, int, float]]:
    merged: List[Tuple[int, int, float]] = []
    for a, b, o in cur_runs:
        if merged and abs(merged[-1][2] - o) < 1e-9:
            pa, _, po = merged[-1]
            merged[-1] = (pa, b, po)
        else:
            merged.append((a, b, o))
    return merged


def alignment_has_positive_signal(
    reference: np.ndarray,
    cues: List[GenericSubtitle],
    offsets: List[Optional[float]],
    *,
    sample_rate: int,
    start_seconds: float,
    length_penalty: float = 0.25,
) -> bool:
    """Return True when kept speech cues show positive overlap lift above chance."""
    if len(cues) == 0 or len(offsets) != len(cues):
        return False
    valid_abs = [abs(off) for off in offsets if off is not None]
    if not valid_abs:
        return False

    starts, ends, is_speech = _cue_sample_bounds(cues, sample_rate, start_seconds)
    durs = np.maximum(ends - starts, 0.0)
    guards = np.minimum(durs, float(_MAX_GUARD_SAMPLES))
    pad = int(np.ceil(max(valid_abs))) + _MAX_GUARD_SAMPLES + 2
    ref, padded = _build_padded_cumsum(reference, pad)
    ref_len = len(ref)
    if ref_len == 0:
        return False

    total_dur = 0.0
    weighted_lift = 0.0
    n = len(cues)
    i = 0
    while i < n:
        off = offsets[i]
        if off is None:
            i += 1
            continue
        j = i + 1
        while j < n and offsets[j] is not None and abs(offsets[j] - off) < 1e-9:
            j += 1
        _, raw_ratio, lift, _, seg_dur = _segment_stats(
            i,
            j,
            off,
            starts=starts,
            ends=ends,
            guards=guards,
            is_speech=is_speech,
            durs=durs,
            padded=padded,
            pad=pad,
            ref_len=ref_len,
            length_penalty=length_penalty,
            sample_rate=sample_rate,
        )
        if seg_dur > 0.0:
            # Compute symmetric max-achievable lift (Loevinger's H) so sparse
            # VAD paired with dense subtitles (q > p) is not capped below 0.08
            # even under near-perfect containment. Reduces identically to
            # `_overlap_lift` whenever q <= p.
            margin = 5.0 * sample_rate
            lo = max(0.0, float(starts[i]) + off - margin)
            hi = min(float(ref_len), float(ends[j - 1]) + off + margin)
            eff_lift = lift
            if hi > lo:
                p_loc = _ref_density(padded, pad, ref_len, lo, hi)
                q_loc = min(1.0, seg_dur / (hi - lo))
                min_pq = min(p_loc, q_loc)
                if min_pq > 0.005:
                    denom = max(0.15 * min_pq, min_pq - p_loc * q_loc)
                    sym_lift = (raw_ratio * q_loc - p_loc * q_loc) / denom
                    eff_lift = max(lift, sym_lift)
            total_dur += seg_dur
            weighted_lift += seg_dur * eff_lift
        i = j

    return total_dur > 0.0 and (weighted_lift / total_dur) > 0.08


def _prune_spurious_splits(
    runs: List[Tuple[int, int, float]],
    *,
    starts: np.ndarray,
    ends: np.ndarray,
    guards: np.ndarray,
    is_speech: np.ndarray,
    padded: np.ndarray,
    pad: int,
    ref_len: int,
    length_penalty: float,
    eff_penalty: float,
    sample_rate: int,
    total_speech_cues: int,
) -> List[Tuple[int, int, float]]:
    """Reject split segments that look like VAD noise rather than real cuts."""
    if len(runs) <= 1:
        return runs

    min_seg_cues = min(10, max(3, total_speech_cues // 8))
    durs = np.maximum(ends - starts, 0.0)

    cur = list(runs)
    changed = True
    while changed and len(cur) > 1:
        changed = False
        # Validate smallest segments first so tiny noise islands are pruned before
        # larger segments instead of absorbing them.
        order = sorted(range(len(cur)), key=lambda k: cur[k][1] - cur[k][0])
        for idx in order:
            a, b, off = cur[idx]
            neighbors = []
            if idx > 0:
                neighbors.append(cur[idx - 1][2])
            if idx + 1 < len(cur):
                neighbors.append(cur[idx + 1][2])

            n_sp, r_own, lift_own, score_own, sp_dur = _segment_stats(
                a,
                b,
                off,
                starts=starts,
                ends=ends,
                guards=guards,
                is_speech=is_speech,
                durs=durs,
                padded=padded,
                pad=pad,
                ref_len=ref_len,
                length_penalty=length_penalty,
                sample_rate=sample_rate,
            )
            best_merge: Optional[Tuple[float, float]] = None  # (net_loss, neighbor_off)
            is_edge = idx == 0 or idx == len(cur) - 1
            for nb_off in neighbors:
                _, r_nb, lift_nb, score_nb, _ = _segment_stats(
                    a,
                    b,
                    nb_off,
                    starts=starts,
                    ends=ends,
                    guards=guards,
                    is_speech=is_speech,
                    durs=durs,
                    padded=padded,
                    pad=pad,
                    ref_len=ref_len,
                    length_penalty=length_penalty,
                    sample_rate=sample_rate,
                )
                gain = score_own - score_nb
                delta = abs(off - nb_off)
                splits_removed = 1 + (
                    1
                    if idx > 0
                    and idx + 1 < len(cur)
                    and abs(cur[idx - 1][2] - cur[idx + 1][2]) < 1e-9
                    else 0
                )
                strong_edge = _is_strong_edge(
                    is_edge, n_sp, r_own, lift_own, lift_nb, delta, sample_rate
                )
                is_tiny = n_sp < min_seg_cues and not strong_edge
                req_penalty = (
                    min(eff_penalty, max(2.5 * sample_rate, 0.35 * sp_dur))
                    if strong_edge
                    else eff_penalty
                )
                eff_min_shift = (
                    0.35 * sample_rate
                    if (n_sp >= 20 and gain >= 1.5 * eff_penalty * splits_removed)
                    else 0.75 * sample_rate
                )
                nb_already_aligned = (lift_nb > 0.35) or (r_nb > 0.50 and lift_nb > 0.25)
                insufficient_improvement = (lift_own - lift_nb) < 0.20 and (
                    r_own - r_nb
                ) < 0.20
                spurious = (
                    delta < eff_min_shift
                    or is_tiny
                    or gain < req_penalty * splits_removed
                    or nb_already_aligned
                    or insufficient_improvement
                )
                if spurious:
                    net_loss = gain - req_penalty * splits_removed
                    if best_merge is None or net_loss < best_merge[0]:
                        best_merge = (net_loss, nb_off)

            if best_merge is not None:
                cur[idx] = (a, b, best_merge[1])
                cur = _coalesce_runs(cur)
                changed = True
                break
    return cur


def _resolve_negative_split_overlaps(
    runs: List[Tuple[int, int, float]],
    *,
    starts: np.ndarray,
    ends: np.ndarray,
    guards: np.ndarray,
    is_speech: np.ndarray,
    padded: np.ndarray,
    pad: int,
    ref_len: int,
    length_penalty: float,
    eff_penalty: float,
    sample_rate: int,
    total_speech_cues: int,
) -> List[Tuple[int, int, float]]:
    """Trim orphan cues (Option B) or collapse spurious negative-jump overlaps."""
    if len(runs) <= 1:
        return runs

    min_seg_cues = min(10, max(3, total_speech_cues // 8))
    durs = np.maximum(ends - starts, 0.0)
    margin = 5.0 * sample_rate
    cur = list(runs)
    changed = True
    while changed and len(cur) > 1:
        changed = False
        for k in range(len(cur) - 1):
            a0, a1, o_a = cur[k]
            b0, b1, o_b = cur[k + 1]
            if a1 <= a0 or b1 <= b0:
                continue
            if not (o_b < o_a and ends[a1 - 1] + o_a > starts[b0] + o_b + 1e-6):
                continue

            span_len = b1 - a0
            p_a = float(
                np.clip(
                    _ref_density(
                        padded,
                        pad,
                        ref_len,
                        float(starts[a0]) + o_a - margin,
                        float(ends[a1 - 1]) + o_a + margin,
                    ),
                    0.05,
                    0.95,
                )
            )
            p_b = float(
                np.clip(
                    _ref_density(
                        padded,
                        pad,
                        ref_len,
                        float(starts[b0]) + o_b - margin,
                        float(ends[b1 - 1]) + o_b + margin,
                    ),
                    0.05,
                    0.95,
                )
            )
            sl = slice(a0, b1)
            excess_a = _cue_excess_ratings(
                starts[sl], ends[sl], guards[sl], is_speech[sl], padded, pad, o_a, length_penalty, p_a
            )
            excess_b = _cue_excess_ratings(
                starts[sl], ends[sl], guards[sl], is_speech[sl], padded, pad, o_b, length_penalty, p_b
            )
            pref_a = np.concatenate([np.zeros(1), np.cumsum(excess_a)])
            suff_b = np.concatenate([np.cumsum(excess_b[::-1])[::-1], np.zeros(1)])
            single_best = max(float(pref_a[-1]), float(suff_b[0]))

            mids = 0.5 * (starts[sl] + ends[sl])
            m_a = np.maximum.accumulate(mids + o_a)
            m_b = np.maximum.accumulate(mids + o_b)
            candidates = np.unique(
                np.concatenate([m_a + 1e-3, m_b - 1e-3, [m_a[0] - 1.0, m_b[-1] + 1.0]])
            )
            cut_a_arr = np.searchsorted(m_a, candidates, side="right")
            cut_b_arr = np.searchsorted(m_b, candidates, side="left")
            interior = (cut_a_arr > 0) & (cut_b_arr < span_len)

            valid_split = False
            cut_a = a1
            cut_b = b0
            if np.any(interior):
                ca_int = cut_a_arr[interior]
                cb_int = cut_b_arr[interior]
                split_scores = pref_a[ca_int] + suff_b[cb_int]
                max_sc = float(np.max(split_scores))
                best_idxs = np.flatnonzero(split_scores >= max_sc - 1e-9)
                best_pos = int(best_idxs[int(np.argmin(cb_int[best_idxs] - ca_int[best_idxs]))])
                best_ca = int(ca_int[best_pos])
                best_cb = int(cb_int[best_pos])
                cut_a = a0 + best_ca
                cut_b = a0 + best_cb
                best_split_excess = float(split_scores[best_pos])

                n_sp_a, r_a, lift_a, _, sp_dur_a = _segment_stats(
                    a0, cut_a, o_a,
                    starts=starts, ends=ends, guards=guards, is_speech=is_speech,
                    durs=durs, padded=padded, pad=pad, ref_len=ref_len,
                    length_penalty=length_penalty, sample_rate=sample_rate,
                )
                n_sp_b, r_b, lift_b, _, sp_dur_b = _segment_stats(
                    cut_b, b1, o_b,
                    starts=starts, ends=ends, guards=guards, is_speech=is_speech,
                    durs=durs, padded=padded, pad=pad, ref_len=ref_len,
                    length_penalty=length_penalty, sample_rate=sample_rate,
                )
                _, _, lift_a_nb, _, _ = _segment_stats(
                    a0, cut_a, o_b,
                    starts=starts, ends=ends, guards=guards, is_speech=is_speech,
                    durs=durs, padded=padded, pad=pad, ref_len=ref_len,
                    length_penalty=length_penalty, sample_rate=sample_rate,
                )
                _, _, lift_b_nb, _, _ = _segment_stats(
                    cut_b, b1, o_a,
                    starts=starts, ends=ends, guards=guards, is_speech=is_speech,
                    durs=durs, padded=padded, pad=pad, ref_len=ref_len,
                    length_penalty=length_penalty, sample_rate=sample_rate,
                )
                delta = abs(o_a - o_b)
                edge_a = _is_strong_edge(
                    k == 0, n_sp_a, r_a, lift_a, lift_a_nb, delta, sample_rate
                )
                edge_b = _is_strong_edge(
                    k + 1 == len(cur) - 1, n_sp_b, r_b, lift_b, lift_b_nb, delta, sample_rate
                )
                ok_a = (n_sp_a >= min_seg_cues or edge_a) and lift_a > 0.25
                ok_b = (n_sp_b >= min_seg_cues or edge_b) and lift_b > 0.25
                if edge_a or edge_b:
                    edge_dur = min(
                        sp_dur_a if edge_a else sp_dur_b,
                        sp_dur_b if edge_b else sp_dur_a,
                    )
                    req_penalty = min(
                        eff_penalty, max(2.5 * sample_rate, 0.35 * edge_dur)
                    )
                else:
                    req_penalty = eff_penalty
                valid_split = ok_a and ok_b and (best_split_excess - req_penalty > single_best)

            if valid_split:
                if cut_a != a1 or cut_b != b0:
                    cur[k] = (a0, cut_a, o_a)
                    cur[k + 1] = (cut_b, b1, o_b)
                    changed = True
                    break
            else:
                winner_off = o_a if float(pref_a[-1]) >= float(suff_b[0]) else o_b
                cur[k : k + 2] = [(a0, b1, winner_off)]
                cur = _coalesce_runs(cur)
                changed = True
                break
    return cur


def _try_rescue_short_edge_splits(
    resolved: List[Tuple[int, int, float]],
    *,
    unhealthy: List[Tuple[int, int]],
    sp_idx: np.ndarray,
    ref: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    guards: np.ndarray,
    is_speech: np.ndarray,
    durs: np.ndarray,
    padded: np.ndarray,
    pad: int,
    ref_len: int,
    min_off: int,
    max_off: int,
    step: float,
    length_penalty: float,
    sample_rate: int,
) -> Tuple[List[Tuple[int, int, float]], float]:
    """Rescue short misaligned cold-open / post-credits boundary segments."""
    if not resolved or len(sp_idx) == 0:
        return resolved, 0.0

    m = len(sp_idx)
    min_seg_cues = min(10, max(3, m // 8))
    cur = list(resolved)
    n = len(starts)

    def _find_edge_split(
        seg_a: int, seg_b: int, base_off: float, is_leading: bool
    ) -> Optional[Tuple[int, float, float]]:
        seg_sp = sp_idx[(sp_idx >= seg_a) & (sp_idx < seg_b)]
        if len(seg_sp) < min_seg_cues + 3:
            return None
        max_edge = min(12, len(seg_sp) // 2)
        if max_edge < 3:
            return None

        check_sp = seg_sp[:max_edge] if is_leading else seg_sp[-max_edge:][::-1]
        ov, _ = _cue_scores_at_offset(
            starts[check_sp],
            ends[check_sp],
            guards[check_sp],
            is_speech[check_sp],
            padded,
            pad,
            base_off,
            length_penalty,
        )
        edge_cnt = 0
        for idx_k, cue_i in enumerate(check_sp):
            if durs[cue_i] > 0 and (ov[idx_k] / durs[cue_i]) < 0.40:
                edge_cnt += 1
            else:
                break
        if edge_cnt < 3 or (len(seg_sp) - edge_cnt) < min_seg_cues:
            return None

        if is_leading:
            cut_idx = int(seg_sp[edge_cnt - 1]) + 1
            edge_a, edge_b = seg_a, cut_idx
            fft_sp = seg_sp[:edge_cnt]
        else:
            cut_idx = int(seg_sp[-edge_cnt])
            edge_a, edge_b = cut_idx, seg_b
            fft_sp = seg_sp[-edge_cnt:]

        if not (seg_a < cut_idx < seg_b):
            return None

        cand = _find_local_offset_fft(
            ref, starts[fft_sp], ends[fft_sp], min_off, max_off
        )
        if cand is None:
            return None

        half_win = 0.5 * sample_rate
        lo = max(float(min_off), float(cand) - half_win)
        hi = min(float(max_off), float(cand) + half_win)
        cnt = max(1, int(round((hi - lo) / step)) + 1)
        best_off = float(cand)
        best_rat = -np.inf
        sl_e = slice(edge_a, edge_b)
        for o_try in np.linspace(lo, hi, cnt):
            _, rat_try = _cue_scores_at_offset(
                starts[sl_e],
                ends[sl_e],
                guards[sl_e],
                is_speech[sl_e],
                padded,
                pad,
                float(o_try),
                length_penalty,
            )
            s_try = float(np.sum(rat_try))
            if s_try > best_rat:
                best_rat = s_try
                best_off = float(o_try)

        if is_leading:
            if ends[cut_idx - 1] + best_off > starts[cut_idx] + base_off + 1e-6:
                return None
        else:
            if ends[cut_idx - 1] + base_off > starts[cut_idx] + best_off + 1e-6:
                return None

        n_sp_e, r_own, lift_own, score_own, sp_dur = _segment_stats(
            edge_a,
            edge_b,
            best_off,
            starts=starts,
            ends=ends,
            guards=guards,
            is_speech=is_speech,
            durs=durs,
            padded=padded,
            pad=pad,
            ref_len=ref_len,
            length_penalty=length_penalty,
            sample_rate=sample_rate,
        )
        _, _, lift_nb, score_nb, _ = _segment_stats(
            edge_a,
            edge_b,
            base_off,
            starts=starts,
            ends=ends,
            guards=guards,
            is_speech=is_speech,
            durs=durs,
            padded=padded,
            pad=pad,
            ref_len=ref_len,
            length_penalty=length_penalty,
            sample_rate=sample_rate,
        )
        delta = abs(best_off - base_off)
        gain = score_own - score_nb
        if not _is_strong_edge(
            True, n_sp_e, r_own, lift_own, lift_nb, delta, sample_rate
        ):
            return None
        edge_pen = max(2.5 * sample_rate, 0.35 * sp_dur)
        if gain < edge_pen:
            return None
        return cut_idx, best_off, gain - edge_pen

    edge_score_delta = 0.0
    if cur and cur[0][0] == 0 and any(a == 0 for a, _ in unhealthy):
        a0, b0, off0 = cur[0]
        res = _find_edge_split(a0, b0, off0, is_leading=True)
        if res is not None:
            cut_idx, edge_off, sc_delta = res
            cur[:1] = [(a0, cut_idx, edge_off), (cut_idx, b0, off0)]
            edge_score_delta += sc_delta

    if cur and cur[-1][1] == n and any(b == m for _, b in unhealthy):
        al, bl, offl = cur[-1]
        res = _find_edge_split(al, bl, offl, is_leading=False)
        if res is not None:
            cut_idx, edge_off, sc_delta = res
            cur[-1:] = [(al, cut_idx, offl), (cut_idx, bl, edge_off)]
            edge_score_delta += sc_delta

    return cur, edge_score_delta


def compute_auto_split_offsets(
    reference: np.ndarray,
    cues: List[GenericSubtitle],
    *,
    global_offset_samples: float,
    sample_rate: int,
    start_seconds: float,
    split_penalty: float,
    length_penalty: float = 0.25,
    offset_step_samples: float = 1.0,
    max_search_samples: int = 30000,
    log_segments: bool = True,
) -> Tuple[List[Optional[float]], float]:
    """Safe two-stage piecewise sync: only split when a real break is detected.

    Stage 1 checks rolling windows across the timeline at ``global_offset_samples``.
    If speech overlap stays consistently healthy throughout, returns the uniform
    global offset with zero splits.

    Stage 2 runs only when one or more windows collapse. It uses fast local FFT
    cross-correlation across a wide search window centered around
    ``global_offset_samples`` to discover alternative offsets, aligns break
    boundaries with unconstrained DP, prunes spurious segments, and removes
    orphan cues left behind by deleted scenes with dialogue (Option B).
    """
    n = len(cues)
    if n == 0:
        return [], 0.0

    starts, ends, is_speech = _cue_sample_bounds(cues, sample_rate, start_seconds)
    durs = np.maximum(ends - starts, 0.0)
    guards = np.minimum(durs, float(_MAX_GUARD_SAMPLES))
    sp_idx = np.flatnonzero(is_speech & (durs > 0))

    g_int = int(round(global_offset_samples))
    search_r = int(abs(max_search_samples))
    min_off = min(-search_r, g_int - search_r)
    max_off = max(search_r, g_int + search_r)
    max_abs = max(
        abs(min_off),
        abs(max_off),
        int(np.ceil(abs(global_offset_samples) + 2.0 * sample_rate)),
    )
    pad = max_abs + _MAX_GUARD_SAMPLES + 2
    ref, padded = _build_padded_cumsum(reference, pad)
    ref_len = len(ref)

    _, global_ratings = _cue_scores_at_offset(
        starts, ends, guards, is_speech, padded, pad, global_offset_samples, length_penalty
    )
    global_score = float(np.sum(global_ratings))

    if len(sp_idx) < 4:
        result: List[Optional[float]] = [float(global_offset_samples)] * n
        if log_segments:
            log_split_segments(result, sample_rate)
        return result, global_score

    # Stage 1: rolling-window health check at global_offset_samples.
    sp_starts = starts[sp_idx]
    sp_ends = ends[sp_idx]
    sp_durs = durs[sp_idx]
    sp_ov, _ = _cue_scores_at_offset(
        sp_starts,
        sp_ends,
        guards[sp_idx],
        np.ones(len(sp_idx), dtype=bool),
        padded,
        pad,
        global_offset_samples,
        length_penalty,
    )

    m = len(sp_idx)
    win = min(14, max(5, m // 4))
    step_w = max(1, win // 3)
    win_starts = list(range(0, max(1, m - win + 1), step_w))
    if win_starts[-1] != max(0, m - win):
        win_starts.append(max(0, m - win))

    window_ranges: List[Tuple[int, int]] = [(a, min(m, a + win)) for a in win_starts]
    tail_w = min(8, m)
    if tail_w < win:
        window_ranges.insert(0, (0, tail_w))
        window_ranges.append((m - tail_w, m))
    small_tail = min(5, m)
    if small_tail < tail_w:
        window_ranges.insert(0, (0, small_tail))
        window_ranges.append((m - small_tail, m))

    margin = 5.0 * sample_rate
    windows: List[Tuple[int, int, float, float]] = []
    for a, b in window_ranges:
        dur_sum = float(np.sum(sp_durs[a:b]))
        ratio = float(np.sum(sp_ov[a:b])) / dur_sum if dur_sum > 0 else 0.0
        p_loc = _ref_density(
            padded,
            pad,
            ref_len,
            float(sp_starts[a]) + global_offset_samples - margin,
            float(sp_ends[b - 1]) + global_offset_samples + margin,
        )
        lift = _overlap_lift(ratio, p_loc)
        windows.append((a, b, ratio, lift))

    ratios = np.array([w[2] for w in windows], dtype=np.float64)
    lifts = np.array([w[3] for w in windows], dtype=np.float64)
    ref_ratio = float(np.percentile(ratios, 85))
    ref_lift = float(np.percentile(lifts, 85))
    if ref_lift < 0.03 and ref_ratio < 0.15:
        unhealthy = [(a, b) for (a, b, _, _) in windows]
    elif ref_lift >= 0.45 and ref_ratio >= 0.65:
        ratio_thresh = min(0.35, 0.45 * ref_ratio)
        lift_thresh = min(0.25, 0.30 * ref_lift)
        unhealthy = [
            (a, b)
            for (a, b, r, lf) in windows
            if (r < ratio_thresh) or (lf < lift_thresh and r < 0.85)
        ]
    else:
        ratio_thresh = min(0.35, 0.45 * ref_ratio)
        lift_thresh = min(0.18, 0.25 * ref_lift)
        unhealthy = [
            (a, b)
            for (a, b, r, lf) in windows
            if (lf < lift_thresh and r < 0.85) and (r < ratio_thresh or lf <= 0.04)
        ]

    if not unhealthy:
        result = [float(global_offset_samples)] * n
        if log_segments:
            log_split_segments(result, sample_rate)
        return result, global_score

    # Merge adjacent unhealthy spans and discover candidate offsets via local FFT.
    spans: List[List[int]] = []
    for a, b in unhealthy:
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])

    centers: List[float] = [float(global_offset_samples)]
    for sa, sb in spans:
        while sb - sa > 4 and sp_durs[sa] > 0 and (sp_ov[sa] / sp_durs[sa]) >= 0.75:
            sa += 1
        while (
            sb - sa > 4
            and sp_durs[sb - 1] > 0
            and (sp_ov[sb - 1] / sp_durs[sb - 1]) >= 0.75
        ):
            sb -= 1
        span_len = sb - sa
        chunk_sz = min(50, span_len)
        chunk_step = max(1, min(35, chunk_sz // 2))
        c_starts = list(range(sa, max(sa + 1, sb - chunk_sz + 1), chunk_step))
        if c_starts[-1] + chunk_sz < sb:
            c_starts.append(max(sa, sb - chunk_sz))
        for ca in c_starts:
            cb = min(sb, ca + chunk_sz)
            cand = _find_local_offset_fft(
                ref, sp_starts[ca:cb], sp_ends[ca:cb], min_off, max_off
            )
            if cand is not None:
                centers.append(float(cand))

    step = float(offset_step_samples) if offset_step_samples > 0 else 1.0
    radius = 1.5 * sample_rate
    candidate_offsets = _build_candidate_offsets(
        centers,
        min_off=min(float(min_off), float(global_offset_samples) - radius),
        max_off=max(float(max_off), float(global_offset_samples) + radius),
        radius_samples=radius,
        step=step,
    )

    total_speech_dur = float(np.sum(sp_durs))
    eff_penalty = min(
        float(split_penalty), max(4.0 * sample_rate, 0.25 * total_speech_dur)
    )

    dp_offsets, _ = compute_split_offsets(
        reference,
        cues,
        sample_rate=sample_rate,
        start_seconds=start_seconds,
        split_penalty=eff_penalty,
        max_offset_samples=max_abs,
        length_penalty=length_penalty,
        offset_step_samples=step,
        candidate_offsets=candidate_offsets,
        enforce_monotonic=False,
        return_score=True,
        log_segments=False,
    )

    # Stage 3: Split Gate — prune spurious splits, trim/collapse negative overlaps,
    # and check short misaligned boundary segments.
    runs = _segment_runs(dp_offsets)
    pruned = _prune_spurious_splits(
        runs,
        starts=starts,
        ends=ends,
        guards=guards,
        is_speech=is_speech,
        padded=padded,
        pad=pad,
        ref_len=ref_len,
        length_penalty=length_penalty,
        eff_penalty=eff_penalty,
        sample_rate=sample_rate,
        total_speech_cues=m,
    )
    resolved = _resolve_negative_split_overlaps(
        pruned,
        starts=starts,
        ends=ends,
        guards=guards,
        is_speech=is_speech,
        padded=padded,
        pad=pad,
        ref_len=ref_len,
        length_penalty=length_penalty,
        eff_penalty=eff_penalty,
        sample_rate=sample_rate,
        total_speech_cues=m,
    )

    final_offsets: List[Optional[float]] = [None] * n
    final_score = 0.0
    for a, b, off in resolved:
        for i in range(a, b):
            final_offsets[i] = off
        _, seg_rat = _cue_scores_at_offset(
            starts[a:b],
            ends[a:b],
            guards[a:b],
            is_speech[a:b],
            padded,
            pad,
            off,
            length_penalty,
        )
        final_score += float(np.sum(seg_rat))
    final_score -= eff_penalty * max(0, len(resolved) - 1)
    dropped = count_dropped_cues(final_offsets)

    # Unconditional "Do No Harm" gate against global_offset_samples before edge rescue.
    reverted = False
    if dropped == 0:
        if final_score <= global_score or (
            len(resolved) <= 1
            and resolved
            and abs(resolved[0][2] - global_offset_samples) < 0.75 * sample_rate
            and final_score < global_score * 1.05
        ):
            reverted = True
    else:
        p_global = float(
            np.clip(
                _ref_density(
                    padded,
                    pad,
                    ref_len,
                    float(starts[0]) + global_offset_samples - margin,
                    float(ends[-1]) + global_offset_samples + margin,
                ),
                0.05,
                0.95,
            )
        )
        global_excess = float(
            np.sum(
                _cue_excess_ratings(
                    starts,
                    ends,
                    guards,
                    is_speech,
                    padded,
                    pad,
                    global_offset_samples,
                    length_penalty,
                    p_global,
                )
            )
        )
        kept_excess = 0.0
        for a, b, off in resolved:
            p_seg = float(
                np.clip(
                    _ref_density(
                        padded,
                        pad,
                        ref_len,
                        float(starts[a]) + off - margin,
                        float(ends[b - 1]) + off + margin,
                    ),
                    0.05,
                    0.95,
                )
            )
            kept_excess += float(
                np.sum(
                    _cue_excess_ratings(
                        starts[a:b],
                        ends[a:b],
                        guards[a:b],
                        is_speech[a:b],
                        padded,
                        pad,
                        off,
                        length_penalty,
                        p_seg,
                    )
                )
            )
        kept_excess -= eff_penalty * max(0, len(resolved) - 1)
        if kept_excess <= global_excess:
            reverted = True

    if reverted:
        resolved = [(0, n, float(global_offset_samples))]
        final_score = global_score

    resolved, edge_score_delta = _try_rescue_short_edge_splits(
        resolved,
        unhealthy=unhealthy,
        sp_idx=sp_idx,
        ref=ref,
        starts=starts,
        ends=ends,
        guards=guards,
        is_speech=is_speech,
        durs=durs,
        padded=padded,
        pad=pad,
        ref_len=ref_len,
        min_off=min_off,
        max_off=max_off,
        step=step,
        length_penalty=length_penalty,
        sample_rate=sample_rate,
    )
    final_offsets = [None] * n
    for a, b, off in resolved:
        for i in range(a, b):
            final_offsets[i] = off
    final_score += edge_score_delta

    if log_segments:
        log_split_segments(final_offsets, sample_rate)
    return final_offsets, final_score
