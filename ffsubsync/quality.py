# -*- coding: utf-8 -*-
"""Post-sync confidence and subtitle quality scoring (0-100).

Produces two independent 0-100 assessments for a sync result:

* ``sync`` — confidence in the alignment itself:
  - ``temporal_consistency``: residual local offset stability across timeline chunks.
  - ``overlap_lift``: speech overlap above random chance + boundary contrast.
  - ``peak_confidence``: global FFT correlation peak SNR and runner-up margin.
  - ``sync_integrity``: healthy window coverage and minimal cue loss/fragmentation.

* ``subtitle`` — intrinsic structural & readability health of the output subtitle:
  - ``timeline_coverage``: span alignment against active speech in the reference.
  - ``reading_speed``: comfortable characters-per-second (CPS) distribution.
  - ``timing_hygiene``: healthy durations and absence of internal overlaps.
  - ``text_integrity``: absence of broken encoding / mojibake / empty / duplicate cues.

Only final aggregate 0-100 scores are returned (no per-chunk raw series) so the
payload stays compact across CLI, desktop, and web/Pyodide builds.
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.split_aligner import (
    _MAX_GUARD_SAMPLES,
    _build_padded_cumsum,
    _cue_sample_bounds,
    _cue_scores_at_offset,
    _interp_cumsum,
    _overlap_lift,
    _ref_density,
)

# Regexes for stripping markup/formatting before CPS & text integrity checks.
_ASS_TAG_RE = re.compile(r"\{[^}]*\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ASS_LINEBREAK_RE = re.compile(r"\\[NnHh]")
_ASS_DRAWING_RE = re.compile(r"\{\\[^}]*\\p[1-9][^}]*\}")
_WHITESPACE_RE = re.compile(r"\s+")
_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Common UTF-8 bytes decoded as Windows-1252 / Latin-1 (mojibake).
_MOJIBAKE_RE = re.compile(
    r"(?:Ã[\x80-\xbf¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿]"
    r"|Â[¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿]"
    r"|â€[œ™˜“”‘’—–…•])"
)
# CJK Unified Ideographs, Hiragana, Katakana, Hangul Syllables.
_CJK_CHAR_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]"
)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, value)))


def _round_score(value: float) -> int:
    return int(round(_clamp(value, 0.0, 100.0)))


def _piecewise(x: float, anchors: Sequence[Tuple[float, float]]) -> float:
    """Piecewise-linear interpolation through monotonically increasing x anchors."""
    if not anchors:
        return 0.0
    if x <= anchors[0][0]:
        return float(anchors[0][1])
    for i in range(1, len(anchors)):
        x0, y0 = anchors[i - 1]
        x1, y1 = anchors[i]
        if x <= x1:
            if x1 == x0:
                return float(y1)
            t = (x - x0) / float(x1 - x0)
            return float(y0 + t * (y1 - y0))
    return float(anchors[-1][1])


def _strip_subtitle_markup(text: str) -> str:
    """Return plain readable subtitle text with ASS/HTML tags and linebreaks removed."""
    if not text:
        return ""
    s = _ASS_LINEBREAK_RE.sub(" ", text)
    s = _ASS_TAG_RE.sub("", s)
    s = _HTML_TAG_RE.sub("", s)
    return _WHITESPACE_RE.sub(" ", s).strip()


def _effective_char_count(clean_text: str) -> float:
    """Count visible characters with CJK weighted for equivalent reading load."""
    if not clean_text:
        return 0.0
    compact = _WHITESPACE_RE.sub("", clean_text)
    if not compact:
        return 0.0
    cjk_count = len(_CJK_CHAR_RE.findall(compact))
    non_cjk = len(compact) - cjk_count
    # Keep spaces lightly weighted (as in standard CPS formulas) and weight CJK ~2.2x.
    spaces = clean_text.count(" ")
    return float(non_cjk + 0.5 * spaces + 2.2 * cjk_count)


def _sym_overlap_lift(
    raw_ratio: float,
    p_loc: float,
    seg_dur: float,
    span_samples: float,
) -> float:
    """Symmetric max-achievable lift (Loevinger's H), matching split_aligner."""
    lift = _overlap_lift(raw_ratio, p_loc)
    if span_samples <= 0.0 or seg_dur <= 0.0:
        return lift
    q_loc = min(1.0, seg_dur / span_samples)
    min_pq = min(p_loc, q_loc)
    if min_pq > 0.005:
        denom = max(0.15 * min_pq, min_pq - p_loc * q_loc)
        sym_lift = (raw_ratio * q_loc - p_loc * q_loc) / denom
        return max(lift, sym_lift)
    return lift


# ---------------------------------------------------------------------------
# Sync confidence sub-metrics
# ---------------------------------------------------------------------------


def _score_overlap_and_edges(
    starts: np.ndarray,
    ends: np.ndarray,
    guards: np.ndarray,
    is_speech: np.ndarray,
    durs: np.ndarray,
    padded: np.ndarray,
    pad: int,
    ref_len: int,
    sample_rate: int,
) -> Tuple[float, float]:
    """Return ``(overlap_lift_score_0_100, effective_lift)`` for synced cues."""
    sp_mask = is_speech & (durs > 0.0)
    sp_idx = np.flatnonzero(sp_mask)
    if len(sp_idx) == 0 or ref_len <= 0:
        return 0.0, 0.0

    sp_starts = starts[sp_idx]
    sp_ends = ends[sp_idx]
    sp_guards = guards[sp_idx]
    sp_durs = durs[sp_idx]
    total_sp_dur = float(np.sum(sp_durs))
    if total_sp_dur <= 0.0:
        return 0.0, 0.0

    ov, _ = _cue_scores_at_offset(
        sp_starts,
        sp_ends,
        sp_guards,
        np.ones(len(sp_idx), dtype=bool),
        padded,
        pad,
        0.0,
        0.0,
    )
    raw_ratio = float(np.sum(ov)) / total_sp_dur

    # Compute local background density & lift across rolling chunks so sparse
    # silent acts don't distort density.
    margin = 5.0 * sample_rate
    m = len(sp_idx)
    n_blocks = min(8, max(1, m // 12))
    edges = np.linspace(0, m, n_blocks + 1, dtype=int)
    weighted_lift = 0.0
    weight_sum = 0.0
    for k in range(n_blocks):
        a, b = int(edges[k]), int(edges[k + 1])
        if b <= a:
            continue
        blk_dur = float(np.sum(sp_durs[a:b]))
        if blk_dur <= 0.0:
            continue
        blk_ratio = float(np.sum(ov[a:b])) / blk_dur
        lo = max(0.0, float(sp_starts[a]) - margin)
        hi = min(float(ref_len), float(sp_ends[b - 1]) + margin)
        if hi <= lo:
            continue
        p_loc = _ref_density(padded, pad, ref_len, lo, hi)
        eff = _sym_overlap_lift(blk_ratio, p_loc, blk_dur, hi - lo)
        weighted_lift += blk_dur * eff
        weight_sum += blk_dur

    eff_lift = weighted_lift / weight_sum if weight_sum > 0.0 else 0.0

    # Guard-band edge contrast: speech density inside cues minus immediately outside.
    guard_sum = float(np.sum(sp_guards))
    if guard_sum > 0.0:
        cs = _interp_cumsum(padded, sp_starts + pad)
        ce = _interp_cumsum(padded, sp_ends + pad)
        gl = cs - _interp_cumsum(padded, sp_starts + pad - sp_guards)
        gr = _interp_cumsum(padded, sp_ends + pad + sp_guards) - ce
        guard_ratio = float(np.sum(gl + gr)) / (2.0 * guard_sum)
        edge_contrast = raw_ratio - guard_ratio
    else:
        edge_contrast = 0.0

    lift_score = _piecewise(
        eff_lift,
        [
            (0.00, 0.0),
            (0.08, 22.0),
            (0.18, 48.0),
            (0.32, 74.0),
            (0.48, 90.0),
            (0.68, 100.0),
        ],
    )
    contrast_score = _piecewise(
        edge_contrast,
        [
            (0.00, 0.0),
            (0.06, 30.0),
            (0.15, 68.0),
            (0.28, 92.0),
            (0.42, 100.0),
        ],
    )
    combined = 0.75 * lift_score + 0.25 * contrast_score
    return _clamp(combined), eff_lift


def _find_local_residual_offset(
    ref: np.ndarray,
    sub_starts: np.ndarray,
    sub_ends: np.ndarray,
    search_samples: int,
) -> Optional[int]:
    """Find local residual offset in ``[-search_samples, +search_samples]``.

    Uses density-centered FFT cross-correlation (matching ``_find_local_offset_fft``),
    and breaks near-ties (within 2% of the peak) in favor of the lag closest to 0
    so periodic/repetitive cue patterns don't report spurious +/-2s residuals.
    """
    if len(sub_starts) == 0 or len(ref) == 0 or search_samples <= 0:
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

    min_off = -int(search_samples)
    max_off = int(search_samples)
    win_lo = sub_lo + min_off
    win_hi = sub_hi + max_off
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

    max_lag = max_off - min_off
    lags = np.arange(max_lag + 1, dtype=np.int64)
    global_offs = min_off + lags
    shifted_lo = sub_lo + global_offs
    shifted_hi = sub_hi + global_offs
    valid = (shifted_hi > 0) & (shifted_lo < len(ref))
    if not np.any(valid):
        return None
    valid_corr = np.where(valid, corr[: max_lag + 1], -np.inf)
    max_val = float(np.max(valid_corr))
    if max_val <= 0.0:
        return None

    # Prefer the offset closest to 0 among lags within 2% of the peak correlation.
    near_best = np.flatnonzero(valid_corr >= max_val * 0.98)
    best_idx = int(near_best[int(np.argmin(np.abs(global_offs[near_best])))])
    return int(global_offs[best_idx])


def _score_temporal_consistency(
    ref: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    is_speech: np.ndarray,
    durs: np.ndarray,
    sample_rate: int,
    fallback_overlap_score: float,
) -> float:
    """Measure residual local offset stability across portions of the video.

    Partitions synced speech cues into contiguous chunks along the timeline and
    runs a narrow local FFT alignment (within +/-2.0s) on each chunk. Low
    dispersion and near-zero residual offsets yield ~100; drifting or shifting
    offsets across the video lower the score proportionally.
    """
    sp_idx = np.flatnonzero(is_speech & (durs > 0.0))
    m = len(sp_idx)
    if m < 4 or len(ref) == 0:
        return _clamp(fallback_overlap_score)

    sp_starts = starts[sp_idx]
    sp_ends = ends[sp_idx]
    sp_durs = durs[sp_idx]

    # Use between 3 and 8 chunks depending on cue count (aim ~12-40 cues/chunk).
    if m < 9:
        n_chunks = 2
    else:
        n_chunks = min(8, max(3, m // 12))

    edges = np.linspace(0, m, n_chunks + 1, dtype=int)
    search_samples = int(round(2.0 * sample_rate))

    residuals_sec: List[float] = []
    chunk_times_sec: List[float] = []
    attempted = 0

    for k in range(n_chunks):
        a, b = int(edges[k]), int(edges[k + 1])
        if b - a < 2:
            continue
        c_starts = sp_starts[a:b]
        c_ends = sp_ends[a:b]
        if float(np.sum(sp_durs[a:b])) < 0.5 * sample_rate:
            continue
        # Skip chunks completely outside the reference timeline.
        if float(c_ends[-1]) <= 0.0 or float(c_starts[0]) >= float(len(ref)):
            attempted += 1
            continue
        attempted += 1
        local_off = _find_local_residual_offset(
            ref, c_starts, c_ends, search_samples
        )
        if local_off is not None:
            residuals_sec.append(float(local_off) / float(sample_rate))
            mid_s = 0.5 * (float(c_starts[0]) + float(c_ends[-1])) / float(sample_rate)
            chunk_times_sec.append(mid_s)

    if attempted == 0:
        return _clamp(fallback_overlap_score)
    if len(residuals_sec) < 2:
        # Couldn't lock onto at least 2 chunks within +/-2s.
        return _clamp(min(45.0, fallback_overlap_score * 0.6))

    res = np.array(residuals_sec, dtype=np.float64)
    med = float(np.median(res))
    dev = np.abs(res - med)

    # Use robust dispersion so a single noisy chunk (e.g. credits or loud music)
    # doesn't sink the score when the rest of the movie is steady.
    if len(res) >= 5:
        trimmed_dev = np.sort(dev)[:-1]
        robust_dev_rms = float(np.sqrt(np.mean(trimmed_dev * trimmed_dev)))
        robust_abs_rms = float(np.sqrt(np.mean(np.sort(np.abs(res))[:-1] ** 2)))
        spread_half = 0.5 * float(np.percentile(res, 90) - np.percentile(res, 10))
    else:
        robust_dev_rms = float(np.sqrt(np.mean(dev * dev)))
        robust_abs_rms = float(np.sqrt(np.mean(res * res)))
        spread_half = 0.5 * float(np.max(res) - np.min(res))

    # Detect monotonic linear drift across measured chunk midpoints.
    linear_drift = 0.0
    if len(residuals_sec) >= 3:
        t = np.array(chunk_times_sec, dtype=np.float64)
        t_span = float(t[-1] - t[0])
        if t_span > 1.0:
            t_centered = t - float(np.mean(t))
            denom = float(np.sum(t_centered * t_centered))
            if denom > 0.0:
                slope = float(np.sum(t_centered * (res - float(np.mean(res))))) / denom
                linear_drift = abs(slope) * t_span

    # When subtitles are much wider/denser than the reference signal (typical of
    # Audio VAD where spoken bursts are shorter than on-screen reading durations),
    # local FFT peaks form a broader plateau (~0.1-0.15s natural jitter). Discount
    # that plateau margin without masking genuine framerate drift or large jumps.
    span_lo = max(0, int(np.floor(float(sp_starts[0]))))
    span_hi = min(len(ref), int(np.ceil(float(sp_ends[-1]))))
    plateau_tol = 0.0
    if span_hi > span_lo:
        ref_dens = float(np.mean(ref[span_lo:span_hi] > 0.0))
        sub_dens = min(1.0, float(np.sum(sp_durs)) / float(span_hi - span_lo))
        if sub_dens > ref_dens * 1.15 and sub_dens > 0.02:
            density_gap = max(0.0, 1.0 - (ref_dens / sub_dens))
            plateau_tol = min(0.16, 0.22 * density_gap)

    raw_jitter = max(
        0.55 * robust_dev_rms + 0.45 * spread_half,
        0.80 * robust_abs_rms,
    )
    adj_jitter = max(0.0, raw_jitter - plateau_tol)
    adj_linear = max(0.0, 0.5 * linear_drift - 0.75 * plateau_tol)
    effective_drift = max(adj_jitter, adj_linear)

    drift_score = _piecewise(
        effective_drift,
        [
            (0.00, 100.0),
            (0.03, 100.0),
            (0.06, 90.0),
            (0.10, 76.0),
            (0.15, 60.0),
            (0.25, 40.0),
            (0.45, 20.0),
            (0.85, 5.0),
            (1.30, 0.0),
        ],
    )

    unmatched_frac = float(attempted - len(residuals_sec)) / float(attempted)
    coverage_factor = 1.0 - 0.75 * unmatched_frac
    return _clamp(drift_score * coverage_factor)


def _score_peak_confidence(
    best_peak_snr: Optional[float],
    best_peak_margin: Optional[float],
    overlap_score: float,
    consistency_score: float,
    n_split_segments: int,
) -> float:
    """Map FFT correlation peak SNR and runner-up margin to 0-100."""
    if best_peak_snr is None and best_peak_margin is None:
        return _clamp(overlap_score)

    snr_score = (
        _piecewise(
            float(best_peak_snr),
            [
                (1.5, 0.0),
                (3.0, 25.0),
                (5.0, 55.0),
                (8.0, 78.0),
                (12.0, 92.0),
                (20.0, 100.0),
            ],
        )
        if best_peak_snr is not None
        else overlap_score
    )

    if best_peak_margin is not None:
        margin_score = _piecewise(
            float(best_peak_margin),
            [
                (0.2, 0.0),
                (1.0, 40.0),
                (2.5, 72.0),
                (5.0, 92.0),
                (10.0, 100.0),
            ],
        )
        score = 0.65 * snr_score + 0.35 * margin_score
    else:
        score = snr_score

    # When piecewise split alignment resolved real mid-file breaks, the single
    # global FFT peak naturally underestimates the post-split confidence.
    if n_split_segments > 1 and consistency_score >= 70.0 and overlap_score >= 60.0:
        score = max(score, 0.85 * overlap_score)

    return _clamp(score)


def _score_sync_integrity(
    starts: np.ndarray,
    ends: np.ndarray,
    guards: np.ndarray,
    is_speech: np.ndarray,
    durs: np.ndarray,
    padded: np.ndarray,
    pad: int,
    ref_len: int,
    sample_rate: int,
    *,
    total_input_cues: int,
    dropped_cues: int,
    clipped_cues: int,
    n_split_segments: int,
) -> float:
    """Evaluate rolling-window health coverage and cue loss during sync."""
    sp_idx = np.flatnonzero(is_speech & (durs > 0.0))
    m = len(sp_idx)
    if m == 0 or ref_len <= 0:
        return 0.0

    sp_starts = starts[sp_idx]
    sp_ends = ends[sp_idx]
    sp_durs = durs[sp_idx]
    sp_ov, _ = _cue_scores_at_offset(
        sp_starts,
        sp_ends,
        guards[sp_idx],
        np.ones(m, dtype=bool),
        padded,
        pad,
        0.0,
        0.0,
    )

    win = min(14, max(4, m // 5))
    step_w = max(1, win // 2)
    win_starts = list(range(0, max(1, m - win + 1), step_w))
    if win_starts[-1] != max(0, m - win):
        win_starts.append(max(0, m - win))

    margin = 5.0 * sample_rate
    healthy = 0
    total_wins = 0
    for a in win_starts:
        b = min(m, a + win)
        dur_sum = float(np.sum(sp_durs[a:b]))
        if dur_sum <= 0.0:
            continue
        total_wins += 1
        raw_ratio = float(np.sum(sp_ov[a:b])) / dur_sum
        lo = max(0.0, float(sp_starts[a]) - margin)
        hi = min(float(ref_len), float(sp_ends[b - 1]) + margin)
        if hi <= lo:
            continue
        p_loc = _ref_density(padded, pad, ref_len, lo, hi)
        eff = _sym_overlap_lift(raw_ratio, p_loc, dur_sum, hi - lo)
        if eff >= 0.10 or (raw_ratio >= 0.55 and eff >= 0.04):
            healthy += 1

    healthy_frac = float(healthy) / float(total_wins) if total_wins > 0 else 0.0
    base = 100.0 * healthy_frac

    denom_cues = max(1, total_input_cues, len(starts))
    loss_frac = float(max(0, dropped_cues) + max(0, clipped_cues)) / float(denom_cues)
    loss_penalty = min(45.0, loss_frac * 250.0)

    n_splits = max(0, n_split_segments - 1)
    # 1-2 clean commercial/scene splits are normal; excessive fragmentation hurts.
    frag_penalty = min(20.0, max(0, n_splits - 2) * 4.0)

    return _clamp(base - loss_penalty - frag_penalty)


# ---------------------------------------------------------------------------
# Intrinsic subtitle quality sub-metrics
# ---------------------------------------------------------------------------


def _score_timeline_coverage(
    cues: List[GenericSubtitle],
    starts: np.ndarray,
    ends: np.ndarray,
    is_speech: np.ndarray,
    durs: np.ndarray,
    ref: Optional[np.ndarray],
    sample_rate: int,
) -> float:
    """Measure how well synced subtitles cover the reference speech timeline."""
    if not cues:
        return 0.0
    if ref is None or len(ref) == 0:
        return 75.0 if len(cues) >= 5 else 50.0

    ref_len = len(ref)
    active_idx = np.flatnonzero(ref > 0.0)
    if len(active_idx) == 0:
        return 75.0

    # Use 1st and 99th percentiles of active speech to ignore opening logos/credits.
    if len(active_idx) >= 50:
        ref_lo = float(np.percentile(active_idx, 1))
        ref_hi = float(np.percentile(active_idx, 99))
    else:
        ref_lo = float(active_idx[0])
        ref_hi = float(active_idx[-1])
    ref_span = max(1.0, ref_hi - ref_lo)

    sp_idx = np.flatnonzero(is_speech & (durs > 0.0))
    use_idx = sp_idx if len(sp_idx) > 0 else np.arange(len(cues))
    sub_lo = float(np.min(starts[use_idx]))
    sub_hi = float(np.max(ends[use_idx]))

    inter_lo = max(ref_lo, sub_lo)
    inter_hi = min(ref_hi, sub_hi)
    span_cov = max(0.0, inter_hi - inter_lo) / ref_span

    score = _piecewise(
        span_cov,
        [
            (0.05, 0.0),
            (0.20, 18.0),
            (0.40, 42.0),
            (0.60, 68.0),
            (0.75, 88.0),
            (0.85, 100.0),
        ],
    )

    # Penalize cues running well beyond the end of the reference media.
    overflow_limit = float(ref_len) + 10.0 * sample_rate
    overflow_cues = int(np.sum(ends[use_idx] > overflow_limit))
    if overflow_cues > 0:
        overflow_frac = float(overflow_cues) / float(len(use_idx))
        score -= min(40.0, overflow_frac * 200.0)

    # Catch sparse forced/signs tracks matched against a long full-length movie.
    ref_dur_s = float(ref_len) / float(sample_rate)
    if ref_dur_s >= 300.0:
        ref_active_samples = float(len(active_idx))
        sub_speech_samples = float(np.sum(durs[use_idx]))
        density_ratio = sub_speech_samples / max(1.0, ref_active_samples)
        if density_ratio < 0.08:
            score *= _piecewise(
                density_ratio,
                [
                    (0.005, 0.20),
                    (0.02, 0.50),
                    (0.05, 0.80),
                    (0.08, 1.00),
                ],
            )

    return _clamp(score)


def _score_reading_speed(
    cues: List[GenericSubtitle],
    is_speech: np.ndarray,
) -> float:
    """Evaluate reading speed (CPS) across dialogue cues."""
    if not cues:
        return 0.0

    cue_scores: List[float] = []
    for i, sub in enumerate(cues):
        if i < len(is_speech) and not bool(is_speech[i]):
            continue
        dur_s = (sub.end - sub.start).total_seconds()
        if dur_s <= 0.0:
            cue_scores.append(0.0)
            continue
        clean = _strip_subtitle_markup(sub.content)
        eff_chars = _effective_char_count(clean)
        if eff_chars <= 0.0:
            continue
        cps = eff_chars / dur_s
        if cps < 4.0:
            # Very short words ("No.", "Hey!") held 1-2s are natural, so only
            # gently penalize ultra-slow stuck cues.
            s = _piecewise(cps, [(0.5, 50.0), (1.5, 72.0), (4.0, 100.0)])
        elif cps <= 20.0:
            s = 100.0
        else:
            s = _piecewise(
                cps,
                [
                    (20.0, 100.0),
                    (25.0, 78.0),
                    (32.0, 42.0),
                    (40.0, 15.0),
                    (55.0, 0.0),
                ],
            )
        cue_scores.append(s)

    if not cue_scores:
        return 50.0
    return _clamp(float(np.mean(cue_scores)))


def _score_timing_hygiene(
    cues: List[GenericSubtitle],
    is_speech: np.ndarray,
    sub_format: str = "srt",
) -> float:
    """Check for invalid durations, flash cues, stuck cues, and overlaps."""
    n = len(cues)
    if n == 0:
        return 0.0

    is_ass = sub_format.lower() in ("ass", "ssa")
    bad_weight = 0.0

    # Track previous end per (layer, style) for ASS/SSA so intentional
    # multi-layer signs/dialogue don't register as timing defects.
    last_end_by_lane: Dict[Tuple[Any, Any], float] = {}

    for i, sub in enumerate(cues):
        start_s = sub.start.total_seconds()
        end_s = sub.end.total_seconds()
        dur_s = end_s - start_s
        dialogue = bool(is_speech[i]) if i < len(is_speech) else True

        if dur_s <= 0.0:
            bad_weight += 1.5
        elif dur_s < 0.35:
            bad_weight += 1.0
        elif dialogue and dur_s > 9.5:
            bad_weight += 0.8

        if is_ass and hasattr(sub.inner, "style"):
            lane = (getattr(sub.inner, "layer", 0), getattr(sub.inner, "style", ""))
        else:
            lane = (0, "")

        prev_end = last_end_by_lane.get(lane)
        if prev_end is not None and start_s < prev_end - 0.05:
            bad_weight += 1.0
        if prev_end is None or end_s > prev_end:
            last_end_by_lane[lane] = end_s

    bad_frac = bad_weight / float(n)
    return _piecewise(
        bad_frac,
        [
            (0.00, 100.0),
            (0.02, 95.0),
            (0.06, 84.0),
            (0.15, 64.0),
            (0.30, 38.0),
            (0.50, 15.0),
            (0.75, 0.0),
        ],
    )


def _score_text_integrity(cues: List[GenericSubtitle]) -> float:
    """Detect broken encoding, mojibake, empty lines, and duplicate cues."""
    n = len(cues)
    if n == 0:
        return 0.0

    corrupt_cues = 0
    mojibake_cues = 0
    empty_cues = 0
    dup_cues = 0

    prev_clean = ""
    prev_end_s: Optional[float] = None

    for sub in cues:
        raw = sub.content or ""
        if "\ufffd" in raw or _CTRL_CHAR_RE.search(raw):
            corrupt_cues += 1
        if _MOJIBAKE_RE.search(raw):
            mojibake_cues += 1

        is_drawing = bool(_ASS_DRAWING_RE.search(raw))
        clean = _strip_subtitle_markup(raw)
        if not clean and not is_drawing:
            empty_cues += 1
        elif clean:
            start_s = sub.start.total_seconds()
            if (
                prev_clean
                and clean == prev_clean
                and prev_end_s is not None
                and abs(start_s - prev_end_s) <= 0.25
            ):
                dup_cues += 1
            prev_clean = clean
            prev_end_s = sub.end.total_seconds()

    corrupt_pen = min(70.0, (float(corrupt_cues) / float(n)) * 400.0)
    mojibake_pen = min(60.0, (float(mojibake_cues) / float(n)) * 300.0)
    empty_pen = min(40.0, (float(empty_cues) / float(n)) * 150.0)
    dup_pen = min(25.0, (float(dup_cues) / float(n)) * 120.0)

    return _clamp(100.0 - corrupt_pen - mojibake_pen - empty_pen - dup_pen)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def compute_quality_metrics(
    reference_speech: Optional[np.ndarray],
    synced_cues: Sequence[GenericSubtitle],
    *,
    sample_rate: int = SAMPLE_RATE,
    start_seconds: float = 0.0,
    best_peak_snr: Optional[float] = None,
    best_peak_margin: Optional[float] = None,
    sync_was_successful: bool = True,
    total_input_cues: Optional[int] = None,
    dropped_cues: int = 0,
    clipped_cues: int = 0,
    n_split_segments: int = 1,
    sub_format: str = "srt",
    reference_kind: str = "audio",
    profile_offset_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Compute final 0-100 ``sync`` and ``subtitle`` quality badges.

    Returns a JSON-serializable dict with final scores only::

        {
            "sync": {
                "score": int,
                "metrics": {
                    "temporal_consistency": int,
                    "overlap_lift": int,
                    "peak_confidence": int,
                    "sync_integrity": int,
                },
            },
            "subtitle": {
                "score": int,
                "metrics": {
                    "timeline_coverage": int,
                    "reading_speed": int,
                    "timing_hygiene": int,
                    "text_integrity": int,
                },
            },
        }
    """
    cues = list(synced_cues)
    n_cues = len(cues)
    in_cues = total_input_cues if total_input_cues is not None else n_cues

    if n_cues > 0:
        starts, ends, is_speech = _cue_sample_bounds(cues, sample_rate, start_seconds)
        durs = np.maximum(ends - starts, 0.0)
        guards = np.minimum(durs, float(_MAX_GUARD_SAMPLES))
    else:
        starts = np.zeros(0, dtype=np.float64)
        ends = np.zeros(0, dtype=np.float64)
        is_speech = np.zeros(0, dtype=bool)
        durs = np.zeros(0, dtype=np.float64)
        guards = np.zeros(0, dtype=np.float64)

    has_ref = reference_speech is not None and len(reference_speech) > 0
    if has_ref:
        pad = _MAX_GUARD_SAMPLES + 4
        ref, padded = _build_padded_cumsum(reference_speech, pad)
        ref_len = len(ref)
    else:
        pad = 0
        ref = None
        padded = None
        ref_len = 0

    # ---- Sync confidence ----
    if not has_ref or n_cues == 0:
        overlap_score = 0.0
        consistency_score = 0.0
        peak_score = 0.0
        integrity_score = 0.0
        sync_total = 0.0
    else:
        assert ref is not None and padded is not None
        overlap_score, _ = _score_overlap_and_edges(
            starts,
            ends,
            guards,
            is_speech,
            durs,
            padded,
            pad,
            ref_len,
            sample_rate,
        )
        # Compensate for any intentional global alignment-profile shift (e.g.
        # "onset" or "early") when measuring symmetric local-FFT residual
        # stability so an intentional edge preference does not look like drift.
        profile_offset_samples = float(profile_offset_seconds) * float(sample_rate)
        consistency_score = _score_temporal_consistency(
            ref,
            starts - profile_offset_samples,
            ends - profile_offset_samples,
            is_speech,
            durs,
            sample_rate,
            fallback_overlap_score=overlap_score,
        )
        peak_score = _score_peak_confidence(
            best_peak_snr,
            best_peak_margin,
            overlap_score=overlap_score,
            consistency_score=consistency_score,
            n_split_segments=n_split_segments,
        )
        integrity_score = _score_sync_integrity(
            starts,
            ends,
            guards,
            is_speech,
            durs,
            padded,
            pad,
            ref_len,
            sample_rate,
            total_input_cues=in_cues,
            dropped_cues=dropped_cues,
            clipped_cues=clipped_cues,
            n_split_segments=n_split_segments,
        )

        sync_total = (
            0.30 * consistency_score
            + 0.30 * overlap_score
            + 0.25 * peak_score
            + 0.15 * integrity_score
        )
        # Bottleneck cap: any single sub-metric below 50 prevents a green score.
        min_sync_metric = min(
            consistency_score, overlap_score, peak_score, integrity_score
        )
        if min_sync_metric < 50.0:
            sync_total = min(sync_total, min(69.0, min_sync_metric + 25.0))
        # Red-flag caps prevent misleading high scores when critical checks fail.
        if not sync_was_successful:
            sync_total = min(sync_total, 35.0)
        if overlap_score < 20.0:
            sync_total = min(sync_total, overlap_score + 25.0)
        if peak_score < 25.0 and consistency_score < 50.0:
            sync_total = min(sync_total, 40.0)

    # ---- Subtitle intrinsic quality ----
    if n_cues == 0:
        cov_score = 0.0
        cps_score = 0.0
        timing_score = 0.0
        text_score = 0.0
        sub_total = 0.0
    else:
        cov_score = _score_timeline_coverage(
            cues, starts, ends, is_speech, durs, ref, sample_rate
        )
        cps_score = _score_reading_speed(cues, is_speech)
        timing_score = _score_timing_hygiene(cues, is_speech, sub_format=sub_format)
        text_score = _score_text_integrity(cues)

        sub_total = (
            0.30 * cov_score
            + 0.25 * cps_score
            + 0.25 * timing_score
            + 0.20 * text_score
        )
        min_sub_metric = min(cov_score, cps_score, timing_score, text_score)
        if min_sub_metric < 50.0:
            sub_total = min(sub_total, min(69.0, min_sub_metric + 25.0))
        if text_score < 40.0:
            sub_total = min(sub_total, text_score + 20.0)
        if cov_score < 35.0:
            sub_total = min(sub_total, cov_score + 25.0)

    norm_ref_kind = "subtitles" if reference_kind == "subtitles" else "audio"
    return {
        "reference_kind": norm_ref_kind,
        "sync": {
            "score": _round_score(sync_total),
            "metrics": {
                "temporal_consistency": _round_score(consistency_score),
                "overlap_lift": _round_score(overlap_score),
                "peak_confidence": _round_score(peak_score),
                "sync_integrity": _round_score(integrity_score),
            },
        },
        "subtitle": {
            "score": _round_score(sub_total),
            "metrics": {
                "timeline_coverage": _round_score(cov_score),
                "reading_speed": _round_score(cps_score),
                "timing_hygiene": _round_score(timing_score),
                "text_integrity": _round_score(text_score),
            },
        },
    }
