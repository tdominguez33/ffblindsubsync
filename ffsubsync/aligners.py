# -*- coding: utf-8 -*-
import logging
import math
from typing import List, Optional, Tuple, Type, Union

import numpy as np

from ffsubsync.golden_section_search import gss
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


MIN_FRAMERATE_RATIO = 0.9
MAX_FRAMERATE_RATIO = 1.1


class FailedToFindAlignmentException(Exception):
    pass


class FFTAligner(TransformerMixin):
    def __init__(self, max_offset_samples: Optional[int] = None) -> None:
        self.max_offset_samples: Optional[int] = max_offset_samples
        self.best_offset_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.best_peak_snr_: Optional[float] = None
        self.best_peak_margin_: Optional[float] = None
        self.get_score_: bool = False

    def _eliminate_extreme_offsets_from_solutions(
        self, convolve: np.ndarray, substring: np.ndarray
    ) -> np.ndarray:
        convolve = np.copy(convolve)
        if self.max_offset_samples is None:
            return convolve

        def _offset_to_index(offset):
            return len(convolve) - 1 + offset - len(substring)

        convolve[: _offset_to_index(-self.max_offset_samples)] = float("-inf")
        convolve[_offset_to_index(self.max_offset_samples) :] = float("-inf")
        return convolve

    def _compute_argmax(self, convolve: np.ndarray, substring: np.ndarray) -> None:
        best_idx = int(np.argmax(convolve))
        self.best_offset_ = len(convolve) - 1 - best_idx - len(substring)
        self.best_score_ = convolve[best_idx]
        snr, margin = self._compute_peak_stats(convolve, best_idx)
        self.best_peak_snr_ = snr
        self.best_peak_margin_ = margin

    @staticmethod
    def _compute_peak_stats(
        convolve: np.ndarray, best_idx: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute peak z-score and runner-up margin relative to valid offsets.

        ``peak_snr`` measures how many standard deviations the winning
        cross-correlation peak stands above the background offsets (excluding a
        small lobe around the peak). ``peak_margin`` measures how many standard
        deviations the winning peak exceeds the strongest competing offset
        outside that lobe, detecting ambiguous multi-peak matches.
        """
        finite_idx = np.flatnonzero(np.isfinite(convolve))
        n_finite = len(finite_idx)
        if n_finite < 8:
            return None, None
        guard = min(200, max(1, n_finite // 10))
        bg_idx = finite_idx[np.abs(finite_idx - best_idx) > guard]
        if len(bg_idx) < 4:
            return None, None
        bg = convolve[bg_idx]
        bg_std = float(np.std(bg))
        if bg_std <= 0.0:
            return None, None
        bg_mean = float(np.mean(bg))
        peak_val = float(convolve[best_idx])
        snr = float((peak_val - bg_mean) / bg_std)
        runner_up = float(np.max(bg))
        margin = float((peak_val - runner_up) / bg_std)
        return snr, margin

    @staticmethod
    def _compute_peak_snr(convolve: np.ndarray, best_idx: int) -> Optional[float]:
        """Compute peak z-score relative to the valid offset search window."""
        snr, _ = FFTAligner._compute_peak_stats(convolve, best_idx)
        return snr

    def fit(self, refstring, substring, get_score: bool = False) -> "FFTAligner":
        refstring, substring = [
            list(map(int, s)) if isinstance(s, str) else s
            for s in [refstring, substring]
        ]
        refstring, substring = map(
            lambda s: 2 * np.array(s).astype(float) - 1, [refstring, substring]
        )
        if len(refstring) == 0 or len(substring) == 0:
            # empty speech on either side has no meaningful alignment, and would
            # otherwise yield a garbage offset (or crash math.log when both empty)
            raise FailedToFindAlignmentException(
                "cannot align empty speech data "
                "(reference length=%d, subtitle length=%d); "
                "the reference or subtitles may contain no detectable speech"
                % (len(refstring), len(substring))
            )
        total_bits = math.log(len(substring) + len(refstring), 2)
        total_length = int(2 ** math.ceil(total_bits))
        extra_zeros = total_length - len(substring) - len(refstring)
        subft = np.fft.fft(np.append(np.zeros(extra_zeros + len(refstring)), substring))
        refft = np.fft.fft(
            np.flip(np.append(refstring, np.zeros(len(substring) + extra_zeros)), 0)
        )
        convolve = np.real(np.fft.ifft(subft * refft))
        self._compute_argmax(
            self._eliminate_extreme_offsets_from_solutions(convolve, substring),
            substring,
        )
        self.get_score_ = get_score
        return self

    def transform(self, *_) -> Union[int, Tuple[float, int]]:
        if self.get_score_:
            return self.best_score_, self.best_offset_
        else:
            return self.best_offset_


class MaxScoreAligner(TransformerMixin):
    def __init__(
        self,
        base_aligner: Union[FFTAligner, Type[FFTAligner]],
        srtin: Optional[str] = None,
        sample_rate=None,
        max_offset_seconds=None,
    ) -> None:
        self.srtin: Optional[str] = srtin
        if sample_rate is None or max_offset_seconds is None:
            self.max_offset_samples: Optional[int] = None
        else:
            self.max_offset_samples = abs(int(max_offset_seconds * sample_rate))
        if isinstance(base_aligner, type):
            self.base_aligner: FFTAligner = base_aligner(
                max_offset_samples=self.max_offset_samples
            )
        else:
            self.base_aligner = base_aligner
        self.max_offset_seconds: Optional[int] = max_offset_seconds
        self._scores: List[Tuple[Tuple[float, int], Pipeline]] = []
        self._snrs: List[Optional[float]] = []
        self._margins: List[Optional[float]] = []
        self.best_peak_snr_: Optional[float] = None
        self.best_peak_margin_: Optional[float] = None

    def fit_gss(self, refstring, subpipe_maker):
        def opt_func(framerate_ratio, is_last_iter):
            subpipe = subpipe_maker(framerate_ratio)
            substring = subpipe.fit_transform(self.srtin)
            score = self.base_aligner.fit_transform(
                refstring, substring, get_score=True
            )
            logger.info(
                "got score %.0f (offset %d) for ratio %.3f",
                score[0],
                score[1],
                framerate_ratio,
            )
            if is_last_iter:
                self._scores.append((score, subpipe))
                self._snrs.append(getattr(self.base_aligner, "best_peak_snr_", None))
                self._margins.append(
                    getattr(self.base_aligner, "best_peak_margin_", None)
                )
            return -score[0]

        gss(opt_func, MIN_FRAMERATE_RATIO, MAX_FRAMERATE_RATIO)
        return self

    def fit(
        self, refstring, subpipes: Union[Pipeline, List[Pipeline]]
    ) -> "MaxScoreAligner":
        if not isinstance(subpipes, list):
            subpipes = [subpipes]
        total = len(subpipes)
        for idx, subpipe in enumerate(subpipes, 1):
            logger.info("computing alignment %d/%d...", idx, total)
            if callable(subpipe):
                self.fit_gss(refstring, subpipe)
                continue
            elif hasattr(subpipe, "transform"):
                substring = subpipe.transform(self.srtin)
            else:
                substring = subpipe
            self._scores.append(
                (
                    self.base_aligner.fit_transform(
                        refstring, substring, get_score=True
                    ),
                    subpipe,
                )
            )
            self._snrs.append(getattr(self.base_aligner, "best_peak_snr_", None))
            self._margins.append(getattr(self.base_aligner, "best_peak_margin_", None))
        return self

    def transform(self, *_) -> Tuple[Tuple[float, float], Pipeline]:
        indexed = list(enumerate(self._scores))
        if self.max_offset_samples is not None:
            indexed = [
                (i, s)
                for i, s in indexed
                if abs(s[0][1]) <= self.max_offset_samples
            ]
        if len(indexed) == 0:
            raise FailedToFindAlignmentException(
                "Synchronization failed; consider passing "
                "--max-offset-seconds with a number larger than "
                "{}".format(self.max_offset_seconds)
            )
        best_i, ((score, offset), subpipe) = max(indexed, key=lambda x: x[1][0][0])
        self.best_peak_snr_ = self._snrs[best_i] if best_i < len(self._snrs) else None
        self.best_peak_margin_ = (
            self._margins[best_i] if best_i < len(self._margins) else None
        )
        return (score, offset), subpipe
