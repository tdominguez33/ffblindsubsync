#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import logging
import os
import shutil
import subprocess
import sys
from typing import cast, Any, Callable, Dict, List, Optional, Tuple, Union

import ffmpeg
import numpy as np

from ffsubsync.aligners import FFTAligner, MaxScoreAligner
from ffsubsync.alignment_profile import (
    compute_profile_offset_adjustment,
    normalize_alignment_profile,
)
from ffsubsync.constants import (
    ALIGNMENT_PROFILE_CHOICES,
    DEFAULT_ALIGNMENT_PROFILE,
    DEFAULT_APPLY_OFFSET_SECONDS,
    DEFAULT_AUTO_SPLIT_MAX_OFFSET_SECONDS,
    DEFAULT_AUTO_SPLIT_PENALTY,
    DEFAULT_FRAME_RATE,
    DEFAULT_MAX_OFFSET_SECONDS,
    DEFAULT_MAX_SUBTITLE_SECONDS,
    DEFAULT_MIN_EMBEDDED_SUBTITLE_CUES,
    DEFAULT_NON_SPEECH_LABEL,
    DEFAULT_SPLIT_PENALTY,
    DEFAULT_SPLIT_LENGTH_PENALTY,
    DEFAULT_SPLIT_SUBSAMPLE,
    DEFAULT_START_SECONDS,
    DEFAULT_VAD,
    DEFAULT_ENCODING,
    FRAMERATE_RATIOS,
    SAMPLE_RATE,
    SUBTITLE_EXTENSIONS,
    DEFAULT_MIN_SCORE,
    DEFAULT_PEAK_SNR_SUCCESS_THRESHOLD,
    DEFAULT_QUALITY_MAX_OFFSET_SECONDS,
    DEFAULT_MAX_FRAMERATE_DEVIATION,
    VAD_CHOICES,
    is_remote_url,
    is_valid_vad,
)
from ffsubsync.ffmpeg_utils import ffmpeg_bin_path, subprocess_args
from ffsubsync.generic_subtitles import SUPPORTED_OUTPUT_FORMATS
from ffsubsync.quality import compute_quality_metrics
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin
from ffsubsync.speech_transformers import (
    VideoSpeechTransformer,
    MultiSegmentVideoSpeechTransformer,
    DeserializeSpeechTransformer,
    PGSSpeechTransformer,
    SubtitleSpeechTransformer,
    WhisperSpeechTransformer,
    ProgressInfo,
    make_subtitle_speech_pipeline,
)
from ffsubsync.split_aligner import (
    alignment_has_positive_signal,
    compute_auto_split_offsets,
    compute_split_offsets,
    count_dropped_cues,
    log_split_segments,
    summarize_split_segments,
)
from ffsubsync.subtitle_parser import make_subtitle_parser
from ffsubsync.subtitle_transformers import (
    SubtitleMerger,
    SubtitleShifter,
    VariableSubtitleShifter,
)
from ffsubsync.version import get_version


logger: logging.Logger = logging.getLogger(__name__)


def override(args: argparse.Namespace, **kwargs: Any) -> Dict[str, Any]:
    args_dict = dict(args.__dict__)
    args_dict.update(kwargs)
    return args_dict


def _ref_format(ref_fname: Optional[str]) -> Optional[str]:
    if ref_fname is None:
        return None
    ext = os.path.splitext(ref_fname)[1].lstrip(".").lower()
    return ext or ref_fname[-3:].lower()


def make_test_case(
    args: argparse.Namespace, npy_savename: Optional[str], sync_was_successful: bool
) -> int:
    if npy_savename is None:
        raise ValueError("need non-null npy_savename")
    tar_dir = "{}.{}".format(
        args.reference, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    logger.info("creating test archive {}.tar.gz...".format(tar_dir))
    os.mkdir(tar_dir)
    try:
        log_path = "ffsubsync.log"
        if args.log_dir_path is not None and os.path.isdir(args.log_dir_path):
            log_path = os.path.join(args.log_dir_path, log_path)
        if os.path.exists(log_path):
            shutil.copy(log_path, tar_dir)
        if args.srtin and args.srtin[0] and os.path.exists(args.srtin[0]):
            shutil.copy(args.srtin[0], tar_dir)
        if sync_was_successful and args.srtout and os.path.exists(args.srtout):
            shutil.move(args.srtout, tar_dir)
        if _ref_format(args.reference) in SUBTITLE_EXTENSIONS:
            if args.reference and os.path.exists(args.reference):
                shutil.copy(args.reference, tar_dir)
        elif os.path.exists(npy_savename):
            if args.serialize_speech or args.reference == npy_savename:
                shutil.copy(npy_savename, tar_dir)
            else:
                shutil.move(npy_savename, tar_dir)
        supported_formats = set(list(zip(*shutil.get_archive_formats()))[0])
        preferred_formats = ["gztar", "bztar", "xztar", "zip", "tar"]
        for archive_format in preferred_formats:
            if archive_format in supported_formats:
                shutil.make_archive(tar_dir, archive_format, os.curdir, tar_dir)
                break
        else:
            logger.error(
                "failed to create test archive; no formats supported "
                "(this should not happen)"
            )
            return 1
        logger.info("...done")
    finally:
        shutil.rmtree(tar_dir)
    return 0


def _resolve_srtout(args: argparse.Namespace, srtin: Optional[str]) -> Optional[str]:
    """Decide where the synced subtitles for ``srtin`` should be written.

    Overwriting the input takes precedence; otherwise, when subtitles were
    auto-detected from the reference, write a `<name>.synced.srt` next to each
    input; otherwise fall back to the explicit (possibly ``None``) output.
    """
    if args.overwrite_input:
        return srtin
    if getattr(args, "auto_srtout", False) and srtin is not None:
        return "{}.synced.srt".format(os.path.splitext(srtin)[0])
    return args.srtout


def _parse_fps_fraction(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in ("0/0", "N/A"):
        return None
    try:
        if "/" in s:
            num_s, den_s = s.split("/", 1)
            den = float(den_s)
            if den <= 0:
                return None
            fps = float(num_s) / den
        else:
            fps = float(s)
    except (TypeError, ValueError):
        return None
    return fps if 1.0 <= fps <= 240.0 else None


def _probe_video_fps(
    fname: Optional[str],
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> Optional[float]:
    """Infer primary video stream FPS from ``fname`` via ffprobe (for MicroDVD .sub)."""
    if not fname:
        return None
    ref_fmt = _ref_format(fname)
    if ref_fmt in SUBTITLE_EXTENSIONS or ref_fmt in ("npy", "npz"):
        return None
    try:
        probe = ffmpeg.probe(
            fname,
            cmd=ffmpeg_bin_path("ffprobe", gui_mode, ffmpeg_resources_path=ffmpeg_path),
        )
    except Exception as e:
        logger.debug("ffprobe failed while inferring video FPS: %s", e)
        return None
    for stream in probe.get("streams") or []:
        if stream.get("codec_type") != "video":
            continue
        if (stream.get("disposition") or {}).get("attached_pic"):
            continue
        fps = _parse_fps_fraction(stream.get("avg_frame_rate")) or _parse_fps_fraction(
            stream.get("r_frame_rate")
        )
        if fps is not None:
            return fps
    return None


def get_srt_pipe_maker(
    args: argparse.Namespace, srtin: Optional[str]
) -> Callable[[Optional[float]], Union[Pipeline, Callable[[float], Pipeline]]]:
    if srtin is None:
        srtin_format = "srt"
    else:
        srtin_format = os.path.splitext(srtin)[-1][1:].lower()
    fallback_fps: Optional[float] = None
    explicit_sub_fps = getattr(args, "sub_fps", None)
    srtout = _resolve_srtout(args, srtin)
    srtout_format = (
        os.path.splitext(srtout)[-1][1:].lower() if srtout is not None else srtin_format
    )
    if explicit_sub_fps is None and (srtin_format == "sub" or srtout_format == "sub"):
        fallback_fps = _probe_video_fps(
            args.reference,
            ffmpeg_path=getattr(args, "ffmpeg_path", None),
            gui_mode=getattr(args, "gui_mode", False),
        )
    parser = make_subtitle_parser(
        fmt=srtin_format,
        caching=True,
        fallback_fps=fallback_fps,
        **args.__dict__,
    )
    return lambda scale_factor: make_subtitle_speech_pipeline(
        **override(args, scale_factor=scale_factor, parser=parser)
    )


def get_framerate_ratios_to_try(args: argparse.Namespace) -> List[Optional[float]]:
    if args.no_fix_framerate:
        return []
    else:
        framerate_ratios = list(
            np.concatenate(
                [np.array(FRAMERATE_RATIOS), 1.0 / np.array(FRAMERATE_RATIOS)]
            )
        )
        if args.gss:
            framerate_ratios.append(None)
        return framerate_ratios


def assess_alignment_quality(
    best_score: float,
    offset_seconds: float,
    scale_factor: float,
    *,
    min_score: float,
    max_offset_seconds: float,
    max_framerate_deviation: float,
) -> List[str]:
    """Return reasons an alignment looks too low-quality to trust (empty = trust it).

    Used by --skip-sync-on-low-quality to decide whether to leave the subtitles
    unmodified rather than apply a probably-wrong shift. A negative score means the
    best alignment is anti-correlated; an implausibly large offset or framerate
    scale suggests a spurious match.
    """
    reasons: List[str] = []
    if best_score < min_score:
        reasons.append("score %.1f < %.1f" % (best_score, min_score))
    if abs(offset_seconds) > max_offset_seconds:
        reasons.append(
            "|offset| %.1fs > %.1fs" % (abs(offset_seconds), max_offset_seconds)
        )
    framerate_deviation = abs(scale_factor - 1.0)
    if framerate_deviation > max_framerate_deviation:
        reasons.append(
            "framerate deviation %.3f > %.3f"
            % (framerate_deviation, max_framerate_deviation)
        )
    return reasons


def try_sync(
    args: argparse.Namespace, reference_pipe: Optional[Pipeline], result: Dict[str, Any]
) -> bool:
    result["sync_was_successful"] = False
    sync_was_successful = True
    logger.info(
        "extracting speech segments from %s...",
        "stdin" if not args.srtin else "subtitles file(s) {}".format(args.srtin),
    )
    if not args.srtin:
        args.srtin = [None]
    for srtin in args.srtin:
        try:
            skip_sync = args.skip_sync or reference_pipe is None
            skip_infer_framerate_ratio = (
                args.skip_infer_framerate_ratio or reference_pipe is None
            )
            srtout = _resolve_srtout(args, srtin)
            srt_pipe_maker = get_srt_pipe_maker(args, srtin)
            framerate_ratios = get_framerate_ratios_to_try(args)
            srt_pipes = [srt_pipe_maker(1.0)] + [
                srt_pipe_maker(rat) for rat in framerate_ratios
            ]
            for srt_pipe in srt_pipes:
                if callable(srt_pipe):
                    continue
                else:
                    srt_pipe.fit(srtin)
            if (
                not skip_infer_framerate_ratio
                and hasattr(reference_pipe[-1], "num_frames")
                and reference_pipe[-1].num_frames is not None
            ):
                inferred_framerate_ratio_from_length = (
                    float(reference_pipe[-1].num_frames)
                    / cast(Pipeline, srt_pipes[0])[-1].num_frames
                )
                logger.info(
                    "inferred frameratio ratio: %.3f"
                    % inferred_framerate_ratio_from_length
                )
                srt_pipes.append(
                    cast(
                        Pipeline, srt_pipe_maker(inferred_framerate_ratio_from_length)
                    ).fit(srtin)
                )
                logger.info("...done")
            logger.info("computing alignments...")
            reference_speech = None
            split_penalty = getattr(args, "split_penalty", None)
            use_split_early = split_penalty is not None and not skip_sync
            use_auto_split_early = (
                bool(getattr(args, "auto_split", False))
                and not use_split_early
                and not skip_sync
            )
            align_max_offset_seconds = (
                max(
                    float(args.max_offset_seconds),
                    DEFAULT_AUTO_SPLIT_MAX_OFFSET_SECONDS,
                )
                if use_auto_split_early
                else args.max_offset_seconds
            )
            aligner: Optional[MaxScoreAligner] = None
            best_peak_snr: Optional[float] = None
            best_peak_margin: Optional[float] = None
            reference_kind: str = "audio"
            if skip_sync:
                best_score = 0.0
                best_srt_pipe = cast(Pipeline, srt_pipes[0])
                offset_samples = 0
            else:
                reference_speech = reference_pipe.transform(args.reference)
                aligner = MaxScoreAligner(
                    FFTAligner, srtin, SAMPLE_RATE, align_max_offset_seconds
                )
                (best_score, offset_samples), best_srt_pipe = aligner.fit_transform(
                    reference_speech,
                    srt_pipes,
                )
                best_peak_snr = getattr(aligner, "best_peak_snr_", None)
                best_peak_margin = getattr(aligner, "best_peak_margin_", None)

                ref_step = (
                    reference_pipe[-1]
                    if hasattr(reference_pipe, "__getitem__")
                    else reference_pipe
                )
                if isinstance(
                    ref_step, (SubtitleSpeechTransformer, PGSSpeechTransformer)
                ) or bool(getattr(ref_step, "used_embedded_subs_", False)):
                    reference_kind = "subtitles"
                else:
                    reference_kind = "audio"
                alt_candidates = list(
                    getattr(ref_step, "alt_video_speech_results_", None) or []
                )
                embedded_peak_weak = (
                    best_peak_snr is not None
                    and best_peak_snr < DEFAULT_PEAK_SNR_SUCCESS_THRESHOLD
                )
                poor_embedded_alignment = (
                    (
                        (best_score < 0 and (best_peak_snr is None or embedded_peak_weak))
                        or embedded_peak_weak
                    )
                    and bool(getattr(ref_step, "can_fallback_to_audio", False))
                )
                if poor_embedded_alignment:
                    logger.warning(
                        "embedded subtitle alignment looked unreliable "
                        "(score=%.1f, peak SNR=%s); falling back to audio VAD",
                        best_score,
                        "%.2f" % best_peak_snr if best_peak_snr is not None else "n/a",
                    )
                    try:
                        audio_speech, audio_alts = ref_step.fallback_to_audio(
                            args.reference
                        )
                        alt_candidates.append(("audio VAD fallback", audio_speech))
                        alt_candidates.extend(audio_alts)
                    except Exception as e:
                        logger.warning(
                            "audio VAD fallback failed (%s); keeping embedded alignment",
                            e,
                        )
                for cand_label, cand_speech in alt_candidates:
                    logger.info("comparing alignment with %s...", cand_label)
                    cand_aligner = MaxScoreAligner(
                        FFTAligner, srtin, SAMPLE_RATE, align_max_offset_seconds
                    )
                    (cand_score, cand_offset), cand_pipe = cand_aligner.fit_transform(
                        cand_speech,
                        srt_pipes,
                    )
                    cand_snr = getattr(cand_aligner, "best_peak_snr_", None)
                    cand_is_valid = cand_score >= 0 or (
                        cand_snr is not None
                        and cand_snr >= DEFAULT_PEAK_SNR_SUCCESS_THRESHOLD
                    )
                    best_is_valid = best_score >= 0 or (
                        best_peak_snr is not None
                        and best_peak_snr >= DEFAULT_PEAK_SNR_SUCCESS_THRESHOLD
                    )
                    prefer_cand = (
                        (cand_is_valid and not best_is_valid)
                        or (
                            cand_snr is not None
                            and (best_peak_snr is None or cand_snr > best_peak_snr)
                        )
                        or (
                            cand_snr is None
                            and best_peak_snr is None
                            and cand_score > best_score
                        )
                    )
                    if prefer_cand:
                        logger.info(
                            "%s alignment scored higher peak SNR (%.2f vs %.2f; "
                            "offset %.3fs vs %.3fs); using %s",
                            cand_label,
                            cand_snr or 0.0,
                            best_peak_snr or 0.0,
                            cand_offset / float(SAMPLE_RATE),
                            offset_samples / float(SAMPLE_RATE),
                            cand_label,
                        )
                        reference_speech = cand_speech
                        aligner = cand_aligner
                        best_score = cand_score
                        offset_samples = cand_offset
                        best_srt_pipe = cand_pipe
                        best_peak_snr = cand_snr
                        best_peak_margin = getattr(
                            cand_aligner, "best_peak_margin_", None
                        )
                        if "audio" in cand_label.lower() or "center channel" in cand_label.lower():
                            reference_kind = "audio"
                    else:
                        logger.info(
                            "keeping primary alignment over %s (peak SNR %.2f vs %.2f)",
                            cand_label,
                            best_peak_snr or 0.0,
                            cand_snr or 0.0,
                        )
            if best_score < 0:
                if (
                    best_peak_snr is not None
                    and best_peak_snr >= DEFAULT_PEAK_SNR_SUCCESS_THRESHOLD
                ):
                    logger.info(
                        "raw score was negative (%.1f) due to speech-density mismatch, "
                        "but peak SNR is strong (%.2f >= %.1f); accepting alignment",
                        best_score,
                        best_peak_snr,
                        DEFAULT_PEAK_SNR_SUCCESS_THRESHOLD,
                    )
                else:
                    sync_was_successful = False
            logger.info("...done")
            offset_seconds = (
                offset_samples / float(SAMPLE_RATE) + args.apply_offset_seconds
            )
            scale_step = best_srt_pipe.named_steps["scale"]
            if best_peak_snr is not None:
                logger.info("score: %.3f (peak SNR: %.2f)", best_score, best_peak_snr)
            else:
                logger.info("score: %.3f", best_score)
            logger.info("offset seconds: %.3f", offset_seconds)
            logger.info("framerate scale factor: %.3f", scale_step.scale_factor)
            low_quality_reasons: List[str] = []
            if getattr(args, "skip_sync_on_low_quality", False):
                low_quality_reasons = assess_alignment_quality(
                    best_score,
                    offset_seconds,
                    scale_step.scale_factor,
                    min_score=args.min_score,
                    max_offset_seconds=args.quality_max_offset_seconds,
                    max_framerate_deviation=args.max_framerate_deviation,
                )
            if low_quality_reasons:
                logger.warning(
                    "low-quality alignment (%s); leaving subtitles unmodified",
                    "; ".join(low_quality_reasons),
                )
                sync_was_successful = False
                # write the original (unscaled, unshifted) subtitles unchanged
                original_subs = best_srt_pipe.named_steps["parse"].subs_
                out_subs = original_subs.clone_props_for_subs(list(original_subs))
                if args.output_encoding != "same":
                    out_subs = out_subs.set_encoding(args.output_encoding)
                logger.info(
                    "writing original (unsynced) output to {}".format(srtout or "stdout")
                )
                out_subs.write_file(srtout)
                continue
            use_split = use_split_early and reference_speech is not None
            use_auto_split = use_auto_split_early and reference_speech is not None
            split_segments: Optional[List[Dict[str, Any]]] = None
            dropped_cues = 0
            alignment_profile = normalize_alignment_profile(
                getattr(args, "alignment_profile", DEFAULT_ALIGNMENT_PROFILE)
            )
            profile_offset_seconds: float = 0.0
            if use_split or use_auto_split:
                length_penalty = getattr(
                    args, "split_length_penalty", DEFAULT_SPLIT_LENGTH_PENALTY
                )
                subsample = max(1, int(getattr(args, "split_subsample", 1) or 1))
                max_off = abs(int(args.max_offset_seconds * SAMPLE_RATE))
                offsets_samples: List[Optional[float]]
                if use_split:
                    # Joint framerate + split search: score the piecewise alignment at
                    # every candidate framerate scale (reusing the pipelines the global
                    # search already fit) and keep the best-scoring one, rather than
                    # committing to the scale the single-offset FFT search preferred.
                    candidate_pipes = [p for p in srt_pipes if not callable(p)]
                    if best_srt_pipe not in candidate_pipes:
                        candidate_pipes.append(cast(Pipeline, best_srt_pipe))
                    best_split: Optional[Tuple[float, List[float], Pipeline]] = None
                    for cand_pipe in candidate_pipes:
                        cand_offsets, cand_score = compute_split_offsets(
                            reference_speech,
                            list(cand_pipe.named_steps["scale"].subs_),
                            sample_rate=SAMPLE_RATE,
                            start_seconds=args.start_seconds,
                            split_penalty=split_penalty * SAMPLE_RATE,
                            length_penalty=length_penalty,
                            offset_step_samples=1.0 / subsample,
                            max_offset_samples=max_off,
                            return_score=True,
                            log_segments=False,
                        )
                        if best_split is None or cand_score > best_split[0]:
                            best_split = (cand_score, cand_offsets, cand_pipe)
                    assert best_split is not None
                    _, split_offs, best_split_pipe = best_split
                    offsets_samples = list(split_offs)
                    if best_split_pipe is not best_srt_pipe:
                        logger.info(
                            "split search preferred framerate scale %.3f (single-offset "
                            "search had chosen %.3f)",
                            best_split_pipe.named_steps["scale"].scale_factor,
                            scale_step.scale_factor,
                        )
                    scale_step = best_split_pipe.named_steps["scale"]
                else:
                    auto_max_search = max(
                        float(args.max_offset_seconds),
                        DEFAULT_AUTO_SPLIT_MAX_OFFSET_SECONDS,
                    )
                    auto_search_samples = int(round(auto_max_search * SAMPLE_RATE))
                    precomputed_offsets: Dict[int, float] = {}
                    if aligner is not None:
                        for cand_score_tuple, cand_pipe in getattr(
                            aligner, "_scores", []
                        ):
                            precomputed_offsets[id(cand_pipe)] = float(
                                cand_score_tuple[1]
                            )

                    def _run_auto_for_pipe(
                        cand_pipe: Pipeline, cand_global: float
                    ) -> Tuple[float, List[Optional[float]], Pipeline]:
                        cand_offsets, cand_score = compute_auto_split_offsets(
                            reference_speech,
                            list(cand_pipe.named_steps["scale"].subs_),
                            global_offset_samples=cand_global,
                            sample_rate=SAMPLE_RATE,
                            start_seconds=args.start_seconds,
                            split_penalty=DEFAULT_AUTO_SPLIT_PENALTY * SAMPLE_RATE,
                            length_penalty=length_penalty,
                            offset_step_samples=1.0 / subsample,
                            max_search_samples=auto_search_samples,
                            log_segments=False,
                        )
                        cand_scale = float(cand_pipe.named_steps["scale"].scale_factor)
                        norm_score = cand_score * min(1.0 / cand_scale, 1.0)
                        return norm_score, cand_offsets, cand_pipe

                    best_auto = _run_auto_for_pipe(
                        cast(Pipeline, best_srt_pipe), float(offset_samples)
                    )
                    first_segs = summarize_split_segments(best_auto[1], SAMPLE_RATE)
                    first_dropped = count_dropped_cues(best_auto[1])
                    needs_alt_scale_check = (
                        len(first_segs) > 1
                        or first_dropped > 0
                        or best_score < 0
                        or abs(scale_step.scale_factor - 1.0) > 1e-6
                    )
                    if needs_alt_scale_check:
                        for cand_pipe in srt_pipes:
                            if callable(cand_pipe) or cand_pipe is best_srt_pipe:
                                continue
                            cand_global = precomputed_offsets.get(id(cand_pipe))
                            if cand_global is None:
                                cand_global = float(
                                    FFTAligner(
                                        max_offset_samples=auto_search_samples
                                    ).fit_transform(
                                        reference_speech, cand_pipe.transform(None)
                                    )
                                )
                            cand_res = _run_auto_for_pipe(cand_pipe, cand_global)
                            if cand_res[0] > best_auto[0]:
                                best_auto = cand_res
                    _, offsets_samples, best_auto_pipe = best_auto
                    if best_auto_pipe is not best_srt_pipe:
                        logger.info(
                            "auto-split preferred framerate scale %.3f (single-offset "
                            "search had chosen %.3f)",
                            best_auto_pipe.named_steps["scale"].scale_factor,
                            scale_step.scale_factor,
                        )
                    scale_step = best_auto_pipe.named_steps["scale"]
                log_split_segments(offsets_samples, SAMPLE_RATE)
                dropped_cues = count_dropped_cues(offsets_samples)
                if not sync_was_successful and alignment_has_positive_signal(
                    reference_speech,
                    list(scale_step.subs_),
                    offsets_samples,
                    sample_rate=SAMPLE_RATE,
                    start_seconds=args.start_seconds,
                    length_penalty=length_penalty,
                ):
                    sync_was_successful = True
                if alignment_profile != DEFAULT_ALIGNMENT_PROFILE:
                    profile_delta_samples = compute_profile_offset_adjustment(
                        reference_speech,
                        list(scale_step.subs_),
                        offsets_samples,
                        profile=alignment_profile,
                        sample_rate=SAMPLE_RATE,
                        start_seconds=args.start_seconds,
                    )
                    profile_offset_seconds = profile_delta_samples / float(SAMPLE_RATE)
                    if abs(profile_delta_samples) > 1e-6:
                        offsets_samples = [
                            None if off is None else off + profile_delta_samples
                            for off in offsets_samples
                        ]
                        logger.info(
                            "alignment profile '%s': adjusted block offset by %+.3fs",
                            alignment_profile,
                            profile_offset_seconds,
                        )
                offsets_seconds: List[Optional[float]] = [
                    None
                    if off is None
                    else off / float(SAMPLE_RATE) + args.apply_offset_seconds
                    for off in offsets_samples
                ]
                valid_offsets = [off for off in offsets_seconds if off is not None]
                split_segments = [
                    {
                        "count": count,
                        "offset_seconds": off_s + args.apply_offset_seconds,
                    }
                    for count, off_s in summarize_split_segments(
                        offsets_samples, SAMPLE_RATE
                    )
                ]
                if len(split_segments) <= 1 and dropped_cues == 0 and valid_offsets:
                    offset_seconds = valid_offsets[0]
                    shift_step: TransformerMixin = SubtitleShifter(offset_seconds)
                else:
                    shift_step = VariableSubtitleShifter(offsets_seconds)
                    if valid_offsets:
                        offset_seconds = float(np.median(valid_offsets))
            else:
                if (
                    not skip_sync
                    and reference_speech is not None
                    and alignment_profile != DEFAULT_ALIGNMENT_PROFILE
                ):
                    profile_delta_samples = compute_profile_offset_adjustment(
                        reference_speech,
                        list(scale_step.subs_),
                        float(offset_samples),
                        profile=alignment_profile,
                        sample_rate=SAMPLE_RATE,
                        start_seconds=args.start_seconds,
                    )
                    profile_offset_seconds = profile_delta_samples / float(SAMPLE_RATE)
                    if abs(profile_delta_samples) > 1e-6:
                        offset_seconds += profile_offset_seconds
                        logger.info(
                            "alignment profile '%s': adjusted block offset by %+.3fs "
                            "(final offset: %.3fs)",
                            alignment_profile,
                            profile_offset_seconds,
                            offset_seconds,
                        )
                shift_step = SubtitleShifter(offset_seconds)
            output_steps: List[Tuple[str, TransformerMixin]] = [("shift", shift_step)]
            if args.merge_with_reference:
                output_steps.append(
                    ("merge", SubtitleMerger(reference_pipe.named_steps["parse"].subs_))
                )
            output_pipe = Pipeline(output_steps)
            out_subs = output_pipe.fit_transform(scale_step.subs_)
            out_subs, clipped_cues = out_subs.sanitize_negative_timestamps()
            if clipped_cues > 0:
                logger.warning(
                    "%d subtitle cue(s) shifted before 00:00:00 and were omitted",
                    clipped_cues,
                )
            quality: Optional[Dict[str, Any]] = None
            if not skip_sync:
                try:
                    quality = compute_quality_metrics(
                        reference_speech,
                        list(out_subs),
                        sample_rate=SAMPLE_RATE,
                        start_seconds=args.start_seconds,
                        best_peak_snr=best_peak_snr,
                        best_peak_margin=best_peak_margin,
                        sync_was_successful=sync_was_successful,
                        total_input_cues=len(scale_step.subs_),
                        dropped_cues=dropped_cues,
                        clipped_cues=clipped_cues,
                        n_split_segments=len(split_segments) if split_segments else 1,
                        sub_format=getattr(out_subs, "_sub_format", "srt"),
                        reference_kind=reference_kind,
                        profile_offset_seconds=profile_offset_seconds,
                    )
                    s_q = quality["sync"]
                    sub_q = quality["subtitle"]
                    logger.info(
                        "sync confidence: %d/100 (consistency=%d, overlap=%d, peak=%d, integrity=%d)",
                        s_q["score"],
                        s_q["metrics"]["temporal_consistency"],
                        s_q["metrics"]["overlap_lift"],
                        s_q["metrics"]["peak_confidence"],
                        s_q["metrics"]["sync_integrity"],
                    )
                    logger.info(
                        "subtitle quality: %d/100 (coverage=%d, speed=%d, timing=%d, text=%d)",
                        sub_q["score"],
                        sub_q["metrics"]["timeline_coverage"],
                        sub_q["metrics"]["reading_speed"],
                        sub_q["metrics"]["timing_hygiene"],
                        sub_q["metrics"]["text_integrity"],
                    )
                except Exception:
                    logger.debug("failed to compute quality metrics", exc_info=True)
            if args.output_encoding != "same":
                out_subs = out_subs.set_encoding(args.output_encoding)
            suppress_output_thresh = args.suppress_output_if_offset_less_than
            if offset_seconds >= (suppress_output_thresh or float("-inf")):
                logger.info("writing output to {}".format(srtout or "stdout"))
                out_subs.write_file(srtout)
            else:
                logger.warning(
                    "suppressing output because offset %s was less than suppression threshold %s",
                    offset_seconds,
                    args.suppress_output_if_offset_less_than,
                )
        except Exception:
            sync_was_successful = False
            logger.exception("failed to sync %s", srtin)
        else:
            result["offset_seconds"] = offset_seconds
            result["framerate_scale_factor"] = scale_step.scale_factor
            result["split_segments"] = split_segments
            result["dropped_cues"] = dropped_cues
            result["clipped_cues"] = clipped_cues
            result["alignment_profile"] = alignment_profile
            result["profile_offset_seconds"] = profile_offset_seconds
            result["quality"] = quality
    result["sync_was_successful"] = sync_was_successful
    return sync_was_successful


def make_reference_pipe(
    args: argparse.Namespace,
    progress_handler: Optional[Callable[[ProgressInfo], None]] = None,
) -> Pipeline:
    pgs_stream = getattr(args, "pgs_ref_stream", None)
    if pgs_stream is not None:
        # "auto" (bare --pgs-ref-stream flag) → let PGSSpeechTransformer auto-detect
        resolved_stream: Optional[str] = None if pgs_stream == "auto" else pgs_stream
        if resolved_stream is not None and not resolved_stream.startswith("0:"):
            resolved_stream = "0:" + resolved_stream
        return Pipeline(
            [
                (
                    "speech_extract",
                    PGSSpeechTransformer(
                        sample_rate=SAMPLE_RATE,
                        start_seconds=args.start_seconds,
                        ffmpeg_path=args.ffmpeg_path,
                        ref_stream=resolved_stream,
                        gui_mode=args.gui_mode,
                    ),
                ),
            ]
        )
    if getattr(args, "whisper_weights", None) is not None:
        # transcribe the reference audio with ffmpeg's whisper filter and use the
        # transcript timings as the sync reference (an alternative to audio VAD)
        return Pipeline(
            [
                (
                    "speech_extract",
                    WhisperSpeechTransformer(
                        model_path=args.whisper_weights,
                        language=args.language,
                        whisper_args=getattr(args, "whisper_args", None),
                        vad=args.vad,  # reused as whisper's optional VAD-model path
                        sample_rate=SAMPLE_RATE,
                        start_seconds=args.start_seconds,
                        max_duration_seconds=getattr(
                            args, "max_duration_seconds", None
                        ),
                        ffmpeg_path=args.ffmpeg_path,
                        ref_stream=args.reference_stream,
                        gui_mode=args.gui_mode,
                    ),
                ),
            ]
        )
    ref_format = _ref_format(args.reference)
    if ref_format in SUBTITLE_EXTENSIONS:
        if args.vad is not None:
            logger.warning("Vad specified, but reference was not a movie")
        return cast(
            Pipeline,
            make_subtitle_speech_pipeline(
                fmt=ref_format,
                **override(args, encoding=args.reference_encoding or DEFAULT_ENCODING),
            ),
        )
    elif ref_format in ("npy", "npz"):
        if args.vad is not None:
            logger.warning("Vad specified, but reference was not a movie")
        return Pipeline(
            [("deserialize", DeserializeSpeechTransformer(args.non_speech_label))]
        )
    else:
        vad = args.vad or DEFAULT_VAD
        if args.reference_encoding is not None:
            logger.warning(
                "Reference srt encoding specified, but reference was a video file"
            )
        ref_stream = args.reference_stream
        if ref_stream is not None and not ref_stream.startswith("0:"):
            ref_stream = "0:" + ref_stream
        if getattr(args, "multi_segment_sync", False):
            return Pipeline(
                [
                    (
                        "speech_extract",
                        MultiSegmentVideoSpeechTransformer(
                            vad=vad,
                            sample_rate=SAMPLE_RATE,
                            frame_rate=args.frame_rate,
                            non_speech_label=args.non_speech_label,
                            segment_count=getattr(args, "segment_count", 8),
                            skip_intro_outro=getattr(
                                args, "skip_intro_outro", False
                            ),
                            parallel_workers=getattr(args, "parallel_workers", 4),
                            ffmpeg_path=args.ffmpeg_path,
                            ref_stream=ref_stream,
                            vlc_mode=args.vlc_mode,
                            gui_mode=args.gui_mode,
                        ),
                    ),
                ]
            )
        return Pipeline(
            [
                (
                    "speech_extract",
                    VideoSpeechTransformer(
                        vad=vad,
                        sample_rate=SAMPLE_RATE,
                        frame_rate=args.frame_rate,
                        non_speech_label=args.non_speech_label,
                        start_seconds=args.start_seconds,
                        ffmpeg_path=args.ffmpeg_path,
                        ref_stream=ref_stream,
                        vlc_mode=args.vlc_mode,
                        gui_mode=args.gui_mode,
                        max_duration_seconds=getattr(
                            args, "max_duration_seconds", None
                        ),
                        extract_audio_first=getattr(
                            args, "extract_audio_first", False
                        ),
                        min_embedded_subtitle_cues=getattr(
                            args,
                            "min_embedded_subtitle_cues",
                            DEFAULT_MIN_EMBEDDED_SUBTITLE_CUES,
                        ),
                        progress_handler=progress_handler,
                    ),
                ),
            ]
        )


def extract_subtitles_from_reference(args: argparse.Namespace) -> int:
    stream = args.extract_subs_from_stream
    if not stream.startswith("0:s:"):
        stream = "0:s:{}".format(stream)
    elif not stream.startswith("0:") and stream.startswith("s:"):
        stream = "0:{}".format(stream)
    if not stream.startswith("0:s:"):
        logger.error(
            "invalid stream for subtitle extraction: %s", args.extract_subs_from_stream
        )
    ffmpeg_args = [
        ffmpeg_bin_path("ffmpeg", args.gui_mode, ffmpeg_resources_path=args.ffmpeg_path)
    ]
    ffmpeg_args.extend(
        [
            "-y",
            "-nostdin",
            "-loglevel",
            "fatal",
            "-i",
            args.reference,
            "-map",
            "{}".format(stream),
            "-f",
            "srt",
        ]
    )
    if args.srtout is None:
        ffmpeg_args.append("-")
    else:
        ffmpeg_args.append(args.srtout)
    logger.info(
        "attempting to extract subtitles to {} ...".format(
            "stdout" if args.srtout is None else args.srtout
        )
    )
    call_kwargs = subprocess_args(include_stdout=False) if args.srtout is not None else {}
    if args.srtout is None and hasattr(subprocess, "CREATE_NO_WINDOW"):
        call_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    retcode = subprocess.call(ffmpeg_args, **call_kwargs)
    if retcode == 0:
        logger.info("...done")
    else:
        logger.error(
            "ffmpeg unable to extract subtitles from reference; return code %d", retcode
        )
    return retcode


def _detect_srtin_from_reference(reference: str) -> List[str]:
    """Find subtitle files that sit next to the reference and share its name.

    Matches `<reference-stem>.srt` and `<reference-stem>.<suffix>.srt` (e.g.
    `movie.srt`, `movie.en.srt` for a `movie.mkv` reference) in the reference's
    own directory, so detection works regardless of the process's current
    working directory. The reference itself is never returned, so this is safe
    even when the reference is already a subtitle file.
    """
    reference_dir = os.path.dirname(reference) or "."
    reference_stem = os.path.splitext(os.path.basename(reference))[0]
    reference_abspath = os.path.abspath(reference)
    matches = []
    for name in sorted(os.listdir(reference_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".srt":
            continue
        if name.endswith(".synced.srt"):
            continue  # skip our own previously-synced outputs so re-runs are idempotent
        if stem != reference_stem and not stem.startswith(reference_stem + "."):
            continue
        path = os.path.join(reference_dir, name)
        if os.path.abspath(path) == reference_abspath:
            continue
        matches.append(path)
    return matches


def validate_args(args: argparse.Namespace) -> None:
    if args.vlc_mode:
        logger.setLevel(logging.CRITICAL)
    # --vad no longer uses argparse `choices` (in whisper mode it carries a model
    # path), so enforce the named-detector choices here when not in whisper mode.
    if getattr(args, "whisper_weights", None) is None and args.vad is not None:
        if not is_valid_vad(args.vad):
            raise ValueError(
                "invalid --vad {!r}; choose one of: {} (auditok also accepts "
                "a parameter, e.g. auditok:75, auditok:p20 or "
                "auditok:webrtc:2 -- see --help)".format(
                    args.vad, ", ".join(VAD_CHOICES)
                )
            )
    if args.reference is None:
        if args.apply_offset_seconds == 0 or not args.srtin:
            raise ValueError(
                "`reference` required unless `--apply-offset-seconds` specified"
            )
    if args.apply_offset_seconds != 0:
        if not args.srtin:
            args.srtin = [args.reference]
        if not args.srtin:
            raise ValueError(
                "at least one of `srtin` or `reference` must be specified to apply offset seconds"
            )
    if args.srtin:
        if len(args.srtin) > 1 and not args.overwrite_input:
            raise ValueError(
                "cannot specify multiple input srt files without overwriting"
            )
        if len(args.srtin) > 1 and args.make_test_case:
            raise ValueError("cannot specify multiple input srt files for test cases")
        if len(args.srtin) > 1 and args.gui_mode:
            raise ValueError("cannot specify multiple input srt files in GUI mode")
    elif (
        args.reference is not None
        and not is_remote_url(args.reference)  # can't list a remote dir for sibling srts
        and args.extract_subs_from_stream is None
        and not args.gui_mode
        and not args.make_test_case
        and sys.stdin.isatty()  # don't hijack subtitles piped in on stdin
    ):
        # No input subtitles were given (and nothing is piped in), so look for
        # subtitle files alongside the reference that share its name.
        logger.info("no input srt specified; detecting input srt from reference")
        detected = _detect_srtin_from_reference(args.reference)
        if detected:
            for path in detected:
                logger.info("detected input srt: %s", path)
            args.srtin = detected
            if len(detected) > 1 and args.srtout is not None:
                raise ValueError(
                    "detected multiple input srt files but an output file was "
                    "specified; re-run with --overwrite-input or a single input"
                )
            if args.srtout is None and not args.overwrite_input:
                args.auto_srtout = True
                logger.info(
                    "writing synced output alongside each input as "
                    "<name>.synced.srt; pass --overwrite-input to overwrite the "
                    "input file(s) in place instead"
                )
        else:
            logger.info("no input srt detected from reference")
    if (
        args.make_test_case and not args.gui_mode
    ):  # this validation not necessary for gui mode
        if not args.srtin or args.srtout is None:
            raise ValueError(
                "need to specify input and output srt files for test cases"
            )
    if args.overwrite_input:
        if args.extract_subs_from_stream is not None:
            raise ValueError(
                "input overwriting not allowed for extracting subtitles from reference"
            )
        if not args.srtin:
            raise ValueError(
                "need to specify input srt if --overwrite-input "
                "is specified since we cannot overwrite stdin"
            )
        if args.srtout is not None:
            raise ValueError(
                "overwrite input set but output file specified; "
                "refusing to run in case this was not intended"
            )
    if args.extract_subs_from_stream is not None:
        if args.make_test_case:
            raise ValueError("test case is for sync and not subtitle extraction")
        if args.srtin:
            raise ValueError(
                "stream specified for reference subtitle extraction; "
                "-i flag for sync input not allowed"
            )
    alignment_profile = getattr(args, "alignment_profile", None)
    if (
        alignment_profile is not None
        and alignment_profile not in ALIGNMENT_PROFILE_CHOICES
    ):
        raise ValueError(
            "invalid --alignment-profile {!r}; choose one of: {}".format(
                alignment_profile, ", ".join(ALIGNMENT_PROFILE_CHOICES)
            )
        )
    sub_fps = getattr(args, "sub_fps", None)
    if sub_fps is not None and sub_fps <= 0:
        raise ValueError("--sub-fps must be positive (got %r)" % (sub_fps,))
    if args.extract_subs_from_stream is None:
        if args.srtout is not None:
            out_ext = os.path.splitext(args.srtout)[-1][1:].lower()
            if out_ext not in SUPPORTED_OUTPUT_FORMATS:
                raise ValueError(
                    "unsupported output subtitle format: {!r} (supported: {})".format(
                        out_ext or args.srtout,
                        ", ".join(sorted(SUPPORTED_OUTPUT_FORMATS)),
                    )
                )
        elif args.overwrite_input and args.srtin:
            for srtin_path in args.srtin:
                if srtin_path is not None:
                    in_ext = os.path.splitext(srtin_path)[-1][1:].lower()
                    if in_ext not in SUPPORTED_OUTPUT_FORMATS:
                        raise ValueError(
                            "unsupported output subtitle format: {!r} (supported: {})".format(
                                in_ext or srtin_path,
                                ", ".join(sorted(SUPPORTED_OUTPUT_FORMATS)),
                            )
                        )


def validate_file_permissions(args: argparse.Namespace) -> None:
    error_string_template = (
        "unable to {action} {file}; "
        "try ensuring file exists and has correct permissions"
    )
    if (
        args.reference is not None
        and not is_remote_url(args.reference)  # ffmpeg streams URLs directly; no local file to check
        and not os.access(args.reference, os.R_OK)
    ):
        raise ValueError(
            error_string_template.format(action="read reference", file=args.reference)
        )
    if args.srtin:
        for srtin in args.srtin:
            if srtin is not None and not os.access(srtin, os.R_OK):
                raise ValueError(
                    error_string_template.format(
                        action="read input subtitles", file=srtin
                    )
                )
    if (
        args.srtout is not None
        and os.path.exists(args.srtout)
        and not os.access(args.srtout, os.W_OK)
    ):
        raise ValueError(
            error_string_template.format(
                action="write output subtitles", file=args.srtout
            )
        )
    if args.make_test_case or args.serialize_speech:
        npy_savename = os.path.splitext(args.reference)[0] + ".npz"
        if os.path.exists(npy_savename) and not os.access(npy_savename, os.W_OK):
            raise ValueError(
                "unable to write test case file archive %s (try checking permissions)"
                % npy_savename
            )


def _setup_logging(
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[logging.FileHandler]]:
    log_handler = None
    log_path = None
    if args.make_test_case or args.log_dir_path is not None:
        log_path = "ffsubsync.log"
        if args.log_dir_path is not None and os.path.isdir(args.log_dir_path):
            log_path = os.path.join(args.log_dir_path, log_path)
        log_handler = logging.FileHandler(log_path)
        logger.addHandler(log_handler)
        logger.info("this log will be written to %s", os.path.abspath(log_path))
    return log_path, log_handler


def _npy_savename(args: argparse.Namespace) -> str:
    return os.path.splitext(args.reference)[0] + ".npz"


def _run_impl(
    args: argparse.Namespace,
    result: Dict[str, Any],
    progress_handler: Optional[Callable[[ProgressInfo], None]] = None,
) -> bool:
    if args.extract_subs_from_stream is not None:
        result["retval"] = extract_subtitles_from_reference(args)
        return True
    if args.srtin is not None and (
        args.reference is None
        or (len(args.srtin) == 1 and args.srtin[0] == args.reference)
    ):
        return try_sync(args, None, result)
    reference_pipe = make_reference_pipe(args, progress_handler=progress_handler)
    logger.info("extracting speech segments from reference '%s'...", args.reference)
    reference_pipe.fit(args.reference)
    logger.info("...done")
    if args.make_test_case or args.serialize_speech:
        logger.info("serializing speech...")
        np.savez_compressed(
            _npy_savename(args), speech=reference_pipe.transform(args.reference)
        )
        logger.info("...done")
        if not args.srtin:
            logger.info(
                "unsynchronized subtitle file not specified; skipping synchronization"
            )
            return False
    return try_sync(args, reference_pipe, result)


def validate_and_transform_args(
    parser_or_args: Union[argparse.ArgumentParser, argparse.Namespace],
) -> Optional[argparse.Namespace]:
    if isinstance(parser_or_args, argparse.Namespace):
        parser = None
        args = parser_or_args
    else:
        parser = parser_or_args
        args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(e)
        if parser is not None:
            parser.print_usage()
        return None
    if args.gui_mode and args.srtout is None:
        args.srtout = "{}.synced.srt".format(os.path.splitext(args.srtin[0])[0])
    try:
        validate_file_permissions(args)
    except ValueError as e:
        logger.error(e)
        return None
    ref_format = _ref_format(args.reference)
    if args.merge_with_reference and ref_format not in SUBTITLE_EXTENSIONS:
        logger.error(
            "merging synced output with reference only valid "
            "when reference composed of subtitles"
        )
        return None
    return args


def run(
    parser_or_args: Union[argparse.ArgumentParser, argparse.Namespace],
    progress_handler: Optional[Callable[[ProgressInfo], None]] = None,
) -> Dict[str, Any]:
    """Synchronize subtitles.

    ``progress_handler``, if given, is called repeatedly during reference speech
    extraction with a :class:`~ffsubsync.speech_transformers.ProgressInfo`
    describing how much of the reference audio has been processed. This lets a
    host application (e.g. a web UI) display sync progress. The handler is only
    invoked for the video-reference path; exceptions it raises are logged and
    swallowed so a buggy handler cannot abort syncing.
    """
    sync_was_successful = False
    result = {
        "retval": 0,
        "offset_seconds": None,
        "framerate_scale_factor": None,
        "split_segments": None,
        "dropped_cues": 0,
        "clipped_cues": 0,
        "alignment_profile": DEFAULT_ALIGNMENT_PROFILE,
        "profile_offset_seconds": 0.0,
    }
    args = validate_and_transform_args(parser_or_args)
    if args is None:
        result["retval"] = 1
        return result
    log_path, log_handler = _setup_logging(args)
    try:
        sync_was_successful = _run_impl(
            args, result, progress_handler=progress_handler
        )
        result["sync_was_successful"] = sync_was_successful
        return result
    finally:
        if log_handler is not None and log_path is not None:
            log_handler.close()
            logger.removeHandler(log_handler)
            if args.make_test_case:
                result["retval"] += make_test_case(
                    args, _npy_savename(args), sync_was_successful
                )
            if args.log_dir_path is None or not os.path.isdir(args.log_dir_path):
                os.remove(log_path)


def add_main_args_for_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "reference",
        nargs="?",
        help=(
            "Reference (video, subtitles, or a numpy array with VAD speech) "
            "to which to synchronize input subtitles."
        ),
    )
    parser.add_argument(
        "-i",
        "--srtin",
        nargs="*",
        help=(
            "Input subtitles file (default=stdin). If omitted (and nothing is "
            "piped in), subtitles sharing the reference's name in its directory "
            "are auto-detected (e.g. `movie.srt`, `movie.en.srt` for `movie.mkv`) "
            "and each is synced to a `<name>.synced.srt` next to it; pass "
            "--overwrite-input to overwrite the detected file(s) in place."
        ),
    )
    parser.add_argument(
        "-o", "--srtout", help="Output subtitles file (default=stdout)."
    )
    parser.add_argument(
        "--merge-with-reference",
        "--merge",
        action="store_true",
        help="Merge reference subtitles with synced output subtitles.",
    )
    parser.add_argument(
        "--make-test-case",
        "--create-test-case",
        action="store_true",
        help="If specified, serialize reference speech to a numpy array, "
        "and create an archive with input/output subtitles "
        "and serialized speech.",
    )
    parser.add_argument(
        "--reference-stream",
        "--refstream",
        "--reference-track",
        "--reftrack",
        default=None,
        help=(
            "Which stream/track in the video file to use as reference, "
            "formatted according to ffmpeg conventions. For example, 0:s:0 "
            "uses the first subtitle track; 0:a:3 would use the third audio track. "
            "You can also drop the leading `0:`; i.e. use s:0 or a:3, respectively. "
            "Example: `ffs ref.mkv -i in.srt -o out.srt --reference-stream s:2`"
        ),
    )
    parser.add_argument(
        "--pgs-ref-stream",
        "--pgsstream",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Use an image-based subtitle track (PGS, VobSub, or DVB) from "
            "the reference video as the sync reference instead of audio VAD. "
            "Optionally specify the stream (leading `0:` is optional, e.g. `s:0` or `3`). "
            "Omit the value to auto-detect one, skipping commentary and forced tracks "
            "when a full track exists. "
            "Example: `ffs ref.mkv -i in.srt -o out.srt --pgs-ref-stream` (auto) "
            "or `ffs ref.mkv -i in.srt -o out.srt --pgs-ref-stream s:2` (explicit)."
        ),
    )


def add_cli_only_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="{package} {version}".format(
            package="ffblindsubsync", version=get_version()
        ),
    )
    parser.add_argument(
        "--overwrite-input",
        action="store_true",
        help=(
            "If specified, will overwrite the input srt "
            "instead of writing the output to a new file."
        ),
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="What encoding to use for reading input subtitles "
        "(default=%s)." % DEFAULT_ENCODING,
    )
    parser.add_argument(
        "--max-subtitle-seconds",
        type=float,
        default=DEFAULT_MAX_SUBTITLE_SECONDS,
        help="Maximum duration for a subtitle to appear on-screen "
        "(default=%.3f seconds)." % DEFAULT_MAX_SUBTITLE_SECONDS,
    )
    parser.add_argument(
        "--start-seconds",
        type=int,
        default=DEFAULT_START_SECONDS,
        help="Start time for processing "
        "(default=%d seconds)." % DEFAULT_START_SECONDS,
    )
    parser.add_argument(
        "--max-offset-seconds",
        type=float,
        default=DEFAULT_MAX_OFFSET_SECONDS,
        help="The max allowed offset seconds for any subtitle segment "
        "(default=%d seconds)." % DEFAULT_MAX_OFFSET_SECONDS,
    )
    parser.add_argument(
        "--split-penalty",
        type=float,
        nargs="?",
        const=DEFAULT_SPLIT_PENALTY,
        default=None,
        help="Enable alass-style piecewise synchronization: instead of a single "
        "global offset, allow the offset to change across the timeline to correct "
        "commercial breaks, inserted/removed scenes (director's cuts), or discs "
        "concatenated into one file. An optional value is the cost (in seconds of "
        "overlap) of introducing each such split -- lower splits more eagerly, "
        "higher stays closer to a single global offset; values around 4-20 are "
        "typical. Pass the flag with no value to use a reasonable default "
        "(%.0f). If the flag is omitted entirely (the default), a single global "
        "offset is used as before." % DEFAULT_SPLIT_PENALTY,
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        help="Safe two-stage piecewise sync: check timeline health at the global "
        "offset and only introduce monotonic, validated mid-file breaks when a "
        "genuine misaligned region is detected.",
    )
    parser.add_argument(
        "--split-length-penalty",
        type=float,
        default=DEFAULT_SPLIT_LENGTH_PENALTY,
        help="Only meaningful with --split-penalty. Weight of the edge/length term "
        "used when scoring each cue's placement ('standard scoring'): a cue is "
        "charged this fraction of the reference speech just outside its edges, so it "
        "prefers a same-sized speech block over sitting anywhere inside a longer one. "
        "0 falls back to pure overlap scoring (default=%.2f)."
        % DEFAULT_SPLIT_LENGTH_PENALTY,
    )
    parser.add_argument(
        "--split-subsample",
        type=int,
        default=DEFAULT_SPLIT_SUBSAMPLE,
        help="Only meaningful with --split-penalty. Sub-sample resolution of the "
        "offset search: 1 aligns to whole %d Hz samples (%dms); higher values refine "
        "the offset to 1/N of a sample via exact linear interpolation of the "
        "reference. 1 is plenty for perceptual sync (default=%d)."
        % (SAMPLE_RATE, 1000 // SAMPLE_RATE, DEFAULT_SPLIT_SUBSAMPLE),
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=None,
        help="If specified, only process the first this-many seconds of the "
        "reference (measured from --start-seconds). Useful for speeding up "
        "long or remote references, since ffmpeg stops reading/downloading "
        "once this duration is reached.",
    )
    parser.add_argument(
        "--extract-audio-first",
        action="store_true",
        help="For remote URL references, first copy the audio track to a local "
        "temp file (no re-encode) and run speech detection on that, instead of "
        "streaming the full container over the network during detection. Can be "
        "more stable on flaky connections; ignored for local references.",
    )
    parser.add_argument(
        "--multi-segment-sync",
        action="store_true",
        help="Sample a few short segments spread across the reference and run "
        "speech detection only on those, instead of the whole reference. Speeds "
        "up long or remote references; the usual framerate and offset search is "
        "unchanged. Only applies to video / audio references.",
    )
    parser.add_argument(
        "--segment-count",
        type=int,
        default=8,
        help="Number of segments to sample for --multi-segment-sync (default=8).",
    )
    parser.add_argument(
        "--skip-intro-outro",
        action="store_true",
        help="With --multi-segment-sync, skip the first 30s and last 60s of the "
        "reference when placing segments (intros/credits often lack dialogue).",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=4,
        help="How many segments to extract in parallel for --multi-segment-sync "
        "(default=4); useful for overlapping downloads of remote references.",
    )
    parser.add_argument(
        "--apply-offset-seconds",
        type=float,
        default=DEFAULT_APPLY_OFFSET_SECONDS,
        help="Apply a predefined offset in seconds to all subtitle segments "
        "(default=%d seconds)." % DEFAULT_APPLY_OFFSET_SECONDS,
    )
    parser.add_argument(
        "--alignment-profile",
        choices=ALIGNMENT_PROFILE_CHOICES,
        default=DEFAULT_ALIGNMENT_PROFILE,
        help="Global block-level timing profile applied after synchronization "
        "(default=%s). Choices: 'balanced' (default symmetric centering), "
        "'onset' (appear right as speech starts, leaving extra reading time at "
        "the end), 'early' (appear slightly before speech starts for early "
        "reading lead-in)." % DEFAULT_ALIGNMENT_PROFILE,
    )
    parser.add_argument(
        "--skip-sync-on-low-quality",
        action="store_true",
        help="If the alignment looks untrustworthy (see the thresholds below), "
        "leave the subtitles unmodified instead of applying a probably-wrong "
        "sync. Useful for batch jobs where a bad sync is worse than none.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="With --skip-sync-on-low-quality, reject alignments scoring below "
        "this. The score's magnitude is not normalized, but its sign is "
        "meaningful, so the default of %.1f rejects only anti-correlated "
        "(clearly wrong) alignments." % DEFAULT_MIN_SCORE,
    )
    parser.add_argument(
        "--quality-max-offset-seconds",
        type=float,
        default=DEFAULT_QUALITY_MAX_OFFSET_SECONDS,
        help="With --skip-sync-on-low-quality, reject alignments whose offset "
        "exceeds this many seconds (default=%.1f)."
        % DEFAULT_QUALITY_MAX_OFFSET_SECONDS,
    )
    parser.add_argument(
        "--max-framerate-deviation",
        type=float,
        default=DEFAULT_MAX_FRAMERATE_DEVIATION,
        help="With --skip-sync-on-low-quality, reject alignments whose framerate "
        "scale deviates from 1.0 by more than this. The default of %.2f permits "
        "every framerate correction ffsubsync can make (so it never rejects a "
        "legitimate one); tighten it only when you know the framerate should not "
        "change." % DEFAULT_MAX_FRAMERATE_DEVIATION,
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=DEFAULT_FRAME_RATE,
        help="Frame rate for audio extraction (default=%d)." % DEFAULT_FRAME_RATE,
    )
    parser.add_argument(
        "--sub-fps",
        type=float,
        default=None,
        help="Framerate for MicroDVD .sub subtitle files (e.g. 23.976 or 25). "
        "When omitted, inferred from the .sub header or the reference video.",
    )
    parser.add_argument(
        "--skip-infer-framerate-ratio",
        action="store_true",
        help="If set, do not try to infer framerate ratio based on duration ratio.",
    )
    parser.add_argument(
        "--non-speech-label",
        type=float,
        default=DEFAULT_NON_SPEECH_LABEL,
        help="Label to use for frames detected as non-speech (default=%f)"
        % DEFAULT_NON_SPEECH_LABEL,
    )
    parser.add_argument(
        "--output-encoding",
        default="utf-8",
        help="What encoding to use for writing output subtitles "
        '(default=utf-8). Can indicate "same" to use same '
        "encoding as that of the input.",
    )
    parser.add_argument(
        "--reference-encoding",
        help="What encoding to use for reading / writing reference subtitles "
        "(if applicable, default=infer).",
    )
    parser.add_argument(
        "--vad",
        # NB: no argparse `choices=` -- validated in validate_args instead, because
        # in --whisper-weights transcription mode --vad is reused to carry a path
        # to whisper's optional ggml VAD model rather than a named detector.
        default=None,
        help="Which voice activity detector to use for speech extraction "
        "(if using video / audio as a reference, default={default}). Choices: "
        "{choices}. The `fused` options combine webrtc and silero and require the "
        "optional silero dependency (torch). `auditok` can use energy (auto "
        "threshold with otsu, `auditok:pXX` for the XXth percentile with XX in "
        "1-99, `auditok:50` for the historical fixed threshold) or a speech model "
        "(`auditok:webrtc[:N]`, N = aggressiveness 0-3). With --whisper-weights "
        "this instead takes a path to a ggml VAD model for whisper's optional "
        "audio fragmentation.".format(
            default=DEFAULT_VAD, choices=", ".join(VAD_CHOICES)
        ),
    )
    parser.add_argument(
        "--min-embedded-subtitle-cues",
        type=int,
        default=DEFAULT_MIN_EMBEDDED_SUBTITLE_CUES,
        help="With a subs_then_* VAD, ignore embedded text subtitle streams that "
        "have fewer cues than this and fall back to bitmap subs or audio instead "
        "(default=%d). Forced-only tracks (signs, foreign lines) are usually far "
        "below this and produce spurious offsets when used as the reference. Set "
        "to 0 to disable the minimum." % DEFAULT_MIN_EMBEDDED_SUBTITLE_CUES,
    )
    parser.add_argument(
        "--whisper-weights",
        "--whisper-model",
        "--ffmpeg-transcription-model-weights",
        default=None,
        help="Path to a whisper.cpp ggml model file (e.g. ggml-base.en.bin). If "
        "given, the reference audio is transcribed with ffmpeg's whisper filter "
        "and the transcript is used as the sync reference instead of audio VAD "
        "(requires ffmpeg >= 8.0 built with --enable-whisper). '~' is expanded "
        "for you. Example: `ffs video.mp4 -i in.srt -o out.srt "
        "--whisper-weights ~/whisper.cpp/models/ggml-base.en.bin`.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code for --whisper-weights transcription (e.g. en, es, "
        "fr), or 'auto' to let whisper detect it. Default: inferred as 'en' for "
        "*.en model files, else 'auto'. Only used with --whisper-weights.",
    )
    parser.add_argument(
        "--whisper-args",
        default=None,
        help="Extra options for ffmpeg's whisper filter as key=value pairs "
        "separated by ':' (e.g. `queue=12`), overriding ffsubsync's defaults. "
        "The model, format, and destination are managed by ffsubsync and cannot "
        "be overridden. Only used with --whisper-weights.",
    )
    parser.add_argument(
        "--no-fix-framerate",
        action="store_true",
        help="If specified, subsync will not attempt to correct a framerate "
        "mismatch between reference and subtitles.",
    )
    parser.add_argument(
        "--serialize-speech",
        action="store_true",
        help="If specified, serialize reference speech to a numpy array.",
    )
    parser.add_argument(
        "--extract-subs-from-stream",
        "--extract-subtitles-from-stream",
        default=None,
        help="If specified, do not attempt sync; instead, just extract subtitles"
        " from the specified stream using the reference.",
    )
    parser.add_argument(
        "--suppress-output-if-offset-less-than",
        type=float,
        default=None,
        help="If specified, do not produce output if offset below provided threshold.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        "--ffmpegpath",
        default=None,
        help="Where to look for ffmpeg and ffprobe. Uses the system PATH by default.",
    )
    parser.add_argument(
        "--log-dir-path",
        default=None,
        help=(
            "If provided, will save log file ffsubsync.log to this path "
            "(must be an existing directory)."
        ),
    )
    parser.add_argument(
        "--gss",
        action="store_true",
        help="If specified, use golden-section search to try to find"
        "the optimal framerate ratio between video and subtitles.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If specified, refuse to parse srt files with formatting issues.",
    )
    parser.add_argument("--vlc-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gui-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-sync", action="store_true", help=argparse.SUPPRESS)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize subtitles with video.")
    add_main_args_for_cli(parser)
    add_cli_only_args(parser)
    return parser


def main() -> int:
    parser = make_parser()
    return run(parser)["retval"]


if __name__ == "__main__":
    sys.exit(main())
