# -*- coding: utf-8 -*-
import os
from concurrent.futures import as_completed, ThreadPoolExecutor
from contextlib import contextmanager
import logging
import io
import re
import subprocess
import sys
import tempfile
import threading
from datetime import timedelta
from typing import cast, Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import ffmpeg
import numpy as np
import tqdm

from ffsubsync.constants import (
    DEFAULT_ENCODING,
    DEFAULT_MAX_SUBTITLE_SECONDS,
    DEFAULT_MIN_EMBEDDED_SUBTITLE_CUES,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_START_SECONDS,
    DEFAULT_WHISPER_LANGUAGE,
    DEFAULT_WHISPER_QUEUE,
    SAMPLE_RATE,
    VAD_CHOICES,
    is_remote_url,
)
from ffsubsync.ffmpeg_utils import (
    ffmpeg_bin_path,
    install_no_window_ffmpeg_probe,
    subprocess_args,
)
from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.sklearn_shim import TransformerMixin
from ffsubsync.sklearn_shim import Pipeline
from ffsubsync.subtitle_parser import make_subtitle_parser
from ffsubsync.subtitle_transformers import SubtitleScaler

install_no_window_ffmpeg_probe()


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


class ProgressInfo(NamedTuple):
    """Progress emitted to a ``progress_handler`` during speech extraction.

    ``processed_seconds`` is the amount of reference audio decoded so far and
    ``total_seconds`` is the reference's total duration (``None`` when ffprobe
    could not determine it). Use :attr:`fraction` for a 0.0-1.0 ratio.
    """

    processed_seconds: float
    total_seconds: Optional[float]

    @property
    def fraction(self) -> Optional[float]:
        if not self.total_seconds:
            return None
        return min(1.0, self.processed_seconds / self.total_seconds)


def make_subtitle_speech_pipeline(
    fmt: str = "srt",
    encoding: str = DEFAULT_ENCODING,
    caching: bool = False,
    max_subtitle_seconds: int = DEFAULT_MAX_SUBTITLE_SECONDS,
    start_seconds: int = DEFAULT_START_SECONDS,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    parser=None,
    **kwargs,
) -> Union[Pipeline, Callable[[float], Pipeline]]:
    if parser is None:
        parser = make_subtitle_parser(
            fmt,
            encoding=encoding,
            caching=caching,
            max_subtitle_seconds=max_subtitle_seconds,
            start_seconds=start_seconds,
            **kwargs,
        )
    assert parser.encoding == encoding
    assert parser.max_subtitle_seconds == max_subtitle_seconds
    assert parser.start_seconds == start_seconds

    def subpipe_maker(framerate_ratio):
        return Pipeline(
            [
                ("parse", parser),
                ("scale", SubtitleScaler(framerate_ratio)),
                (
                    "speech_extract",
                    SubtitleSpeechTransformer(
                        sample_rate=SAMPLE_RATE,
                        start_seconds=start_seconds,
                        framerate_ratio=framerate_ratio,
                    ),
                ),
            ]
        )

    if scale_factor is None:
        return subpipe_maker
    else:
        return subpipe_maker(scale_factor)


def _parse_auditok_spec(vad: str) -> str:
    """Extract the parameter from an ``auditok[:spec]`` vad name (with or
    without a ``subs_then_`` prefix). Bare ``auditok`` means ``auto``."""
    base = vad.split("subs_then_")[-1]
    return base.split(":", 1)[1] if ":" in base else "auto"


def _make_auditok_detector(
    sample_rate: int,
    frame_rate: int,
    non_speech_label: float,
    vad_spec: str = "auto",
) -> Callable[[bytes], np.ndarray]:
    try:
        import auditok
    except ImportError as e:
        logger.error(
            "Error: auditok not installed!\n"
            "        Consider installing it with `pip install auditok` "
            "(MIT-licensed since v0.2)."
        )
        raise e
    if vad_spec.startswith("webrtc"):
        try:
            import webrtcvad  # noqa: F401
        except ImportError as e:
            logger.error(
                "Error: the auditok:webrtc detector requires the webrtcvad "
                "package (`pip install webrtcvad-wheels`)."
            )
            raise e
    # A numeric spec is a fixed energy threshold in dB on auditok's scale
    # (the historical hard-coded value was 50); anything else ("otsu",
    # "pXX", "webrtc[:N]") is an auditok validator spec. "auto" is otsu,
    # auditok's default estimation method; estimation runs per chunk
    # (~100 s), so the threshold adapts across the reference.
    if vad_spec == "auto":
        vad_spec = "otsu"
    try:
        threshold_kwargs: Dict[str, Any] = {"energy_threshold": float(vad_spec)}
    except ValueError:
        threshold_kwargs = {"validator": vad_spec}
    # threshold estimation degenerates on silent chunks (see the guard in
    # _detect); fixed thresholds and the webrtc backend need no guard
    estimates_threshold = "validator" in threshold_kwargs and not vad_spec.startswith(
        "webrtc"
    )
    bytes_per_frame = 2
    frames_per_window = frame_rate // sample_rate

    def _detect(asegment) -> np.ndarray:
        if isinstance(asegment, np.ndarray):
            asegment = asegment.tobytes()
        length = (
            len(asegment) // bytes_per_frame + frames_per_window - 1
        ) // frames_per_window
        media_bstring = np.full(length, non_speech_label)
        # auditok >= 0.5.1 yields no events when estimating on digitally
        # silent input, but a *near*-silent chunk -- e.g. a dithered black
        # leader or credits -- still has no usable energy distribution, and
        # an estimated threshold can mark the whole chunk as speech. 25 dB
        # on auditok's scale (peak ~ -65 dBFS) is far below any speech.
        if estimates_threshold:
            peak = np.abs(np.frombuffer(asegment, np.int16)).max(initial=1)
            if 20 * np.log10(peak) < 25.0:
                return media_bstring
        events = auditok.split(
            asegment,
            sampling_rate=frame_rate,
            sample_width=bytes_per_frame,
            channels=1,
            analysis_window=1.0 / sample_rate,
            min_dur=0.2,
            max_dur=5,
            max_silence=0.25,
            **threshold_kwargs,
        )
        for event in events:
            start = int(event.start * sample_rate)
            end = min(int(round(event.end * sample_rate)), length)
            media_bstring[start:end] = 1.0
        return media_bstring

    return _detect


def _make_webrtcvad_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float
) -> Callable[[bytes], np.ndarray]:
    import webrtcvad

    vad = webrtcvad.Vad()
    vad.set_mode(3)  # set non-speech pruning aggressiveness from 0 to 3
    window_duration = 1.0 / sample_rate  # duration in seconds
    frames_per_window = int(window_duration * frame_rate + 0.5)
    bytes_per_frame = 2

    def _detect(asegment: bytes) -> np.ndarray:
        media_bstring = []
        failures = 0
        for start in range(0, len(asegment) // bytes_per_frame, frames_per_window):
            stop = min(start + frames_per_window, len(asegment) // bytes_per_frame)
            try:
                is_speech = vad.is_speech(
                    asegment[start * bytes_per_frame : stop * bytes_per_frame],
                    sample_rate=frame_rate,
                )
            except Exception:
                is_speech = False
                failures += 1
            # webrtcvad has low recall on mode 3, so treat non-speech as "not sure"
            media_bstring.append(1.0 if is_speech else non_speech_label)
        return np.array(media_bstring)

    return _detect


def _make_silero_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float
) -> Callable[[bytes], np.ndarray]:
    try:
        import torch
    except ImportError as e:
        logger.error(
            "Error: the silero VAD requires PyTorch, which is not installed!\n"
            "        Install it with `pip install ffsubsync[torch]`, or install\n"
            "        torch directly with `pip install torch` (see https://pytorch.org\n"
            "        for platform-specific instructions). torch is an optional\n"
            "        dependency and is not installed with ffsubsync by default."
        )
        raise e

    window_duration = 1.0 / sample_rate  # duration in seconds
    frames_per_window = int(window_duration * frame_rate + 0.5)
    bytes_per_frame = 1

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )

    exception_logged = False

    def _detect(asegment) -> np.ndarray:
        asegment = np.frombuffer(asegment, np.int16).astype(np.float32) / (1 << 15)
        asegment = torch.FloatTensor(asegment)
        media_bstring = []
        failures = 0
        for start in range(0, len(asegment) // bytes_per_frame, frames_per_window):
            stop = min(start + frames_per_window, len(asegment))
            try:
                speech_prob = model(
                    asegment[start * bytes_per_frame : stop * bytes_per_frame],
                    frame_rate,
                ).item()
            except Exception:
                nonlocal exception_logged
                if not exception_logged:
                    exception_logged = True
                    logger.exception("exception occurred during speech detection")
                speech_prob = 0.0
                failures += 1
            media_bstring.append(1.0 - (1.0 - speech_prob) * (1.0 - non_speech_label))
        return np.array(media_bstring)

    return _detect


# bitmap (image-based) subtitle codecs cannot be muxed to SRT; mapping one
# into an SRT extraction aborts the whole ffmpeg invocation, so they are
# skipped when enumerating embedded subtitle streams
_BITMAP_SUBTITLE_CODECS: frozenset = frozenset(
    {
        "hdmv_pgs_subtitle",
        "dvd_subtitle",
        "dvb_subtitle",
        "dvb_teletext",
        "xsub",
    }
)


def _is_deprioritized_sub_track(
    title: str = "", forced: Any = 0, comment: Any = 0
) -> bool:
    """Return True if a subtitle track appears to be forced/commentary/signs."""
    if str(forced).strip() in ("1", "true", "True") or str(comment).strip() in (
        "1",
        "true",
        "True",
    ):
        return True
    low = (title or "").lower()
    return any(tok in low for tok in ("forced", "commentary", "signs", "songs"))


def _is_sdh_sub_track(title: str = "", hearing_impaired: Any = 0) -> bool:
    """Return True if a subtitle track is SDH / closed-captions (hearing impaired)."""
    if str(hearing_impaired).strip() in ("1", "true", "True"):
        return True
    low = (title or "").lower()
    return bool(
        re.search(r"\b(sdh|cc|hi)\b", low)
        or "hearing impaired" in low
        or "closed caption" in low
    )


def _normalize_lang_code(lang: Any) -> str:
    code = str(lang or "").strip().lower()
    return "" if code in ("", "und", "unk", "unknown") else code


def _primary_audio_language(streams: List[Dict[str, Any]]) -> str:
    """Return the normalized language code of the primary audio stream, or ''."""
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        return ""
    for s in audio_streams:
        disp = s.get("disposition") or {}
        if disp.get("default") and not disp.get("comment"):
            return _normalize_lang_code((s.get("tags") or {}).get("language"))
    for s in audio_streams:
        disp = s.get("disposition") or {}
        title = str((s.get("tags") or {}).get("title") or "").lower()
        if not disp.get("comment") and "commentary" not in title:
            return _normalize_lang_code((s.get("tags") or {}).get("language"))
    return _normalize_lang_code((audio_streams[0].get("tags") or {}).get("language"))


def _probe_embedded_subtitle_streams_with_meta(
    fname: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> Tuple[Optional[List[str]], str, Dict[str, Dict[str, Any]]]:
    """Enumerate usable text subtitle streams and metadata in a single ffprobe pass.

    Returns ``(streams, primary_audio_lang, meta_by_spec)`` where ``streams`` is:
    - ``None`` when ffprobe is unavailable or fails (signal to degrade to blind
      per-stream probing),
    - ``[]`` when ffprobe worked and found no usable text subtitle streams, or
    - a list of ``"0:<index>"`` specifiers for usable text subtitle streams.
    """
    ffprobe_args = [
        ffmpeg_bin_path("ffprobe", gui_mode, ffmpeg_resources_path=ffmpeg_path),
        "-loglevel",
        "fatal",
        "-show_entries",
        "stream=index,codec_type,codec_name:stream_disposition=default,forced,comment,hearing_impaired:stream_tags=language,title",
        "-of",
        "json",
        fname,
    ]
    try:
        process = subprocess.Popen(ffprobe_args, **subprocess_args(include_stdout=True))
        output = process.communicate()[0]
    except OSError as e:
        logger.warning("ffprobe unavailable while enumerating subtitles: %s", e)
        return None, "", {}
    if process.returncode != 0:
        return None, "", {}

    raw_text = output.decode("utf-8", errors="replace").strip()
    if not raw_text:
        return [], "", {}

    # Real ffprobe emits JSON; test stubs may still emit legacy CSV lines.
    try:
        import json

        data = json.loads(raw_text)
        raw_streams = data.get("streams") if isinstance(data, dict) else None
    except ValueError:
        raw_streams = None

    if isinstance(raw_streams, list):
        audio_lang = _primary_audio_language(raw_streams)
        streams: List[str] = []
        meta: Dict[str, Dict[str, Any]] = {}
        for s in raw_streams:
            if s.get("codec_type") != "subtitle":
                continue
            idx = s.get("index")
            if idx is None:
                continue
            codec_name = str(s.get("codec_name") or "").lower()
            if codec_name in _BITMAP_SUBTITLE_CODECS:
                continue
            disp = s.get("disposition") or {}
            tags = s.get("tags") or {}
            title = str(tags.get("title") or "")
            spec = "0:{}".format(idx)
            if _is_deprioritized_sub_track(
                title=title,
                forced=disp.get("forced", 0),
                comment=disp.get("comment", 0),
            ):
                logger.info(
                    "skipping forced/commentary/signs embedded subtitle stream %s",
                    spec,
                )
                continue
            streams.append(spec)
            meta[spec] = {
                "is_sdh": _is_sdh_sub_track(
                    title=title,
                    hearing_impaired=disp.get("hearing_impaired", 0),
                ),
                "lang": _normalize_lang_code(tags.get("language")),
                "is_default": bool(disp.get("default")),
            }
        return streams, audio_lang, meta

    # Fallback parser for legacy CSV ffprobe output (used by unit-test mocks).
    streams = []
    for line in raw_text.splitlines():
        parts = [p.strip() for p in line.strip().split(",", 4)]
        if not parts or not parts[0].isdigit():
            continue
        index = parts[0]
        codec_name = parts[1].lower() if len(parts) > 1 else ""
        if codec_name in _BITMAP_SUBTITLE_CODECS:
            continue
        disp_flags = [p for p in parts[2:4] if p in ("0", "1")]
        is_flagged = any(p == "1" for p in disp_flags)
        title = ",".join(parts[2 + len(disp_flags) :])
        spec = "0:{}".format(index)
        if _is_deprioritized_sub_track(title=title, forced="1" if is_flagged else "0"):
            logger.info(
                "skipping forced/commentary/signs embedded subtitle stream %s", spec
            )
            continue
        streams.append(spec)
    return streams, "", {}


def _probe_embedded_subtitle_streams(
    fname: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> Optional[List[str]]:
    """Enumerate usable text-based subtitle streams in ``fname`` via ffprobe."""
    streams, _, _ = _probe_embedded_subtitle_streams_with_meta(
        fname, ffmpeg_path=ffmpeg_path, gui_mode=gui_mode
    )
    return streams


_FUSION_STRATEGIES: Tuple[str, ...] = ("weighted", "intersection", "union")


def _make_fused_detector(
    sample_rate: int,
    frame_rate: int,
    non_speech_label: float,
    fusion_strategy: str = "weighted",
) -> Callable[[bytes], np.ndarray]:
    """Combine the webrtc and silero VADs into a single detector.

    Requires the optional silero dependency (torch); a clear error is raised if
    it is missing. ``fusion_strategy`` controls how the two are combined:
    ``intersection`` (speech only where both agree -- conservative),
    ``union`` (speech where either fires -- aggressive), or ``weighted``
    (default; ``0.6 * silero + 0.4 * webrtc``).
    """
    if fusion_strategy not in _FUSION_STRATEGIES:
        raise ValueError(
            "unknown fused VAD strategy %r; choose one of %s"
            % (fusion_strategy, ", ".join(_FUSION_STRATEGIES))
        )
    webrtc_detector = _make_webrtcvad_detector(
        sample_rate, frame_rate, non_speech_label
    )
    silero_detector = _make_silero_detector(
        sample_rate, frame_rate, non_speech_label
    )

    def _detect(asegment) -> np.ndarray:
        webrtc_result = webrtc_detector(asegment)
        silero_result = silero_detector(asegment)
        # the two detectors can disagree by a frame at the tail; clip to common length
        min_len = min(len(webrtc_result), len(silero_result))
        webrtc_result = webrtc_result[:min_len]
        silero_result = silero_result[:min_len]
        if fusion_strategy == "intersection":
            return np.minimum(webrtc_result, silero_result)
        elif fusion_strategy == "union":
            return np.maximum(webrtc_result, silero_result)
        else:  # weighted
            return 0.6 * silero_result + 0.4 * webrtc_result

    return _detect


class ComputeSpeechFrameBoundariesMixin:
    def __init__(self) -> None:
        self.start_frame_: Optional[int] = None
        self.end_frame_: Optional[int] = None

    @property
    def num_frames(self) -> Optional[int]:
        if self.start_frame_ is None or self.end_frame_ is None:
            return None
        return self.end_frame_ - self.start_frame_

    def fit_boundaries(
        self, speech_frames: np.ndarray
    ) -> "ComputeSpeechFrameBoundariesMixin":
        nz = np.nonzero(speech_frames > 0.5)[0]
        if len(nz) > 0:
            self.start_frame_ = int(np.min(nz))
            self.end_frame_ = int(np.max(nz))
        return self


_CENTER_CHANNEL_LAYOUT_PREFIXES = (
    "3.0",
    "3.1",
    "4.0",
    "4.1",
    "5.0",
    "5.1",
    "6.0",
    "6.1",
    "7.0",
    "7.1",
)
_CENTER_CHANNEL_LAYOUT_NAMES = frozenset({"hexagonal", "octagonal"})
_NO_CENTER_LAYOUTS = frozenset({"mono", "stereo", "2.1", "2.2", "quad", "quad(side)", "2.1.2"})


def _layout_has_center_channel(channel_layout: str) -> bool:
    """Return True if an ffprobe ``channel_layout`` includes a dedicated ``FC`` channel."""
    low = (channel_layout or "").strip().lower()
    if not low or low == "unknown" or low in _NO_CENTER_LAYOUTS:
        return False
    if low in _CENTER_CHANNEL_LAYOUT_NAMES:
        return True
    if low.startswith(_CENTER_CHANNEL_LAYOUT_PREFIXES):
        return True
    return "fc" in re.split(r"[^a-z0-9]+", low)


def _probe_audio_stream_has_center(
    probe_data: Optional[Dict[str, Any]], ref_stream: Optional[str] = None
) -> bool:
    """Return True when the target audio stream in ``probe_data`` is surround with ``FC``."""
    if not isinstance(probe_data, dict):
        return False
    streams = probe_data.get("streams")
    if not isinstance(streams, list):
        return False
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        return False

    target = None
    if ref_stream:
        spec = ref_stream[2:] if ref_stream.startswith("0:") else ref_stream
        if spec.startswith("a:"):
            try:
                idx = int(spec.split(":", 1)[1])
                if 0 <= idx < len(audio_streams):
                    target = audio_streams[idx]
            except ValueError:
                pass
        elif spec.isdigit():
            global_idx = int(spec)
            for s in audio_streams:
                if s.get("index") == global_idx:
                    target = s
                    break
    if target is None:
        for s in audio_streams:
            if (s.get("disposition") or {}).get("default"):
                target = s
                break
        if target is None:
            target = audio_streams[0]

    try:
        channels = int(target.get("channels") or 0)
    except (TypeError, ValueError):
        channels = 0
    if channels < 3:
        return False
    return _layout_has_center_channel(str(target.get("channel_layout") or ""))


class VideoSpeechTransformer(TransformerMixin):
    def __init__(
        self,
        vad: str,
        sample_rate: int,
        frame_rate: int,
        non_speech_label: float,
        start_seconds: int = 0,
        ffmpeg_path: Optional[str] = None,
        ref_stream: Optional[str] = None,
        vlc_mode: bool = False,
        gui_mode: bool = False,
        max_duration_seconds: Optional[float] = None,
        extract_audio_first: bool = False,
        progress_handler: Optional[Callable[["ProgressInfo"], None]] = None,
        min_embedded_subtitle_cues: int = DEFAULT_MIN_EMBEDDED_SUBTITLE_CUES,
    ) -> None:
        super(VideoSpeechTransformer, self).__init__()
        self.vad: str = vad
        self.min_embedded_subtitle_cues: int = min_embedded_subtitle_cues
        self.sample_rate: int = sample_rate
        self.frame_rate: int = frame_rate
        self._non_speech_label: float = non_speech_label
        self.start_seconds: int = start_seconds
        self.ffmpeg_path: Optional[str] = ffmpeg_path
        self.ref_stream: Optional[str] = ref_stream
        self.vlc_mode: bool = vlc_mode
        self.gui_mode: bool = gui_mode
        self.max_duration_seconds: Optional[float] = max_duration_seconds
        self.extract_audio_first: bool = extract_audio_first
        self.progress_handler: Optional[
            Callable[["ProgressInfo"], None]
        ] = progress_handler
        self.video_speech_results_: Optional[np.ndarray] = None
        self.alt_video_speech_results_: List[Tuple[str, np.ndarray]] = []
        self.used_embedded_subs_: bool = False
        self.reference_source_: Optional[str] = None
        self._last_ffmpeg_stderr_: str = ""
        self._audio_fallback_cache_: Optional[
            Tuple[str, np.ndarray, List[Tuple[str, np.ndarray]]]
        ] = None

    @property
    def can_fallback_to_audio(self) -> bool:
        """Return True when alignment used auto-selected embedded subs and can retry audio."""
        return self.used_embedded_subs_ and self.ref_stream is None

    def _probe_embedded_subtitle_streams(self, fname: str) -> Optional[List[str]]:
        streams, audio_lang, meta = _probe_embedded_subtitle_streams_with_meta(
            fname, ffmpeg_path=self.ffmpeg_path, gui_mode=self.gui_mode
        )
        self._last_probe_audio_lang_ = audio_lang
        self._last_probe_sub_meta_ = meta
        return streams

    def _extract_embedded_subs_single_pass(
        self, fname: str, streams: List[str]
    ) -> Optional[List[Tuple[str, io.BytesIO]]]:
        """Extract several subtitle streams in one ffmpeg invocation.

        ffmpeg can only send a single output to stdout, so each stream is
        written to a temporary file (in the system temp dir -- never next to
        the source media), read back into memory, and then deleted along with
        the temp dir. Returns one ``(stream, buffer)`` pair per stream that
        produced data, or ``None`` if the ffmpeg invocation failed wholesale
        (the caller then falls back to per-stream extraction).
        """
        with tempfile.TemporaryDirectory(prefix="ffsubsync_subs_") as tmpdir:
            ffmpeg_args = [
                ffmpeg_bin_path(
                    "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
                ),
                "-loglevel",
                "fatal",
                "-nostdin",
                "-i",
                fname,
            ]
            out_paths: List[str] = []
            for i, stream in enumerate(streams):
                out_path = os.path.join(tmpdir, "embedded.{}.srt".format(i))
                out_paths.append(out_path)
                ffmpeg_args.extend(
                    ["-map", "{}".format(stream), "-f", "srt", out_path]
                )
            process = subprocess.Popen(
                ffmpeg_args, **subprocess_args(include_stdout=True)
            )
            process.communicate()
            if process.returncode != 0:
                return None
            buffers: List[Tuple[str, io.BytesIO]] = []
            for stream, out_path in zip(streams, out_paths):
                if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                    continue
                with open(out_path, "rb") as f:
                    buffers.append((stream, io.BytesIO(f.read())))
            return buffers

    def _extract_embedded_subs_per_stream(
        self, fname: str, streams: List[str]
    ) -> List[Tuple[str, io.BytesIO]]:
        """Extract subtitle streams one ffmpeg invocation at a time (to stdout).

        This preserves the original extraction behavior and is used as a
        fallback whenever single-pass extraction is unavailable (no ffprobe) or
        fails. Stops at the first stream ffmpeg cannot extract. Returns
        ``(stream, buffer)`` pairs.
        """
        buffers: List[Tuple[str, io.BytesIO]] = []
        for stream in streams:
            ffmpeg_args = [
                ffmpeg_bin_path(
                    "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
                ),
                "-loglevel",
                "fatal",
                "-nostdin",
                "-i",
                fname,
                "-map",
                "{}".format(stream),
                "-f",
                "srt",
                "-",
            ]
            process = subprocess.Popen(
                ffmpeg_args, **subprocess_args(include_stdout=True)
            )
            output = process.communicate()[0]
            if process.returncode != 0:
                break
            buffers.append((stream, io.BytesIO(output)))
        return buffers

    def try_fit_using_embedded_subs(self, fname: str) -> None:
        if self.ref_stream is not None:
            # a specific stream was requested; extract just that one
            subtitle_buffers = self._extract_embedded_subs_per_stream(
                fname, [self.ref_stream]
            )
        else:
            # enumerate the subtitle streams so they can all be extracted in a
            # single ffmpeg pass (~5x faster than one invocation per stream)
            streams = self._probe_embedded_subtitle_streams(fname)
            if streams is None:
                # ffprobe unavailable/failed: fall back to probing the first 5
                # streams individually (should cover 99% of movies)
                subtitle_buffers = self._extract_embedded_subs_per_stream(
                    fname, list(map("0:s:{}".format, range(5)))
                )
            elif not streams:
                # ffprobe worked and every subtitle stream is bitmap or forced;
                # do not probe blindly, that would pick the skipped streams up
                # again
                raise ValueError(
                    "Video file appears to lack a usable text subtitle stream "
                    "(none, bitmap-only, or forced-only)"
                )
            else:
                subtitle_buffers = self._extract_embedded_subs_single_pass(
                    fname, streams
                )
                if subtitle_buffers is None:
                    # single pass failed; degrade to per-stream over the same
                    # (known-present) streams
                    subtitle_buffers = self._extract_embedded_subs_per_stream(
                        fname, streams
                    )
        embedded_subs: List[
            Tuple[str, int, float, float, SubtitleSpeechTransformer]
        ] = []
        for stream, buffer in subtitle_buffers:
            pipe = cast(
                Pipeline,
                make_subtitle_speech_pipeline(start_seconds=self.start_seconds),
            ).fit(buffer)
            speech_step = pipe.steps[-1][1]
            num_cues = len(pipe.named_steps["parse"].subs_)
            speech_samples = float(np.sum(speech_step.subtitle_speech_results_ > 0))
            embedded_subs.append(
                (
                    stream,
                    num_cues,
                    speech_samples,
                    float(speech_step.max_time_ or 0.0),
                    speech_step,
                )
            )
        if len(embedded_subs) == 0:
            if self.ref_stream is None:
                error_msg = "Video file appears to lack subtitle stream"
            else:
                error_msg = "Stream {} not found".format(self.ref_stream)
            raise ValueError(error_msg)
        # Rank full dialogue tracks cleanly:
        # 1. Full tracks beat sparse/stub tracks (e.g. unflagged forced tracks).
        # 2. Clean non-SDH tracks beat SDH tracks (no [MUSIC]/[SFX] pollution).
        # 3. Tracks matching the primary audio language beat foreign translations
        #    (which often have different line segmentation and timing offsets).
        # 4. Default track disposition, then higher cue count / speech / max time.
        audio_lang = getattr(self, "_last_probe_audio_lang_", "")
        stream_meta = getattr(self, "_last_probe_sub_meta_", None) or {}
        max_cues = max(entry[1] for entry in embedded_subs)
        full_cue_floor = max(200, int(0.4 * max_cues))

        def _embedded_rank(
            entry: Tuple[str, int, float, float, SubtitleSpeechTransformer]
        ) -> Tuple[bool, bool, bool, bool, int, float, float]:
            s_spec, s_cues, s_speech, s_max_t, _ = entry
            info = stream_meta.get(s_spec) or {}
            is_full = s_cues >= full_cue_floor
            not_sdh = not info.get("is_sdh", False)
            matches_audio = bool(audio_lang and info.get("lang") == audio_lang)
            is_default = bool(info.get("is_default", False))
            return (
                is_full,
                not_sdh,
                matches_audio,
                is_default,
                s_cues,
                s_speech,
                s_max_t,
            )

        stream, num_cues, _, _, subs_to_use = max(embedded_subs, key=_embedded_rank)
        if self.ref_stream is None and num_cues < self.min_embedded_subtitle_cues:
            raise ValueError(
                "best embedded subtitle stream {} has only {} cues "
                "(< {}); falling back to bitmap/audio".format(
                    stream, num_cues, self.min_embedded_subtitle_cues
                )
            )
        logger.info(
            "using embedded subtitle stream %s (%d cues) as reference",
            stream,
            num_cues,
        )
        self.video_speech_results_ = subs_to_use.subtitle_speech_results_
        self.used_embedded_subs_ = True
        self.reference_source_ = "embedded_text"

    def _build_ffmpeg_args(
        self, fname: str, dual_center: bool = False
    ) -> List[str]:
        ffmpeg_args = [
            ffmpeg_bin_path(
                "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
            )
        ]
        if self.start_seconds > 0:
            ffmpeg_args.extend(
                [
                    "-ss",
                    str(timedelta(seconds=self.start_seconds)),
                ]
            )
        if self.max_duration_seconds is not None:
            # input-side -t: stop reading (and, for remote URLs, downloading)
            # after this many seconds past the seek point
            ffmpeg_args.extend(
                ["-t", str(timedelta(seconds=self.max_duration_seconds))]
            )
        ffmpeg_args.extend(["-loglevel", "error", "-nostdin", "-i", fname])
        if self.ref_stream is not None and self.ref_stream.startswith("0:a:"):
            ffmpeg_args.extend(["-map", self.ref_stream])
        if dual_center:
            # Single-pass 2ch output for surround audio:
            #   ch0 = standard mono downmix, ch1 = isolated Front Center (FC)
            # In 5.1/7.1/Atmos mixes, dialogue lives in FC and -ac 1 downmix
            # attenuates it by ~10-14 dB, hurting WebRTC VAD mode 3 recall.
            af_expr = (
                "aresample=async=1,asplit=2[a][b];"
                "[a]aresample=ochl=mono[m];"
                "[b]pan=mono|c0=FC[c];"
                "[m][c]amerge=inputs=2"
            )
            out_channels = "2"
        else:
            af_expr = "aresample=async=1"
            out_channels = "1"
        ffmpeg_args.extend(
            [
                "-vn",
                "-sn",
                "-dn",
                "-threads",
                "0",
                "-f",
                "s16le",
                "-ac",
                out_channels,
                "-acodec",
                "pcm_s16le",
                "-af",
                af_expr,
                "-ar",
                str(self.frame_rate),
                "-",
            ]
        )
        return ffmpeg_args

    def _extract_audio_to_temp(self, url: str) -> Optional[str]:
        """Copy the reference's audio to a local temp file (no re-encode).

        Returns the temp path, or None if extraction fails (the caller then
        falls back to streaming the URL directly). Uses a Matroska (.mka)
        container, which accepts essentially any audio codec, so ``-acodec
        copy`` is safe regardless of the source's codec.
        """
        fd, temp_path = tempfile.mkstemp(suffix=".mka")
        os.close(fd)
        ffmpeg_args = [
            ffmpeg_bin_path(
                "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
            ),
            "-loglevel",
            "fatal",
            "-nostdin",
            "-y",
            "-i",
            url,
            "-vn",
            "-acodec",
            "copy",
        ]
        if self.max_duration_seconds is not None:
            # extract from t=0 up to start+max so the main pass can still seek
            # --start-seconds accurately within the local file
            limit = self.start_seconds + self.max_duration_seconds
            ffmpeg_args.extend(["-t", str(timedelta(seconds=limit))])
        ffmpeg_args.append(temp_path)
        logger.info("extracting audio from remote reference to %s...", temp_path)
        retcode = subprocess.call(ffmpeg_args, **subprocess_args(include_stdout=False))
        if retcode != 0 or not os.path.getsize(temp_path):
            logger.warning(
                "audio extraction failed (ffmpeg returned %d); "
                "falling back to streaming the reference directly",
                retcode,
            )
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return None
        return temp_path

    def _run_audio_pipeline(self, fname: str) -> None:
        temp_audio = None
        if self.extract_audio_first and is_remote_url(fname):
            temp_audio = self._extract_audio_to_temp(fname)
            if temp_audio is not None:
                fname = temp_audio
        try:
            self._fit_using_audio(fname)
        finally:
            if temp_audio is not None and os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except OSError:
                    logger.warning("failed to remove temp audio file %s", temp_audio)
        self.used_embedded_subs_ = False
        self.reference_source_ = "audio"

    def fallback_to_audio(
        self, fname: str
    ) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
        """Re-extract reference speech from audio VAD after poor embedded-sub alignment.

        Results are cached per ``fname`` and the primary embedded-subtitle speech
        signal is preserved so multi-subtitle batches neither re-decode audio nor
        lose embedded subtitles for subsequent candidates.
        """
        if self._audio_fallback_cache_ is not None and self._audio_fallback_cache_[0] == fname:
            return self._audio_fallback_cache_[1], list(self._audio_fallback_cache_[2])
        prev_results = self.video_speech_results_
        prev_alts = list(self.alt_video_speech_results_)
        prev_used_embedded = self.used_embedded_subs_
        prev_source = self.reference_source_
        try:
            self._run_audio_pipeline(fname)
            assert self.video_speech_results_ is not None
            audio_speech = self.video_speech_results_
            audio_alts = list(self.alt_video_speech_results_)
            self._audio_fallback_cache_ = (fname, audio_speech, audio_alts)
            return audio_speech, list(audio_alts)
        finally:
            if prev_used_embedded and prev_results is not None:
                self.video_speech_results_ = prev_results
                self.alt_video_speech_results_ = prev_alts
                self.used_embedded_subs_ = prev_used_embedded
                self.reference_source_ = prev_source

    def fit(self, fname: str, *_) -> "VideoSpeechTransformer":
        self.used_embedded_subs_ = False
        self.reference_source_ = None
        self._audio_fallback_cache_ = None
        if "subs" in self.vad and (
            self.ref_stream is None or self.ref_stream.startswith("0:s:")
        ):
            try:
                logger.info("Checking video for subtitles stream...")
                self.try_fit_using_embedded_subs(fname)
                logger.info("...success!")
                return self
            except Exception as e:
                logger.info(e)
            if self.ref_stream is None and not is_remote_url(fname):
                try:
                    bitmap_streams = _find_bitmap_timing_streams(
                        fname,
                        ffmpeg_path=self.ffmpeg_path,
                        gui_mode=self.gui_mode,
                        include_deprioritized=False,
                    )
                    for pgs_stream, _, codec_name in bitmap_streams[:3]:
                        try:
                            pgs = PGSSpeechTransformer(
                                sample_rate=self.sample_rate,
                                start_seconds=self.start_seconds,
                                ffmpeg_path=self.ffmpeg_path,
                                ref_stream=pgs_stream,
                                gui_mode=self.gui_mode,
                                progress_handler=self.progress_handler,
                            ).fit(fname)
                            pgs_res = pgs.transform(fname)
                            active_sec = float(np.sum(pgs_res > 0)) / float(
                                self.sample_rate
                            )
                            num_timings = getattr(pgs, "num_timings_", 0)
                            if active_sec >= 50.0 and num_timings >= 20:
                                if codec_name == "hdmv_pgs_subtitle":
                                    kind_label = "PGS"
                                elif codec_name == "dvb_subtitle":
                                    kind_label = "DVB"
                                else:
                                    kind_label = "VobSub"
                                logger.info(
                                    "using %s subtitle stream %s as reference "
                                    "(%d cues, %.1fs active captions)",
                                    kind_label,
                                    pgs_stream,
                                    num_timings,
                                    active_sec,
                                )
                                self.video_speech_results_ = pgs_res
                                self.used_embedded_subs_ = True
                                self.reference_source_ = kind_label.lower()
                                return self
                            logger.info(
                                "skipping bitmap stream %s (%d cues, %.1fs active captions)",
                                pgs_stream,
                                num_timings,
                                active_sec,
                            )
                        except Exception as e:
                            logger.info(
                                "bitmap stream %s failed (%s)", pgs_stream, e
                            )
                except Exception as e:
                    logger.info(
                        "PGS/VobSub/DVB fallback failed (%s); falling back to audio VAD",
                        e,
                    )
        self._run_audio_pipeline(fname)
        return self

    def _make_detector(self) -> Callable[[Any], np.ndarray]:
        if "fused" in self.vad:
            # e.g. "fused" or "fused:intersection"; default strategy is weighted
            fusion_strategy = (
                self.vad.split(":", 1)[1] if ":" in self.vad else "weighted"
            )
            return _make_fused_detector(
                self.sample_rate,
                self.frame_rate,
                self._non_speech_label,
                fusion_strategy,
            )
        elif "auditok" in self.vad:
            # NB: checked before webrtc -- "auditok:webrtc:N" selects auditok's
            # event segmentation over webrtc frame decisions, not plain webrtc
            return _make_auditok_detector(
                self.sample_rate,
                self.frame_rate,
                self._non_speech_label,
                vad_spec=_parse_auditok_spec(self.vad),
            )
        elif "webrtc" in self.vad:
            return _make_webrtcvad_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label
            )
        elif "silero" in self.vad:
            return _make_silero_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label
            )
        else:
            raise ValueError("unknown vad: %s" % self.vad)

    def _decode_audio_vad(
        self,
        fname: str,
        total_duration: Optional[float],
        dual_center: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        detector = self._make_detector()
        detector_fc = self._make_detector() if dual_center else None
        media_bstring: List[np.ndarray] = []
        fc_bstring: List[np.ndarray] = []
        self._last_ffmpeg_stderr_ = ""
        ffmpeg_args = (
            self._build_ffmpeg_args(fname, dual_center=True)
            if dual_center
            else self._build_ffmpeg_args(fname)
        )
        process = subprocess.Popen(ffmpeg_args, **subprocess_args(include_stdout=True))
        stderr_chunks: List[bytes] = []
        stderr_thread: Optional[threading.Thread] = None
        proc_stderr = getattr(process, "stderr", None)
        if proc_stderr is not None:

            def _drain_stderr() -> None:
                try:
                    data = proc_stderr.read()
                    if data:
                        stderr_chunks.append(data)
                except Exception:
                    pass

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

        out_channels = 2 if dual_center else 1
        bytes_per_frame = 2
        bytes_per_sample_step = bytes_per_frame * out_channels
        frames_per_window = bytes_per_sample_step * self.frame_rate // self.sample_rate
        windows_per_buffer = 2500
        simple_progress = 0.0
        carry = b""

        redirect_stderr = None
        tqdm_extra_args = {}
        should_print_redirected_stderr = self.gui_mode
        if self.gui_mode:
            try:
                from contextlib import redirect_stderr  # type: ignore

                tqdm_extra_args["file"] = sys.stdout
            except ImportError:
                should_print_redirected_stderr = False
        if redirect_stderr is None:

            @contextmanager
            def redirect_stderr(enter_result=None):
                yield enter_result

        assert redirect_stderr is not None
        pbar_output = io.StringIO()
        with redirect_stderr(pbar_output):
            with tqdm.tqdm(
                total=total_duration, disable=self.vlc_mode, **tqdm_extra_args
            ) as pbar:
                while True:
                    in_bytes = process.stdout.read(
                        frames_per_window * windows_per_buffer
                    )
                    if not in_bytes:
                        break
                    newstuff = (
                        len(in_bytes)
                        / float(bytes_per_sample_step)
                        / self.frame_rate
                    )
                    if (
                        total_duration is not None
                        and simple_progress + newstuff > total_duration
                    ):
                        newstuff = total_duration - simple_progress
                    simple_progress += newstuff
                    pbar.update(newstuff)
                    if self.progress_handler is not None:
                        try:
                            self.progress_handler(
                                ProgressInfo(
                                    processed_seconds=simple_progress,
                                    total_seconds=total_duration,
                                )
                            )
                        except Exception as e:
                            # a host-supplied callback must never break syncing
                            logger.warning("progress_handler raised: %s", e)
                    if self.vlc_mode and total_duration is not None:
                        print("%d" % int(simple_progress * 100.0 / total_duration))
                        sys.stdout.flush()
                    if should_print_redirected_stderr:
                        assert self.gui_mode
                        # no need to flush since we pass -u to do unbuffered output for gui mode
                        print(pbar_output.read())
                    if dual_center:
                        assert detector_fc is not None
                        buf = carry + in_bytes
                        usable = (len(buf) // frames_per_window) * frames_per_window
                        carry = buf[usable:]
                        if usable == 0:
                            continue
                        stereo = np.frombuffer(buf[:usable], dtype=np.int16).reshape(
                            -1, 2
                        )
                        mix_u8 = np.ascontiguousarray(stereo[:, 0]).view(np.uint8)
                        fc_u8 = np.ascontiguousarray(stereo[:, 1]).view(np.uint8)
                        media_bstring.append(detector(mix_u8))
                        fc_bstring.append(detector_fc(fc_u8))
                    else:
                        # silero (and fused, which wraps it) needs the raw s16le
                        # bytes for its own int16 reinterpretation
                        if "silero" not in self.vad and "fused" not in self.vad:
                            in_bytes = np.frombuffer(in_bytes, np.uint8)
                        media_bstring.append(detector(in_bytes))
        rc = process.wait()
        if stderr_thread is not None:
            stderr_thread.join(timeout=2.0)
        if stderr_chunks:
            self._last_ffmpeg_stderr_ = (
                b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
            )
        return media_bstring, fc_bstring, rc

    def _fit_using_audio(self, fname: str) -> None:
        self.alt_video_speech_results_ = []
        probe_data = None
        try:
            probe_data = ffmpeg.probe(
                fname,
                cmd=ffmpeg_bin_path(
                    "ffprobe",
                    self.gui_mode,
                    ffmpeg_resources_path=self.ffmpeg_path,
                ),
            )
            total_duration: Optional[float] = (
                float(probe_data["format"]["duration"]) - self.start_seconds
            )
        except Exception as e:
            logger.warning(e)
            total_duration = None
        if self.max_duration_seconds is not None and total_duration is not None:
            total_duration = min(total_duration, self.max_duration_seconds)

        use_dual_center = (
            "silero" not in self.vad
            and "fused" not in self.vad
            and _probe_audio_stream_has_center(probe_data, self.ref_stream)
        )
        if use_dual_center:
            logger.info(
                "surround audio with center channel (FC) detected; extracting "
                "downmix + FC in a single pass"
            )
        media_bstring, fc_bstring, rc = self._decode_audio_vad(
            fname, total_duration, dual_center=use_dual_center
        )
        if use_dual_center and len(media_bstring) == 0:
            logger.info(
                "dual-channel surround extraction returned no audio (rc=%s); "
                "falling back to standard mono downmix",
                rc,
            )
            media_bstring, fc_bstring, rc = self._decode_audio_vad(
                fname, total_duration, dual_center=False
            )
        if len(media_bstring) == 0:
            msg = (
                "Unable to detect speech. "
                "Perhaps try specifying a different stream / track, or a different vad."
            )
            stderr_tail = "\n".join(self._last_ffmpeg_stderr_.splitlines()[-5:]).strip()
            if rc != 0 or stderr_tail:
                details = []
                if rc != 0:
                    details.append("ffmpeg exited with code %d" % rc)
                if stderr_tail:
                    details.append("ffmpeg error: %s" % stderr_tail)
                low_err = stderr_tail.lower()
                if any(
                    tok in low_err
                    for tok in ("no such filter", "option not found", "unrecognized option")
                ):
                    details.append(
                        "your ffmpeg build may be too old; try upgrading ffmpeg"
                    )
                msg = "%s (%s)" % (msg, "; ".join(details))
            raise ValueError(msg)
        self.video_speech_results_ = np.concatenate(media_bstring)
        if fc_bstring:
            fc_results = np.concatenate(fc_bstring)
            if np.any(fc_results > self._non_speech_label):
                self.alt_video_speech_results_ = [
                    ("center channel (FC)", fc_results)
                ]
            logger.info(
                "total of speech segments: %s (center channel FC: %s)",
                np.sum(self.video_speech_results_),
                np.sum(fc_results),
            )
        else:
            logger.info(
                "total of speech segments: %s", np.sum(self.video_speech_results_)
            )

    def transform(self, *_) -> np.ndarray:
        return self.video_speech_results_


class MultiSegmentVideoSpeechTransformer(TransformerMixin):
    """Produce a reference speech signal by sampling only a few short segments.

    Instead of extracting and running VAD over the entire reference, this samples
    ``segment_count`` short windows spread across it, runs VAD on each, and places
    the results into a full-length array that is zero everywhere else. That sparse
    signal can be fed straight into the normal alignment path: because it preserves
    each segment's true position on the reference timeline, the existing
    framerate-ratio search and offset cross-correlation recover the same answer
    they would from the full signal -- but only the sampled audio has to be
    extracted (and, for remote URLs, downloaded). Useful for long / remote
    references; for typical local files the plain VideoSpeechTransformer is fine.
    """

    # margins skipped at the very start/end when skip_intro_outro is set, since
    # intros/credits often lack dialogue
    START_MARGIN_SECONDS: int = 30
    END_MARGIN_SECONDS: int = 60

    def __init__(
        self,
        vad: str,
        sample_rate: int,
        frame_rate: int,
        non_speech_label: float,
        segment_count: int = 8,
        segment_duration: int = 60,
        skip_intro_outro: bool = False,
        parallel_workers: int = 4,
        ffmpeg_path: Optional[str] = None,
        ref_stream: Optional[str] = None,
        vlc_mode: bool = False,
        gui_mode: bool = False,
    ) -> None:
        super(MultiSegmentVideoSpeechTransformer, self).__init__()
        # sampling is audio-only, so drop any "subs_then_" prefix (embedded-subtitle
        # extraction ignores the per-segment time window)
        self.vad: str = vad.split("subs_then_")[-1]
        self.sample_rate: int = sample_rate
        self.frame_rate: int = frame_rate
        self._non_speech_label: float = non_speech_label
        self.segment_count: int = segment_count
        self.segment_duration: int = segment_duration
        self.skip_intro_outro: bool = skip_intro_outro
        self.parallel_workers: int = parallel_workers
        self.ffmpeg_path: Optional[str] = ffmpeg_path
        self.ref_stream: Optional[str] = ref_stream
        self.vlc_mode: bool = vlc_mode
        self.gui_mode: bool = gui_mode
        self.video_speech_results_: Optional[np.ndarray] = None

    def _segment_starts(self, total_duration: float) -> List[int]:
        """Evenly-spaced segment start times (seconds) across the reference."""
        duration = self.segment_duration
        if total_duration <= duration:
            return [0]
        start_margin = self.START_MARGIN_SECONDS if self.skip_intro_outro else 0
        end_margin = self.END_MARGIN_SECONDS if self.skip_intro_outro else 0
        lo = float(start_margin)
        hi = total_duration - end_margin
        if hi - lo < duration:  # margins too large for this reference; ignore them
            lo, hi = 0.0, total_duration
        usable = hi - lo - duration
        n = max(1, self.segment_count)
        if usable <= 0 or n == 1:
            return [int(max(0.0, min(lo, total_duration - duration)))]
        step = usable / (n - 1)
        starts = [int(round(lo + i * step)) for i in range(n)]
        # clamp into range and drop duplicates that clamping may create
        starts = [max(0, min(s, int(total_duration) - duration)) for s in starts]
        return sorted(set(starts))

    def _extract_segment_speech(self, fname: str, start: int) -> Tuple[int, np.ndarray]:
        """Run VAD over a single window, returning (start_seconds, speech array)."""
        segment = VideoSpeechTransformer(
            vad=self.vad,
            sample_rate=self.sample_rate,
            frame_rate=self.frame_rate,
            non_speech_label=self._non_speech_label,
            start_seconds=start,
            ffmpeg_path=self.ffmpeg_path,
            ref_stream=self.ref_stream,
            vlc_mode=self.vlc_mode,
            gui_mode=self.gui_mode,
            max_duration_seconds=self.segment_duration,
        )
        segment.fit(fname)
        return start, segment.transform()

    def fit(self, fname: str, *_) -> "MultiSegmentVideoSpeechTransformer":
        try:
            total_duration = float(
                ffmpeg.probe(
                    fname,
                    cmd=ffmpeg_bin_path(
                        "ffprobe", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
                    ),
                )["format"]["duration"]
            )
        except Exception as e:
            raise ValueError(
                "multi-segment sync needs the reference duration, but probing "
                "'%s' failed: %s" % (fname, e)
            )
        starts = self._segment_starts(total_duration)
        logger.info(
            "multi-segment sync: sampling %d segment(s) of up to %ds at %s",
            len(starts),
            self.segment_duration,
            [int(s) for s in starts],
        )
        sparse = np.zeros(int(total_duration * self.sample_rate) + 2, dtype=float)
        workers = max(1, min(self.parallel_workers, len(starts)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_start = {
                executor.submit(self._extract_segment_speech, fname, start): start
                for start in starts
            }
            for future in as_completed(future_to_start):
                start = future_to_start[future]
                try:
                    start, seg_speech = future.result()
                except Exception as e:
                    # a single flaky segment shouldn't sink the whole sync; the
                    # remaining segments still localize the offset
                    logger.warning("failed to extract segment at %ds: %s", start, e)
                    continue
                begin = int(start * self.sample_rate)
                end = min(begin + len(seg_speech), len(sparse))
                if end > begin:
                    sparse[begin:end] = seg_speech[: end - begin]
        if not np.any(sparse > 0):
            raise ValueError(
                "Unable to detect speech in any sampled segment. "
                "Perhaps try specifying a different stream / track, or a different vad."
            )
        self.video_speech_results_ = sparse
        logger.info(
            "total of speech segments: %s", np.sum(self.video_speech_results_)
        )
        return self

    def transform(self, *_) -> np.ndarray:
        return self.video_speech_results_


_PAIRED_NESTER: Dict[str, str] = {
    "(": ")",
    "{": "}",
    "[": "]",
    "（": "）",  # full-width / CJK brackets, common in non-English subtitles
    "【": "】",
    "「": "」",
}

# Markup tags (e.g. <i>, </i>, <font ...>) and ASS override tags ({\an8},
# {\pos(x,y)}, {\i1}) carry no speech. Stripping them before classifying a line
# lets us recognize wrapped cues like "<i>[music]</i>" or "{\an8}(applause)" as
# non-dialogue while still treating "<i>Hello?</i>" as dialogue -- which is why
# '<' is intentionally not a paired nester above, and ASS stripping requires the
# leading backslash so "{laughter}" stays intact.
_MARKUP_TAG: "re.Pattern[str]" = re.compile(r"<[^>]+>")
_ASS_OVERRIDE_TAG: "re.Pattern[str]" = re.compile(r"\{\\[^}]*\}")
_ASS_NEWLINE: "re.Pattern[str]" = re.compile(r"\\[Nn]")
_ASS_HARD_SPACE: "re.Pattern[str]" = re.compile(r"\\h")
_LEADING_DIALOGUE_DASH: "re.Pattern[str]" = re.compile(r"^[\s\-–—•*]+")
_SPEAKER_LABEL_RE: "re.Pattern[str]" = re.compile(r"^[A-Z0-9][A-Z0-9\s.'-]{0,28}:\s*")
_PAIRED_BLOCK_RE: "re.Pattern[str]" = re.compile(
    r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}|（[^（）]*）|【[^【】]*】|「[^「」]*」"
)
_WATERMARK_RE: "re.Pattern[str]" = re.compile(
    r"(?:opensubtitles\.org|subscene\.com|addic7ed\.com|yifysubtitles|"
    r"subdl\.com|podnapisi\.net|advertise your product or brand here)",
    re.IGNORECASE,
)
_BOUNDARY_CREDIT_RE: "re.Pattern[str]" = re.compile(
    r"(?:https?://|www\.|\b(?:synced|sync|subtitles?|resync(?:ed)?|corrected|"
    r"translated|ripped|encoded)\s+by\b|\bopensubtitles\b|\baddic7ed\b|\bsubscene\b)",
    re.IGNORECASE,
)

# Symbols that, on their own, denote a musical / non-speech cue.
_NON_DIALOGUE_SYMBOLS: frozenset = frozenset("♪♫♬♩🎵🎶")

# Ultra-short duplicate-text transition cues (e.g. 10 ms YouTube rolling-caption
# frames) add timing noise without representing separate speech.
_ROLLING_TRANSITION_MAX_SECONDS: float = 0.05


def _strip_subtitle_formatting(content: str) -> str:
    text = _ASS_OVERRIDE_TAG.sub("", content)
    text = _MARKUP_TAG.sub("", text)
    text = _ASS_NEWLINE.sub("\n", text)
    text = _ASS_HARD_SPACE.sub(" ", text)
    return text.strip()


def _is_non_dialogue_line(line: str) -> bool:
    cleaned = _LEADING_DIALOGUE_DASH.sub("", line).strip()
    if not cleaned:
        return True
    if (
        cleaned[0] in _PAIRED_NESTER
        and cleaned[-1] == _PAIRED_NESTER[cleaned[0]]
    ):
        return True
    # lines consisting only of music notes / sound symbols are cues, not speech
    if all(ch.isspace() or ch in _NON_DIALOGUE_SYMBOLS for ch in cleaned):
        return True
    body = cleaned
    m_speaker = _SPEAKER_LABEL_RE.match(body)
    if m_speaker:
        remainder = body[m_speaker.end() :].strip()
        if remainder:
            body = remainder
    if body[0] in _PAIRED_NESTER and body[-1] == _PAIRED_NESTER[body[0]]:
        return True
    if all(ch.isspace() or ch in _NON_DIALOGUE_SYMBOLS for ch in body):
        return True
    residual = body
    for _ in range(3):
        updated = _PAIRED_BLOCK_RE.sub("", residual)
        if updated == residual:
            break
        residual = updated
    if residual != body and not any(ch.isalnum() for ch in residual):
        return True
    return False


def _is_metadata(content: str, is_beginning_or_end: bool) -> bool:
    content = _strip_subtitle_formatting(content)
    if len(content) == 0:
        return True
    if (
        content[0] in _PAIRED_NESTER.keys()
        and content[-1] == _PAIRED_NESTER[content[0]]
    ):
        return True
    if _WATERMARK_RE.search(content):
        return True
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not lines or all(_is_non_dialogue_line(ln) for ln in lines):
        return True
    if is_beginning_or_end:
        if "english" in content.lower():
            return True
        if " - " in content:
            return True
        if _BOUNDARY_CREDIT_RE.search(content):
            return True
    return False


def _classify_speech_cues(subs: List[GenericSubtitle]) -> np.ndarray:
    """Return a boolean mask of cues that represent alignable speech.

    Filters out SDH sound descriptions, release credits/watermarks, and
    ultra-short YouTube rolling-caption duplicate transition frames without
    ever removing cues from the output file. Includes a safety fallback so an
    unusual subtitle file (e.g. mostly bracketed lines) never collapses to an
    empty signal.
    """
    n = len(subs)
    if n == 0:
        return np.zeros(0, dtype=bool)
    is_speech = np.array(
        [not _is_metadata(sub.content, i == 0 or i + 1 == n) for i, sub in enumerate(subs)],
        dtype=bool,
    )
    if n >= 2:
        norm_lines: List[frozenset] = [
            frozenset(
                ln.strip().lower()
                for ln in _strip_subtitle_formatting(sub.content).splitlines()
                if ln.strip()
            )
            for sub in subs
        ]
        for i, sub in enumerate(subs):
            if not is_speech[i]:
                continue
            dur = (sub.end - sub.start).total_seconds()
            if 0.0 <= dur <= _ROLLING_TRANSITION_MAX_SECONDS and norm_lines[i]:
                if (i > 0 and norm_lines[i].issubset(norm_lines[i - 1])) or (
                    i + 1 < n and norm_lines[i].issubset(norm_lines[i + 1])
                ):
                    is_speech[i] = False
    if n >= 10 and int(np.sum(is_speech)) < max(1, int(0.15 * n)):
        logger.info(
            "metadata filtering excluded too many cues (%d/%d kept); "
            "falling back to unfiltered cues for alignment",
            int(np.sum(is_speech)),
            n,
        )
        is_speech = np.array(
            [bool(_strip_subtitle_formatting(sub.content)) for sub in subs],
            dtype=bool,
        )
    return is_speech


class SubtitleSpeechTransformer(TransformerMixin, ComputeSpeechFrameBoundariesMixin):
    def __init__(
        self, sample_rate: int, start_seconds: int = 0, framerate_ratio: float = 1.0
    ) -> None:
        super(SubtitleSpeechTransformer, self).__init__()
        self.sample_rate: int = sample_rate
        self.start_seconds: int = start_seconds
        self.framerate_ratio: float = framerate_ratio
        self.subtitle_speech_results_: Optional[np.ndarray] = None
        self.max_time_: Optional[int] = None

    def fit(self, subs: List[GenericSubtitle], *_) -> "SubtitleSpeechTransformer":
        max_time = 0
        for sub in subs:
            max_time = max(max_time, sub.end.total_seconds())
        self.max_time_ = max_time - self.start_seconds
        samples = np.zeros(int(max_time * self.sample_rate) + 2, dtype=float)
        start_frame = float("inf")
        end_frame = 0
        speech_mask = _classify_speech_cues(subs)
        for i, sub in enumerate(subs):
            if not speech_mask[i]:
                continue
            start = int(
                round(
                    (sub.start.total_seconds() - self.start_seconds) * self.sample_rate
                )
            )
            start_frame = min(start_frame, start)
            duration = sub.end.total_seconds() - sub.start.total_seconds()
            end = start + int(round(duration * self.sample_rate))
            end_frame = max(end_frame, end)
            samples[start:end] = min(1.0 / self.framerate_ratio, 1.0)
        self.subtitle_speech_results_ = samples
        self.fit_boundaries(self.subtitle_speech_results_)
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.subtitle_speech_results_ is not None
        return self.subtitle_speech_results_


class DeserializeSpeechTransformer(TransformerMixin):
    def __init__(self, non_speech_label: float) -> None:
        super(DeserializeSpeechTransformer, self).__init__()
        self._non_speech_label: float = non_speech_label
        self.deserialized_speech_results_: Optional[np.ndarray] = None
        self.alt_video_speech_results_: List[Tuple[str, np.ndarray]] = []

    def fit(self, fname, *_) -> "DeserializeSpeechTransformer":
        loaded = np.load(fname)
        self.alt_video_speech_results_ = []
        try:
            if hasattr(loaded, "files"):
                if "speech" in loaded.files:
                    speech = np.array(loaded["speech"], copy=True)
                else:
                    raise ValueError(
                        'could not find "speech" array in '
                        "serialized file; only contains: %s" % loaded.files
                    )
                if "alt_fc_speech" in loaded.files:
                    fc_speech = np.array(loaded["alt_fc_speech"], copy=True)
                    fc_speech[fc_speech < 1.0] = self._non_speech_label
                    self.alt_video_speech_results_.append(
                        ("center channel (FC)", fc_speech)
                    )
            else:
                speech = loaded
        finally:
            if hasattr(loaded, "close"):
                loaded.close()
        speech[speech < 1.0] = self._non_speech_label
        self.deserialized_speech_results_ = speech
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.deserialized_speech_results_ is not None
        return self.deserialized_speech_results_


_TIMING_BITMAP_SUBTITLE_CODECS = frozenset(
    {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"}
)


def _number_of_frames(stream: Dict[str, Any]) -> int:
    """Return the cue count mkvmerge records in a stream's statistics tags, or 0."""
    for key, value in (stream.get("tags") or {}).items():
        if str(key).upper().startswith("NUMBER_OF_FRAMES"):
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _find_bitmap_timing_streams(
    fname: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
    include_deprioritized: bool = True,
) -> List[Tuple[str, Any, str]]:
    """Return ``(specifier, stream_index, codec_name)`` tuples for PGS/VobSub/DVB tracks.

    Non-forced, non-commentary tracks come first; within each tier, tracks are
    ordered by mkvmerge's ``NUMBER_OF_FRAMES`` cue count descending (preserving
    original stream order on ties).
    """
    try:
        probe = ffmpeg.probe(
            fname,
            cmd=ffmpeg_bin_path("ffprobe", gui_mode, ffmpeg_resources_path=ffmpeg_path),
        )
    except Exception as e:
        logger.warning(
            "ffprobe failed while searching for PGS/VobSub/DVB streams: %s", e
        )
        return []

    raw_streams = probe.get("streams", [])
    audio_lang = _primary_audio_language(raw_streams)

    sub_index = 0
    preferred: List[Tuple[int, bool, bool, bool, Tuple[str, Any, str]]] = []
    fallback: List[Tuple[int, bool, bool, bool, Tuple[str, Any, str]]] = []
    for stream in raw_streams:
        if stream.get("codec_type") == "subtitle":
            codec_name = (stream.get("codec_name") or "").lower()
            if codec_name in _TIMING_BITMAP_SUBTITLE_CODECS:
                specifier = "0:s:{}".format(sub_index)
                disp = stream.get("disposition") or {}
                tags = stream.get("tags") or {}
                entry = (specifier, stream.get("index"), codec_name)
                n_frames = _number_of_frames(stream)
                is_sdh = _is_sdh_sub_track(
                    title=tags.get("title", ""),
                    hearing_impaired=disp.get("hearing_impaired", 0),
                )
                matches_audio = bool(
                    audio_lang and _normalize_lang_code(tags.get("language")) == audio_lang
                )
                is_default = bool(disp.get("default"))
                record = (n_frames, is_sdh, matches_audio, is_default, entry)
                if not _is_deprioritized_sub_track(
                    title=tags.get("title", ""),
                    forced=disp.get("forced", 0),
                    comment=disp.get("comment", 0),
                ):
                    preferred.append(record)
                else:
                    fallback.append(record)
            sub_index += 1

    def _sort_bitmap_tier(
        items: List[Tuple[int, bool, bool, bool, Tuple[str, Any, str]]]
    ) -> List[Tuple[str, Any, str]]:
        if not items:
            return []
        max_frames = max(item[0] for item in items)
        full_floor = max(200, int(0.4 * max_frames)) if max_frames > 0 else 0

        def _rank(
            item: Tuple[int, bool, bool, bool, Tuple[str, Any, str]]
        ) -> Tuple[bool, bool, bool, bool, int]:
            n_frames, is_sdh, matches_audio, is_default, _ = item
            # Treat tracks with unknown (0) frame count or >= full_floor as full so
            # sparse unflagged forced stubs (e.g. 40 cues) lose to full tracks (1500),
            # without letting a translated/SDH track (1861) beat the primary clean track (1500).
            is_full = (max_frames == 0) or (n_frames == 0) or (n_frames >= full_floor)
            return (
                is_full,
                not is_sdh,
                matches_audio,
                is_default,
                0 if is_full else n_frames,
            )

        # Python's sort is stable, so ties preserve the container's stream order.
        ordered = sorted(items, key=_rank, reverse=True)
        return [item[4] for item in ordered]

    pref_entries = _sort_bitmap_tier(preferred)
    if not include_deprioritized:
        return pref_entries
    return pref_entries + _sort_bitmap_tier(fallback)


def find_pgs_stream(
    fname: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> Optional[str]:
    """Return the ffmpeg stream specifier for the best image-based subtitle track in *fname*.

    Considers ``hdmv_pgs_subtitle``, ``dvd_subtitle``, and ``dvb_subtitle``.
    Skips forced, commentary, and signs/songs tracks when a full track exists,
    and prefers the track with the most cues when ``NUMBER_OF_FRAMES`` is present.
    """
    candidates = _find_bitmap_timing_streams(
        fname,
        ffmpeg_path=ffmpeg_path,
        gui_mode=gui_mode,
        include_deprioritized=True,
    )
    if candidates:
        specifier, stream_idx, codec_name = candidates[0]
        logger.info(
            "auto-detected %s stream: %s (ffmpeg stream index %s)",
            codec_name,
            specifier,
            stream_idx,
        )
        return specifier
    return None


def _probe_stream_codec(
    fname: str,
    stream: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> Optional[str]:
    """Return the codec name of *stream* in *fname*, or ``None`` if unknown."""
    probe_stream = stream[2:] if stream.startswith("0:") else stream
    try:
        probe = ffmpeg.probe(
            fname,
            cmd=ffmpeg_bin_path("ffprobe", gui_mode, ffmpeg_resources_path=ffmpeg_path),
            select_streams=probe_stream,
            show_entries="stream=codec_name",
        )
    except Exception:
        return None
    streams = probe.get("streams", [])
    if not streams:
        return None
    codec = streams[0].get("codec_name")
    return codec.lower() if isinstance(codec, str) else None


def _probe_stream_codec_and_duration(
    fname: str,
    stream: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> Tuple[Optional[str], Optional[float]]:
    """Read both stream codec and container duration in a single ffprobe call."""
    probe_stream = stream[2:] if stream.startswith("0:") else stream
    try:
        probe = ffmpeg.probe(
            fname,
            cmd=ffmpeg_bin_path("ffprobe", gui_mode, ffmpeg_resources_path=ffmpeg_path),
            select_streams=probe_stream,
            show_entries="format=duration:stream=codec_name",
        )
    except Exception:
        return None, None
    codec: Optional[str] = None
    streams = probe.get("streams", [])
    if streams:
        raw_codec = streams[0].get("codec_name")
        if isinstance(raw_codec, str):
            codec = raw_codec.lower()
    duration: Optional[float] = None
    raw_dur = (probe.get("format") or {}).get("duration")
    if raw_dur is not None:
        try:
            val = float(raw_dur)
            if val > 0.0 and np.isfinite(val):
                duration = val
        except (TypeError, ValueError):
            pass
    return codec, duration


def _probe_media_duration(
    fname: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> Optional[float]:
    """Quickly read container duration in seconds via ffprobe, or return ``None``."""
    ffprobe_cmd = ffmpeg_bin_path(
        "ffprobe", gui_mode, ffmpeg_resources_path=ffmpeg_path
    )
    kw = subprocess_args(include_stdout=True)
    kw["stdin"] = subprocess.DEVNULL
    kw["stderr"] = subprocess.DEVNULL
    try:
        out = subprocess.check_output(
            [
                ffprobe_cmd,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                fname,
            ],
            timeout=10,
            **{k: v for k, v in kw.items() if k != "stdout"},
        )
        val = float(out.decode("utf-8", errors="replace").strip().splitlines()[0])
        return val if val > 0.0 and np.isfinite(val) else None
    except Exception:
        return None


def _parse_compact_ffprobe_line(line: str) -> Dict[str, str]:
    """Parse one line of ``ffprobe -of compact=p=0:nk=0`` output into a dict."""
    row: Dict[str, str] = {}
    for part in line.strip().split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            row[k.strip()] = v.strip()
    return row


def _stream_ffprobe_compact(
    ffprobe_cmd: str,
    args: List[str],
    total_seconds: Optional[float],
    progress_handler: Callable[[ProgressInfo], None],
) -> Optional[List[Dict[str, str]]]:
    """Stream ``ffprobe -of compact=p=0:nk=0`` line-by-line, emitting live progress."""
    import time

    kw = subprocess_args(include_stdout=True)
    kw["stdin"] = subprocess.DEVNULL
    kw["stderr"] = subprocess.DEVNULL
    try:
        proc = subprocess.Popen([ffprobe_cmd] + args, **kw)
    except Exception:
        return None

    rows: List[Dict[str, str]] = []
    last_emit = 0.0
    last_pts = 0.0
    try:
        try:
            progress_handler(ProgressInfo(0.0, total_seconds))
        except Exception:
            pass
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            text = raw_line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            row = _parse_compact_ffprobe_line(text)
            if not row:
                continue
            rows.append(row)
            pts_raw = row.get("pts_time")
            if pts_raw and pts_raw != "N/A":
                try:
                    pts = float(pts_raw)
                except ValueError:
                    continue
                if pts > last_pts:
                    last_pts = pts
                now = time.monotonic()
                if now - last_emit >= 0.12:
                    last_emit = now
                    try:
                        progress_handler(ProgressInfo(last_pts, total_seconds))
                    except Exception:
                        pass
        proc.wait()
        if proc.returncode != 0 and not rows:
            return None
        if rows:
            done_sec = (
                total_seconds
                if (total_seconds is not None and total_seconds > 0)
                else last_pts
            )
            try:
                progress_handler(ProgressInfo(done_sec, total_seconds))
            except Exception:
                pass
        return rows
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return None
    finally:
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass


def _get_dvd_timings_via_ffprobe(
    fname: str,
    stream: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
    progress_handler: Optional[Callable[[ProgressInfo], None]] = None,
    total_seconds: Optional[float] = None,
) -> Optional[List[Tuple[float, float]]]:
    """Read DVD (VobSub) subtitle timings by letting ffprobe decode the stream.

    Unlike PGS/DVB, a DVD subtitle packet has no separate "clear" packet: its
    stop-display timestamp lives inside the packet itself, and MKV muxers
    commonly leave packet ``duration_time=N/A``. Decoding with ``-show_frames``
    recovers ``start_display_time`` and ``end_display_time`` directly.
    """
    ffprobe_cmd = ffmpeg_bin_path(
        "ffprobe", gui_mode, ffmpeg_resources_path=ffmpeg_path
    )
    probe_stream = stream[2:] if stream.startswith("0:") else stream
    frames: List[Dict[str, Any]]
    if progress_handler is not None:
        if total_seconds is None:
            total_seconds = _probe_media_duration(
                fname, ffmpeg_path=ffmpeg_path, gui_mode=gui_mode
            )
        streamed = _stream_ffprobe_compact(
            ffprobe_cmd,
            [
                "-v",
                "error",
                "-show_frames",
                "-select_streams",
                probe_stream,
                "-show_entries",
                "subtitle=pts_time,start_display_time,end_display_time,num_rects",
                "-of",
                "compact=p=0:nk=0",
                fname,
            ],
            total_seconds=total_seconds,
            progress_handler=progress_handler,
        )
        if streamed is None:
            return None
        frames = streamed
    else:
        try:
            probe_data = ffmpeg.probe(
                fname,
                cmd=ffprobe_cmd,
                show_frames=None,
                select_streams=probe_stream,
                show_entries=(
                    "subtitle=pts_time,start_display_time,end_display_time,num_rects"
                ),
            )
        except Exception:
            return None
        frames = probe_data.get("frames", [])

    captions: List[Tuple[float, float, float]] = []
    for frame in frames:
        try:
            pts_time = float(frame["pts_time"])
            start_ms = int(frame.get("start_display_time", 0))
            end_ms = int(frame["end_display_time"])
            num_rects = int(frame.get("num_rects", 1))
        except (KeyError, TypeError, ValueError):
            continue
        if num_rects == 0:
            # Empty frame clears the screen: ends the previous caption and shows nothing.
            start_ms = end_ms = 0
        captions.append((pts_time, start_ms / 1000.0, end_ms / 1000.0))
    captions.sort()

    results: List[Tuple[float, float]] = []
    for i, (pts_time, start_offset, end_offset) in enumerate(captions):
        start = pts_time + start_offset
        end = pts_time + end_offset
        if i + 1 < len(captions):
            end = min(end, captions[i + 1][0])
        if end > start:
            results.append((start, end))

    return results or None


_PGS_CLEAR_MAX_BYTES = 50
_PGS_FALLBACK_MAX_CUE_SECONDS = 15.0


def _parse_pgs_timings_from_packets(
    packets: List[Dict[str, Any]],
) -> Optional[List[Tuple[float, float]]]:
    """Extract ``(start_s, end_s)`` intervals from ffprobe subtitle packets.

    Handles both numeric ``duration_time`` packets and durationless (``N/A`` or
    omitted) PGS/DVB packets in a single forward pass, including streams that
    mix both representations:

    - Numeric SHOW packet (``size > 50`` and numeric ``duration_time``): emits
      ``(pts_time, pts_time + duration_time)`` when ``duration_time > 0`` and
      clears any pending durationless SHOW.
    - Durationless SHOW packet (``size > 50`` and ``duration_time`` missing or
      ``"N/A"``): closes any earlier pending durationless SHOW (capped at 15s)
      and becomes the pending SHOW awaiting a CLEAR packet.
    - Durationless CLEAR packet (``size <= 50`` and ``duration_time`` missing or
      ``"N/A"``): closes a pending durationless SHOW at ``pts_time`` (capped at 15s).
    """
    parsed_packets: List[Tuple[float, int, Optional[float]]] = []

    for packet in packets:
        pts_time_str = packet.get("pts_time")
        size_str = packet.get("size")
        if pts_time_str is None or size_str is None:
            continue
        try:
            pts_time = float(pts_time_str)
            size = int(size_str)
        except ValueError:
            continue

        duration_time_str = packet.get("duration_time", "N/A")
        duration_time: Optional[float] = None
        if duration_time_str is not None and duration_time_str != "N/A":
            try:
                duration_time = float(duration_time_str)
            except ValueError:
                continue
        parsed_packets.append((pts_time, size, duration_time))

    parsed_packets.sort(key=lambda x: x[0])

    results: List[Tuple[float, float]] = []
    pending_show_pts: Optional[float] = None

    for pts_time, size, duration_time in parsed_packets:
        if size <= _PGS_CLEAR_MAX_BYTES:
            if (
                pending_show_pts is not None
                and duration_time is None
                and pts_time > pending_show_pts
            ):
                end_time = min(pts_time, pending_show_pts + _PGS_FALLBACK_MAX_CUE_SECONDS)
                results.append((pending_show_pts, end_time))
                pending_show_pts = None
            continue

        if duration_time is None:
            if pending_show_pts is not None and pts_time > pending_show_pts:
                end_time = min(pts_time, pending_show_pts + _PGS_FALLBACK_MAX_CUE_SECONDS)
                results.append((pending_show_pts, end_time))
            pending_show_pts = pts_time
            continue

        pending_show_pts = None
        if duration_time > 0:
            results.append((pts_time, pts_time + duration_time))

    return results or None


def _get_pgs_timings_via_ffprobe(
    fname: str,
    stream: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
    progress_handler: Optional[Callable[[ProgressInfo], None]] = None,
    total_seconds: Optional[float] = None,
) -> Optional[List[Tuple[float, float]]]:
    """Read PGS timings from container metadata using ffprobe.

    MKV stores per-packet PTS (and sometimes duration) for subtitle streams, so
    we can get start/end timestamps without extracting or parsing the raw SUP
    binary. Show events are large packets; clear events are tiny (~30-byte)
    packets with ``duration_time=N/A`` (or omitted).

    Returns a list of ``(start_seconds, end_seconds)`` tuples, or ``None`` if
    ffprobe fails or returns no usable timings.
    """
    ffprobe_cmd = ffmpeg_bin_path(
        "ffprobe", gui_mode, ffmpeg_resources_path=ffmpeg_path
    )
    # ffprobe -select_streams does not accept the "0:" input-index prefix;
    # strip it so "0:s:0" → "s:0" and "0:3" → "3".
    probe_stream = stream[2:] if stream.startswith("0:") else stream
    if progress_handler is not None:
        if total_seconds is None:
            total_seconds = _probe_media_duration(
                fname, ffmpeg_path=ffmpeg_path, gui_mode=gui_mode
            )
        streamed = _stream_ffprobe_compact(
            ffprobe_cmd,
            [
                "-v",
                "error",
                "-show_packets",
                "-select_streams",
                probe_stream,
                "-show_entries",
                "packet=pts_time,duration_time,size",
                "-of",
                "compact=p=0:nk=0",
                fname,
            ],
            total_seconds=total_seconds,
            progress_handler=progress_handler,
        )
        if streamed is None:
            return None
        return _parse_pgs_timings_from_packets(streamed)

    try:
        probe_data = ffmpeg.probe(
            fname,
            cmd=ffprobe_cmd,
            show_packets=None,
            select_streams=probe_stream,
            show_entries="packet=pts_time,duration_time,size",
        )
    except Exception:
        return None

    return _parse_pgs_timings_from_packets(probe_data.get("packets", []))


class PGSSpeechTransformer(TransformerMixin, ComputeSpeechFrameBoundariesMixin):
    """Use PGS (Presentation Graphic Stream) subtitle timings as a sync reference.

    PGS subtitles are bitmap-based (e.g. Blu-ray / HDMV) and cannot be converted
    to text by ffmpeg, so they can't be fed through the normal subtitle pipeline.
    However, when muxed into an MKV the container still stores a presentation
    timestamp (``pts_time``) and ``duration_time`` for every subtitle packet, so
    we can recover *when* each caption is on screen without decoding the bitmaps
    or parsing the raw SUP/PCS binary at all.

    This transformer reads those per-packet timings via ``ffprobe`` (see
    :func:`_get_pgs_timings_via_ffprobe`), filtering out the tiny "clear" packets
    that carry no image, and builds the same kind of sparse binary speech signal
    that :class:`SubtitleSpeechTransformer` produces for text subtitles: 1.0 while
    a caption is displayed, 0.0 otherwise. That signal can then be aligned against
    the input subtitle file by the normal ffsubsync pipeline.

    The reference stream may be given explicitly via ``ref_stream`` (with or
    without a leading ``0:``), or left as ``None`` to auto-detect the first
    ``hdmv_pgs_subtitle`` track in the file.
    """

    # PGS is already in the MKV timebase so its duration cannot be compared
    # against the SRT to infer a framerate ratio.  Returning None here prevents
    # the duration-based framerate inference in try_sync from running.
    @property
    def num_frames(self) -> None:
        return None

    def __init__(
        self,
        sample_rate: int,
        start_seconds: int = 0,
        ffmpeg_path: Optional[str] = None,
        ref_stream: Optional[str] = None,
        gui_mode: bool = False,
        progress_handler: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> None:
        super(PGSSpeechTransformer, self).__init__()
        self.sample_rate: int = sample_rate
        self.start_seconds: int = start_seconds
        self.ffmpeg_path: Optional[str] = ffmpeg_path
        self.ref_stream: Optional[str] = ref_stream
        self.gui_mode: bool = gui_mode
        self.progress_handler: Optional[Callable[[ProgressInfo], None]] = progress_handler
        self.pgs_speech_results_: Optional[np.ndarray] = None
        self.num_timings_: int = 0

    def fit(self, fname: str, *_) -> "PGSSpeechTransformer":
        if self.ref_stream is None:
            stream = find_pgs_stream(fname, self.ffmpeg_path, self.gui_mode)
            if stream is None:
                raise ValueError(
                    "No image-based (PGS, VobSub, or DVB) subtitle stream found in {}. "
                    "Specify one explicitly with --pgs-ref-stream.".format(fname)
                )
        else:
            stream = self.ref_stream
            if not stream.startswith("0:"):
                stream = "0:" + stream

        total_seconds: Optional[float] = None
        if self.progress_handler is not None:
            codec, total_seconds = _probe_stream_codec_and_duration(
                fname, stream, self.ffmpeg_path, self.gui_mode
            )
        else:
            codec = _probe_stream_codec(fname, stream, self.ffmpeg_path, self.gui_mode)
        logger.info(
            "reading %s timings for stream %s from %s...",
            codec or "PGS/VobSub/DVB",
            stream,
            fname,
        )
        if codec == "dvd_subtitle":
            timings = _get_dvd_timings_via_ffprobe(
                fname,
                stream,
                self.ffmpeg_path,
                self.gui_mode,
                progress_handler=self.progress_handler,
                total_seconds=total_seconds,
            )
            if timings is None:
                timings = _get_pgs_timings_via_ffprobe(
                    fname,
                    stream,
                    self.ffmpeg_path,
                    self.gui_mode,
                    progress_handler=self.progress_handler,
                    total_seconds=total_seconds,
                )
        else:
            timings = _get_pgs_timings_via_ffprobe(
                fname,
                stream,
                self.ffmpeg_path,
                self.gui_mode,
                progress_handler=self.progress_handler,
                total_seconds=total_seconds,
            )
        if timings is None:
            raise ValueError(
                "Failed to get image subtitle timings via ffprobe for stream {} from {}. "
                "Make sure the stream exists and is an image-based subtitle track "
                "(hdmv_pgs_subtitle, dvd_subtitle, or dvb_subtitle; check with: "
                "ffprobe -show_streams {}).".format(stream, fname, fname)
            )

        if not timings:
            raise ValueError(
                "No subtitle timings found in image subtitle stream {}.".format(stream)
            )

        self.num_timings_ = len(timings)

        logger.info("found %d PGS subtitle segments", len(timings))
        for i, (s, e) in enumerate(timings[:8]):
            logger.debug(
                "  PGS[%d]: %s --> %s (%.3fs)",
                i,
                str(timedelta(seconds=s)),
                str(timedelta(seconds=e)),
                e - s,
            )

        max_time = max(end for _, end in timings)
        num_samples = int(max_time * self.sample_rate) + 2
        samples = np.zeros(num_samples, dtype=float)

        for start, end in timings:
            start_sample = int(round((start - self.start_seconds) * self.sample_rate))
            end_sample = int(round((end - self.start_seconds) * self.sample_rate))
            start_sample = max(start_sample, 0)
            end_sample = min(end_sample, num_samples)
            if start_sample < end_sample:
                samples[start_sample:end_sample] = 1.0

        self.pgs_speech_results_ = samples
        self.fit_boundaries(self.pgs_speech_results_)
        logger.info(
            "total PGS subtitle frames: %d", int(np.sum(self.pgs_speech_results_))
        )
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.pgs_speech_results_ is not None
        return self.pgs_speech_results_


def _infer_whisper_language(model_path: str, language: Optional[str]) -> str:
    """Pick the language to pass to the whisper filter.

    An explicit ``language`` always wins. Otherwise, whisper.cpp English-only
    models are named ``ggml-<size>.en.bin``, so a ``.en`` marker in the filename
    implies English; anything else falls back to ``auto`` (whisper detects it).
    """
    if language:
        return language
    base = os.path.basename(model_path).lower()
    if base.endswith(".en.bin") or ".en." in base:
        return "en"
    return DEFAULT_WHISPER_LANGUAGE


def _escape_ffmpeg_filter_value(value: str) -> str:
    """Escape a value for use inside an ``-af`` filter option (``key=value``).

    Within a filtergraph ``:`` separates a filter's options and ``\\`` is the
    escape character, so a literal ``:`` or ``\\`` (e.g. a Windows path like
    ``C:\\models\\ggml.bin``) must be backslash-escaped. ``\\`` is escaped first
    so the backslashes added for ``:``/``'`` are not themselves doubled.
    """
    for ch in ("\\", ":", "'"):
        value = value.replace(ch, "\\" + ch)
    return value


def _parse_filter_opts(opts_str: str) -> Dict[str, str]:
    """Parse ``key=value:key=value`` (ffmpeg filter option syntax) into a dict.

    Splits on ``:`` that is not backslash-escaped, preserving insertion order.
    Malformed segments (no ``=``) are skipped with a warning. Values are taken
    verbatim -- the caller is passing raw ffmpeg filter syntax and may pre-escape
    if needed.
    """
    result: Dict[str, str] = {}
    for part in re.split(r"(?<!\\):", opts_str):
        if not part.strip():
            continue
        if "=" not in part:
            logger.warning(
                "ignoring malformed whisper option %r (expected key=value)", part
            )
            continue
        key, val = part.split("=", 1)
        result[key.strip()] = val
    return result


def _build_whisper_filter(
    model_path: str,
    language: str,
    destination: str,
    queue: int = DEFAULT_WHISPER_QUEUE,
    vad_model: Optional[str] = None,
    extra_opts: Optional[str] = None,
) -> str:
    """Assemble the ``whisper=...`` ``-af`` filter string.

    ``format=srt`` and ``destination`` are always set by us (they are structural
    -- we parse the emitted SRT), so a user-supplied ``--whisper-args`` cannot
    override them (nor ``model``); it can override everything else, e.g.
    ``queue``. Paths are escaped for the filtergraph syntax.
    """
    opts: Dict[str, str] = {}
    opts["model"] = _escape_ffmpeg_filter_value(model_path)
    opts["language"] = language
    opts["queue"] = str(queue)
    if vad_model:
        opts["vad_model"] = _escape_ffmpeg_filter_value(vad_model)
    if extra_opts:
        for key, val in _parse_filter_opts(extra_opts).items():
            if key in ("model", "format", "destination"):
                logger.warning(
                    "ignoring whisper option %r (managed by ffsubsync)", key
                )
                continue
            opts[key] = val
    # structural options set last so --whisper-args can never clobber them
    opts["format"] = "srt"
    opts["destination"] = _escape_ffmpeg_filter_value(destination)
    return "whisper=" + ":".join("%s=%s" % (k, v) for k, v in opts.items())


def _ffmpeg_supports_whisper(ffmpeg_bin: str) -> bool:
    """Return True if ``ffmpeg_bin`` was built with the whisper audio filter.

    ``ffmpeg -h filter=whisper`` prints the filter's option help on a supported
    build and ``Unknown filter 'whisper'`` otherwise.
    """
    try:
        process = subprocess.Popen(
            [ffmpeg_bin, "-hide_banner", "-h", "filter=whisper"],
            **subprocess_args(include_stdout=True),
        )
        stdout, stderr = process.communicate()
    except OSError:
        return False
    combined = (stdout or b"").decode("utf-8", errors="replace") + (
        stderr or b""
    ).decode("utf-8", errors="replace")
    return "Unknown filter" not in combined and "whisper" in combined.lower()


class WhisperSpeechTransformer(TransformerMixin):
    """Use ffmpeg's whisper audio filter to transcribe the reference for sync.

    ffmpeg (>= 8.0, built with ``--enable-whisper``) can transcribe audio to SRT
    in a single pass via its ``whisper`` audio filter. This transformer runs that
    filter, writes the transcript to a temp SRT, and rasterizes the cue timings
    into the same 100 Hz binary speech signal that
    :class:`SubtitleSpeechTransformer` produces for real subtitle files -- so the
    normal FFT + framerate-ratio search aligns the input subtitles against the
    transcript with no other changes. This is a reference *source*, analogous to
    :class:`PGSSpeechTransformer`, selected via ``--whisper-weights``.

    ffsubsync reduces the ffmpeg burden: the model path has ``~`` expanded (ffmpeg
    won't), the language is inferred/defaulted, and clear errors are raised when
    the weights are missing or ffmpeg lacks the whisper filter.
    """

    # whisper may clip speech at the very start/end of the reference, which would
    # skew a duration-based framerate estimate; like PGS, return None so try_sync
    # skips length-ratio inference and relies on the discrete-ratio / --gss search.
    @property
    def num_frames(self) -> None:
        return None

    def __init__(
        self,
        model_path: str,
        language: Optional[str] = None,
        whisper_args: Optional[str] = None,
        vad: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
        start_seconds: int = 0,
        max_duration_seconds: Optional[float] = None,
        ffmpeg_path: Optional[str] = None,
        ref_stream: Optional[str] = None,
        gui_mode: bool = False,
    ) -> None:
        super(WhisperSpeechTransformer, self).__init__()
        # expand ~ up front -- ffmpeg does not do this itself
        self.model_path: str = (
            os.path.expanduser(model_path) if model_path else model_path
        )
        self.language: Optional[str] = language
        self.whisper_args: Optional[str] = whisper_args
        self.vad: Optional[str] = vad
        self.sample_rate: int = sample_rate
        self.start_seconds: int = start_seconds
        self.max_duration_seconds: Optional[float] = max_duration_seconds
        self.ffmpeg_path: Optional[str] = ffmpeg_path
        self.ref_stream: Optional[str] = ref_stream
        self.gui_mode: bool = gui_mode
        self.speech_results_: Optional[np.ndarray] = None

    def _resolve_vad_model(self) -> Optional[str]:
        """Interpret the reused --vad flag as whisper's optional VAD model path.

        In transcription mode whisper does its own speech detection, so the named
        VAD choices (webrtc/silero/...) don't apply; instead ``--vad`` may point
        at a ggml VAD model that whisper uses to fragment the audio. A non-path
        value is ignored (with a warning explaining the repurposing).
        """
        vad = self.vad
        if not vad:
            return None
        expanded = os.path.expanduser(vad)
        if os.path.exists(expanded):
            return expanded
        if vad in VAD_CHOICES:
            logger.warning(
                "--vad '%s' is ignored in whisper-transcription mode (whisper "
                "does its own speech detection); to enable whisper's optional VAD "
                "fragmentation, pass a path to a ggml VAD model as --vad instead.",
                vad,
            )
        else:
            logger.warning(
                "--vad '%s' is not an existing file, so it is not used as a "
                "whisper VAD model; ignoring.",
                vad,
            )
        return None

    def _build_ffmpeg_args(
        self, ffmpeg_bin: str, fname: str, whisper_filter: str
    ) -> List[str]:
        ffmpeg_args = [ffmpeg_bin]
        if self.start_seconds > 0:
            ffmpeg_args.extend(["-ss", str(timedelta(seconds=self.start_seconds))])
        if self.max_duration_seconds is not None:
            ffmpeg_args.extend(
                ["-t", str(timedelta(seconds=self.max_duration_seconds))]
            )
        ffmpeg_args.extend(["-loglevel", "error", "-nostdin", "-i", fname])
        if self.ref_stream is not None and self.ref_stream.startswith("0:a:"):
            ffmpeg_args.extend(["-map", self.ref_stream])
        # -vn: whisper only needs audio; -f null: we consume the SRT via destination
        ffmpeg_args.extend(["-vn", "-af", whisper_filter, "-f", "null", "-"])
        return ffmpeg_args

    def fit(self, fname: str, *_) -> "WhisperSpeechTransformer":
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError(
                "whisper weights not found: {!r}. Pass an existing model file "
                "via --whisper-weights (e.g. a whisper.cpp ggml-*.bin); '~' is "
                "expanded for you.".format(self.model_path)
            )
        ffmpeg_bin = ffmpeg_bin_path(
            "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
        )
        if not _ffmpeg_supports_whisper(ffmpeg_bin):
            raise ValueError(
                "the ffmpeg at '{}' does not support the whisper filter. You need "
                "ffmpeg >= 8.0 built with --enable-whisper (libwhisper). Point "
                "--ffmpeg-path at such a build, or install one.".format(ffmpeg_bin)
            )
        language = _infer_whisper_language(self.model_path, self.language)
        vad_model = self._resolve_vad_model()
        try:
            embedded_streams = _probe_embedded_subtitle_streams(
                fname, ffmpeg_path=self.ffmpeg_path, gui_mode=self.gui_mode
            )
        except Exception:
            embedded_streams = None
        if embedded_streams:
            logger.warning(
                "reference '%s' already contains %d embedded subtitle stream(s); "
                "those may be a better sync reference than whisper transcription "
                "-- consider '--vad subs_then_webrtc' or '--reference-stream s:0'.",
                fname,
                len(embedded_streams),
            )
        with tempfile.TemporaryDirectory(prefix="ffsubsync_whisper_") as tmpdir:
            destination = os.path.join(tmpdir, "whisper.srt")
            whisper_filter = _build_whisper_filter(
                self.model_path,
                language,
                destination,
                queue=DEFAULT_WHISPER_QUEUE,
                vad_model=vad_model,
                extra_opts=self.whisper_args,
            )
            ffmpeg_args = self._build_ffmpeg_args(ffmpeg_bin, fname, whisper_filter)
            logger.info(
                "transcribing reference audio with whisper (language=%s); "
                "this can take a while...",
                language,
            )
            process = subprocess.Popen(
                ffmpeg_args, **subprocess_args(include_stdout=True)
            )
            _, stderr = process.communicate()
            if process.returncode != 0:
                tail = "\n".join(
                    (stderr or b"")
                    .decode("utf-8", errors="replace")
                    .strip()
                    .splitlines()[-5:]
                )
                raise ValueError(
                    "ffmpeg whisper transcription failed (return code {}):\n{}".format(
                        process.returncode, tail
                    )
                )
            if not os.path.exists(destination) or os.path.getsize(destination) == 0:
                raise ValueError(
                    "whisper transcription produced no subtitles; the audio may be "
                    "silent, or the model/language may not match it."
                )
            with open(destination, "rb") as f:
                buffer = io.BytesIO(f.read())
        pipe = cast(
            Pipeline,
            make_subtitle_speech_pipeline(start_seconds=self.start_seconds),
        ).fit(buffer)
        speech_step = pipe.steps[-1][1]
        self.speech_results_ = speech_step.subtitle_speech_results_
        logger.info(
            "...done; total of speech segments: %s", np.sum(self.speech_results_)
        )
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.speech_results_ is not None
        return self.speech_results_
