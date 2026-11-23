# -*- coding: utf-8 -*-
from datetime import timedelta
import logging
import re
from typing import Any, cast, List, Optional

try:
    import cchardet
except:  # noqa: E722
    cchardet = None
try:
    import chardet
except:  # noqa: E722
    chardet = None
try:
    import charset_normalizer
except:  # noqa: E722
    charset_normalizer = None
import pysubs2
from ffsubsync.sklearn_shim import TransformerMixin
import srt

from ffsubsync.constants import (
    DEFAULT_ENCODING,
    DEFAULT_MAX_SUBTITLE_SECONDS,
    DEFAULT_START_SECONDS,
)
from ffsubsync.file_utils import open_file
from ffsubsync.generic_subtitles import GenericSubtitle, GenericSubtitlesFile, SubsMixin

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

_MICRODVD_FPS_LINE_RE = re.compile(
    r"^\{\s*(\d+)\s*\}\{\s*(\d+)\s*\}\s*(\d+(?:[.,]\d+)?)\s*$"
)


def _detect_microdvd_fps(decoded_subs: str) -> Optional[float]:
    """Scan the opening lines of a MicroDVD ``.sub`` file for an FPS cue.

    Standard MicroDVD puts ``{1}{1}23.976`` on line 1, which ``pysubs2`` parses
    automatically; some files place a site ad/watermark ahead of it (e.g.
    OpenSubtitles banners), causing ``pysubs2`` to raise ``UnknownFPSError``.
    Scanning the first 10 non-empty lines recovers the FPS in both cases.
    """
    seen = 0
    for raw_line in decoded_subs.splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line:
            continue
        seen += 1
        m = _MICRODVD_FPS_LINE_RE.match(line)
        if m:
            start_f, end_f = int(m.group(1)), int(m.group(2))
            if abs(end_f - start_f) <= 1:
                try:
                    val = float(m.group(3).replace(",", "."))
                except ValueError:
                    val = 0.0
                if 10.0 <= val <= 120.0:
                    return val
        if seen >= 10:
            break
    return None


def _preprocess_subs(
    subs,
    max_subtitle_seconds: Optional[int] = None,
    start_seconds: int = 0,
    tolerant: bool = True,
) -> List[GenericSubtitle]:
    subs_list = []
    start_time = timedelta(seconds=start_seconds)
    max_duration = timedelta(days=1)
    if max_subtitle_seconds is not None:
        max_duration = timedelta(seconds=max_subtitle_seconds)
    subs = iter(subs)
    while True:
        try:
            next_sub = GenericSubtitle.wrap_inner_subtitle(next(subs))
            if next_sub.start < start_time:
                continue
            next_sub.end = min(next_sub.end, next_sub.start + max_duration)
            subs_list.append(next_sub)
        # We don't catch SRTParseError here b/c that is typically raised when we
        # are trying to parse with the wrong encoding, in which case we might
        # be able to try another one on the *entire* set of subtitles elsewhere.
        except ValueError as e:
            if tolerant:
                logger.warning(e)
                continue
            else:
                raise
        except StopIteration:
            break
    return subs_list


class GenericSubtitleParser(SubsMixin, TransformerMixin):
    def __init__(
        self,
        fmt: str = "srt",
        encoding: str = "infer",
        caching: bool = False,
        max_subtitle_seconds: Optional[int] = None,
        start_seconds: int = 0,
        skip_ssa_info: bool = False,
        strict: bool = False,
        fps: Optional[float] = None,
        fallback_fps: Optional[float] = None,
    ) -> None:
        super(self.__class__, self).__init__()
        self.sub_format: str = fmt.lower()
        self.encoding: str = encoding
        self.caching: bool = caching
        self.fit_fname: Optional[str] = None
        self.detected_encoding_: Optional[str] = None
        self.max_subtitle_seconds: Optional[int] = max_subtitle_seconds
        self.start_seconds: int = start_seconds
        # FIXME: hack to get tests to pass; remove
        self._skip_ssa_info: bool = skip_ssa_info
        self._strict: bool = strict
        self.fps: Optional[float] = fps
        self.fallback_fps: Optional[float] = fallback_fps

    def fit(self, fname: str, *_) -> "GenericSubtitleParser":
        if self.caching and self.fit_fname == ("<stdin>" if fname is None else fname):
            return self
        encodings_to_try = (self.encoding,)
        with open_file(fname, "rb") as f:
            subs = f.read()
        if self.encoding == "infer":
            for chardet_lib in (cchardet, charset_normalizer, chardet):
                if chardet_lib is not None:
                    try:
                        detected_encoding = cast(
                            Optional[str], chardet_lib.detect(subs)["encoding"]
                        )
                    except:  # noqa: E722
                        continue
                    if detected_encoding is not None:
                        self.detected_encoding_ = detected_encoding
                        encodings_to_try = (detected_encoding,)
                        break
            assert self.detected_encoding_ is not None
            logger.info("detected encoding: %s" % self.detected_encoding_)
        exc = None
        for encoding in encodings_to_try:
            try:
                decoded_subs = subs.decode(encoding, errors="replace").strip()
                resolved_fps: Optional[float] = self.fps
                if self.sub_format == "srt":
                    parsed_subs = srt.parse(
                        decoded_subs, ignore_errors=not self._strict
                    )
                elif self.sub_format == "sub":
                    if self.fps is not None:
                        parsed_subs = pysubs2.SSAFile.from_string(
                            decoded_subs, fps=self.fps
                        )
                    else:
                        try:
                            parsed_subs = pysubs2.SSAFile.from_string(decoded_subs)
                        except pysubs2.exceptions.UnknownFPSError:
                            detected_fps = _detect_microdvd_fps(decoded_subs)
                            if detected_fps is not None:
                                logger.info(
                                    "detected MicroDVD FPS %.3f from subtitle header",
                                    detected_fps,
                                )
                                resolved_fps = detected_fps
                                parsed_subs = pysubs2.SSAFile.from_string(
                                    decoded_subs, fps=detected_fps
                                )
                            elif self.fallback_fps is not None:
                                logger.info(
                                    "MicroDVD .sub has no FPS header; using reference video FPS %.3f",
                                    self.fallback_fps,
                                )
                                resolved_fps = self.fallback_fps
                                parsed_subs = pysubs2.SSAFile.from_string(
                                    decoded_subs, fps=self.fallback_fps
                                )
                            else:
                                raise pysubs2.exceptions.UnknownFPSError(
                                    "Framerate was not specified and cannot be read from the "
                                    "MicroDVD .sub file. Specify it with --sub-fps "
                                    "(e.g. --sub-fps 23.976 or --sub-fps 25)."
                                )
                    if resolved_fps is None:
                        resolved_fps = getattr(parsed_subs, "fps", None)
                elif self.sub_format in ("ass", "ssa", "vtt"):
                    parsed_subs = pysubs2.SSAFile.from_string(decoded_subs)
                else:
                    raise NotImplementedError(
                        "unsupported format: %s" % self.sub_format
                    )
                if resolved_fps is None and self.fallback_fps is not None:
                    resolved_fps = self.fallback_fps
                extra_generic_subtitle_file_kwargs: dict = {}
                if resolved_fps is not None:
                    extra_generic_subtitle_file_kwargs["fps"] = resolved_fps
                if isinstance(parsed_subs, pysubs2.SSAFile):
                    extra_generic_subtitle_file_kwargs.update(
                        dict(
                            styles=parsed_subs.styles,
                            # pysubs2 on Python >= 3.6 doesn't support this
                            fonts_opaque=getattr(parsed_subs, "fonts_opaque", None),
                            info=parsed_subs.info if not self._skip_ssa_info else None,
                        )
                    )
                self.subs_ = GenericSubtitlesFile(
                    _preprocess_subs(
                        parsed_subs,
                        max_subtitle_seconds=self.max_subtitle_seconds,
                        start_seconds=self.start_seconds,
                    ),
                    sub_format=self.sub_format,
                    encoding=encoding,
                    **extra_generic_subtitle_file_kwargs,
                )
                self.fit_fname = "<stdin>" if fname is None else fname
                if len(encodings_to_try) > 1:
                    self.detected_encoding_ = encoding
                    logger.info("detected encoding: %s" % self.detected_encoding_)
                return self
            except Exception as e:
                exc = e
                continue
        raise exc

    def transform(self, *_) -> GenericSubtitlesFile:
        return self.subs_


def make_subtitle_parser(
    fmt: str,
    encoding: str = DEFAULT_ENCODING,
    caching: bool = False,
    max_subtitle_seconds: int = DEFAULT_MAX_SUBTITLE_SECONDS,
    start_seconds: int = DEFAULT_START_SECONDS,
    **kwargs: Any,
) -> GenericSubtitleParser:
    return GenericSubtitleParser(
        fmt=fmt,
        encoding=encoding,
        caching=caching,
        max_subtitle_seconds=max_subtitle_seconds,
        start_seconds=start_seconds,
        skip_ssa_info=kwargs.get("skip_ssa_info", False),
        strict=kwargs.get("strict", False),
        fps=kwargs.get("fps", kwargs.get("sub_fps", None)),
        fallback_fps=kwargs.get("fallback_fps", None),
    )
