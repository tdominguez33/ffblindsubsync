"""Browser bridge for ffsubsync.

Loaded into the Pyodide runtime; exposes JSON-friendly entry points the JS worker
calls. Three sync paths:

* ``sync_subtitles`` — subtitle-vs-subtitle (or extracted embedded text subs).
* ``sync_with_audio`` — video/audio reference (ffmpeg.wasm PCM -> WebRTC VAD -> align).
* ``sync_with_timings`` — PGS/VobSub packet timings -> align.

All delegate alignment to the *real*, vendored ffsubsync code so results match
the CLI. Input/output files live in Pyodide's in-memory MEMFS.
"""

import logging
import os
import re
import traceback

import numpy as np

from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.ffsubsync import make_parser, run
from ffsubsync.speech_transformers import (
    _make_auditok_detector,
    _make_webrtcvad_detector,
)

_WORK_DIR = "/work"
_ALIGN_STEP_RE = re.compile(r"computing alignment (\d+)/(\d+)")
_GSS_STEP_RE = re.compile(r"got score .* for ratio ([0-9.]+)")


def _ensure_workdir() -> None:
    os.makedirs(_WORK_DIR, exist_ok=True)


def _as_bytes(data) -> bytes:
    """Coerce whatever the JS worker handed us into real Python bytes."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    to_py = getattr(data, "to_py", None)
    if to_py is not None:
        data = to_py()
    return bytes(data)


def _write(name: str, data) -> str:
    path = os.path.join(_WORK_DIR, name)
    with open(path, "wb") as f:
        f.write(_as_bytes(data))
    return path


def _synced_name(in_name: str) -> str:
    stem, _ext = os.path.splitext(os.path.basename(in_name))
    return "{}.synced.srt".format(stem)


class _ProgressLogHandler(logging.Handler):
    """Translate ffsubsync INFO log milestones into normalized 0..1 progress."""

    def __init__(self, progress_cb, start_frac: float = 0.0, end_frac: float = 1.0):
        super().__init__(level=logging.INFO)
        self._cb = progress_cb
        self._start = float(start_frac)
        self._span = max(0.0, float(end_frac) - self._start)
        self._gss_steps = 0

    def _emit_scaled(self, local_frac: float, message: str) -> None:
        clamped = max(0.0, min(1.0, float(local_frac)))
        try:
            self._cb(self._start + self._span * clamped, message)
        except Exception:
            pass

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "extracting speech segments from subtitles" in msg:
            self._emit_scaled(0.08, "Parsing subtitles…")
            return
        m = _ALIGN_STEP_RE.search(msg)
        if m:
            cur, tot = int(m.group(1)), max(1, int(m.group(2)))
            frac = 0.15 + 0.72 * ((cur - 1) / float(tot))
            self._emit_scaled(frac, "Computing alignment (%d/%d)…" % (cur, tot))
            return
        gm = _GSS_STEP_RE.search(msg)
        if gm:
            self._gss_steps += 1
            frac = min(0.87, 0.15 + 0.72 * (self._gss_steps / 14.0))
            self._emit_scaled(frac, "Refining framerate ratio (%s)…" % gm.group(1))
            return
        if "split alignment" in msg:
            self._emit_scaled(0.92, "Checking mid-file breaks…")
            return
        if "writing output" in msg:
            self._emit_scaled(0.97, "Writing synced subtitles…")
            return


def _common_argv(in_path, out_path, *, output_encoding, no_fix_framerate, gss,
                 max_offset_seconds, split_sync=False, auto_split=False):
    argv = ["-i", in_path, "-o", out_path]
    if output_encoding:
        argv += ["--output-encoding", output_encoding]
    if no_fix_framerate:
        argv += ["--no-fix-framerate"]
    if gss:
        argv += ["--gss"]
    if split_sync:
        argv += ["--split-penalty"]
    elif auto_split:
        argv += ["--auto-split"]
    if max_offset_seconds is not None:
        argv += ["--max-offset-seconds", str(max_offset_seconds)]
    return argv


def _run_and_collect(
    argv,
    out_path,
    in_name,
    progress_cb=None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
):
    """Run ffsubsync with ``argv`` and package a JS-friendly result dict."""
    ff_logger = logging.getLogger("ffsubsync")
    handler = None
    if progress_cb is not None:
        handler = _ProgressLogHandler(progress_cb, start_frac=start_frac, end_frac=end_frac)
        ff_logger.addHandler(handler)
    try:
        args = make_parser().parse_args(argv)
        result = run(args)
    except Exception:  # pragma: no cover - defensive; surfaced to the UI
        return {
            "ok": False,
            "offset_seconds": None,
            "framerate_scale_factor": None,
            "output_name": _synced_name(in_name),
            "output_text": "",
            "error": traceback.format_exc(),
        }
    finally:
        if handler is not None:
            ff_logger.removeHandler(handler)

    ok = bool(result.get("sync_was_successful")) and os.path.exists(out_path)
    output_text = ""
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8", errors="replace") as f:
            output_text = f.read()

    if ok and progress_cb is not None:
        try:
            progress_cb(end_frac, "Done.")
        except Exception:
            pass

    return {
        "ok": ok,
        "offset_seconds": result.get("offset_seconds"),
        "framerate_scale_factor": result.get("framerate_scale_factor"),
        "split_segments": result.get("split_segments"),
        "dropped_cues": result.get("dropped_cues", 0),
        "output_name": _synced_name(in_name),
        "output_text": output_text,
        "error": None if ok else "sync did not succeed (see console logs)",
    }


def sync_subtitles(
    ref_name: str,
    ref_bytes,
    in_name: str,
    in_bytes,
    *,
    reference_encoding=None,
    output_encoding: str = "utf-8",
    no_fix_framerate: bool = False,
    gss: bool = False,
    split_sync: bool = False,
    auto_split: bool = False,
    max_offset_seconds=None,
    progress_cb=None,
):
    """Sync ``in_bytes`` against subtitle reference ``ref_bytes``."""
    _ensure_workdir()
    if progress_cb is not None:
        progress_cb(0.02, "Preparing subtitles…")
    ref_path = _write("reference_" + os.path.basename(ref_name), ref_bytes)
    in_path = _write("input_" + os.path.basename(in_name), in_bytes)
    out_path = os.path.join(_WORK_DIR, "output.srt")

    argv = [ref_path] + _common_argv(
        in_path, out_path, output_encoding=output_encoding,
        no_fix_framerate=no_fix_framerate, gss=gss, split_sync=split_sync,
        auto_split=auto_split, max_offset_seconds=max_offset_seconds,
    )
    if reference_encoding:
        argv += ["--reference-encoding", reference_encoding]
    return _run_and_collect(argv, out_path, in_name, progress_cb=progress_cb)


# ---- Phase 2: video/audio reference -----------------------------------------

_DETECTOR_MAKERS = {
    "webrtc": _make_webrtcvad_detector,
    "auditok": _make_auditok_detector,
}


def _pcm_to_speech_signal(
    pcm: bytes,
    frame_rate: int,
    vad: str,
    non_speech_label: float,
    progress_cb=None,
) -> np.ndarray:
    """Turn mono s16le PCM into ffsubsync's speech signal via the chosen VAD."""
    maker = _DETECTOR_MAKERS.get(vad)
    if maker is None:
        raise ValueError(
            "unsupported browser vad %r; expected one of %s"
            % (vad, ", ".join(sorted(_DETECTOR_MAKERS)))
        )
    detector = maker(SAMPLE_RATE, frame_rate, non_speech_label)

    bytes_per_frame = 2
    frames_per_window = bytes_per_frame * frame_rate // SAMPLE_RATE
    windows_per_buffer = 10000
    step = frames_per_window * windows_per_buffer  # bytes per chunk (window-aligned)

    total_bytes = len(pcm)
    parts = []
    for start in range(0, total_bytes, step):
        chunk = pcm[start:start + step]
        if not chunk:
            break
        parts.append(detector(np.frombuffer(chunk, np.uint8)))
        if progress_cb is not None and total_bytes > 0:
            done_frac = min(1.0, (start + len(chunk)) / float(total_bytes))
            pct = int(round(done_frac * 100))
            progress_cb(done_frac * 0.50, "Detecting speech (%d%%)…" % pct)
    if not parts:
        raise ValueError("no audio samples to analyze")
    return np.concatenate(parts)


def sync_with_audio(
    ref_pcm,
    frame_rate: int,
    in_name: str,
    in_bytes,
    *,
    vad: str = "webrtc",
    non_speech_label: float = 0.0,
    output_encoding: str = "utf-8",
    no_fix_framerate: bool = False,
    gss: bool = False,
    split_sync: bool = False,
    auto_split: bool = False,
    max_offset_seconds=None,
    progress_cb=None,
):
    """Sync ``in_bytes`` against decoded audio PCM from a video/audio reference."""
    _ensure_workdir()
    try:
        if progress_cb is not None:
            progress_cb(0.0, "Detecting speech (0%)…")
        signal = _pcm_to_speech_signal(
            _as_bytes(ref_pcm),
            int(frame_rate),
            vad,
            non_speech_label,
            progress_cb=progress_cb,
        )
    except Exception:
        return {
            "ok": False,
            "offset_seconds": None,
            "framerate_scale_factor": None,
            "output_name": _synced_name(in_name),
            "output_text": "",
            "error": traceback.format_exc(),
        }

    npz_path = os.path.join(_WORK_DIR, "reference.npz")
    np.savez_compressed(npz_path, speech=signal)
    in_path = _write("input_" + os.path.basename(in_name), in_bytes)
    out_path = os.path.join(_WORK_DIR, "output.srt")

    argv = [npz_path] + _common_argv(
        in_path, out_path, output_encoding=output_encoding,
        no_fix_framerate=no_fix_framerate, gss=gss, split_sync=split_sync,
        auto_split=auto_split, max_offset_seconds=max_offset_seconds,
    )
    argv += ["--non-speech-label", str(non_speech_label)]
    return _run_and_collect(
        argv,
        out_path,
        in_name,
        progress_cb=progress_cb,
        start_frac=0.50,
        end_frac=1.0,
    )


def sync_with_timings(
    timings,
    in_name: str,
    in_bytes,
    *,
    output_encoding: str = "utf-8",
    no_fix_framerate: bool = False,
    gss: bool = False,
    split_sync: bool = False,
    auto_split: bool = False,
    max_offset_seconds=None,
    progress_cb=None,
):
    """Sync ``in_bytes`` against ``(start_s, end_s)`` timing intervals (PGS/VobSub)."""
    _ensure_workdir()
    try:
        if progress_cb is not None:
            progress_cb(0.03, "Building subtitle timeline…")
        pairs = []
        for item in timings:
            start = float(item[0])
            end = float(item[1])
            if end > start:
                pairs.append((start, end))
        if not pairs:
            raise ValueError("no subtitle timings to align against")
        max_time = max(end for _, end in pairs)
        num_samples = int(max_time * SAMPLE_RATE) + 2
        signal = np.zeros(num_samples, dtype=float)
        for start, end in pairs:
            start_sample = max(0, int(round(start * SAMPLE_RATE)))
            end_sample = min(num_samples, int(round(end * SAMPLE_RATE)))
            if start_sample < end_sample:
                signal[start_sample:end_sample] = 1.0
    except Exception:
        return {
            "ok": False,
            "offset_seconds": None,
            "framerate_scale_factor": None,
            "output_name": _synced_name(in_name),
            "output_text": "",
            "error": traceback.format_exc(),
        }

    npz_path = os.path.join(_WORK_DIR, "reference.npz")
    np.savez_compressed(npz_path, speech=signal)
    in_path = _write("input_" + os.path.basename(in_name), in_bytes)
    out_path = os.path.join(_WORK_DIR, "output.srt")

    argv = [npz_path] + _common_argv(
        in_path, out_path, output_encoding=output_encoding,
        no_fix_framerate=no_fix_framerate, gss=gss, split_sync=split_sync,
        auto_split=auto_split, max_offset_seconds=max_offset_seconds,
    )
    return _run_and_collect(argv, out_path, in_name, progress_cb=progress_cb)
