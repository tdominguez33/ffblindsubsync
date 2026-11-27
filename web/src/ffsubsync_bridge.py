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
    _parse_auditok_spec,
)
from ffsubsync.subtitle_parser import (
    _detect_microdvd_fps,
    make_subtitle_parser,
)

_WORK_DIR = "/work"
_ALIGN_STEP_RE = re.compile(r"computing alignment (\d+)/(\d+)")
_GSS_STEP_RE = re.compile(r"got score .* for ratio ([0-9.]+)")
_PRESERVED_OUT_EXTS = (".srt", ".ass", ".ssa", ".vtt")


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


def _output_ext(in_name: str) -> str:
    """Preserve .srt/.ass/.ssa/.vtt output format; convert other inputs (e.g. .sub) to .srt."""
    _stem, ext = os.path.splitext(os.path.basename(in_name))
    low = ext.lower()
    return low if low in _PRESERVED_OUT_EXTS else ".srt"


def _synced_name(in_name: str) -> str:
    stem, _ext = os.path.splitext(os.path.basename(in_name))
    return "{}.synced{}".format(stem, _output_ext(in_name))


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
                 max_offset_seconds, split_sync=False, auto_split=False, sub_fps=None,
                 alignment_profile=None):
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
    if sub_fps is not None:
        argv += ["--sub-fps", str(sub_fps)]
    if alignment_profile:
        argv += ["--alignment-profile", str(alignment_profile)]
    return argv


def _resolve_fallback_sub_fps(in_name: str, in_bytes, inferred_sub_fps):
    """Return ``inferred_sub_fps`` only for MicroDVD ``.sub`` inputs lacking an FPS header."""
    if inferred_sub_fps is None or not str(in_name or "").lower().endswith(".sub"):
        return None
    try:
        raw = _as_bytes(in_bytes)
        sample = raw[:4096].decode("utf-8", errors="ignore")
        if _detect_microdvd_fps(sample) is not None:
            return None
    except Exception:
        pass
    return float(inferred_sub_fps)


def _run_and_collect(
    argv,
    out_path,
    in_name,
    progress_cb=None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    reference_kind=None,
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

    quality = result.get("quality")
    if isinstance(quality, dict) and reference_kind in ("audio", "subtitles"):
        quality = dict(quality)
        quality["reference_kind"] = reference_kind

    return {
        "ok": ok,
        "offset_seconds": result.get("offset_seconds"),
        "framerate_scale_factor": result.get("framerate_scale_factor"),
        "split_segments": result.get("split_segments"),
        "dropped_cues": result.get("dropped_cues", 0),
        "clipped_cues": result.get("clipped_cues", 0),
        "alignment_profile": result.get("alignment_profile", "balanced"),
        "profile_offset_seconds": result.get("profile_offset_seconds", 0.0),
        "quality": quality,
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
    min_ref_cues: int = 0,
    inferred_sub_fps=None,
    alignment_profile=None,
    progress_cb=None,
):
    """Sync ``in_bytes`` against subtitle reference ``ref_bytes``."""
    _ensure_workdir()
    if progress_cb is not None:
        progress_cb(0.02, "Preparing subtitles…")
    ref_path = _write("reference_" + os.path.basename(ref_name), ref_bytes)
    in_path = _write("input_" + os.path.basename(in_name), in_bytes)
    out_path = os.path.join(_WORK_DIR, "output" + _output_ext(in_name))

    if min_ref_cues and int(min_ref_cues) > 0:
        try:
            ref_fmt = os.path.splitext(ref_path)[-1].lstrip(".").lower() or "srt"
            ref_parser = make_subtitle_parser(
                fmt=ref_fmt,
                encoding=reference_encoding or "infer",
            ).fit(ref_path)
            ref_cue_count = len(ref_parser.subs_ or [])
            if ref_cue_count < int(min_ref_cues):
                return {
                    "ok": False,
                    "offset_seconds": None,
                    "framerate_scale_factor": None,
                    "split_segments": None,
                    "dropped_cues": 0,
                    "clipped_cues": 0,
                    "output_name": _synced_name(in_name),
                    "output_text": "",
                    "error": (
                        "reference subtitle has only %d cues (< %d)"
                        % (ref_cue_count, int(min_ref_cues))
                    ),
                }
        except Exception:
            pass

    argv = [ref_path] + _common_argv(
        in_path, out_path, output_encoding=output_encoding,
        no_fix_framerate=no_fix_framerate, gss=gss, split_sync=split_sync,
        auto_split=auto_split, max_offset_seconds=max_offset_seconds,
        sub_fps=_resolve_fallback_sub_fps(in_name, in_bytes, inferred_sub_fps),
        alignment_profile=alignment_profile,
    )
    if reference_encoding:
        argv += ["--reference-encoding", reference_encoding]
    return _run_and_collect(
        argv, out_path, in_name, progress_cb=progress_cb, reference_kind="subtitles"
    )


# ---- Phase 2: video/audio reference -----------------------------------------

_DETECTOR_MAKERS = {
    "webrtc": _make_webrtcvad_detector,
    "auditok": _make_auditok_detector,
}


def _make_browser_detector(vad: str, frame_rate: int, non_speech_label: float):
    if vad.startswith("auditok"):
        return _make_auditok_detector(
            SAMPLE_RATE,
            frame_rate,
            non_speech_label,
            vad_spec=_parse_auditok_spec(vad),
        )
    maker = _DETECTOR_MAKERS.get(vad)
    if maker is None:
        raise ValueError(
            "unsupported browser vad %r; expected one of %s"
            % (vad, ", ".join(sorted(_DETECTOR_MAKERS)))
        )
    return maker(SAMPLE_RATE, frame_rate, non_speech_label)


def _iter_pcm_chunks(ref_pcm, step: int):
    """Yield ``(chunk_u8, bytes_done, total_bytes)`` without copying the whole buffer.

    In Pyodide, ``ref_pcm`` is a JS ``Uint8Array`` proxy (often 500 MB–1.5 GB
    for a 2-hour 48 kHz movie). Slicing via ``subarray(start, end).to_py()``
    keeps only one ~10 MB chunk inside the 32-bit WASM heap at a time instead
    of duplicating the entire audio buffer into WASM linear memory.
    """
    is_js_typed_array = (
        hasattr(ref_pcm, "subarray")
        and hasattr(ref_pcm, "byteLength")
        and hasattr(ref_pcm, "to_py")
    )
    if is_js_typed_array:
        total_bytes = int(ref_pcm.byteLength)
        for start in range(0, total_bytes, step):
            end = min(total_bytes, start + step)
            sub_proxy = ref_pcm.subarray(start, end)
            try:
                chunk_py = sub_proxy.to_py()
            finally:
                if hasattr(sub_proxy, "destroy"):
                    sub_proxy.destroy()
            yield np.frombuffer(chunk_py, dtype=np.uint8), end, total_bytes
            chunk_py = None
    else:
        raw = (
            memoryview(ref_pcm)
            if isinstance(ref_pcm, (bytes, bytearray, memoryview))
            else memoryview(_as_bytes(ref_pcm))
        )
        total_bytes = raw.nbytes
        for start in range(0, total_bytes, step):
            end = min(total_bytes, start + step)
            yield np.frombuffer(raw[start:end], dtype=np.uint8), end, total_bytes


def _pcm_to_speech_signals(
    ref_pcm,
    frame_rate: int,
    vad: str,
    non_speech_label: float,
    channels: int = 1,
    progress_cb=None,
):
    """Stream mono or interleaved 2-ch (mix + FC) s16le PCM through VAD.

    Returns ``(mix_signal, fc_signal_or_None)``. Processes window-aligned chunks
    directly from the source buffer so 2-hour 48 kHz surround tracks stay well
    within Pyodide's 32-bit WASM memory ceiling.
    """
    n_ch = 2 if int(channels or 1) == 2 else 1
    mix_detector = _make_browser_detector(vad, frame_rate, non_speech_label)
    fc_detector = (
        _make_browser_detector(vad, frame_rate, non_speech_label) if n_ch == 2 else None
    )

    bytes_per_sample_frame = 2 * n_ch
    bytes_per_window = bytes_per_sample_frame * frame_rate // SAMPLE_RATE
    # 5000 windows = 50 s of audio per chunk (~1.6–19.2 MB depending on rate & channels).
    windows_per_buffer = 5000
    step = bytes_per_window * windows_per_buffer

    mix_parts = []
    fc_parts = []
    for chunk_u8, bytes_done, total_bytes in _iter_pcm_chunks(ref_pcm, step):
        usable = len(chunk_u8) - (len(chunk_u8) % bytes_per_sample_frame)
        if usable < bytes_per_sample_frame:
            continue
        if n_ch == 2:
            stereo = np.frombuffer(chunk_u8[:usable], dtype="<i2").reshape(-1, 2)
            mix_chunk = np.ascontiguousarray(stereo[:, 0]).view(np.uint8)
            fc_chunk = np.ascontiguousarray(stereo[:, 1]).view(np.uint8)
            mix_parts.append(mix_detector(mix_chunk))
            fc_parts.append(fc_detector(fc_chunk))
        else:
            mix_parts.append(mix_detector(chunk_u8[:usable]))
        if progress_cb is not None and total_bytes > 0:
            done_frac = min(1.0, bytes_done / float(total_bytes))
            pct = int(round(done_frac * 100))
            progress_cb(done_frac * 0.50, "Detecting speech (%d%%)…" % pct)

    if not mix_parts:
        raise ValueError("no audio samples to analyze")
    mix_signal = np.concatenate(mix_parts)
    fc_signal = np.concatenate(fc_parts) if (n_ch == 2 and fc_parts) else None
    return mix_signal, fc_signal


def sync_with_audio(
    ref_pcm,
    frame_rate: int,
    in_name: str,
    in_bytes,
    *,
    vad: str = "webrtc",
    channels: int = 1,
    non_speech_label: float = 0.0,
    output_encoding: str = "utf-8",
    no_fix_framerate: bool = False,
    gss: bool = False,
    split_sync: bool = False,
    auto_split: bool = False,
    max_offset_seconds=None,
    inferred_sub_fps=None,
    alignment_profile=None,
    progress_cb=None,
):
    """Sync ``in_bytes`` against decoded audio PCM from a video/audio reference.

    When ``channels == 2`` (interleaved s16le downmix + surround center FC),
    runs VAD on both channels and serializes ``alt_fc_speech`` alongside
    ``speech`` so ffsubsync picks the higher peak-SNR alignment identically to
    the CLI and desktop builds.
    """
    _ensure_workdir()
    fc_signal = None
    try:
        if progress_cb is not None:
            progress_cb(0.0, "Detecting speech (0%)…")
        signal, fc_signal = _pcm_to_speech_signals(
            ref_pcm,
            int(frame_rate),
            vad,
            non_speech_label,
            channels=int(channels or 1),
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
    if fc_signal is not None:
        np.savez_compressed(npz_path, speech=signal, alt_fc_speech=fc_signal)
    else:
        np.savez_compressed(npz_path, speech=signal)
    in_path = _write("input_" + os.path.basename(in_name), in_bytes)
    out_path = os.path.join(_WORK_DIR, "output" + _output_ext(in_name))

    argv = [npz_path] + _common_argv(
        in_path, out_path, output_encoding=output_encoding,
        no_fix_framerate=no_fix_framerate, gss=gss, split_sync=split_sync,
        auto_split=auto_split, max_offset_seconds=max_offset_seconds,
        sub_fps=_resolve_fallback_sub_fps(in_name, in_bytes, inferred_sub_fps),
        alignment_profile=alignment_profile,
    )
    argv += ["--non-speech-label", str(non_speech_label)]
    return _run_and_collect(
        argv,
        out_path,
        in_name,
        progress_cb=progress_cb,
        start_frac=0.50,
        end_frac=1.0,
        reference_kind="audio",
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
    inferred_sub_fps=None,
    alignment_profile=None,
    progress_cb=None,
):
    """Sync ``in_bytes`` against ``(start_s, end_s)`` timing intervals (PGS/VobSub/DVB)."""
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
    out_path = os.path.join(_WORK_DIR, "output" + _output_ext(in_name))

    argv = [npz_path] + _common_argv(
        in_path, out_path, output_encoding=output_encoding,
        no_fix_framerate=no_fix_framerate, gss=gss, split_sync=split_sync,
        auto_split=auto_split, max_offset_seconds=max_offset_seconds,
        sub_fps=_resolve_fallback_sub_fps(in_name, in_bytes, inferred_sub_fps),
        alignment_profile=alignment_profile,
    )
    return _run_and_collect(
        argv, out_path, in_name, progress_cb=progress_cb, reference_kind="subtitles"
    )


# ---- Cached reference & multi-subtitle ranking (Phase 3) --------------------

_REF_CACHE = {
    "key": None,
    "path": None,
    "kind": None,
    "non_speech_label": 0.0,
    "reference_encoding": None,
    "pipe": None,
}


def has_cached_reference(cache_key: str) -> bool:
    return bool(cache_key) and _REF_CACHE["key"] == str(cache_key) and bool(_REF_CACHE["path"])


def _set_ref_cache(
    cache_key: str,
    ref_path: str,
    ref_kind: str,
    *,
    non_speech_label: float = 0.0,
    reference_encoding=None,
) -> None:
    _REF_CACHE["key"] = str(cache_key)
    _REF_CACHE["path"] = ref_path
    _REF_CACHE["kind"] = ref_kind
    _REF_CACHE["non_speech_label"] = float(non_speech_label)
    _REF_CACHE["reference_encoding"] = reference_encoding
    _REF_CACHE["pipe"] = None


def prepare_reference_subtitles(
    cache_key: str,
    ref_name: str,
    ref_bytes,
    *,
    reference_encoding=None,
    min_ref_cues: int = 0,
):
    """Write and validate a subtitle reference in MEMFS and cache it by ``cache_key``."""
    _ensure_workdir()
    ext = os.path.splitext(os.path.basename(ref_name))[-1] or ".srt"
    ref_path = _write("cached_reference" + ext, ref_bytes)
    if min_ref_cues and int(min_ref_cues) > 0:
        try:
            ref_fmt = ext.lstrip(".").lower() or "srt"
            ref_parser = make_subtitle_parser(
                fmt=ref_fmt,
                encoding=reference_encoding or "infer",
            ).fit(ref_path)
            ref_cue_count = len(ref_parser.subs_ or [])
            if ref_cue_count < int(min_ref_cues):
                return {
                    "ok": False,
                    "error": (
                        "reference subtitle has only %d cues (< %d)"
                        % (ref_cue_count, int(min_ref_cues))
                    ),
                }
        except Exception:
            pass
    _set_ref_cache(
        cache_key,
        ref_path,
        "subtitles",
        reference_encoding=reference_encoding,
    )
    return {"ok": True, "error": None}


def prepare_reference_audio(
    cache_key: str,
    ref_pcm,
    frame_rate: int,
    *,
    vad: str = "webrtc",
    channels: int = 1,
    non_speech_label: float = 0.0,
    progress_cb=None,
):
    """Run VAD once on decoded PCM and cache the speech signal in MEMFS."""
    _ensure_workdir()
    try:
        if progress_cb is not None:
            progress_cb(0.0, "Detecting speech (0%)…")
        signal, fc_signal = _pcm_to_speech_signals(
            ref_pcm,
            int(frame_rate),
            vad,
            non_speech_label,
            channels=int(channels or 1),
            progress_cb=progress_cb,
        )
    except Exception:
        return {"ok": False, "error": traceback.format_exc()}

    npz_path = os.path.join(_WORK_DIR, "cached_reference.npz")
    if fc_signal is not None:
        np.savez_compressed(npz_path, speech=signal, alt_fc_speech=fc_signal)
    else:
        np.savez_compressed(npz_path, speech=signal)
    _set_ref_cache(
        cache_key,
        npz_path,
        "audio",
        non_speech_label=non_speech_label,
    )
    return {"ok": True, "error": None}


def prepare_reference_timings(cache_key: str, timings):
    """Build a binary reference timeline from (start_s, end_s) intervals and cache it."""
    _ensure_workdir()
    try:
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
        return {"ok": False, "error": traceback.format_exc()}

    npz_path = os.path.join(_WORK_DIR, "cached_reference.npz")
    np.savez_compressed(npz_path, speech=signal)
    _set_ref_cache(cache_key, npz_path, "subtitles")
    return {"ok": True, "error": None}


def _serialize_subs_to_text(out_subs, out_ext: str) -> str:
    tmp_path = os.path.join(_WORK_DIR, "_ser_out" + out_ext)
    out_subs.write_file(tmp_path)
    with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def sync_with_cached_reference(
    cache_key: str,
    inputs,
    *,
    output_encoding: str = "utf-8",
    no_fix_framerate: bool = False,
    gss: bool = False,
    split_sync: bool = False,
    auto_split: bool = False,
    max_offset_seconds=None,
    inferred_sub_fps=None,
    alignment_profile=None,
    progress_cb=None,
):
    """Sync 1 or N subtitle inputs against the previously cached reference.

    ``inputs`` is a sequence of ``{"name": str, "bytes": bytes/Uint8Array}`` dicts.
    When ``len(inputs) == 1``, returns the standard single-file result dict; when
    ``len(inputs) > 1``, returns ``{"ok": ..., "multi_mode": True, "candidates": [...]}``
    ranked best-first with each candidate's ``output_text`` ready for download.
    """
    _ensure_workdir()
    if not has_cached_reference(cache_key):
        return {
            "ok": False,
            "multi_mode": len(inputs or []) > 1,
            "error": "cached reference not found",
        }
    items = list(inputs or [])
    if not items:
        return {"ok": False, "multi_mode": False, "error": "no input subtitles provided"}

    ref_path = _REF_CACHE["path"]
    ref_kind = _REF_CACHE["kind"]
    ref_enc = _REF_CACHE["reference_encoding"]
    non_speech_label = _REF_CACHE["non_speech_label"]
    multi_mode = len(items) > 1

    # Write input files with unique prefixes so duplicate basenames don't collide.
    in_paths = []
    name_by_path = {}
    fallback_fps = None
    for idx, entry in enumerate(items):
        in_name = str(entry["name"])
        raw_bytes = _as_bytes(entry["bytes"])
        ext = os.path.splitext(os.path.basename(in_name))[-1] or ".srt"
        safe_name = "multi_{}_{}".format(idx, os.path.basename(in_name))
        if not safe_name.lower().endswith(ext.lower()):
            safe_name += ext
        in_p = _write(safe_name, raw_bytes)
        in_paths.append(in_p)
        name_by_path[in_p] = in_name
        if fallback_fps is None:
            fallback_fps = _resolve_fallback_sub_fps(in_name, raw_bytes, inferred_sub_fps)

    if not multi_mode:
        in_name = name_by_path[in_paths[0]]
        out_path = os.path.join(_WORK_DIR, "output" + _output_ext(in_name))
        argv = [ref_path] + _common_argv(
            in_paths[0],
            out_path,
            output_encoding=output_encoding,
            no_fix_framerate=no_fix_framerate,
            gss=gss,
            split_sync=split_sync,
            auto_split=auto_split,
            max_offset_seconds=max_offset_seconds,
            sub_fps=fallback_fps,
            alignment_profile=alignment_profile,
        )
        if ref_kind == "audio":
            argv += ["--non-speech-label", str(non_speech_label)]
        if ref_enc:
            argv += ["--reference-encoding", ref_enc]
        ff_logger = logging.getLogger("ffsubsync")
        handler = None
        if progress_cb is not None:
            handler = _ProgressLogHandler(progress_cb, start_frac=0.0, end_frac=1.0)
            ff_logger.addHandler(handler)
        try:
            args = make_parser().parse_args(argv)
            result = run(
                args,
                reference_pipe=_REF_CACHE["pipe"],
                return_reference_pipe=True,
            )
            if result.get("reference_pipe") is not None:
                _REF_CACHE["pipe"] = result["reference_pipe"]
        except Exception:
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
        quality = result.get("quality")
        if isinstance(quality, dict) and ref_kind in ("audio", "subtitles"):
            quality = dict(quality)
            quality["reference_kind"] = ref_kind
        return {
            "ok": ok,
            "multi_mode": False,
            "offset_seconds": result.get("offset_seconds"),
            "framerate_scale_factor": result.get("framerate_scale_factor"),
            "split_segments": result.get("split_segments"),
            "dropped_cues": result.get("dropped_cues", 0),
            "clipped_cues": result.get("clipped_cues", 0),
            "alignment_profile": result.get("alignment_profile", "balanced"),
            "profile_offset_seconds": result.get("profile_offset_seconds", 0.0),
            "quality": quality,
            "output_name": _synced_name(in_name),
            "output_text": output_text,
            "error": None if ok else "sync did not succeed (see console logs)",
        }

    # Multi-subtitle batch sync against the single cached reference.
    argv = [ref_path, "-i"] + in_paths
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
    if fallback_fps is not None:
        argv += ["--sub-fps", str(fallback_fps)]
    if alignment_profile:
        argv += ["--alignment-profile", str(alignment_profile)]
    if ref_kind == "audio":
        argv += ["--non-speech-label", str(non_speech_label)]
    if ref_enc:
        argv += ["--reference-encoding", ref_enc]

    def _on_cand(idx: int, total: int, srtin):
        if progress_cb is not None:
            disp = name_by_path.get(srtin, os.path.basename(srtin or ""))
            frac = idx / float(max(1, total))
            try:
                progress_cb(frac, "Analyzing subtitle %d/%d (%s)…" % (idx + 1, total, disp))
            except Exception:
                pass

    try:
        args = make_parser().parse_args(argv)
        result = run(
            args,
            reference_pipe=_REF_CACHE["pipe"],
            candidate_progress_cb=_on_cand,
            return_reference_pipe=True,
            in_memory_only=True,
        )
        if result.get("reference_pipe") is not None:
            _REF_CACHE["pipe"] = result["reference_pipe"]
    except Exception:
        return {
            "ok": False,
            "multi_mode": True,
            "candidates": [],
            "error": traceback.format_exc(),
        }

    raw_cands = result.get("candidates") or []
    serialized = []
    any_ok = False
    for idx, cand in enumerate(raw_cands):
        in_p = cand.get("input_path") or ""
        orig_name = name_by_path.get(in_p, cand.get("input_name") or os.path.basename(in_p))
        out_name = _synced_name(orig_name)
        out_ext = _output_ext(orig_name)
        out_subs = cand.get("output_subs")
        out_text = ""
        err = cand.get("error")
        if out_subs is not None and not err:
            try:
                out_text = _serialize_subs_to_text(out_subs, out_ext)
            except Exception as exc:
                err = str(exc)
        has_output = bool(out_text) and not err
        if has_output:
            any_ok = True
        quality = cand.get("quality")
        if isinstance(quality, dict) and ref_kind in ("audio", "subtitles"):
            quality = dict(quality)
            quality["reference_kind"] = ref_kind
        serialized.append(
            {
                "id": "web-cand-%d" % idx,
                "rank": cand.get("rank", idx + 1),
                "input_name": orig_name,
                "output_name": out_name,
                "output_text": out_text,
                "ok": bool(cand.get("sync_was_successful")) and has_output,
                "can_save": has_output,
                "sync_was_successful": bool(cand.get("sync_was_successful")),
                "overall_score": int(cand.get("overall_score") or 0),
                "offset_seconds": cand.get("offset_seconds"),
                "framerate_scale_factor": cand.get("framerate_scale_factor"),
                "split_segments": cand.get("split_segments"),
                "dropped_cues": cand.get("dropped_cues", 0),
                "clipped_cues": cand.get("clipped_cues", 0),
                "alignment_profile": cand.get("alignment_profile", "balanced"),
                "profile_offset_seconds": cand.get("profile_offset_seconds", 0.0),
                "quality": quality,
                "error": err,
            }
        )

    if progress_cb is not None:
        try:
            progress_cb(1.0, "Done.")
        except Exception:
            pass

    return {
        "ok": any_ok,
        "multi_mode": True,
        "candidates": serialized,
        "error": None if any_ok else "All subtitle candidates failed to analyze.",
    }
