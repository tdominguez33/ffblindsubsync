# -*- coding: utf-8 -*-
"""Native Windows desktop GUI for ffsubsync (pywebview + native FFmpeg)."""

import io
import json
import locale
import logging
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

if not getattr(sys, "frozen", False):
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import webview
from webview.dom import DOMEventHandler

from ffsubsync.constants import SUBTITLE_EXTENSIONS, SUBSYNC_RESOURCES_ENV_MAGIC

if SUBSYNC_RESOURCES_ENV_MAGIC not in os.environ:
    os.environ[SUBSYNC_RESOURCES_ENV_MAGIC] = getattr(sys, "_MEIPASS", "")

from ffsubsync.ffmpeg_downloader import (
    download_latest_ffmpeg,
    find_ffmpeg_binaries,
    get_portable_dir,
)
from ffsubsync.ffmpeg_utils import subprocess_args
from ffsubsync.ffsubsync import make_parser, run
from ffsubsync.speech_transformers import ProgressInfo
from ffsubsync.version import get_version

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

logger = logging.getLogger(__name__)


def _ui_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "ui"
    return Path(__file__).resolve().parent.parent / "ui"


def _icon_path() -> Optional[Path]:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidate = Path(sys._MEIPASS) / "icon.ico"
    else:
        candidate = Path(__file__).resolve().parent / "icon.ico"
    return candidate if candidate.is_file() else None


def _is_subtitle_ref(path: str) -> bool:
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return ext in SUBTITLE_EXTENSIONS or ext in ("npz", "npy")


_DEFAULT_WIN_WIDTH = 760
_DEFAULT_WIN_HEIGHT = 720
_MIN_WIN_WIDTH = 620
_MIN_WIN_HEIGHT = 580


def _is_rect_visible_on_screens(x: int, y: int, width: int, height: int) -> bool:
    """Return True if at least a draggable portion of the window is on a connected screen."""
    try:
        screens = list(webview.screens)
    except Exception:
        screens = []
    if not screens:
        return True
    for s in screens:
        sx, sy, sw, sh = int(s.x), int(s.y), int(s.width), int(s.height)
        overlap_w = min(x + width, sx + sw) - max(x, sx)
        overlap_h = min(y + height, sy + sh) - max(y, sy)
        # Ensure the titlebar area isn't completely off the top of the screen.
        if overlap_w >= 80 and overlap_h >= 40 and y >= sy - 16:
            return True
    return False


def _resolve_initial_window_geometry(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Read and validate persisted window size/position from desktop config."""
    win = cfg.get("window") if isinstance(cfg.get("window"), dict) else {}
    try:
        width = int(win.get("width", _DEFAULT_WIN_WIDTH))
    except (TypeError, ValueError):
        width = _DEFAULT_WIN_WIDTH
    try:
        height = int(win.get("height", _DEFAULT_WIN_HEIGHT))
    except (TypeError, ValueError):
        height = _DEFAULT_WIN_HEIGHT

    width = max(_MIN_WIN_WIDTH, min(width, 10000))
    height = max(_MIN_WIN_HEIGHT, min(height, 10000))
    maximized = bool(win.get("maximized", False))

    x = win.get("x")
    y = win.get("y")
    try:
        x_int = int(x) if x is not None else None
        y_int = int(y) if y is not None else None
    except (TypeError, ValueError):
        x_int, y_int = None, None

    if (
        x_int is not None
        and y_int is not None
        and x_int > -10000
        and y_int > -10000
        and _is_rect_visible_on_screens(x_int, y_int, width, height)
    ):
        pos_x, pos_y = x_int, y_int
    else:
        pos_x, pos_y = None, None

    return {
        "width": width,
        "height": height,
        "x": pos_x,
        "y": pos_y,
        "maximized": maximized,
    }


def _config_path() -> Path:
    return get_portable_dir() / "config.json"


def _load_desktop_config() -> Dict[str, Any]:
    try:
        p = _config_path()
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_desktop_config(cfg: Dict[str, Any]) -> None:
    try:
        p = _config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save desktop config: %s", exc)


def _get_system_languages() -> List[str]:
    """Return ordered BCP-47 system/OS UI language tags (e.g. ['es-AR', 'en-US'])."""
    found: List[str] = []

    def _add(tag: Optional[str]) -> None:
        if not tag or not isinstance(tag, str):
            return
        cleaned = tag.strip().split(".")[0].split("@")[0].replace("_", "-")
        if not cleaned or cleaned.lower() in ("c", "posix"):
            return
        if cleaned not in found:
            found.append(cleaned)

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            MUI_LANGUAGE_NAME = 0x8
            num_langs = wintypes.ULONG(0)
            buf_len = wintypes.ULONG(0)
            # First call queries required buffer size (in UTF-16 code units).
            if kernel32.GetUserPreferredUILanguages(
                MUI_LANGUAGE_NAME,
                ctypes.byref(num_langs),
                None,
                ctypes.byref(buf_len),
            ) and buf_len.value > 0:
                buf = ctypes.create_unicode_buffer(buf_len.value)
                if kernel32.GetUserPreferredUILanguages(
                    MUI_LANGUAGE_NAME,
                    ctypes.byref(num_langs),
                    buf,
                    ctypes.byref(buf_len),
                ):
                    raw_slice = ctypes.wstring_at(buf, buf_len.value)
                    for item in raw_slice.split("\x00"):
                        if item:
                            _add(item)
        except Exception:
            pass

        if not found:
            try:
                import ctypes

                lcid = int(ctypes.windll.kernel32.GetUserDefaultUILanguage())
                win_loc = locale.windows_locale.get(lcid)
                _add(win_loc)
            except Exception:
                pass

    for env_key in ("LC_ALL", "LC_MESSAGES", "LANG"):
        _add(os.environ.get(env_key))
    language_env = os.environ.get("LANGUAGE")
    if language_env:
        for part in language_env.split(":"):
            _add(part)

    if not found:
        try:
            loc_code, _ = locale.getlocale()
            _add(loc_code)
        except Exception:
            pass

    return found


_ALIGN_STEP_RE = re.compile(r"computing alignment (\d+)/(\d+)")


class _PhaseLogHandler(logging.Handler):
    """Forward key ffsubsync phase lines and collect warning/error messages."""

    def __init__(self, on_message):
        super().__init__(level=logging.INFO)
        self._on_message = on_message
        self.error_lines = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if record.levelno >= logging.WARNING and msg:
            self.error_lines.append(msg)
        m = _ALIGN_STEP_RE.search(msg)
        if m:
            cur, tot = int(m.group(1)), max(1, int(m.group(2)))
            self._on_message(
                "Computing alignment (%d/%d)…" % (cur, tot),
                (cur - 1) / float(tot),
                "progress.computingAlignmentStep",
                {"cur": cur, "tot": tot},
            )
            return
        if "split alignment" in msg:
            self._on_message("Checking mid-file breaks…", 0.92, "progress.checkingBreaks")
            return
        if "writing output" in msg:
            self._on_message("Writing output…", 0.98, "progress.writingOutput")
            return
        if "Checking video for subtitles" in msg:
            self._on_message("Checking video for subtitles…", None, "progress.checkingVideoSubs")
            return
        if "using PGS subtitle stream" in msg:
            self._on_message("Using PGS subtitle timings…", None, "progress.usingPgsTimings")
            return
        if "using VobSub subtitle stream" in msg:
            self._on_message("Using VobSub subtitle timings…", None, "progress.usingVobSubTimings")
            return
        if "using DVB subtitle stream" in msg:
            self._on_message("Using DVB subtitle timings…", None, "progress.usingDvbTimings")
            return
        if "falling back to audio VAD" in msg:
            self._on_message("Falling back to audio VAD…", None, "progress.fallingBackAudio")
            return
        if "extracting speech segments from subtitles" in msg:
            self._on_message("Extracting subtitle timings…", None, "progress.extractingSubTimings")
            return
        if "extracting speech segments" in msg:
            self._on_message("Extracting speech segments…", None, "progress.extractingSpeech")
            return
        if "computing alignments" in msg:
            self._on_message("Computing alignments…", None, "progress.computingAlignments")
            return


class DesktopApi:
    """Thread-safe JS <-> Python bridge for the native Windows desktop UI."""

    def __init__(self, initial_geom: Optional[Dict[str, Any]] = None) -> None:
        self._window: Optional[webview.Window] = None
        self._lock = threading.Lock()
        self._dl_seq = 0
        self._sync_seq = 0
        self._dl_done_event = threading.Event()
        self._dl_done_event.set()
        self._ffmpeg_dl: Dict[str, Any] = {"active": False, "done": False, "seq": 0}
        self._sync_state: Dict[str, Any] = {"active": False, "done": False, "seq": 0}
        self._config: Dict[str, Any] = _load_desktop_config()
        self._custom_ffmpeg_path: str = str(self._config.get("ffmpeg_path") or "").strip()
        self._language: str = str(self._config.get("language") or "system").strip() or "system"
        geom = initial_geom or _resolve_initial_window_geometry(self._config)
        self._win_state: Dict[str, Any] = {
            "width": geom["width"],
            "height": geom["height"],
            "x": geom["x"],
            "y": geom["y"],
            "maximized": geom["maximized"],
        }
        self._win_save_timer: Optional[threading.Timer] = None

    def attach_window(self, window: webview.Window) -> None:
        self._window = window

    def _find_ffmpeg(self) -> Dict[str, Any]:
        with self._lock:
            custom = self._custom_ffmpeg_path
        return find_ffmpeg_binaries(custom_path=custom or None)

    def _get_download_target_dir(self) -> Path:
        with self._lock:
            custom = self._custom_ffmpeg_path
        return Path(custom) if custom else get_portable_dir()

    def get_init_state(self) -> Dict[str, Any]:
        return {
            "version": get_version(),
            "ffmpeg": self._find_ffmpeg(),
            "portable_dir": str(get_portable_dir()),
            "custom_ffmpeg_path": self._custom_ffmpeg_path,
            "language": self._language,
            "system_languages": _get_system_languages(),
        }

    def set_language(self, language: str = "system") -> Dict[str, Any]:
        cleaned = str(language or "system").strip() or "system"
        with self._lock:
            self._language = cleaned
            self._config["language"] = cleaned
            _save_desktop_config(self._config)
        return {"language": cleaned}

    def pick_ffmpeg_dir(self) -> Optional[str]:
        assert self._window is not None
        initial = self._custom_ffmpeg_path
        if initial and os.path.isfile(initial):
            initial = os.path.dirname(initial)
        directory = initial if (initial and os.path.isdir(initial)) else ""
        res = self._window.create_file_dialog(
            webview.FileDialog.FOLDER,
            directory=directory,
        )
        if not res:
            return None
        return res[0] if isinstance(res, (list, tuple)) else str(res)

    def set_ffmpeg_path(self, path: str = "") -> Dict[str, Any]:
        cleaned = str(path or "").strip()
        if cleaned and os.path.isfile(cleaned):
            cleaned = os.path.dirname(cleaned)
        with self._lock:
            self._custom_ffmpeg_path = cleaned
            self._config["ffmpeg_path"] = cleaned
            _save_desktop_config(self._config)
        return {
            "ffmpeg": self._find_ffmpeg(),
            "custom_ffmpeg_path": cleaned,
        }

    def pick_reference(self, file_types: Optional[List[str]] = None) -> Optional[str]:
        assert self._window is not None
        default_types = (
            "Media files (*.mkv;*.mp4;*.avi;*.mov;*.webm;*.m4v;*.mka;*.mp3;*.aac;*.flac;*.wav;*.srt;*.ass;*.ssa;*.sub;*.vtt)",
            "Video files (*.mkv;*.mp4;*.avi;*.mov;*.webm;*.m4v)",
            "Subtitle files (*.srt;*.ass;*.ssa;*.sub;*.vtt)",
            "All files (*.*)",
        )
        filters = (
            tuple(str(x) for x in file_types if x)
            if isinstance(file_types, (list, tuple)) and file_types
            else default_types
        )
        res = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=filters,
        )
        return res[0] if res else None

    def probe_streams(self, ref_path: str) -> Dict[str, Any]:
        """Inspect audio and subtitle streams in ``ref_path`` with ffprobe."""
        if not ref_path or not os.path.isfile(ref_path) or _is_subtitle_ref(ref_path):
            return {"streams": [], "has_text_subs": False, "has_pgs_subs": False}
        ff_info = self._find_ffmpeg()
        ffprobe = ff_info.get("ffprobe_path")
        if not ffprobe:
            return {"streams": [], "has_text_subs": False, "has_pgs_subs": False}
        try:
            out = subprocess.check_output(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=index,codec_type,codec_name:stream_disposition=forced,comment:stream_tags=language,title",
                    "-of",
                    "json",
                    ref_path,
                ],
                **subprocess_args(include_stdout=False),
                timeout=10,
            )
            data = json.loads(out.decode("utf-8", errors="replace"))
        except Exception:
            return {"streams": [], "has_text_subs": False, "has_pgs_subs": False}

        raw_streams = data.get("streams") or []
        audio_idx = 0
        sub_idx = 0
        items = []
        has_text_subs = False
        has_pgs_subs = False
        bitmap_codecs = {
            "dvb_teletext",
            "xsub",
        }
        for s in raw_streams:
            ctype = s.get("codec_type")
            codec = (s.get("codec_name") or "unknown").lower()
            tags = s.get("tags") or {}
            disp = s.get("disposition") or {}
            lang = tags.get("language") or ""
            title = tags.get("title") or ""
            meta = " · ".join(p for p in (lang, title) if p)
            flags = []
            if disp.get("forced"):
                flags.append("forced")
            if disp.get("comment"):
                flags.append("commentary")
            flag_note = " (%s)" % ", ".join(flags) if flags else ""
            if ctype == "audio":
                spec = "a:%d" % audio_idx
                label = "Audio #%d (%s): %s%s [%s]" % (
                    audio_idx + 1,
                    spec,
                    meta or "default",
                    flag_note,
                    codec,
                )
                items.append(
                    {
                        "value": spec,
                        "label": label,
                        "kind": "audio",
                        "index": audio_idx + 1,
                        "spec": spec,
                        "codec": codec,
                        "lang": lang,
                        "title": title,
                        "forced": bool(disp.get("forced")),
                        "comment": bool(disp.get("comment")),
                    }
                )
                audio_idx += 1
            elif ctype == "subtitle":
                spec = "s:%d" % sub_idx
                is_pgs = codec in ("hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle")
                if is_pgs:
                    kind = "pgs"
                    is_bitmap = False
                    has_pgs_subs = True
                    if codec == "hdmv_pgs_subtitle":
                        note_kind = "pgs"
                        note = "PGS timings"
                    elif codec == "dvb_subtitle":
                        note_kind = "dvb"
                        note = "DVB timings"
                    else:
                        note_kind = "vobsub"
                        note = "VobSub timings"
                elif codec in bitmap_codecs:
                    kind = "subtitle"
                    is_bitmap = True
                    note_kind = "image"
                    note = "image: %s" % codec
                else:
                    kind = "subtitle"
                    is_bitmap = False
                    has_text_subs = True
                    note_kind = "text"
                    note = "text: %s" % codec
                label = "Subtitle #%d (%s): %s%s [%s]" % (
                    sub_idx + 1,
                    spec,
                    meta or "default",
                    flag_note,
                    note,
                )
                items.append(
                    {
                        "value": spec,
                        "label": label,
                        "kind": kind,
                        "index": sub_idx + 1,
                        "spec": spec,
                        "codec": codec,
                        "lang": lang,
                        "title": title,
                        "forced": bool(disp.get("forced")),
                        "comment": bool(disp.get("comment")),
                        "note_kind": note_kind,
                        "is_bitmap": is_bitmap,
                        "is_pgs": is_pgs,
                    }
                )
                sub_idx += 1
        return {
            "streams": items,
            "has_text_subs": has_text_subs,
            "has_pgs_subs": has_pgs_subs,
        }

    def pick_subtitle(self, file_types: Optional[List[str]] = None) -> Optional[str]:
        assert self._window is not None
        default_types = (
            "Subtitles (*.srt;*.ass;*.ssa;*.sub;*.vtt)",
            "All files (*.*)",
        )
        filters = (
            tuple(str(x) for x in file_types if x)
            if isinstance(file_types, (list, tuple)) and file_types
            else default_types
        )
        res = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=filters,
        )
        return res[0] if res else None

    def pick_output(
        self, default_path: str = "", file_types: Optional[List[str]] = None
    ) -> Optional[str]:
        assert self._window is not None
        directory = os.path.dirname(default_path) if default_path else ""
        filename = os.path.basename(default_path) if default_path else "synced.srt"
        default_types = ("SubRip (*.srt)", "All files (*.*)")
        filters = (
            tuple(str(x) for x in file_types if x)
            if isinstance(file_types, (list, tuple)) and file_types
            else default_types
        )
        res = self._window.create_file_dialog(
            webview.FileDialog.SAVE,
            directory=directory,
            save_filename=filename,
            file_types=filters,
        )
        if not res:
            return None
        return res[0] if isinstance(res, (list, tuple)) else str(res)

    def start_ffmpeg_download(self) -> Dict[str, Any]:
        with self._lock:
            if self._ffmpeg_dl.get("active"):
                return {"started": False}
            self._dl_seq += 1
            seq = self._dl_seq
            self._dl_done_event.clear()
            self._ffmpeg_dl = {
                "active": True,
                "done": False,
                "seq": seq,
                "phase": "connecting",
                "downloaded": 0,
                "total": 0,
                "fraction": None,
                "error": None,
                "info": None,
            }

        def _worker() -> None:
            try:
                def _on_dl(ev: Dict[str, Any]) -> None:
                    with self._lock:
                        self._ffmpeg_dl.update(
                            phase=ev.get("phase", "downloading"),
                            downloaded=ev.get("downloaded", 0),
                            total=ev.get("total", 0),
                            fraction=ev.get("fraction"),
                        )

                target_dir = self._get_download_target_dir()
                download_latest_ffmpeg(target_dir, progress_cb=_on_dl)
                info = self._find_ffmpeg()
                with self._lock:
                    self._ffmpeg_dl.update(
                        active=False,
                        done=True,
                        seq=seq,
                        fraction=1.0,
                        info=info,
                        error=None,
                    )
            except Exception as exc:
                with self._lock:
                    self._ffmpeg_dl.update(
                        active=False, done=True, seq=seq, error=str(exc)
                    )
            finally:
                self._dl_done_event.set()

        threading.Thread(target=_worker, daemon=True).start()
        return {"started": True}

    def start_sync(self, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._sync_state.get("active"):
                return {"started": False}
            self._sync_seq += 1
            sync_seq = self._sync_seq
            self._sync_state = {
                "active": True,
                "done": False,
                "seq": sync_seq,
                "message": "Preparing…",
                "message_key": "progress.preparing",
                "message_params": None,
                "fraction": None,
            }

        def _worker() -> None:
            t0 = time.perf_counter()
            phase_handler = None
            ff_logger = logging.getLogger("ffsubsync")
            try:
                ref_path = str(params["ref_path"])
                in_path = str(params["in_path"])
                out_path = str(params["out_path"])
                if not os.path.isfile(ref_path):
                    with self._lock:
                        self._sync_state = {
                            "active": False,
                            "done": True,
                            "seq": sync_seq,
                            "ok": False,
                            "error": "Reference file not found: %s" % ref_path,
                            "error_key": "errors.refNotFound",
                            "error_params": {"path": ref_path},
                            "elapsed_seconds": time.perf_counter() - t0,
                        }
                    return
                if not os.path.isfile(in_path):
                    with self._lock:
                        self._sync_state = {
                            "active": False,
                            "done": True,
                            "seq": sync_seq,
                            "ok": False,
                            "error": "Subtitle file not found: %s" % in_path,
                            "error_key": "errors.subNotFound",
                            "error_params": {"path": in_path},
                            "elapsed_seconds": time.perf_counter() - t0,
                        }
                    return

                ff_info = self._find_ffmpeg()
                if not _is_subtitle_ref(ref_path) and not ff_info["available"]:
                    need_download = False
                    while True:
                        with self._lock:
                            if not self._ffmpeg_dl.get("active"):
                                ff_info = find_ffmpeg_binaries(
                                    custom_path=self._custom_ffmpeg_path or None
                                )
                                if not ff_info["available"]:
                                    self._dl_seq += 1
                                    dl_seq = self._dl_seq
                                    self._dl_done_event.clear()
                                    self._ffmpeg_dl = {
                                        "active": True,
                                        "done": False,
                                        "seq": dl_seq,
                                        "phase": "connecting",
                                        "downloaded": 0,
                                        "total": 0,
                                        "fraction": None,
                                        "error": None,
                                        "info": None,
                                    }
                                    need_download = True
                                break
                        with self._lock:
                            self._sync_state.update(
                                message="Waiting for FFmpeg download…",
                                message_key="progress.waitingFfmpeg",
                                message_params=None,
                                fraction=None,
                            )
                        self._dl_done_event.wait()

                    if need_download:
                        try:
                            def _on_dl(ev: Dict[str, Any]) -> None:
                                has_frac = ev.get("fraction") is not None
                                pct_num = int(round(ev["fraction"] * 100)) if has_frac else None
                                pct = " %d%%" % pct_num if pct_num is not None else ""
                                with self._lock:
                                    self._ffmpeg_dl.update(
                                        phase=ev.get("phase", "downloading"),
                                        downloaded=ev.get("downloaded", 0),
                                        total=ev.get("total", 0),
                                        fraction=ev.get("fraction"),
                                    )
                                    self._sync_state.update(
                                        message="Downloading FFmpeg%s…" % pct,
                                        message_key=(
                                            "progress.downloadingFfmpegPct"
                                            if pct_num is not None
                                            else "progress.downloadingFfmpeg"
                                        ),
                                        message_params=(
                                            {"pct": pct_num} if pct_num is not None else None
                                        ),
                                        fraction=ev.get("fraction"),
                                    )

                            target_dir = self._get_download_target_dir()
                            download_latest_ffmpeg(target_dir, progress_cb=_on_dl)
                            ff_info = self._find_ffmpeg()
                            with self._lock:
                                self._ffmpeg_dl.update(
                                    active=False,
                                    done=True,
                                    seq=dl_seq,
                                    fraction=1.0,
                                    info=ff_info,
                                    error=None,
                                )
                        except Exception as exc:
                            with self._lock:
                                self._ffmpeg_dl.update(
                                    active=False, done=True, seq=dl_seq, error=str(exc)
                                )
                            raise
                        finally:
                            self._dl_done_event.set()

                with self._lock:
                    self._sync_state.update(
                        message="Analyzing reference…",
                        message_key="progress.analyzingRef",
                        message_params=None,
                        fraction=None,
                    )

                ref_stream = params.get("ref_stream")
                ref_stream_kind = params.get("ref_stream_kind")
                vad = str(params.get("vad") or "subs_then_webrtc")
                if (
                    ref_stream
                    and ref_stream_kind != "pgs"
                    and (
                        str(ref_stream).startswith("s:")
                        or str(ref_stream).startswith("0:s:")
                    )
                    and "subs" not in vad
                ):
                    vad = "subs_then_webrtc"

                argv = [
                    ref_path,
                    "-i",
                    in_path,
                    "-o",
                    out_path,
                    "--output-encoding",
                    "utf-8",
                    "--frame-rate",
                    str(int(params.get("frame_rate") or 48000)),
                ]
                if ff_info.get("resource_dir"):
                    argv += ["--ffmpeg-path", str(ff_info["resource_dir"])]
                if ref_stream:
                    if ref_stream_kind == "pgs":
                        argv += ["--pgs-ref-stream", str(ref_stream)]
                    else:
                        argv += ["--vad", vad, "--reference-stream", str(ref_stream)]
                else:
                    argv += ["--vad", vad]
                if params.get("no_fix_framerate"):
                    argv += ["--no-fix-framerate"]
                if params.get("gss"):
                    argv += ["--gss"]

                split_mode = params.get("split_mode", "auto")
                if split_mode == "force":
                    argv += ["--split-penalty"]
                elif split_mode == "auto":
                    argv += ["--auto-split"]

                args = make_parser().parse_args(argv)
                args.gui_mode = True

                def _on_progress(info: ProgressInfo) -> None:
                    frac = info.fraction
                    with self._lock:
                        if frac is not None:
                            pct = int(round(frac * 100))
                            self._sync_state.update(
                                message="Decoding audio + VAD (%d%%)…" % pct,
                                message_key="progress.decodingAudioVadPct",
                                message_params={"pct": pct},
                                fraction=frac,
                            )
                        else:
                            sec = "%.0f" % info.processed_seconds
                            self._sync_state.update(
                                message="Decoding audio + VAD (%ss)…" % sec,
                                message_key="progress.decodingAudioVadSec",
                                message_params={"sec": sec},
                                fraction=None,
                            )

                def _on_log(
                    msg: str,
                    frac: Optional[float] = None,
                    msg_key: Optional[str] = None,
                    msg_params: Optional[Dict[str, Any]] = None,
                ) -> None:
                    with self._lock:
                        self._sync_state["message"] = msg
                        self._sync_state["message_key"] = msg_key
                        self._sync_state["message_params"] = msg_params
                        if frac is not None:
                            self._sync_state["fraction"] = frac

                phase_handler = _PhaseLogHandler(_on_log)
                ff_logger.addHandler(phase_handler)

                result = run(args, progress_handler=_on_progress)
                elapsed = time.perf_counter() - t0
                ok = bool(result.get("sync_was_successful")) and os.path.isfile(out_path)
                last_err = (
                    phase_handler.error_lines[-1]
                    if phase_handler.error_lines
                    else None
                )
                err_msg = None
                err_key = None
                err_params = None
                if not ok:
                    if last_err:
                        err_msg = "Sync failed: %s" % last_err
                        err_key = "errors.syncFailedWithReason"
                        err_params = {"reason": last_err}
                    else:
                        err_msg = "Sync failed (check input tracks/format)."
                        err_key = "errors.syncFailedCheckTracks"
                with self._lock:
                    self._sync_state = {
                        "active": False,
                        "done": True,
                        "seq": sync_seq,
                        "ok": ok,
                        "offset_seconds": result.get("offset_seconds"),
                        "framerate_scale_factor": result.get("framerate_scale_factor"),
                        "split_segments": result.get("split_segments"),
                        "dropped_cues": result.get("dropped_cues", 0),
                        "clipped_cues": result.get("clipped_cues", 0),
                        "output_path": out_path,
                        "elapsed_seconds": elapsed,
                        "ffmpeg_info": ff_info,
                        "error": err_msg,
                        "error_key": err_key,
                        "error_params": err_params,
                    }
            except Exception as exc:
                logger.exception("Desktop sync failed")
                with self._lock:
                    self._sync_state = {
                        "active": False,
                        "done": True,
                        "seq": sync_seq,
                        "ok": False,
                        "error": "%s\n%s" % (exc, traceback.format_exc()),
                        "elapsed_seconds": time.perf_counter() - t0,
                    }
            finally:
                if phase_handler is not None:
                    ff_logger.removeHandler(phase_handler)

        threading.Thread(target=_worker, daemon=True).start()
        return {"started": True}

    def get_poll_state(self) -> Dict[str, Any]:
        with self._lock:
            dl = dict(self._ffmpeg_dl)
            sy = dict(self._sync_state)
            # Clear one-shot done flags only after reading? Keep idempotent; UI checks active.
            return {"ffmpeg_dl": dl, "sync": sy}

    def open_output_folder(self, path: str) -> None:
        norm = os.path.normpath(path)
        if os.path.exists(norm):
            subprocess.Popen(["explorer", "/select,", norm])

    def open_ffmpeg_folder(self) -> None:
        ff_info = self._find_ffmpeg()
        folder = None
        if ff_info.get("resource_dir") and os.path.isdir(ff_info["resource_dir"]):
            folder = ff_info["resource_dir"]
        elif ff_info.get("ffmpeg_path") and os.path.exists(ff_info["ffmpeg_path"]):
            folder = os.path.dirname(ff_info["ffmpeg_path"])
        elif ff_info.get("custom_path") and os.path.isdir(ff_info["custom_path"]):
            folder = ff_info["custom_path"]
        else:
            pdir = get_portable_dir()
            pdir.mkdir(parents=True, exist_ok=True)
            folder = str(pdir)
        norm = os.path.normpath(folder)
        if os.path.isdir(norm):
            os.startfile(norm)

    def _get_native_winforms(self):
        if sys.platform != "win32" or self._window is None:
            return None, 0
        native = getattr(self._window, "native", None)
        if native is None:
            return None, 0
        handle_obj = getattr(native, "Handle", None)
        hwnd = int(handle_obj.ToInt64()) if handle_obj is not None else 0
        return native, hwnd

    def _begin_nc_drag(self, hit_code: int) -> None:
        native, hwnd = self._get_native_winforms()
        if native is None or not hwnd:
            return
        try:
            import ctypes
            from System import Action

            user32 = ctypes.windll.user32

            def _do_drag():
                user32.ReleaseCapture()
                user32.SendMessageW(hwnd, 0x00A1, int(hit_code), 0)

            native.BeginInvoke(Action(_do_drag))
        except Exception:
            pass

    def window_start_drag(self) -> None:
        """Start native Win32 titlebar drag (supports smooth move & Aero Snap)."""
        self._begin_nc_drag(2)  # HTCAPTION

    def window_start_resize(self, hit_code: int) -> None:
        """Start native Win32 borderless window resize for hit_code in 10..17."""
        code = int(hit_code)
        if 10 <= code <= 17:
            self._begin_nc_drag(code)

    def window_minimize(self) -> None:
        if self._window is not None:
            self._window.minimize()

    def window_is_maximized(self) -> bool:
        native, _ = self._get_native_winforms()
        if native is None:
            return False
        try:
            import System.Windows.Forms as WinForms

            return native.WindowState == WinForms.FormWindowState.Maximized
        except Exception:
            return False

    def _window_is_minimized(self) -> bool:
        native, _ = self._get_native_winforms()
        if native is None:
            return False
        try:
            import System.Windows.Forms as WinForms

            return native.WindowState == WinForms.FormWindowState.Minimized
        except Exception:
            return False

    def _schedule_window_state_save(self) -> None:
        with self._lock:
            if self._win_save_timer is not None:
                self._win_save_timer.cancel()
            timer = threading.Timer(0.4, self._persist_window_state)
            timer.daemon = True
            self._win_save_timer = timer
            timer.start()

    def _persist_window_state(self) -> None:
        with self._lock:
            if self._win_save_timer is not None:
                self._win_save_timer.cancel()
                self._win_save_timer = None
            state = dict(self._win_state)
            self._config["window"] = state
            _save_desktop_config(self._config)

    def record_window_resize(self, width: int, height: int) -> None:
        if self._window_is_minimized() or self.window_is_maximized():
            return
        w, h = int(width), int(height)
        if w < _MIN_WIN_WIDTH or h < _MIN_WIN_HEIGHT:
            return
        with self._lock:
            self._win_state["width"] = w
            self._win_state["height"] = h
        self._schedule_window_state_save()

    def record_window_move(self, x: int, y: int) -> None:
        if self._window_is_minimized() or self.window_is_maximized():
            return
        xi, yi = int(x), int(y)
        if xi <= -10000 or yi <= -10000:
            return
        with self._lock:
            self._win_state["x"] = xi
            self._win_state["y"] = yi
        self._schedule_window_state_save()

    def record_window_maximized(self, is_max: bool) -> None:
        with self._lock:
            self._win_state["maximized"] = bool(is_max)
        self._schedule_window_state_save()

    def flush_window_state(self) -> None:
        # Capture latest normal bounds directly from WinForms if available.
        native, _ = self._get_native_winforms()
        if native is not None:
            try:
                import System.Windows.Forms as WinForms

                ws = native.WindowState
                with self._lock:
                    if ws == WinForms.FormWindowState.Maximized:
                        self._win_state["maximized"] = True
                    elif ws == WinForms.FormWindowState.Normal:
                        scale = getattr(native, "_scale", 1.0) or 1.0
                        w = int(round(native.Width / scale))
                        h = int(round(native.Height / scale))
                        x = int(round(native.Location.X / scale))
                        y = int(round(native.Location.Y / scale))
                        self._win_state["maximized"] = False
                        if w >= _MIN_WIN_WIDTH and h >= _MIN_WIN_HEIGHT:
                            self._win_state["width"] = w
                            self._win_state["height"] = h
                        if x > -10000 and y > -10000:
                            self._win_state["x"] = x
                            self._win_state["y"] = y
            except Exception:
                pass
        self._persist_window_state()

    def window_toggle_maximize(self) -> bool:
        native, _ = self._get_native_winforms()
        if native is None:
            return False
        try:
            from System import Action
            from System.Drawing import Rectangle
            import System.Windows.Forms as WinForms

            currently_max = native.WindowState == WinForms.FormWindowState.Maximized
            target_max = not currently_max

            def _toggle():
                if target_max:
                    screen = WinForms.Screen.FromControl(native)
                    wa = screen.WorkingArea
                    sb = screen.Bounds
                    # Keep maximized borderless window inside the working area (don't cover taskbar).
                    native.MaximizedBounds = Rectangle(
                        wa.X - sb.X, wa.Y - sb.Y, wa.Width, wa.Height
                    )
                    native.WindowState = WinForms.FormWindowState.Maximized
                else:
                    native.WindowState = WinForms.FormWindowState.Normal

            native.Invoke(Action(_toggle))
            return target_max
        except Exception:
            return False

    def window_close(self) -> None:
        self.flush_window_state()
        if self._window is not None:
            self._window.destroy()

    def apply_window_icon(self) -> None:
        native, _ = self._get_native_winforms()
        icon = _icon_path()
        if native is None or icon is None:
            return
        try:
            from System import Action
            import System.Drawing as Drawing

            icon_str = str(icon)

            def _apply():
                native.Icon = Drawing.Icon(icon_str)
                native.ShowIcon = True

            if native.InvokeRequired:
                native.BeginInvoke(Action(_apply))
            else:
                _apply()
        except Exception:
            pass

    def set_theme(self, is_dark: bool) -> None:
        """Sync native Windows 10/11 DWM border/caption theme with light/dark mode."""
        _, hwnd = self._get_native_winforms()
        if not hwnd:
            return
        try:
            import ctypes

            val = ctypes.c_int(1 if is_dark else 0)
            dwmapi = ctypes.windll.dwmapi
            if dwmapi.DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(val), ctypes.sizeof(val)
            ) != 0:
                dwmapi.DwmSetWindowAttribute(
                    hwnd, 19, ctypes.byref(val), ctypes.sizeof(val)
                )
        except Exception:
            pass


def _wire_drop_zones(window: webview.Window) -> None:
    """Attach native pywebview DOM drop handlers so files pass full disk paths.

    Note: dragover preventDefault is handled purely in JS (ui/app.js) — wiring
    dragover through pywebview's DOMEventHandler would serialize and IPC every
    60-120Hz mousemove event to Python and stall drops by seconds.
    """
    def _make_drop(target: str):
        def _on_drop(e):
            files = (e.get("dataTransfer") or {}).get("files") or []
            if not files:
                return
            full_path = files[0].get("pywebviewFullPath") or ""
            if full_path:
                window.evaluate_js(
                    "window.setDroppedPath(%s, %s)"
                    % (json.dumps(target), json.dumps(full_path))
                )

        return _on_drop

    drop_ref = window.dom.get_element("#drop-ref")
    drop_in = window.dom.get_element("#drop-in")
    if drop_ref is not None:
        drop_ref.events.drop += DOMEventHandler(_make_drop("ref"), prevent_default=True)
    if drop_in is not None:
        drop_in.events.drop += DOMEventHandler(_make_drop("in"), prevent_default=True)


def _wire_desktop_window(
    window: webview.Window, api: DesktopApi, start_maximized: bool = False
) -> None:
    """Wire native drop zones and window geometry persistence for frameless mode."""
    api.apply_window_icon()
    _wire_drop_zones(window)

    def _notify_max(is_max: bool) -> None:
        api.record_window_maximized(is_max)
        try:
            window.evaluate_js(
                "window.setWindowMaximized && window.setWindowMaximized(%s)"
                % ("true" if is_max else "false")
            )
        except Exception:
            pass

    window.events.maximized += lambda: _notify_max(True)
    window.events.restored += lambda: _notify_max(False)
    window.events.resized += lambda w, h: api.record_window_resize(w, h)
    window.events.moved += lambda x, y: api.record_window_move(x, y)
    window.events.closing += lambda: api.flush_window_state()

    if start_maximized:
        api.window_toggle_maximize()
        _notify_max(True)


def main() -> int:
    # Ensure Windows registry quirks never serve ES modules or CSS as text/plain.
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/javascript", ".mjs")
    mimetypes.add_type("text/css", ".css")

    cfg = _load_desktop_config()
    geom = _resolve_initial_window_geometry(cfg)
    ui_entry = str(_ui_dir() / "index.html") + "#desktop"
    api = DesktopApi(initial_geom=geom)
    window = webview.create_window(
        "ffblindsubsync — Windows Desktop",
        url=ui_entry,
        js_api=api,
        width=geom["width"],
        height=geom["height"],
        x=geom["x"],
        y=geom["y"],
        min_size=(_MIN_WIN_WIDTH, _MIN_WIN_HEIGHT),
        frameless=True,
        easy_drag=False,
        background_color="#09090b",
    )
    api.attach_window(window)
    webview.start(_wire_desktop_window, (window, api, geom["maximized"]), http_server=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
