# -*- coding: utf-8 -*-
"""Native Windows desktop GUI for ffsubsync (pywebview + native FFmpeg)."""

import atexit
import io
import json
import locale
import logging
import mimetypes
import os
from pathlib import Path
import re
import shutil
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

from ffsubsync.alignment_profile import normalize_alignment_profile
from ffsubsync.constants import (
    DEFAULT_ALIGNMENT_PROFILE,
    SUBTITLE_EXTENSIONS,
    SUBSYNC_RESOURCES_ENV_MAGIC,
)

if SUBSYNC_RESOURCES_ENV_MAGIC not in os.environ:
    os.environ[SUBSYNC_RESOURCES_ENV_MAGIC] = getattr(sys, "_MEIPASS", "")

from ffsubsync.ffmpeg_downloader import (
    download_latest_ffmpeg,
    find_ffmpeg_binaries,
    get_portable_dir,
)
from ffsubsync.ffmpeg_utils import subprocess_args
from ffsubsync.archive_utils import is_archive_filename
from ffsubsync.ffsubsync import (
    _default_synced_output_path,
    discover_subtitle_files,
    make_parser,
    run,
)
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


_DEFAULT_WIN_WIDTH = 860
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
_GSS_STEP_RE = re.compile(r"got score .* for ratio ([0-9.]+)")
_PGS_READ_RE = re.compile(r"reading .+ timings for stream (\S+)")


class _PhaseLogHandler(logging.Handler):
    """Forward key ffsubsync phase lines and collect warning/error messages."""

    def __init__(self, on_message):
        super().__init__(level=logging.INFO)
        self._on_message = on_message
        self.error_lines = []
        self._gss_steps = 0

    def reset_candidate(self) -> None:
        self._gss_steps = 0

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if record.levelno >= logging.WARNING and msg:
            self.error_lines.append(msg)
        pm = _PGS_READ_RE.search(msg)
        if pm:
            stream = pm.group(1)
            self._on_message(
                "Reading PGS/VobSub/DVB timings (%s)…" % stream,
                None,
                "progress.readingPgsStream",
                {"stream": stream},
            )
            return
        if "comparing alignment with" in msg:
            self._on_message(
                "Computing alignments…",
                None,
                "progress.comparingAlignment",
            )
            return
        m = _ALIGN_STEP_RE.search(msg)
        if m:
            cur, tot = int(m.group(1)), max(1, int(m.group(2)))
            self._on_message(
                "Computing alignment (%d/%d)…" % (cur, tot),
                (cur - 0.5) / float(tot),
                "progress.computingAlignmentStep",
                {"cur": cur, "tot": tot},
            )
            return
        gm = _GSS_STEP_RE.search(msg)
        if gm:
            self._gss_steps += 1
            self._on_message(
                "Refining framerate ratio…",
                min(1.0, self._gss_steps / 14.0),
                "progress.refiningFramerate",
            )
            return
        if "split alignment" in msg:
            self._on_message("Checking mid-file breaks…", 0.90, "progress.checkingBreaks")
            return
        if "writing output" in msg:
            self._on_message("Writing output…", 0.96, "progress.writingOutput")
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
        self._ref_cache_key: Optional[tuple] = None
        self._ref_cache_pipe: Optional[Any] = None
        self._multi_outputs: Dict[str, Dict[str, Any]] = {}
        self._archive_temp_dirs: List[str] = []
        self._archive_meta: Dict[str, Dict[str, Any]] = {}
        atexit.register(self._cleanup_archive_temps)
        self._config: Dict[str, Any] = _load_desktop_config()
        self._custom_ffmpeg_path: str = str(self._config.get("ffmpeg_path") or "").strip()
        self._language: str = str(self._config.get("language") or "system").strip() or "system"
        self._alignment_profile: str = normalize_alignment_profile(
            self._config.get("alignment_profile")
        )
        geom = initial_geom or _resolve_initial_window_geometry(self._config)
        self._win_state: Dict[str, Any] = {
            "width": geom["width"],
            "height": geom["height"],
            "x": geom["x"],
            "y": geom["y"],
            "maximized": geom["maximized"],
        }
        self._win_save_timer: Optional[threading.Timer] = None
        self._window_revealed: bool = False
        self._reveal_timer: Optional[threading.Timer] = None

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
            "alignment_profile": self._alignment_profile,
            "system_languages": _get_system_languages(),
            "maximized": self.window_is_maximized(),
        }

    def set_language(self, language: str = "system") -> Dict[str, Any]:
        cleaned = str(language or "system").strip() or "system"
        with self._lock:
            self._language = cleaned
            self._config["language"] = cleaned
            _save_desktop_config(self._config)
        return {"language": cleaned}

    def set_alignment_profile(
        self, profile: str = DEFAULT_ALIGNMENT_PROFILE
    ) -> Dict[str, Any]:
        cleaned = normalize_alignment_profile(profile)
        with self._lock:
            self._alignment_profile = cleaned
            self._config["alignment_profile"] = cleaned
            _save_desktop_config(self._config)
        return {"alignment_profile": cleaned}

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

    def _cleanup_archive_temps(self) -> None:
        with self._lock:
            dirs = list(self._archive_temp_dirs)
            self._archive_temp_dirs.clear()
            self._archive_meta.clear()
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)

    def pick_subtitle(self, file_types: Optional[List[str]] = None) -> List[str]:
        assert self._window is not None
        default_types = (
            "Subtitles & Archives (*.srt;*.ass;*.ssa;*.sub;*.vtt;*.zip;*.rar;*.7z;*.tar;*.gz;*.tgz;*.bz2;*.xz)",
            "Subtitles (*.srt;*.ass;*.ssa;*.sub;*.vtt)",
            "Archives (*.zip;*.rar;*.7z;*.tar;*.gz;*.tgz;*.bz2;*.xz)",
            "All files (*.*)",
        )
        filters = (
            tuple(str(x) for x in file_types if x)
            if isinstance(file_types, (list, tuple)) and file_types
            else default_types
        )
        res = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=True,
            file_types=filters,
        )
        if not res:
            return []
        if isinstance(res, (list, tuple)):
            return [str(p) for p in res if p]
        return [str(res)]

    def pick_subtitle_folder(self, ref_path: Optional[str] = None) -> Dict[str, Any]:
        assert self._window is not None
        res = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        if not res:
            return {"cancelled": True, "folder": None, "files": [], "entries": [], "error": None}
        folder = res[0] if isinstance(res, (list, tuple)) else str(res)
        return self.resolve_subtitle_inputs([folder], ref_path=ref_path)

    def resolve_subtitle_inputs(
        self, paths: List[str], ref_path: Optional[str] = None
    ) -> Dict[str, Any]:
        cleaned = [str(p).strip() for p in (paths or []) if p and str(p).strip()]
        if not cleaned:
            return {"cancelled": False, "files": [], "entries": [], "error": None}
        had_explicit_archive = any(
            os.path.isfile(p) and is_archive_filename(p) for p in cleaned
        )
        archive_ctx: Dict[str, Any] = {"temp_dirs": [], "meta_by_path": {}}
        try:
            discovered, had_expanded = discover_subtitle_files(
                cleaned,
                recursive=False,
                exclude_path=ref_path or None,
                archive_context=archive_ctx,
            )
            files = [p for p in discovered if p is not None]
            with self._lock:
                self._archive_temp_dirs.extend(archive_ctx["temp_dirs"])
                self._archive_meta.update(archive_ctx["meta_by_path"])
                meta_map = dict(self._archive_meta)
            entries = []
            for p in files:
                meta = meta_map.get(p)
                if meta:
                    entries.append(
                        {
                            "value": p,
                            "display": meta["display_name"],
                            "name": meta["display_name"],
                            "default_output_path": meta["default_output_path"],
                        }
                    )
                else:
                    entries.append({"value": p, "display": p})
            return {
                "cancelled": False,
                "files": files,
                "entries": entries,
                "had_dir": had_expanded,
                "error": None,
            }
        except ValueError as exc:
            for d in archive_ctx.get("temp_dirs") or []:
                shutil.rmtree(d, ignore_errors=True)
            msg = str(exc)
            if "no subtitle files found" in msg.lower():
                err_key = (
                    "errors.noSubsInArchive"
                    if had_explicit_archive
                    else "errors.noSubsInFolder"
                )
            else:
                err_key = None
            return {
                "cancelled": False,
                "files": [],
                "entries": [],
                "had_dir": True,
                "error": msg,
                "error_key": err_key,
            }

    def save_candidate(
        self, candidate_id: str, file_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        assert self._window is not None
        with self._lock:
            entry = self._multi_outputs.get(str(candidate_id))
        if not entry or entry.get("subs") is None:
            return {
                "saved": False,
                "cancelled": False,
                "error": "Candidate output is no longer available.",
            }
        default_path = entry.get("default_path") or "synced.srt"
        chosen = self.pick_output(default_path=default_path, file_types=file_types)
        if not chosen:
            return {"saved": False, "cancelled": True, "path": None, "error": None}
        try:
            entry["subs"].write_file(chosen)
            return {"saved": True, "cancelled": False, "path": chosen, "error": None}
        except Exception as exc:
            logger.exception("Failed to save candidate %s to %s", candidate_id, chosen)
            return {
                "saved": False,
                "cancelled": False,
                "path": None,
                "error": str(exc),
            }

    def _build_ref_cache_key(
        self,
        ref_path: str,
        vad: str,
        frame_rate: int,
        ref_stream: Optional[str],
        ref_stream_kind: Optional[str],
    ) -> Optional[tuple]:
        try:
            st = os.stat(ref_path)
        except OSError:
            return None
        return (
            os.path.normcase(os.path.abspath(ref_path)),
            int(st.st_mtime_ns),
            int(st.st_size),
            str(vad),
            int(frame_rate),
            str(ref_stream or ""),
            str(ref_stream_kind or ""),
        )

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
                raw_inputs = params.get("in_paths")
                if isinstance(raw_inputs, list) and raw_inputs:
                    in_paths_raw = [str(p).strip() for p in raw_inputs if p and str(p).strip()]
                else:
                    single_in = str(params.get("in_path") or "").strip()
                    in_paths_raw = [single_in] if single_in else []
                out_path = str(params.get("out_path") or "").strip()

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
                archive_ctx: Dict[str, Any] = {"temp_dirs": [], "meta_by_path": {}}
                try:
                    discovered, _had_dir = discover_subtitle_files(
                        in_paths_raw,
                        recursive=False,
                        exclude_path=ref_path,
                        archive_context=archive_ctx,
                    )
                    in_paths = [p for p in discovered if p is not None]
                    with self._lock:
                        self._archive_temp_dirs.extend(archive_ctx["temp_dirs"])
                        self._archive_meta.update(archive_ctx["meta_by_path"])
                        run_archive_meta = dict(self._archive_meta)
                except ValueError as exc:
                    for d in archive_ctx.get("temp_dirs") or []:
                        shutil.rmtree(d, ignore_errors=True)
                    with self._lock:
                        self._sync_state = {
                            "active": False,
                            "done": True,
                            "seq": sync_seq,
                            "ok": False,
                            "error": str(exc),
                            "error_key": "errors.noSubsInFolder"
                            if "no subtitle files found" in str(exc).lower()
                            else None,
                            "elapsed_seconds": time.perf_counter() - t0,
                        }
                    return
                if not in_paths:
                    with self._lock:
                        self._sync_state = {
                            "active": False,
                            "done": True,
                            "seq": sync_seq,
                            "ok": False,
                            "error": "No subtitle files selected.",
                            "error_key": "errors.noSubsInFolder",
                            "elapsed_seconds": time.perf_counter() - t0,
                        }
                    return
                for p in in_paths:
                    if not os.path.isfile(p):
                        with self._lock:
                            self._sync_state = {
                                "active": False,
                                "done": True,
                                "seq": sync_seq,
                                "ok": False,
                                "error": "Subtitle file not found: %s" % p,
                                "error_key": "errors.subNotFound",
                                "error_params": {"path": p},
                                "elapsed_seconds": time.perf_counter() - t0,
                            }
                        return

                multi_mode = len(in_paths) > 1
                if not multi_mode and not out_path:
                    meta0 = run_archive_meta.get(in_paths[0])
                    out_path = (
                        meta0["default_output_path"]
                        if meta0 and meta0.get("default_output_path")
                        else _default_synced_output_path(in_paths[0])
                    )

                ff_info = self._find_ffmpeg()
                sync_start = 0.0
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
                        sync_start = 0.25
                        try:
                            def _on_dl(ev: Dict[str, Any]) -> None:
                                dl_frac = ev.get("fraction")
                                pct_num = (
                                    int(round(max(0.0, min(1.0, float(dl_frac))) * 100))
                                    if dl_frac is not None
                                    else None
                                )
                                overall_dl = (
                                    max(0.0, min(1.0, float(dl_frac))) * sync_start
                                    if dl_frac is not None
                                    else None
                                )
                                with self._lock:
                                    self._ffmpeg_dl.update(
                                        phase=ev.get("phase", "downloading"),
                                        downloaded=ev.get("downloaded", 0),
                                        total=ev.get("total", 0),
                                        fraction=dl_frac,
                                    )
                                    if pct_num is not None:
                                        self._sync_state["message"] = (
                                            "Downloading FFmpeg (%d%%)…" % pct_num
                                        )
                                        self._sync_state["message_key"] = (
                                            "progress.downloadingFfmpegPct"
                                        )
                                        self._sync_state["message_params"] = {"pct": pct_num}
                                    else:
                                        self._sync_state["message"] = "Downloading FFmpeg…"
                                        self._sync_state["message_key"] = (
                                            "progress.downloadingFfmpeg"
                                        )
                                        self._sync_state["message_params"] = None
                                    if overall_dl is not None:
                                        prev_f = self._sync_state.get("fraction") or 0.0
                                        self._sync_state["fraction"] = max(prev_f, overall_dl)

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
                                prev_f = self._sync_state.get("fraction") or 0.0
                                self._sync_state["fraction"] = max(prev_f, sync_start)
                        except Exception as exc:
                            with self._lock:
                                self._ffmpeg_dl.update(
                                    active=False, done=True, seq=dl_seq, error=str(exc)
                                )
                            raise
                        finally:
                            self._dl_done_event.set()

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

                frame_rate = int(params.get("frame_rate") or 48000)
                cache_key = self._build_ref_cache_key(
                    ref_path, vad, frame_rate, ref_stream, ref_stream_kind
                )
                with self._lock:
                    cached_pipe = (
                        self._ref_cache_pipe
                        if (cache_key is not None and self._ref_cache_key == cache_key)
                        else None
                    )

                sync_span = 1.0 - sync_start
                if cached_pipe is not None:
                    ref_end_local = 0.05
                elif _is_subtitle_ref(ref_path):
                    ref_end_local = 0.10
                elif multi_mode:
                    ref_end_local = 0.55
                else:
                    ref_end_local = 0.72
                align_base = sync_start + ref_end_local * sync_span

                prog_state: Dict[str, Any] = {
                    "max_frac": sync_start,
                    "has_numeric": sync_start > 0.0,
                    "in_candidates": False,
                    "cand_base": align_base,
                    "cand_idx": 0,
                    "cand_total": max(1, len(in_paths)),
                    "cand_name": "",
                    "alt_pass": False,
                    "fallback_audio": False,
                    "fallback_base": None,
                    "ref_mode": "audio",
                    "ref_pass_base": sync_start + 0.02 * sync_span,
                }

                def _update_progress(
                    msg: Optional[str] = None,
                    raw_frac: Optional[float] = None,
                    msg_key: Optional[str] = None,
                    msg_params: Optional[Dict[str, Any]] = None,
                    update_msg: bool = True,
                ) -> None:
                    with self._lock:
                        if update_msg and (msg is not None or msg_key is not None):
                            if msg is not None:
                                self._sync_state["message"] = msg
                            self._sync_state["message_key"] = msg_key
                            self._sync_state["message_params"] = msg_params
                        if raw_frac is not None:
                            clamped = max(0.0, min(0.99, float(raw_frac)))
                            if not prog_state["has_numeric"] or clamped > prog_state["max_frac"]:
                                prog_state["max_frac"] = clamped
                                prog_state["has_numeric"] = True
                            self._sync_state["fraction"] = prog_state["max_frac"]
                        elif prog_state["has_numeric"]:
                            self._sync_state["fraction"] = prog_state["max_frac"]

                if cached_pipe is not None:
                    _update_progress(
                        "Reusing analyzed video audio…",
                        align_base,
                        "progress.reusingCachedRef",
                    )
                else:
                    _update_progress(
                        "Analyzing reference…",
                        sync_start + 0.02 * sync_span,
                        "progress.analyzingRef",
                    )

                argv = [ref_path, "-i"] + in_paths
                if not multi_mode:
                    argv += ["-o", out_path]
                argv += [
                    "--output-encoding",
                    "utf-8",
                    "--frame-rate",
                    str(frame_rate),
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

                with self._lock:
                    default_profile = self._alignment_profile
                alignment_profile = normalize_alignment_profile(
                    params.get("alignment_profile") or default_profile
                )
                argv += ["--alignment-profile", alignment_profile]

                args = make_parser().parse_args(argv)
                args.gui_mode = True
                args._archive_context = {
                    "temp_dirs": [],
                    "meta_by_path": run_archive_meta,
                }
                args._keep_archive_temp = True

                def _cand_slice() -> tuple:
                    tot = max(1, int(prog_state["cand_total"]))
                    idx = max(0, min(tot - 1, int(prog_state["cand_idx"])))
                    c_base = float(prog_state["cand_base"])
                    span = max(0.01, 0.99 - c_base) / float(tot)
                    return c_base + idx * span, c_base + (idx + 1) * span

                def _map_cand(local_0_to_1: float) -> float:
                    c_start, c_end = _cand_slice()
                    clamped = max(0.0, min(1.0, float(local_0_to_1)))
                    return c_start + clamped * (c_end - c_start)

                def _on_progress(info: ProgressInfo) -> None:
                    frac = info.fraction
                    is_pgs = prog_state["ref_mode"] == "pgs"
                    if frac is not None:
                        f_clamped = max(0.0, min(1.0, float(frac)))
                        pct = int(round(f_clamped * 100))
                        if prog_state["fallback_audio"] or prog_state["in_candidates"]:
                            if prog_state["fallback_base"] is None:
                                prog_state["fallback_base"] = prog_state["max_frac"]
                            fb_base = float(prog_state["fallback_base"])
                            fb_end = min(0.92, fb_base + max(0.08, (0.92 - fb_base) * 0.65))
                            overall = fb_base + f_clamped * max(0.0, fb_end - fb_base)
                        else:
                            ref_start = float(prog_state["ref_pass_base"])
                            if ref_start >= align_base - 0.02:
                                ref_end_dyn = min(0.88, ref_start + max(0.08, (0.88 - ref_start) * 0.6))
                            else:
                                ref_end_dyn = align_base
                            overall = ref_start + f_clamped * max(0.0, ref_end_dyn - ref_start)
                        if is_pgs:
                            _update_progress(
                                "Reading PGS/VobSub/DVB timings (%d%%)…" % pct,
                                overall,
                                "progress.readingPgsPct",
                                {"pct": pct},
                            )
                        else:
                            _update_progress(
                                "Decoding audio + VAD (%d%%)…" % pct,
                                overall,
                                "progress.decodingAudioVadPct",
                                {"pct": pct},
                            )
                    else:
                        sec = "%.0f" % info.processed_seconds
                        if is_pgs:
                            # Advance gently even if duration is unknown so bar shows life.
                            ref_start = float(prog_state["ref_pass_base"])
                            ref_end_dyn = max(ref_start + 0.05, align_base)
                            asymp = 1.0 - (1.0 / (1.0 + max(0.0, float(info.processed_seconds)) / 1800.0))
                            overall = ref_start + asymp * (ref_end_dyn - ref_start) * 0.9
                            _update_progress(
                                "Reading PGS/VobSub/DVB timings (%ss)…" % sec,
                                overall,
                                "progress.readingPgsSec",
                                {"sec": sec},
                            )
                        else:
                            _update_progress(
                                "Decoding audio + VAD (%ss)…" % sec,
                                None,
                                "progress.decodingAudioVadSec",
                                {"sec": sec},
                            )

                def _on_candidate(idx: int, total: int, srtin: Optional[str]) -> None:
                    meta = run_archive_meta.get(srtin) if srtin else None
                    name = (
                        meta["output_name"]
                        if (meta and meta.get("output_name"))
                        else (os.path.basename(srtin) if srtin else "")
                    )
                    if not prog_state["in_candidates"]:
                        prog_state["cand_base"] = min(
                            0.92, max(align_base, float(prog_state["max_frac"]))
                        )
                    prog_state["in_candidates"] = True
                    prog_state["cand_idx"] = idx
                    prog_state["cand_total"] = max(1, total)
                    prog_state["cand_name"] = name
                    prog_state["alt_pass"] = False
                    prog_state["fallback_audio"] = False
                    prog_state["fallback_base"] = None
                    if phase_handler is not None:
                        phase_handler.reset_candidate()
                    c_start, _ = _cand_slice()
                    if multi_mode:
                        _update_progress(
                            "Analyzing subtitle %d/%d (%s)…" % (idx + 1, total, name),
                            c_start,
                            "progress.analyzingSubtitle",
                            {"cur": idx + 1, "tot": total, "name": name},
                        )
                    else:
                        _update_progress(raw_frac=c_start, update_msg=False)

                def _on_log(
                    msg: str,
                    frac: Optional[float] = None,
                    msg_key: Optional[str] = None,
                    msg_params: Optional[Dict[str, Any]] = None,
                ) -> None:
                    if msg_key == "progress.readingPgsStream":
                        prog_state["ref_mode"] = "pgs"
                        prog_state["ref_pass_base"] = max(
                            sync_start + 0.02 * sync_span, prog_state["max_frac"]
                        )
                        _update_progress(msg, prog_state["ref_pass_base"], msg_key, msg_params)
                        return
                    if msg_key == "progress.comparingAlignment":
                        prog_state["alt_pass"] = True
                        _update_progress(
                            "Computing alignments…",
                            _map_cand(0.65),
                            "progress.computingAlignments",
                            update_msg=not multi_mode,
                        )
                        return
                    if msg_key == "progress.fallingBackAudio":
                        prog_state["ref_mode"] = "audio"
                        if prog_state["in_candidates"]:
                            prog_state["fallback_audio"] = True
                            prog_state["fallback_base"] = prog_state["max_frac"]
                        else:
                            prog_state["ref_pass_base"] = max(
                                sync_start + 0.02 * sync_span, prog_state["max_frac"]
                            )
                        _update_progress(msg, None, msg_key, msg_params)
                        return
                    if msg_key == "progress.extractingSpeech" and not prog_state["in_candidates"]:
                        prog_state["ref_mode"] = "audio"
                        prog_state["ref_pass_base"] = max(
                            sync_start + 0.02 * sync_span, prog_state["max_frac"]
                        )
                    if msg_key == "progress.computingAlignmentStep":
                        step_f = max(0.0, min(1.0, float(frac or 0.0)))
                        local_c = (
                            0.65 + 0.20 * step_f
                            if prog_state["alt_pass"]
                            else 0.08 + 0.57 * step_f
                        )
                        _update_progress(
                            msg,
                            _map_cand(local_c),
                            msg_key,
                            msg_params,
                            update_msg=not multi_mode,
                        )
                        return
                    if msg_key == "progress.refiningFramerate":
                        step_f = max(0.0, min(1.0, float(frac or 0.0)))
                        local_c = 0.15 + 0.70 * step_f
                        _update_progress(
                            msg,
                            _map_cand(local_c),
                            msg_key,
                            msg_params,
                            update_msg=not multi_mode,
                        )
                        return
                    if msg_key in ("progress.checkingBreaks", "progress.writingOutput"):
                        local_c = 0.90 if msg_key == "progress.checkingBreaks" else 0.96
                        _update_progress(
                            msg,
                            _map_cand(local_c),
                            msg_key,
                            msg_params,
                            update_msg=not multi_mode,
                        )
                        return
                    if msg_key in (
                        "progress.extractingSubTimings",
                        "progress.computingAlignments",
                    ):
                        local_c = 0.03 if msg_key == "progress.extractingSubTimings" else 0.06
                        target_f = _map_cand(local_c) if prog_state["in_candidates"] else None
                        _update_progress(
                            msg,
                            target_f,
                            msg_key,
                            msg_params,
                            update_msg=not multi_mode,
                        )
                        return
                    # Reference-stage status logs (checkingVideoSubs, usingPgsTimings, etc.)
                    ref_milestones = {
                        "progress.checkingVideoSubs": 0.04,
                        "progress.usingPgsTimings": 0.10,
                        "progress.usingVobSubTimings": 0.10,
                        "progress.usingDvbTimings": 0.10,
                        "progress.extractingSpeech": 0.06,
                    }
                    milestone = ref_milestones.get(msg_key or "")
                    target_f = (
                        sync_start + min(ref_end_local, milestone) * sync_span
                        if (milestone is not None and not prog_state["in_candidates"])
                        else None
                    )
                    _update_progress(msg, target_f, msg_key, msg_params)

                phase_handler = _PhaseLogHandler(_on_log)
                ff_logger.addHandler(phase_handler)

                if cached_pipe is not None:
                    ref_step = (
                        cached_pipe[-1]
                        if hasattr(cached_pipe, "__getitem__")
                        else cached_pipe
                    )
                    if hasattr(ref_step, "progress_handler"):
                        ref_step.progress_handler = _on_progress

                result = run(
                    args,
                    progress_handler=_on_progress,
                    reference_pipe=cached_pipe,
                    candidate_progress_cb=_on_candidate,
                    return_reference_pipe=True,
                    in_memory_only=multi_mode,
                )
                elapsed = time.perf_counter() - t0

                fitted_pipe = result.get("reference_pipe")
                if fitted_pipe is not None and cache_key is not None:
                    with self._lock:
                        self._ref_cache_key = cache_key
                        self._ref_cache_pipe = fitted_pipe

                if multi_mode:
                    raw_cands = result.get("candidates") or []
                    stored_outputs: Dict[str, Dict[str, Any]] = {}
                    serialized_cands: List[Dict[str, Any]] = []
                    any_ok = False
                    for i, cand in enumerate(raw_cands):
                        cid = "cand-%d-%d" % (sync_seq, i)
                        in_p = cand.get("input_path") or ""
                        default_out = (
                            cand.get("default_output_path")
                            or (_default_synced_output_path(in_p) if in_p else "synced.srt")
                        )
                        out_subs = cand.get("output_subs")
                        has_output = out_subs is not None and not cand.get("error")
                        if has_output:
                            stored_outputs[cid] = {
                                "subs": out_subs,
                                "default_path": default_out,
                                "input_path": in_p,
                            }
                        cand_ok = bool(cand.get("sync_was_successful")) and has_output
                        if has_output:
                            any_ok = True
                        serialized_cands.append(
                            {
                                "id": cid,
                                "rank": cand.get("rank", i + 1),
                                "input_path": in_p,
                                "input_name": cand.get("input_name") or os.path.basename(in_p),
                                "output_name": os.path.basename(default_out),
                                "ok": cand_ok,
                                "can_save": has_output,
                                "sync_was_successful": bool(cand.get("sync_was_successful")),
                                "overall_score": int(cand.get("overall_score") or 0),
                                "offset_seconds": cand.get("offset_seconds"),
                                "framerate_scale_factor": cand.get("framerate_scale_factor"),
                                "split_segments": cand.get("split_segments"),
                                "dropped_cues": cand.get("dropped_cues", 0),
                                "clipped_cues": cand.get("clipped_cues", 0),
                                "alignment_profile": cand.get("alignment_profile", alignment_profile),
                                "profile_offset_seconds": cand.get("profile_offset_seconds", 0.0),
                                "quality": cand.get("quality"),
                                "error": cand.get("error"),
                            }
                        )
                    with self._lock:
                        self._multi_outputs = stored_outputs
                        self._sync_state = {
                            "active": False,
                            "done": True,
                            "seq": sync_seq,
                            "ok": any_ok,
                            "multi_mode": True,
                            "candidates": serialized_cands,
                            "ref_cached": cached_pipe is not None,
                            "elapsed_seconds": elapsed,
                            "ffmpeg_info": ff_info,
                            "error": None if any_ok else "All subtitle candidates failed to analyze.",
                            "error_key": None if any_ok else "errors.allCandidatesFailed",
                        }
                else:
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
                            "multi_mode": False,
                            "ref_cached": cached_pipe is not None,
                            "offset_seconds": result.get("offset_seconds"),
                            "framerate_scale_factor": result.get("framerate_scale_factor"),
                            "split_segments": result.get("split_segments"),
                            "dropped_cues": result.get("dropped_cues", 0),
                            "clipped_cues": result.get("clipped_cues", 0),
                            "alignment_profile": result.get(
                                "alignment_profile", alignment_profile
                            ),
                            "profile_offset_seconds": result.get(
                                "profile_offset_seconds", 0.0
                            ),
                            "quality": result.get("quality"),
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

    def prepare_before_show(self, start_maximized: bool = False) -> None:
        """Apply icon, maximized bounds, and hide window until first UI paint.

        Runs inside ``window.events.before_show`` on the WinForms STA thread:
        setting ``Opacity = 0.0`` keeps the HWND visible to WebView2 so Chromium
        commits its first painted frame invisibly (no blank black window flash),
        then ``reveal_window()`` fades it in atomically once ``boot()`` renders.
        """
        self.apply_window_icon()
        native, _ = self._get_native_winforms()
        if native is None:
            return
        try:
            from System.Drawing import Rectangle
            import System.Windows.Forms as WinForms

            if start_maximized:
                screen = WinForms.Screen.FromControl(native)
                wa = screen.WorkingArea
                sb = screen.Bounds
                native.MaximizedBounds = Rectangle(
                    wa.X - sb.X, wa.Y - sb.Y, wa.Width, wa.Height
                )
                native.WindowState = WinForms.FormWindowState.Maximized
                native.old_state = native.WindowState

            native.Opacity = 0.0
            with self._lock:
                self._window_revealed = False
                if self._reveal_timer is not None:
                    self._reveal_timer.cancel()
                # Safety fallback so the window always appears even if JS boot stalls.
                timer = threading.Timer(2.5, self.reveal_window)
                timer.daemon = True
                self._reveal_timer = timer
                timer.start()
        except Exception:
            pass

    def reveal_window(self) -> None:
        """Make the window visible once the frontend has rendered its first frame."""
        with self._lock:
            if self._window_revealed:
                return
            self._window_revealed = True
            if self._reveal_timer is not None:
                self._reveal_timer.cancel()
                self._reveal_timer = None
        native, _ = self._get_native_winforms()
        if native is None:
            return
        try:
            from System import Action

            def _reveal():
                native.Opacity = 1.0
                native.Activate()

            if native.InvokeRequired:
                native.BeginInvoke(Action(_reveal))
            else:
                _reveal()
        except Exception:
            pass

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
        with self._lock:
            if self._reveal_timer is not None:
                self._reveal_timer.cancel()
                self._reveal_timer = None
            self._window_revealed = True
        self.flush_window_state()
        self._cleanup_archive_temps()
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
            if target == "in":
                paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
                if paths:
                    window.evaluate_js(
                        "window.setDroppedPaths && window.setDroppedPaths(%s, %s)"
                        % (json.dumps(target), json.dumps(paths))
                    )
            else:
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
        if not api.window_is_maximized():
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
    window.events.before_show += lambda: api.prepare_before_show(
        start_maximized=bool(geom["maximized"])
    )
    webview.start(_wire_desktop_window, (window, api, geom["maximized"]), http_server=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
