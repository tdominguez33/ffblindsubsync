# -*- coding: utf-8 -*-
"""Portable FFmpeg locator and auto-downloader for Windows desktop builds."""

import logging
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, Optional
import urllib.request
import zipfile

from ffsubsync.ffmpeg_utils import subprocess_args

logger = logging.getLogger(__name__)

GYAN_VERSION_URL = "https://www.gyan.dev/ffmpeg/builds/release-version"
GYAN_ESSENTIALS_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
BTBN_LATEST_ZIP_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl.zip"
)

_REQUIRED_BINS = ("ffmpeg.exe", "ffprobe.exe") if platform.system() == "Windows" else ("ffmpeg", "ffprobe")


def _preferred_portable_dir() -> Path:
    """Return the folder next to the executable (or ``gui/`` when running from source)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent / "gui"


def _get_fallback_dir() -> Path:
    """Return a user-writable fallback directory when the app dir is read-only."""
    if platform.system() == "Windows":
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            return Path(local_app) / "ffblindsubsync"
        return Path.home() / "AppData" / "Local" / "ffblindsubsync"
    return Path.home() / ".local" / "share" / "ffblindsubsync"


def _is_dir_writable(path: Path) -> bool:
    """Return True if a temporary file can be created and removed in ``path``."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".write_test_", dir=str(path))
        os.close(fd)
        os.remove(probe)
        return True
    except OSError:
        return False


def get_portable_dir() -> Path:
    """Return a writable portable folder next to the executable, or fallback app-data."""
    preferred = _preferred_portable_dir()
    if _is_dir_writable(preferred):
        return preferred
    return _get_fallback_dir()


def _probe_ffmpeg_version(ffmpeg_exe: str) -> Optional[str]:
    """Return the first line of ``ffmpeg -version`` without flashing a console."""
    try:
        out = subprocess.check_output(
            [ffmpeg_exe, "-version"],
            **subprocess_args(include_stdout=False),
            timeout=5,
        )
        first = out.decode("utf-8", errors="replace").splitlines()[0].strip()
        if first.startswith("ffmpeg version "):
            return first[len("ffmpeg version ") :].split()[0]
        return first
    except Exception:
        return None


def find_ffmpeg_binaries(portable_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Locate ``ffmpeg`` and ``ffprobe`` in portable dirs first, then system PATH."""
    ffmpeg_name, ffprobe_name = _REQUIRED_BINS
    if portable_dir is not None:
        base = Path(portable_dir)
        candidates = [base, base / "ffmpeg-bin"]
    else:
        pref = _preferred_portable_dir()
        fallback = _get_fallback_dir()
        base = pref if _is_dir_writable(pref) else fallback
        raw_candidates = [pref, pref / "ffmpeg-bin"]
        if not getattr(sys, "frozen", False):
            raw_candidates.extend(
                [pref / "dist" / "FFblindsubsync", pref / "dist" / "FFsubsync"]
            )
        raw_candidates.extend([fallback, fallback / "ffmpeg-bin"])
        candidates = []
        seen = set()
        for c in raw_candidates:
            key = str(c)
            if key not in seen:
                seen.add(key)
                candidates.append(c)

    for candidate_dir in candidates:
        ff = candidate_dir / ffmpeg_name
        fp = candidate_dir / ffprobe_name
        if ff.is_file() and fp.is_file():
            return {
                "available": True,
                "source": "portable",
                "portable_dir": str(base),
                "resource_dir": str(candidate_dir),
                "ffmpeg_path": str(ff),
                "ffprobe_path": str(fp),
                "version": _probe_ffmpeg_version(str(ff)),
            }

    sys_ff = shutil.which(ffmpeg_name)
    sys_fp = shutil.which(ffprobe_name)
    if sys_ff and sys_fp:
        return {
            "available": True,
            "source": "system",
            "portable_dir": str(base),
            "resource_dir": None,
            "ffmpeg_path": sys_ff,
            "ffprobe_path": sys_fp,
            "version": _probe_ffmpeg_version(sys_ff),
        }

    return {
        "available": False,
        "source": None,
        "portable_dir": str(base),
        "resource_dir": None,
        "ffmpeg_path": None,
        "ffprobe_path": None,
        "version": None,
    }


def get_latest_ffmpeg_version(timeout: float = 5.0) -> Optional[str]:
    """Fetch the latest Windows FFmpeg release version string from gyan.dev."""
    try:
        req = urllib.request.Request(
            GYAN_VERSION_URL,
            headers={"User-Agent": "ffsubsync-desktop"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace").strip() or None
    except Exception:
        return None


def _extract_binaries_from_zip(
    zip_path: Path, target_dir: Path, required: tuple
) -> None:
    """Extract only ``ffmpeg`` and ``ffprobe`` binaries atomically into ``target_dir``."""
    want = {name.lower(): name for name in required}
    found = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            leaf = PurePosixPath(info.filename).name.lower()
            if leaf in want and leaf not in found:
                found[leaf] = info

        missing = [orig for low, orig in want.items() if low not in found]
        if missing:
            raise RuntimeError(
                "FFmpeg archive is missing expected binary: %s" % ", ".join(missing)
            )

        for low, info in found.items():
            dest = target_dir / want[low]
            fd, tmp_bin = tempfile.mkstemp(prefix=".tmp_", dir=str(target_dir))
            try:
                with os.fdopen(fd, "wb") as out_f, zf.open(info, "r") as src_f:
                    shutil.copyfileobj(src_f, out_f, length=1024 * 256)
                os.replace(tmp_bin, dest)
                if platform.system() != "Windows":
                    os.chmod(dest, 0o755)
            except Exception:
                if os.path.exists(tmp_bin):
                    os.remove(tmp_bin)
                raise


def download_latest_ffmpeg(
    target_dir: Optional[Path] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    urls: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Download the latest FFmpeg Windows build and extract it into ``target_dir``.

    Leaves ``ffmpeg.exe`` and ``ffprobe.exe`` directly inside ``target_dir``
    (the executable's directory) so the app folder stays completely portable.
    """
    if platform.system() != "Windows" and urls is None:
        raise RuntimeError(
            "Automatic FFmpeg download is currently supported on Windows builds only."
        )

    dest_dir = Path(target_dir) if target_dir is not None else get_portable_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate_urls = urls or (GYAN_ESSENTIALS_ZIP_URL, BTBN_LATEST_ZIP_URL)

    def _emit(payload: Dict[str, Any]) -> None:
        if progress_cb is not None:
            try:
                progress_cb(payload)
            except Exception:
                logger.warning("ffmpeg progress callback failed", exc_info=True)

    last_err: Optional[Exception] = None
    for url in candidate_urls:
        fd, tmp_zip = tempfile.mkstemp(prefix="ffmpeg_", suffix=".zip", dir=str(dest_dir))
        os.close(fd)
        try:
            _emit({"phase": "connecting", "downloaded": 0, "total": 0, "fraction": None, "url": url})
            req = urllib.request.Request(url, headers={"User-Agent": "ffsubsync-desktop"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_zip, "wb") as out_f:
                total_hdr = resp.headers.get("Content-Length")
                total = int(total_hdr) if total_hdr and total_hdr.isdigit() else 0
                downloaded = 0
                while True:
                    chunk = resp.read(262144)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    frac = min(1.0, downloaded / total) if total > 0 else None
                    _emit({
                        "phase": "downloading",
                        "downloaded": downloaded,
                        "total": total,
                        "fraction": frac,
                        "url": url,
                    })

            _emit({"phase": "extracting", "downloaded": downloaded, "total": total, "fraction": 1.0})
            _extract_binaries_from_zip(Path(tmp_zip), dest_dir, _REQUIRED_BINS)
            info = find_ffmpeg_binaries(dest_dir)
            if not info["available"]:
                raise RuntimeError("FFmpeg binaries were not found after extraction")
            _emit({"phase": "ready", "downloaded": downloaded, "total": total, "fraction": 1.0, "info": info})
            return info
        except Exception as exc:
            last_err = exc
            logger.warning("FFmpeg download from %s failed: %s", url, exc)
        finally:
            if os.path.exists(tmp_zip):
                try:
                    os.remove(tmp_zip)
                except OSError:
                    pass

    raise RuntimeError("Failed to download FFmpeg: %s" % last_err)
