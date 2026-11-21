# -*- coding: utf-8 -*-
"""Portable FFmpeg locator and auto-downloader for Windows desktop builds."""

import json
import logging
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.error
import urllib.request
import zipfile
import zlib

from ffsubsync.ffmpeg_utils import subprocess_args

logger = logging.getLogger(__name__)

GYAN_GITHUB_LATEST_API = "https://api.github.com/repos/GyanD/codexffmpeg/releases/latest"
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


def _get_appdata_dir() -> Path:
    """Return the per-user AppData directory (default storage location)."""
    if platform.system() == "Windows":
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            return Path(local_app) / "ffblindsubsync"
        return Path.home() / "AppData" / "Local" / "ffblindsubsync"
    return Path.home() / ".local" / "share" / "ffblindsubsync"


# Backwards-compatible alias
_get_fallback_dir = _get_appdata_dir


def is_portable_mode(app_dir: Optional[Path] = None) -> bool:
    """Return True if a ``portable.txt`` marker (case-insensitive) exists next to the app."""
    target = Path(app_dir) if app_dir is not None else _preferred_portable_dir()
    try:
        if not target.is_dir():
            return False
        for entry in target.iterdir():
            if entry.is_file() and entry.name.lower() == "portable.txt":
                return True
    except OSError:
        return False
    return False


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
    """Return the active data directory (AppData by default, app folder if ``portable.txt`` exists)."""
    preferred = _preferred_portable_dir()
    if is_portable_mode(preferred) and _is_dir_writable(preferred):
        return preferred
    return _get_appdata_dir()


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


def _which_in_system_path(
    binary_name: str, exclude_dir: Optional[Path] = None
) -> Optional[str]:
    """Find ``binary_name`` on system PATH without implicitly matching Windows CWD ('.')."""
    raw = shutil.which(binary_name)
    if not raw:
        return None
    try:
        excl = exclude_dir.resolve() if exclude_dir is not None else None
        is_dot = os.path.dirname(raw) in ("", ".")
        is_excluded = excl is not None and os.path.exists(raw) and Path(raw).resolve().parent == excl
    except OSError:
        is_dot = False
        is_excluded = False

    if not is_dot and not is_excluded:
        return str(Path(raw).resolve()) if os.path.exists(raw) else raw

    path_env = os.environ.get("PATH") or ""
    for entry in path_env.split(os.pathsep):
        cleaned = entry.strip().strip('"')
        if not cleaned or cleaned == ".":
            continue
        try:
            cand_dir = Path(cleaned).resolve()
        except OSError:
            continue
        if exclude_dir is not None:
            try:
                if cand_dir == exclude_dir.resolve():
                    continue
            except OSError:
                pass
        cand = cand_dir / binary_name
        if cand.is_file():
            return str(cand)
    return None


def find_ffmpeg_binaries(
    portable_dir: Optional[Path] = None,
    custom_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Locate ``ffmpeg`` and ``ffprobe`` in custom path, data/app dirs, or system PATH."""
    ffmpeg_name, ffprobe_name = _REQUIRED_BINS
    if portable_dir is not None:
        base = Path(portable_dir)
        candidates: List[Tuple[Path, str]] = [
            (base, "portable"),
            (base / "ffmpeg-bin", "portable"),
        ]
    else:
        pref = _preferred_portable_dir()
        appdata = _get_appdata_dir()
        portable = is_portable_mode(pref)
        if portable and _is_dir_writable(pref):
            base = pref
            raw_candidates: List[Tuple[Path, str]] = [
                (pref, "portable"),
                (pref / "ffmpeg-bin", "portable"),
            ]
        else:
            base = appdata
            raw_candidates = [
                (appdata, "appdata"),
                (appdata / "ffmpeg-bin", "appdata"),
            ]
        candidates = []
        seen = set()
        for c, src in raw_candidates:
            key = str(c)
            if key not in seen:
                seen.add(key)
                candidates.append((c, src))

    cleaned_custom = (custom_path or "").strip()
    if cleaned_custom:
        custom_p = Path(cleaned_custom)
        search_dirs = [custom_p.parent] if custom_p.is_file() else [custom_p, custom_p / "bin"]
        for cdir in search_dirs:
            ff = cdir / ffmpeg_name
            fp = cdir / ffprobe_name
            if ff.is_file() and fp.is_file():
                return {
                    "available": True,
                    "source": "custom",
                    "custom_path": str(cdir),
                    "custom_error": None,
                    "portable_dir": str(base),
                    "resource_dir": str(cdir),
                    "ffmpeg_path": str(ff),
                    "ffprobe_path": str(fp),
                    "version": _probe_ffmpeg_version(str(ff)),
                }
        return {
            "available": False,
            "source": None,
            "custom_path": cleaned_custom,
            "custom_error": "ffmpeg and ffprobe were not found in: %s" % cleaned_custom,
            "portable_dir": str(base),
            "resource_dir": None,
            "ffmpeg_path": None,
            "ffprobe_path": None,
            "version": None,
        }

    for candidate_dir, source_kind in candidates:
        ff = candidate_dir / ffmpeg_name
        fp = candidate_dir / ffprobe_name
        if ff.is_file() and fp.is_file():
            return {
                "available": True,
                "source": source_kind,
                "custom_path": "",
                "custom_error": None,
                "portable_dir": str(base),
                "resource_dir": str(candidate_dir),
                "ffmpeg_path": str(ff),
                "ffprobe_path": str(fp),
                "version": _probe_ffmpeg_version(str(ff)),
            }

    exclude_dir = None
    if portable_dir is None and not is_portable_mode(_preferred_portable_dir()):
        exclude_dir = _preferred_portable_dir()
    sys_ff = _which_in_system_path(ffmpeg_name, exclude_dir=exclude_dir)
    sys_fp = _which_in_system_path(ffprobe_name, exclude_dir=exclude_dir)
    if sys_ff and sys_fp:
        return {
            "available": True,
            "source": "system",
            "custom_path": "",
            "custom_error": None,
            "portable_dir": str(base),
            "resource_dir": None,
            "ffmpeg_path": sys_ff,
            "ffprobe_path": sys_fp,
            "version": _probe_ffmpeg_version(sys_ff),
        }

    return {
        "available": False,
        "source": None,
        "custom_path": "",
        "custom_error": None,
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


def _resolve_gyan_github_zip_url(timeout: float = 5.0) -> Optional[str]:
    """Resolve the latest GyanD essentials zip URL hosted on GitHub Releases CDN."""
    try:
        req = urllib.request.Request(
            GYAN_GITHUB_LATEST_API,
            headers={
                "User-Agent": "ffsubsync-desktop",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        for asset in data.get("assets") or []:
            name = str(asset.get("name") or "").lower()
            dl_url = asset.get("browser_download_url")
            if dl_url and "essentials" in name and name.endswith(".zip"):
                return str(dl_url)
    except Exception as exc:
        logger.debug("Could not resolve GyanD GitHub release URL: %s", exc)
    return None


def _default_download_urls() -> Tuple[str, ...]:
    """Return mirrors ordered fastest-first (GitHub CDN before throttled gyan.dev)."""
    urls: List[str] = []
    gh_gyan = _resolve_gyan_github_zip_url()
    if gh_gyan:
        urls.append(gh_gyan)
    urls.append(BTBN_LATEST_ZIP_URL)
    urls.append(GYAN_ESSENTIALS_ZIP_URL)
    return tuple(urls)


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("Unexpected EOF while reading %d bytes (got %d)" % (n, len(buf)))
        buf.extend(chunk)
    return bytes(buf)


def _parse_zip_central_directory(
    cd_bytes: bytes, required: Tuple[str, ...]
) -> Optional[Dict[str, Dict[str, int]]]:
    """Parse ZIP Central Directory records and locate required binary entries."""
    want = {name.lower(): name for name in required}
    found: Dict[str, Dict[str, int]] = {}
    pos = 0
    end = len(cd_bytes)
    while pos + 46 <= end:
        sig = struct.unpack_from("<I", cd_bytes, pos)[0]
        if sig != 0x02014B50:
            break
        (
            comp_method,
            _mtime,
            _mdate,
            crc32_val,
            comp_size,
            uncomp_size,
            fn_len,
            ex_len,
            cm_len,
            _disk,
            _iattr,
            _eattr,
            loc_off,
        ) = struct.unpack_from("<HHHIIIHHHHHII", cd_bytes, pos + 10)
        name_end = pos + 46 + fn_len
        if name_end > end:
            return None
        entry_name = cd_bytes[pos + 46 : name_end].decode("utf-8", errors="replace")
        pos = name_end + ex_len + cm_len
        if entry_name.endswith("/"):
            continue
        leaf = PurePosixPath(entry_name).name.lower()
        if leaf in want and leaf not in found:
            if comp_method not in (0, 8):
                return None
            if comp_size == 0xFFFFFFFF or uncomp_size == 0xFFFFFFFF or loc_off == 0xFFFFFFFF:
                return None
            found[leaf] = {
                "target_name": want[leaf],
                "comp_method": comp_method,
                "crc32": crc32_val,
                "comp_size": comp_size,
                "uncomp_size": uncomp_size,
                "fn_len": fn_len,
                "loc_off": loc_off,
            }
    if len(found) != len(want):
        return None
    return found


def _try_range_download_and_extract(
    url: str,
    dest_dir: Path,
    required: Tuple[str, ...],
    emit: Callable[[Dict[str, Any]], None],
) -> bool:
    """Download and decompress only required binaries via HTTP Range requests.

    Skips unused large entries in the archive (e.g. ffplay.exe ~37 MB) and
    decompresses directly to disk with CRC32 verification. Returns True on
    success, or False if the server does not support Range requests so the
    caller can fall back to a full-archive download.
    """
    head_req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "ffsubsync-desktop"}
    )
    with urllib.request.urlopen(head_req, timeout=15) as head_resp:
        final_url = getattr(head_resp, "geturl", lambda: url)() or url
        total_hdr = head_resp.headers.get("Content-Length")
        accept_ranges = (head_resp.headers.get("Accept-Ranges") or "").lower()
    if not total_hdr or not total_hdr.isdigit() or "bytes" not in accept_ranges:
        return False
    archive_size = int(total_hdr)
    if archive_size < 1024:
        return False

    # 1. Fetch ZIP End of Central Directory (last <=64 KB).
    tail_len = min(65536, archive_size)
    tail_start = archive_size - tail_len
    tail_req = urllib.request.Request(
        final_url,
        headers={
            "User-Agent": "ffsubsync-desktop",
            "Range": "bytes=%d-%d" % (tail_start, archive_size - 1),
        },
    )
    with urllib.request.urlopen(tail_req, timeout=20) as tail_resp:
        if getattr(tail_resp, "status", None) != 206:
            return False
        tail = tail_resp.read()

    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0 or eocd + 22 > len(tail):
        return False
    cd_size, cd_offset = struct.unpack_from("<II", tail, eocd + 12)
    if cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF or cd_size == 0:
        return False

    if cd_offset >= tail_start and (cd_offset - tail_start) + cd_size <= len(tail):
        rel = cd_offset - tail_start
        cd_bytes = tail[rel : rel + cd_size]
    else:
        cd_req = urllib.request.Request(
            final_url,
            headers={
                "User-Agent": "ffsubsync-desktop",
                "Range": "bytes=%d-%d" % (cd_offset, cd_offset + cd_size - 1),
            },
        )
        with urllib.request.urlopen(cd_req, timeout=20) as cd_resp:
            if getattr(cd_resp, "status", None) != 206:
                return False
            cd_bytes = cd_resp.read()

    entries = _parse_zip_central_directory(cd_bytes, required)
    if not entries:
        return False

    # Sort entries by offset in the archive and stream only their compressed bytes.
    ordered = sorted(entries.values(), key=lambda e: e["loc_off"])
    total_to_download = sum(e["comp_size"] for e in ordered)
    downloaded = 0
    staged: List[Tuple[str, Path]] = []

    try:
        for entry in ordered:
            loc_off = entry["loc_off"]
            comp_size = entry["comp_size"]
            uncomp_size = entry["uncomp_size"]
            comp_method = entry["comp_method"]
            expected_crc = entry["crc32"]
            # Request local header + generous headroom for local extra field + compressed data.
            range_end = min(archive_size - 1, loc_off + 30 + entry["fn_len"] + 4096 + comp_size - 1)
            part_req = urllib.request.Request(
                final_url,
                headers={
                    "User-Agent": "ffsubsync-desktop",
                    "Range": "bytes=%d-%d" % (loc_off, range_end),
                },
            )
            fd, tmp_bin = tempfile.mkstemp(prefix=".tmp_", dir=str(dest_dir))
            os.close(fd)
            staged.append((tmp_bin, dest_dir / entry["target_name"]))
            with urllib.request.urlopen(part_req, timeout=30) as resp, open(tmp_bin, "wb") as out_f:
                if getattr(resp, "status", None) != 206:
                    raise RuntimeError("Server did not honor byte range request")
                lh = _read_exact(resp, 30)
                if lh[:4] != b"PK\x03\x04":
                    raise RuntimeError("Invalid ZIP local file header")
                lh_fn_len, lh_ex_len = struct.unpack_from("<HH", lh, 26)
                if lh_fn_len + lh_ex_len > 0:
                    _read_exact(resp, lh_fn_len + lh_ex_len)

                decomp = zlib.decompressobj(-zlib.MAX_WBITS) if comp_method == 8 else None
                remaining = comp_size
                crc = 0
                written = 0
                while remaining > 0:
                    chunk = resp.read(min(262144, remaining))
                    if not chunk:
                        raise EOFError("Truncated compressed stream for %s" % entry["target_name"])
                    remaining -= len(chunk)
                    downloaded += len(chunk)
                    plain = decomp.decompress(chunk) if decomp is not None else chunk
                    if plain:
                        out_f.write(plain)
                        crc = zlib.crc32(plain, crc)
                        written += len(plain)
                    frac = min(1.0, downloaded / total_to_download) if total_to_download > 0 else None
                    emit({
                        "phase": "downloading",
                        "downloaded": downloaded,
                        "total": total_to_download,
                        "fraction": frac,
                        "url": url,
                    })

                if decomp is not None:
                    tail_plain = decomp.flush()
                    if tail_plain:
                        out_f.write(tail_plain)
                        crc = zlib.crc32(tail_plain, crc)
                        written += len(tail_plain)

                if (crc & 0xFFFFFFFF) != expected_crc or written != uncomp_size:
                    raise RuntimeError(
                        "CRC/size mismatch for %s (got %d/%#x, expected %d/%#x)"
                        % (entry["target_name"], written, crc & 0xFFFFFFFF, uncomp_size, expected_crc)
                    )

        # Atomically move all verified binaries into place.
        for tmp_bin, dest in staged:
            os.replace(tmp_bin, dest)
            if platform.system() != "Windows":
                os.chmod(dest, 0o755)
        staged.clear()
        return True
    finally:
        for tmp_bin, _ in staged:
            if os.path.exists(tmp_bin):
                try:
                    os.remove(tmp_bin)
                except OSError:
                    pass


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
    candidate_urls = urls or _default_download_urls()

    def _emit(payload: Dict[str, Any]) -> None:
        if progress_cb is not None:
            try:
                progress_cb(payload)
            except Exception:
                logger.warning("ffmpeg progress callback failed", exc_info=True)

    last_err: Optional[Exception] = None
    for url in candidate_urls:
        _emit({"phase": "connecting", "downloaded": 0, "total": 0, "fraction": None, "url": url})
        # Fast path: stream only ffmpeg + ffprobe via HTTP Range (skips ffplay.exe ~37 MB).
        try:
            if _try_range_download_and_extract(url, dest_dir, _REQUIRED_BINS, _emit):
                info = find_ffmpeg_binaries(dest_dir)
                if info["available"]:
                    _emit({"phase": "ready", "downloaded": 0, "total": 0, "fraction": 1.0, "info": info})
                    return info
        except urllib.error.HTTPError as exc:
            logger.debug("Range HEAD/GET rejected (%s) for %s, falling back to full zip", exc.code, url)
        except OSError as exc:
            last_err = exc
            logger.warning("FFmpeg download from %s failed: %s", url, exc)
            continue
        except Exception as exc:
            logger.debug("Range download for %s fell back to full archive: %s", url, exc)

        # Fallback path: download full zip and extract required binaries.
        fd, tmp_zip = tempfile.mkstemp(prefix="ffmpeg_", suffix=".zip", dir=str(dest_dir))
        os.close(fd)
        try:
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
