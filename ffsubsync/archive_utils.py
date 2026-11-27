# -*- coding: utf-8 -*-
"""Safe extraction of subtitle files from compressed archives (.zip, .rar, .7z, .tar.*, .gz)."""

try:
    import bz2
except ImportError:  # pragma: no cover - unvendored in Pyodide WASM
    bz2 = None  # type: ignore

import gzip
import logging

try:
    import lzma
except ImportError:  # pragma: no cover - unvendored in Pyodide WASM
    lzma = None  # type: ignore
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import zipfile

from ffsubsync.constants import ARCHIVE_EXTENSIONS, SUBTITLE_EXTENSIONS
from ffsubsync.ffmpeg_utils import subprocess_args

logger = logging.getLogger(__name__)

# Guard rails against zip bombs / huge non-subtitle entries.
MAX_ARCHIVE_SUBTITLE_BYTES: int = 25 * 1024 * 1024  # 25 MB per subtitle file
MAX_ARCHIVE_SUBTITLES: int = 200
MAX_ARCHIVE_NESTING_DEPTH: int = 2

_PRESERVED_OUT_EXTS = frozenset({".srt", ".ass", ".ssa", ".vtt"})
_COMPOUND_TAR_EXTS = (".tar.gz", ".tar.bz2", ".tar.xz")
_SINGLE_COMP_EXTS = (".gz", ".bz2", ".xz")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def is_archive_filename(name: str) -> bool:
    """Return True if ``name`` has a supported compressed archive extension."""
    if not name:
        return False
    low = os.path.basename(name).lower()
    for ext in _COMPOUND_TAR_EXTS:
        if low.endswith(ext):
            return True
    _, ext = os.path.splitext(low)
    ext_clean = ext.lstrip(".")
    if ext_clean not in ARCHIVE_EXTENSIONS:
        return False
    # For standalone .gz/.bz2/.xz (not .tar.*), only treat as subtitle archive
    # when the inner filename has a subtitle extension (e.g. movie.srt.gz) or
    # could be a tar archive; still allow trying tarfile first if unknown.
    return True


def is_synced_subtitle_name(name: str) -> bool:
    """Return True if ``name`` looks like a previously synced ``*.synced.<ext>`` output."""
    stem, _ = os.path.splitext(os.path.basename(name))
    return stem.lower().endswith(".synced")


def _normalize_member_path(raw: str) -> str:
    return str(raw or "").replace("\\", "/").lstrip("/")


def _is_junk_member(norm_path: str) -> bool:
    parts = [p for p in norm_path.split("/") if p]
    if not parts:
        return True
    for part in parts:
        if part == "__MACOSX":
            return True
    base = parts[-1]
    if base.startswith("._"):
        return True
    return False


def _is_valid_subtitle_member(norm_path: str) -> bool:
    if _is_junk_member(norm_path):
        return False
    base = norm_path.rsplit("/", 1)[-1]
    ext = os.path.splitext(base)[1].lstrip(".").lower()
    if ext not in SUBTITLE_EXTENSIONS:
        return False
    if is_synced_subtitle_name(base):
        return False
    return True


def _is_nested_archive_member(norm_path: str) -> bool:
    if _is_junk_member(norm_path):
        return False
    base = norm_path.rsplit("/", 1)[-1]
    return is_archive_filename(base)


def _safe_temp_filename(idx: int, original_name: str) -> str:
    base = os.path.basename(original_name)
    stem, ext = os.path.splitext(base)
    safe_stem = _SAFE_FILENAME_RE.sub("_", stem).strip("._") or "subtitle"
    safe_stem = safe_stem[:64]
    safe_ext = ext.lower() if ext else ".srt"
    return "{:03d}_{}{}".format(idx, safe_stem, safe_ext)


def _default_output_for_member(archive_path: str, inner_basename: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(inner_basename))
    low = ext.lower()
    out_ext = low if low in _PRESERVED_OUT_EXTS else ".srt"
    parent_dir = os.path.dirname(os.path.abspath(archive_path))
    return os.path.join(parent_dir, "{}.synced{}".format(stem, out_ext))


def _read_stream_limited(stream, limit: int = MAX_ARCHIVE_SUBTITLE_BYTES) -> Optional[bytes]:
    data = stream.read(limit + 1)
    if len(data) > limit:
        return None
    return data


def _find_cli_tool(names: Tuple[str, ...]) -> Optional[str]:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    if os.name == "nt":
        sys32_tar = os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "tar.exe"
        )
        if "tar" in names or "bsdtar" in names:
            if os.path.isfile(sys32_tar):
                return sys32_tar
        for prog_env in ("ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(prog_env)
            if not base:
                continue
            if "7z" in names or "7za" in names:
                cand = os.path.join(base, "7-Zip", "7z.exe")
                if os.path.isfile(cand):
                    return cand
            if "unrar" in names:
                cand = os.path.join(base, "WinRAR", "UnRAR.exe")
                if os.path.isfile(cand):
                    return cand
    return None


def _collect_extracted_tree(
    staging_dir: str,
) -> List[Tuple[str, bytes]]:
    results: List[Tuple[str, bytes]] = []
    for root, dirnames, filenames in os.walk(staging_dir):
        dirnames.sort()
        for fname in sorted(filenames):
            full = os.path.join(root, fname)
            rel = _normalize_member_path(os.path.relpath(full, staging_dir))
            if not (_is_valid_subtitle_member(rel) or _is_nested_archive_member(rel)):
                continue
            try:
                if os.path.getsize(full) > MAX_ARCHIVE_SUBTITLE_BYTES:
                    continue
                with open(full, "rb") as f:
                    data = f.read(MAX_ARCHIVE_SUBTITLE_BYTES + 1)
                if len(data) <= MAX_ARCHIVE_SUBTITLE_BYTES:
                    results.append((rel, data))
            except OSError:
                continue
    return results


def _extract_with_bsdtar(archive_path: str) -> Optional[List[Tuple[str, bytes]]]:
    tar_bin = _find_cli_tool(("bsdtar", "tar"))
    if not tar_bin:
        return None
    staging = tempfile.mkdtemp(prefix="ffsubsync_bsdtar_")
    try:
        proc = subprocess.run(
            [tar_bin, "-xf", archive_path, "-C", staging],
            timeout=60,
            **subprocess_args(include_stdout=True),
        )
        if proc.returncode != 0:
            return None
        return _collect_extracted_tree(staging)
    except Exception as exc:
        logger.debug("bsdtar extraction failed for %s: %s", archive_path, exc)
        return None
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _extract_with_7z_cli(archive_path: str) -> Optional[List[Tuple[str, bytes]]]:
    seven_bin = _find_cli_tool(("7z", "7za", "7zz"))
    if not seven_bin:
        return None
    staging = tempfile.mkdtemp(prefix="ffsubsync_7zcli_")
    try:
        proc = subprocess.run(
            [seven_bin, "x", "-y", "-p", "-o" + staging, archive_path],
            timeout=60,
            **subprocess_args(include_stdout=True),
        )
        if proc.returncode != 0:
            return None
        return _collect_extracted_tree(staging)
    except Exception as exc:
        logger.debug("7z CLI extraction failed for %s: %s", archive_path, exc)
        return None
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _extract_with_unrar_cli(archive_path: str) -> Optional[List[Tuple[str, bytes]]]:
    unrar_bin = _find_cli_tool(("unrar", "unar"))
    if not unrar_bin:
        return None
    staging = tempfile.mkdtemp(prefix="ffsubsync_unrarcli_")
    try:
        base = os.path.basename(unrar_bin).lower()
        if "unar" in base and "unrar" not in base:
            cmd = [unrar_bin, "-f", "-o", staging, archive_path]
        else:
            cmd = [unrar_bin, "x", "-y", "-p-", archive_path, staging + os.sep]
        proc = subprocess.run(
            cmd,
            timeout=60,
            **subprocess_args(include_stdout=True),
        )
        if proc.returncode != 0:
            return None
        return _collect_extracted_tree(staging)
    except Exception as exc:
        logger.debug("unrar CLI extraction failed for %s: %s", archive_path, exc)
        return None
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _read_zip_entries(archive_path: str) -> List[Tuple[str, bytes]]:
    entries: List[Tuple[str, bytes]] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            norm = _normalize_member_path(info.filename)
            if not (_is_valid_subtitle_member(norm) or _is_nested_archive_member(norm)):
                continue
            if info.file_size > MAX_ARCHIVE_SUBTITLE_BYTES:
                logger.warning("skipping oversized archive member %s", norm)
                continue
            try:
                with zf.open(info, "r") as stream:
                    data = _read_stream_limited(stream)
            except RuntimeError as exc:
                if "encrypted" in str(exc).lower() or "password" in str(exc).lower():
                    raise ValueError(
                        "archive is password-protected: {}".format(
                            os.path.basename(archive_path)
                        )
                    ) from exc
                raise
            if data is not None:
                entries.append((norm, data))
    return entries


def _read_tar_entries(archive_path: str) -> List[Tuple[str, bytes]]:
    entries: List[Tuple[str, bytes]] = []
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            norm = _normalize_member_path(member.name)
            if not (_is_valid_subtitle_member(norm) or _is_nested_archive_member(norm)):
                continue
            if member.size > MAX_ARCHIVE_SUBTITLE_BYTES:
                continue
            stream = tf.extractfile(member)
            if stream is None:
                continue
            with stream:
                data = _read_stream_limited(stream)
            if data is not None:
                entries.append((norm, data))
    return entries


def _read_single_compressed(archive_path: str, ext: str) -> List[Tuple[str, bytes]]:
    # Try tarfile first in case a .gz/.bz2/.xz is actually a tar archive.
    try:
        if tarfile.is_tarfile(archive_path):
            return _read_tar_entries(archive_path)
    except Exception:
        pass
    base = os.path.basename(archive_path)
    inner_name = base[: -len(ext)] if base.lower().endswith(ext) else base
    if not _is_valid_subtitle_member(inner_name):
        return []
    openers = {
        ".gz": gzip.open,
        ".bz2": bz2.open if bz2 is not None else None,
        ".xz": lzma.open if lzma is not None else None,
    }
    opener = openers.get(ext)
    if opener is None:
        raise ValueError(
            "compression module for '{}' is not available in this Python runtime".format(
                ext
            )
        )
    with opener(archive_path, "rb") as stream:
        data = _read_stream_limited(stream)
    return [(inner_name, data)] if data is not None else []


def _read_7z_entries(archive_path: str) -> List[Tuple[str, bytes]]:
    last_err: Optional[Exception] = None
    try:
        import py7zr  # type: ignore

        staging = tempfile.mkdtemp(prefix="ffsubsync_py7zr_")
        try:
            with py7zr.SevenZipFile(archive_path, mode="r") as sz:
                if sz.needs_password():
                    raise ValueError(
                        "archive is password-protected: {}".format(
                            os.path.basename(archive_path)
                        )
                    )
                wanted: List[str] = []
                for info in sz.list():
                    if getattr(info, "is_directory", False):
                        continue
                    norm = _normalize_member_path(info.filename)
                    if not (
                        _is_valid_subtitle_member(norm)
                        or _is_nested_archive_member(norm)
                    ):
                        continue
                    uncomp = getattr(info, "uncompressed", 0) or 0
                    if uncomp > MAX_ARCHIVE_SUBTITLE_BYTES:
                        continue
                    wanted.append(info.filename)
                if not wanted:
                    return []
                sz.extract(path=staging, targets=wanted)
            return _collect_extracted_tree(staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    except ValueError:
        raise
    except Exception as exc:
        last_err = exc
        logger.debug("py7zr failed for %s (%s); trying CLI fallbacks", archive_path, exc)

    for fn in (_extract_with_bsdtar, _extract_with_7z_cli):
        res = fn(archive_path)
        if res is not None:
            return res

    raise ValueError(
        "unable to extract 7z archive '{}': {}".format(
            os.path.basename(archive_path),
            last_err or "install py7zr or 7-Zip",
        )
    )


def _read_rar_entries(archive_path: str) -> List[Tuple[str, bytes]]:
    last_err: Optional[Exception] = None
    try:
        from unrar.cffi import rarfile as cffi_rarfile  # type: ignore

        entries: List[Tuple[str, bytes]] = []
        rf = cffi_rarfile.RarFile(archive_path)
        for info in rf.infolist():
            is_dir_fn = getattr(info, "is_dir", None)
            if callable(is_dir_fn) and is_dir_fn():
                continue
            if getattr(info, "flag_bits", 0) & 0xE0 == 0xE0:
                continue
            norm = _normalize_member_path(info.filename)
            if not norm or norm.endswith("/"):
                continue
            if not (
                _is_valid_subtitle_member(norm)
                or _is_nested_archive_member(norm)
            ):
                continue
            if getattr(info, "file_size", 0) > MAX_ARCHIVE_SUBTITLE_BYTES:
                continue
            try:
                data = rf.read(info)
            except Exception as exc:
                msg = str(exc).lower()
                if "password" in msg or "encrypt" in msg:
                    raise ValueError(
                        "archive is password-protected: {}".format(
                            os.path.basename(archive_path)
                        )
                    ) from exc
                raise
            if len(data) <= MAX_ARCHIVE_SUBTITLE_BYTES:
                entries.append((norm, data))
        return entries
    except ValueError:
        raise
    except Exception as exc:
        last_err = exc
        logger.debug("unrar2-cffi failed for %s (%s); trying CLI fallbacks", archive_path, exc)

    for fn in (_extract_with_bsdtar, _extract_with_unrar_cli, _extract_with_7z_cli):
        res = fn(archive_path)
        if res is not None:
            return res

    raise ValueError(
        "unable to extract RAR archive '{}': {}".format(
            os.path.basename(archive_path),
            last_err or "install unrar2-cffi, 7-Zip, or UnRAR",
        )
    )


def _read_raw_archive_entries(archive_path: str) -> List[Tuple[str, bytes]]:
    low = os.path.basename(archive_path).lower()
    if low.endswith(".zip"):
        return _read_zip_entries(archive_path)
    if low.endswith(".rar"):
        return _read_rar_entries(archive_path)
    if low.endswith(".7z"):
        return _read_7z_entries(archive_path)
    if low.endswith(_COMPOUND_TAR_EXTS) or low.endswith((".tar", ".tgz", ".tbz2", ".txz")):
        return _read_tar_entries(archive_path)
    for ext in _SINGLE_COMP_EXTS:
        if low.endswith(ext):
            return _read_single_compressed(archive_path, ext)
    raise ValueError("unsupported archive format: {}".format(archive_path))


def extract_subtitles_from_archive(
    archive_path: str,
    *,
    dest_dir: Optional[str] = None,
    _depth: int = 0,
    _label_prefix: Optional[str] = None,
    _root_archive: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Extract all valid subtitle files from ``archive_path`` into a temp directory.

    Returns ``(records, temp_dir)`` where each record is a dict containing:
      - ``path``: path on disk to the extracted subtitle file
      - ``display_name``: human-readable label (e.g. ``subs.rar › movie.es.srt``)
      - ``output_name``: clean inner subtitle basename (e.g. ``movie.es.srt``)
      - ``archive_path``: path to the top-level archive file
      - ``member_path``: normalized path inside the archive
      - ``default_output_path``: suggested ``<inner>.synced.<ext>`` in the archive's folder
    """
    if not os.path.isfile(archive_path):
        raise ValueError("archive file not found: {}".format(archive_path))

    created_dir = dest_dir is None
    out_dir = dest_dir or tempfile.mkdtemp(prefix="ffsubsync_archive_")
    root_archive = _root_archive or archive_path
    prefix = _label_prefix or os.path.basename(archive_path)

    try:
        raw_entries = _read_raw_archive_entries(archive_path)
        sub_items: List[Tuple[str, bytes]] = []
        nested_items: List[Tuple[str, bytes]] = []
        for norm_path, data in raw_entries:
            if _is_valid_subtitle_member(norm_path):
                sub_items.append((norm_path, data))
            elif _depth < MAX_ARCHIVE_NESTING_DEPTH and _is_nested_archive_member(norm_path):
                nested_items.append((norm_path, data))

        # Check basename collisions inside this archive so display labels stay unambiguous.
        base_counts: Dict[str, int] = {}
        for norm_path, _ in sub_items:
            b = norm_path.rsplit("/", 1)[-1].lower()
            base_counts[b] = base_counts.get(b, 0) + 1

        records: List[Dict[str, Any]] = []
        idx = len(os.listdir(out_dir))

        for norm_path, data in sorted(sub_items, key=lambda pair: pair[0].lower()):
            if len(records) >= MAX_ARCHIVE_SUBTITLES:
                break
            inner_base = norm_path.rsplit("/", 1)[-1]
            idx += 1
            dest_path = os.path.join(out_dir, _safe_temp_filename(idx, inner_base))
            with open(dest_path, "wb") as f:
                f.write(data)
            shown_inner = (
                norm_path if base_counts.get(inner_base.lower(), 0) > 1 else inner_base
            )
            records.append(
                {
                    "path": dest_path,
                    "display_name": "{} › {}".format(prefix, shown_inner),
                    "output_name": inner_base,
                    "archive_path": root_archive,
                    "member_path": norm_path,
                    "default_output_path": _default_output_for_member(
                        root_archive, inner_base
                    ),
                }
            )

        for norm_path, data in sorted(nested_items, key=lambda pair: pair[0].lower()):
            if len(records) >= MAX_ARCHIVE_SUBTITLES:
                break
            nested_base = norm_path.rsplit("/", 1)[-1]
            nested_tmp_dir = tempfile.mkdtemp(prefix="ffsubsync_nested_")
            try:
                nested_file = os.path.join(
                    nested_tmp_dir, _safe_temp_filename(0, nested_base)
                )
                with open(nested_file, "wb") as f:
                    f.write(data)
                nested_recs, _ = extract_subtitles_from_archive(
                    nested_file,
                    dest_dir=out_dir,
                    _depth=_depth + 1,
                    _label_prefix="{} › {}".format(prefix, nested_base),
                    _root_archive=root_archive,
                )
                remaining = MAX_ARCHIVE_SUBTITLES - len(records)
                records.extend(nested_recs[:remaining])
            except Exception as exc:
                logger.debug("skipping nested archive %s: %s", norm_path, exc)
            finally:
                shutil.rmtree(nested_tmp_dir, ignore_errors=True)

        if created_dir and not records:
            shutil.rmtree(out_dir, ignore_errors=True)
        return records, out_dir
    except Exception:
        if created_dir:
            shutil.rmtree(out_dir, ignore_errors=True)
        raise
