# FFblindsubsync

[![License: MIT](https://img.shields.io/badge/License-MIT-maroon.svg)](LICENSE)

A hardened fork of [ffsubsync](https://github.com/smacke/ffsubsync) focused on
reliable, one-shot **blind subtitle synchronization** -- featuring a portable
Windows desktop app, automatic PGS/VobSub timing extraction, and safer
mid-file break alignment.

> Forked from [smacke/ffsubsync](https://github.com/smacke/ffsubsync) by
> Stephen Macke.

## What's New in This Fork

- **Portable Windows Desktop App (`FFblindsubsync.exe`)**:
  Native desktop WebView UI (`gui/desktop_app.py` + shared `ui/`) with
  drag-and-drop file inputs, automatic track/stream inspection (embedded text
  subtitles, PGS/VobSub timing streams, and audio tracks), built-in portable
  `ffmpeg`/`ffprobe` downloader and updater (with `%LOCALAPPDATA%\ffblindsubsync`
  fallback when the app folder is read-only), `48 KHz` Audio VAD by default
  (`32 KHz` / `16 KHz` optional), and split/dropped-cue reporting.
- **Automatic PGS & VobSub Timing Fallback**:
  When no usable embedded text subtitle track is available, `subs_then_webrtc`
  automatically inspects packet timings from `hdmv_pgs_subtitle` (PGS) and
  `dvd_subtitle` (VobSub) streams via `ffprobe` before falling back to Audio
  VAD. `forced`, `commentary`, `signs`, and `songs` tracks are automatically
  deprioritized in favor of full dialogue tracks.
- **Hardened `--auto-split` Alignment**:
  - Unconditional **Do No Harm** gate so a split solution never beats a better
    global alignment.
  - Adaptive SNR thresholds so sparse or noisy Audio VAD signals do not shatter
    into spurious splits.
  - Negative-overlap / deleted-scene handling (drops orphan cues belonging to
    cut scenes and clamps/trims boundary overlaps).
  - Short boundary-split rescue (cold opens / post-credits), automatic
    framerate-ratio re-check, and an expanded `300s` alignment window when
    `--auto-split` is enabled.
- **Broader Format Support**:
  Added WebVTT (`.vtt`) extension support, case-insensitive extension handling
  (`.SRT`, `.VTT`, etc.), and automatic `.sub` output conversion to
  `.synced.srt` when writing text subtitles.

## Windows Desktop App

The bundled app (`gui/dist/FFblindsubsync/FFblindsubsync.exe`) runs directly
without installing Python. By default, configuration (`config.json`, including
saved window size/position and custom FFmpeg path) and downloaded `ffmpeg` /
`ffprobe` binaries are stored in `%LOCALAPPDATA%\ffblindsubsync`. To run in
**portable mode** (storing everything next to `FFblindsubsync.exe`), place an
empty `portable.txt` file (case-insensitive) next to the executable.

To build the portable executable from source (Windows PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\gui\build-windows-native.ps1
```

This outputs the single-file executable to `gui/dist/FFblindsubsync.exe`.

## Web Version (Browser / WebAssembly)

This fork preserves the client-side browser application in `web/` (powered by
Pyodide and `ffmpeg.wasm`, so media and subtitle files never leave the user's
machine) and shares the same unified UI (`ui/`) as the native desktop app:

- The shared UI (`ui/index.html`, `ui/styles.css`, `ui/app.js`) renders the
  same layout, drag-and-drop workflow, options, progress bars, and result
  reporting on both desktop (`ui/adapters/desktop.js`) and web
  (`ui/adapters/web.js`).
- The Python web bridge (`web/src/ffsubsync_bridge.py`) runs against this
  fork's updated alignment core, including hardened `--auto-split` and
  `dropped_cues` reporting.
- GitHub Pages deployment (`.github/workflows/build-site.yml`) assembles the
  static site from `ui/` and `web/` via `make -C web site`.

For local web build and test instructions, see [web/README.md](web/README.md)
(`make -C web site`, `make -C web serve`).

## CLI Installation & Quick Usage

### Prerequisites

- Python >= 3.8
- `ffmpeg` and `ffprobe` available on `PATH` (or passed via `--ffmpeg-path`)

### Install

```bash
pip install git+https://github.com/tdominguez33/ffblindsubsync
```

### Quick Examples

All `ffs`, `ffsubsync`, and `subsync` CLI entrypoints work:

```bash
# Auto sync (embedded text -> PGS/VobSub timings -> 48 KHz audio VAD)
ffs movie.mkv -i unsynced.srt -o synced.srt

# Enable safe mid-file break / commercial-cut detection
ffs movie.mkv -i unsynced.srt -o synced.srt --auto-split

# Explicitly target an embedded PGS or VobSub stream as timing reference
ffs movie.mkv -i unsynced.srt -o synced.srt --pgs-ref-stream s:0

# Sync against an already-synced reference subtitle file
ffs reference.srt -i unsynced.srt -o synced.srt
```

All upstream options (`--vad`, `--gss`, `--max-offset-seconds`,
`--start-seconds`, `--max-duration-seconds`, remote URLs, multi-segment sync,
etc.) remain supported. See [smacke/ffsubsync](https://github.com/smacke/ffsubsync)
for the complete upstream CLI reference.

## Development & Testing

```bash
# Run Python test suite
pytest -q

# Run web bridge parity tests
python web/tests/native_bridge_test.py
```

```powershell
# Build Windows portable desktop app
powershell -ExecutionPolicy Bypass -File .\gui\build-windows-native.ps1
```

## Credits & License

Built on [ffsubsync](https://github.com/smacke/ffsubsync) by Stephen Macke.
Distributed under the [MIT License](LICENSE).
