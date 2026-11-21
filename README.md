# FFblindsubsync

[![License: MIT](https://img.shields.io/badge/License-MIT-maroon.svg)](LICENSE)

Automatically synchronize out-of-sync subtitles to any video, audio track, or
reference subtitle — available as a **standalone Windows app**, a **browser
app** (100% local, nothing uploaded), and a **command-line tool**.

**Why "blind"?** When you download subtitles for a movie or episode you haven't
watched yet, you don't want to scrub through scenes, tweak flags, or risk
spoilers just to get the timing right. Unlike the original `ffsubsync`,
**FFblindsubsync** is built around doing the best possible **one-shot,
hands-off sync** out of the box — automatically choosing the cleanest reference
in your video, safely handling mid-file cuts or release differences, and giving
you a ready-to-use desktop and browser interface with zero guesswork.

> Based on [smacke/ffsubsync](https://github.com/smacke/ffsubsync) **v0.5.1**
> by Stephen Macke.

## What to Expect

- **Standalone Windows Desktop App**: A single `FFblindsubsync.exe` file with no
  Python installation required. Drag and drop your files, switch between light
  and dark themes, pick specific tracks from a video, and download or update
  FFmpeg right from Settings.
- **Browser Version (100% Private)**: Run it directly in your browser with zero
  installation. All processing happens locally on your machine — your videos and
  subtitles are never uploaded anywhere.
- **Smarter Automatic Sync**: Automatically uses the best reference available in
  your video — full embedded text subtitles first, then Blu-ray (PGS) and DVD
  (VobSub) subtitle timings (ignoring forced, signs, or commentary tracks when
  possible), and finally audio speech detection.
- **Mid-File Cut & Break Recovery**: Automatically detects and fixes sudden
  timing jumps caused by commercial breaks, different intros, or cut/extended
  scenes, while safely avoiding unnecessary splits on unbroken files.
- **Broad Format Support**: Works with `.srt`, `.ass`, `.ssa`, `.vtt`, and
  `.sub` subtitles alongside common video and audio formats (`.mkv`, `.mp4`,
  `.avi`, `.mov`, etc.).

## Ways to Use It

### 1. In the Browser

Open **[tdominguez33.github.io/ffblindsubsync](https://tdominguez33.github.io/ffblindsubsync)**,
select your reference video or subtitle and your unsynced subtitle, and click
**Sync subtitles**.

### 2. Windows Desktop App

Download `FFblindsubsync.exe` from the
**[Releases](https://github.com/tdominguez33/ffblindsubsync/releases)** page
and run it directly.

- On first launch, you can download FFmpeg automatically from the app (or point
  Settings to an existing FFmpeg folder).
- By default, your settings (including window size and position) and downloaded
  FFmpeg binaries are stored in `%LOCALAPPDATA%\ffblindsubsync`.
- **Portable mode**: Place an empty `portable.txt` file next to
  `FFblindsubsync.exe` to keep configuration and FFmpeg in the same folder as
  the executable.

### 3. Command Line (CLI)

Requires Python 3.8+ and `ffmpeg` / `ffprobe` on `PATH`:

```bash
pip install git+https://github.com/tdominguez33/ffblindsubsync
```

Examples (`ffblindsubsync`, `ffs`, `ffsubsync`, and `subsync` all work):

```bash
# Automatic sync against a video
ffblindsubsync movie.mkv -i unsynced.srt -o synced.srt

# Enable automatic mid-file break / commercial-cut detection
ffblindsubsync movie.mkv -i unsynced.srt -o synced.srt --auto-split

# Sync against an already-synced subtitle file
ffblindsubsync reference.srt -i unsynced.srt -o synced.srt
```

All upstream CLI options remain supported.

## Building from Source

```bash
# Install with desktop and test dependencies
pip install -e ".[desktop,dev]"

# Run unit tests
pytest -q
```

Build the single-file Windows app (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\gui\build-windows-native.ps1
```

For local web builds, see [web/README.md](web/README.md).

## Credits & License

Based on [ffsubsync](https://github.com/smacke/ffsubsync) (v0.5.1) by Stephen
Macke. Distributed under the [MIT License](LICENSE).
