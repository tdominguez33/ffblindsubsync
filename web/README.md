# ffblindsubsync in the browser (WebAssembly / Pyodide)

A backendless static site that runs [ffblindsubsync](https://github.com/tdominguez33/ffblindsubsync)
entirely client-side via [Pyodide](https://pyodide.org). Subtitle files never leave
the user's machine.

## 👉 Try it: https://tdominguez33.github.io/ffblindsubsync

No install, no upload. Pick a correctly-synced reference subtitle and an out-of-sync
subtitle, click **Sync subtitles**, and download the corrected file. First load pulls
the WebAssembly runtime (a few MB) from a CDN and caches it. To run it locally instead,
see [Build & run](#build--run) below.

**Status.**
- **Phase 1 — subtitle-vs-subtitle:** stable. Pure numpy + subtitle parsing (no ffmpeg,
  no VAD).
- **Phase 2 — video/audio reference:** implemented. Audio is decoded in-browser by
  `ffmpeg.wasm` (large files mounted lazily via WORKERFS, never loaded whole into
  memory), then WebRTC VAD builds the reference speech signal. The VAD ships as a
  cross-compiled `webrtcvad` wasm wheel (built in CI); when it is unavailable, video/
  audio references are disabled and the UI says so (subtitle references still work).
  ffsubsync's other VAD, `auditok`, is **not** bundled: it has no PyPI wheel and is
  GPLv3, which would relicense this MIT site. The Python pipeline is covered by native
  tests; the ffmpeg.wasm decode is exercised in a real browser via CI.

## How it works

- The **local** `ffsubsync/` package (this checkout, not the PyPI wheel) is bundled by
  `scripts/vendor.mjs` into `vendor/py_sources.json` and written into the Pyodide
  filesystem at runtime — so the site matches this repo's ffsubsync exactly.
- `src/ffmpeg_decode.mjs` decodes a video/audio reference to mono PCM with ffmpeg.wasm;
  `src/ffsubsync_bridge.py:sync_with_audio` runs the VAD over that PCM and rejoins the
  standard alignment path.
- `src/ffsubsync_bridge.py` is a thin, runtime-agnostic entry point that drives the real
  `ffsubsync.run(...)` the same way the CLI does, over Pyodide's in-memory filesystem.
- `src/ffsubsync_engine.mjs` boots Pyodide (numpy via `loadPackage`; `pysubs2` / `srt` /
  `ffmpeg-python` / `auditok` via `micropip`), loads the bundle, and calls the bridge.
- `src/worker.js` runs all of that in a Web Worker; the UI is shared with the desktop
  app in `../ui/` (`index.html`, `styles.css`, `app.js`) wired through `../ui/adapters/web.js`.

Version pins (Pyodide, pip packages) live in one place: `build.config.json`.

## Build & run

Requires **Node.js** and **Python 3.8+** (commands work on Windows PowerShell, macOS, and Linux from the repository root):

```sh
# 1. (First time on a fresh clone) Download the pure-Python wheels into web/vendor/wheels
python -m pip download --only-binary=:all: --implementation py --abi none --platform any pysubs2 ffmpeg-python -d web/vendor/wheels

# 2. Bundle local ffsubsync + bridge + ffmpeg.wasm glue -> web/vendor/
node web/scripts/vendor.mjs

# 3. Assemble the static site -> web/dist/site/
node web/scripts/build_site.mjs

# 4. Serve locally at http://localhost:8000
python -m http.server 8000 --directory web/dist/site
```

> **Tip:** If you only edited UI assets in `ui/` (HTML, CSS, JS, or locales) and didn't change Python code, you only need step 3 (`node web/scripts/build_site.mjs`).  
> On Windows with the project virtualenv, replace `python` with `.venv\Scripts\python.exe`. On Linux/macOS, `make site` and `make serve` inside `web/` remain available as shorthand.

Open http://localhost:8000, pick a reference video or subtitle and an out-of-sync subtitle, and download the synced result. First load pulls the Pyodide runtime (a few MB) from the CDN; after that it is cached.

## Test

Run from the repository root (with the project dependencies installed, e.g. `pip install -e ".[dev]"`):

```sh
# Native bridge test: sub-vs-sub, auto-split, PGS timings, and alignment profiles
python web/tests/native_bridge_test.py

# Native audio path test: PCM -> real WebRTC VAD -> align
python web/tests/native_audio_test.py

# UI i18n locale parity and matching tests
node web/tests/i18n_test.mjs

# Full headless Chromium Pyodide test (requires `npm install` in web/ + network)
npm --prefix web run test:browser
```

CI (`.github/workflows/build-site.yml`) runs the native + browser tests, cross-compiles
the webrtcvad wasm wheel, builds the bundle, and deploys `dist/site` to GitHub Pages. The
ffmpeg core is kept single-threaded so Pages needs no COOP/COEP headers.

## The wasm wheels (webrtcvad + cchardet)

`make wheels` cross-compiles ffsubsync's C-extension deps to Pyodide `wasm32` wheels (see
`scripts/build_wheels.sh`) so the browser uses the same native libraries as the CLI:

- **webrtcvad** — from [py-webrtcvad](https://github.com/wiseman/py-webrtcvad); the same
  default VAD as the CLI. When absent, video/audio references are disabled (subtitle
  references still work).
- **cchardet** — from [faust-cchardet](https://github.com/faust-streaming/cChardet) (a
  Cython wrapper around the uchardet C++ library, kept in a git submodule). Gives exact
  CLI-parity encoding detection: `subtitle_parser` prefers `cchardet` over
  `charset-normalizer`. When absent, `charset-normalizer` (a Pyodide core package) handles
  detection in-browser, so this wheel is best-effort — a build failure never blocks deploy.

Building needs network + a POSIX toolchain (emsdk is installed on demand) and a Python 3.13
host, so it normally runs in CI; the built `.whl`s land in `vendor/wheels/` and are served
with the site. `make webrtcvad-wheel` / `make cchardet-wheel` build just one. Version pins
(Pyodide, py-webrtcvad, faust-cchardet, ffmpeg.wasm) all live in `build.config.json`.
