// Browser WASM adapter (Pyodide Web Worker + ffmpeg.wasm, 100% client-side).
// Mirrors the native desktop feature set: stream inspection (ffprobe), embedded
// text subtitle extraction, PGS/VobSub timing extraction, and 48/32/16 KHz Audio VAD.

const SUBTITLE_EXTS = new Set(["srt", "ass", "ssa", "ttml", "vtt", "sub", "npz", "npy"]);

export const REF_ACCEPT =
  "video/*,audio/*,.mkv,.mp4,.avi,.mov,.webm,.m4v,.mka,.mp3,.aac,.flac,.wav,.ogg,.srt,.ass,.ssa,.ttml,.vtt,.sub";
export const SUBTITLE_ACCEPT = ".srt,.ass,.ssa,.ttml,.vtt,.sub";

function isSubtitleFile(filename) {
  if (!filename) return false;
  const dot = filename.lastIndexOf(".");
  if (dot < 0) return false;
  return SUBTITLE_EXTS.has(filename.slice(dot + 1).toLowerCase());
}

function capitalize(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function formatFileLabel(file) {
  if (!file) return "";
  const size = file.size;
  if (!Number.isFinite(size) || size <= 0) return file.name;
  if (size >= 1024 * 1024 * 1024) {
    return `${file.name} (${(size / (1024 * 1024 * 1024)).toFixed(2)} GB)`;
  }
  if (size >= 1024 * 1024) {
    return `${file.name} (${(size / (1024 * 1024)).toFixed(1)} MB)`;
  }
  if (size >= 1024) {
    return `${file.name} (${(size / 1024).toFixed(1)} KB)`;
  }
  return `${file.name} (${size} B)`;
}

function activeSecondsForTimings(timings) {
  let total = 0;
  for (const [s, e] of timings) {
    if (e > s) total += e - s;
  }
  return total;
}

function sortByPriority(streams) {
  const preferred = streams.filter((s) => !s.deprioritized);
  const deprioritized = streams.filter((s) => s.deprioritized);
  return [...preferred, ...deprioritized];
}

export function createAdapter({ withVersion, callbacks }) {
  let worker = null;
  let engineReady = false;
  let workerCaps = { webrtcvad: false };
  let ffmpegConfig = null;
  let configPromise = null;
  let ffmpegModPromise = null;
  let pendingSync = null;
  let lastBlobUrl = null;
  const probeCache = new WeakMap();

  function finishPending(result) {
    if (!pendingSync) return;
    const current = pendingSync;
    pendingSync = null;
    current.resolve(result);
  }

  function getFfmpegModule() {
    if (!ffmpegModPromise) {
      const decodeUrl = withVersion(new URL("./src/ffmpeg_decode.mjs", document.baseURI).href);
      ffmpegModPromise = import(decodeUrl);
    }
    return ffmpegModPromise;
  }

  async function resolveFfmpegConfig(frameRateOverride) {
    let base = ffmpegConfig;
    if (!base) {
      if (!configPromise) {
        const configUrl = withVersion(new URL("./build.config.json", document.baseURI).href);
        configPromise = fetch(configUrl)
          .then((r) => r.json())
          .then((cfg) => (cfg && cfg.ffmpeg) || null);
      }
      base = await configPromise;
      if (base && !ffmpegConfig) ffmpegConfig = base;
    }
    if (!base) throw new Error("Missing ffmpeg configuration");
    return {
      ...base,
      frameRate: Number(frameRateOverride) || base.frameRate || 16000,
      module: withVersion(new URL(base.module, document.baseURI).href),
      util: withVersion(new URL(base.util, document.baseURI).href),
    };
  }

  function postWorkerSync(message, transferables, onProgress, t0, range = [0, 1]) {
    const [baseFrac, endFrac] = range;
    return new Promise((resolve) => {
      pendingSync = { resolve, onProgress, t0, baseFrac, endFrac };
      worker.postMessage(message, transferables);
    });
  }

  async function probeStreamsCached(file, onStatus) {
    const empty = { streams: [], has_text_subs: false, has_pgs_subs: false };
    if (!(file instanceof File) || isSubtitleFile(file.name)) {
      return empty;
    }
    if (probeCache.has(file)) {
      return probeCache.get(file);
    }
    const p = (async () => {
      try {
        const [mod, ff] = await Promise.all([getFfmpegModule(), resolveFfmpegConfig()]);
        return await mod.probeStreams(ff, file, { status: onStatus });
      } catch (err) {
        console.warn("probeStreams failed:", err);
        probeCache.delete(file);
        return empty;
      }
    })();
    probeCache.set(file, p);
    return p;
  }

  return {
    meta: {
      pageTitle: "ffblindsubsync — In the Browser",
      subtitle: "In the Browser",
      badgeText: "100% Local · WASM",
      badgeTitle: "Runs entirely on your device via Pyodide & ffmpeg.wasm. Nothing is uploaded.",
      refPlaceholder: "Drop or browse movie.mkv, audio, or reference.srt",
      inPlaceholder: "Drop or browse subtitles.srt",
      footerText:
        "Runs locally in your browser via WebAssembly (Pyodide + ffmpeg.wasm). Nothing is uploaded.",
    },
    capabilities: {
      editablePaths: false,
      outputPath: false,
      streamProbe: true,
      runtimeAction: false,
      refAccept: REF_ACCEPT,
      subtitleAccept: SUBTITLE_ACCEPT,
    },

    formatFileDisplay(file) {
      return formatFileLabel(file);
    },

    async probeStreams(file) {
      return probeStreamsCached(file);
    },

    async init() {
      callbacks.onRuntimeStatus({
        level: "warn",
        text: "Loading Python runtime (first load downloads a few MB)…",
        ready: false,
      });
      callbacks.onRuntimeProgress({ active: true, fraction: null });

      const configUrl = withVersion(new URL("./build.config.json", document.baseURI).href);
      const workerUrl = withVersion(new URL("./src/worker.js", document.baseURI).href);
      const engineUrl = withVersion(new URL("./src/ffsubsync_engine.mjs", document.baseURI).href);

      worker = new Worker(workerUrl, { type: "module" });

      worker.onmessage = (event) => {
        const msg = event.data || {};
        if (msg.type === "progress") {
          if (pendingSync && pendingSync.onProgress) {
            const raw = Number(msg.fraction);
            const scaled = Number.isFinite(raw)
              ? Math.max(
                  0,
                  Math.min(
                    1,
                    pendingSync.baseFrac +
                      (pendingSync.endFrac - pendingSync.baseFrac) * Math.max(0, Math.min(1, raw)),
                  ),
                )
              : null;
            pendingSync.onProgress({
              message: msg.message ? capitalize(msg.message) : undefined,
              fraction: scaled,
            });
          }
        } else if (msg.type === "status") {
          if (pendingSync && pendingSync.onProgress) {
            pendingSync.onProgress({ message: capitalize(msg.status), fraction: null });
          } else if (!engineReady) {
            callbacks.onRuntimeStatus({
              level: "warn",
              text: capitalize(msg.status),
              ready: false,
            });
          }
        } else if (msg.type === "ready") {
          engineReady = true;
          workerCaps = msg.capabilities || workerCaps;
          ffmpegConfig = msg.ffmpeg || ffmpegConfig;
          callbacks.onRuntimeProgress({ active: false, fraction: null });
          callbacks.onRuntimeStatus({
            level: "ok",
            text: "Ready. Pick a reference and a subtitle file to sync.",
            ready: true,
          });
        } else if (msg.type === "result") {
          const raw = msg.result || {};
          const elapsed = pendingSync ? (performance.now() - pendingSync.t0) / 1000 : 0;
          if (!raw.ok) {
            finishPending({
              ok: false,
              elapsed_seconds: elapsed,
              error: raw.error || "Sync failed.",
            });
            return;
          }
          if (lastBlobUrl) {
            URL.revokeObjectURL(lastBlobUrl);
            lastBlobUrl = null;
          }
          const blob = new Blob([raw.output_text || ""], {
            type: "application/x-subrip;charset=utf-8",
          });
          lastBlobUrl = URL.createObjectURL(blob);
          const outName = raw.output_name || "synced.srt";
          finishPending({
            ok: true,
            offset_seconds: raw.offset_seconds,
            framerate_scale_factor: raw.framerate_scale_factor,
            split_segments: raw.split_segments,
            dropped_cues: raw.dropped_cues || 0,
            elapsed_seconds: elapsed,
            output_label: outName,
            download_url: lastBlobUrl,
            download_name: outName,
            error: null,
          });
        } else if (msg.type === "error") {
          if (pendingSync) {
            const elapsed = (performance.now() - pendingSync.t0) / 1000;
            finishPending({
              ok: false,
              elapsed_seconds: elapsed,
              error: "Error: " + msg.error,
            });
          } else {
            callbacks.onRuntimeProgress({ active: false, fraction: null });
            callbacks.onRuntimeStatus({
              level: "err",
              text: "Error: " + msg.error,
              ready: false,
            });
          }
        }
      };

      worker.onerror = (e) => {
        const message = "Worker failed to load: " + (e.message || e.filename || e);
        if (pendingSync) {
          finishPending({ ok: false, error: message });
        } else {
          callbacks.onRuntimeProgress({ active: false, fraction: null });
          callbacks.onRuntimeStatus({ level: "err", text: "Error: " + message, ready: false });
        }
      };

      worker.onmessageerror = () => {
        const message = "Worker message error";
        if (pendingSync) {
          finishPending({ ok: false, error: message });
        } else {
          callbacks.onRuntimeProgress({ active: false, fraction: null });
          callbacks.onRuntimeStatus({ level: "err", text: "Error: " + message, ready: false });
        }
      };

      worker.postMessage({ type: "init", configUrl, engineUrl });
    },

    async sync(request, { onProgress }) {
      if (!worker || !engineReady) {
        return { ok: false, error: "Runtime is not ready yet." };
      }

      const refFile = request.ref;
      const inFile = request.input;
      if (!(refFile instanceof File) || !(inFile instanceof File)) {
        return { ok: false, error: "Please select both a reference file and a subtitle file." };
      }

      const t0 = performance.now();
      const options = {
        output_encoding: "utf-8",
        no_fix_framerate: Boolean(request.noFixFramerate),
        gss: Boolean(request.gss),
        split_sync: request.splitMode === "force",
        auto_split: request.splitMode === "auto",
      };

      onProgress({ message: "Reading input subtitles…", fraction: 0.02 });
      const inBytesRaw = new Uint8Array(await inFile.arrayBuffer());

      // 1. Direct subtitle reference file (.srt, .ass, .vtt, etc.)
      if (isSubtitleFile(refFile.name)) {
        onProgress({ message: "Reading reference…", fraction: 0.05 });
        const refBytes = new Uint8Array(await refFile.arrayBuffer());
        return postWorkerSync(
          {
            type: "sync",
            payload: {
              refName: refFile.name,
              inName: inFile.name,
              refBytes,
              inBytes: inBytesRaw,
              options,
            },
          },
          [refBytes.buffer, inBytesRaw.buffer],
          onProgress,
          t0,
          [0.05, 1.0],
        );
      }

      // 2. Video / audio reference file via ffmpeg.wasm
      const [mod, ff] = await Promise.all([
        getFfmpegModule(),
        resolveFfmpegConfig(request.frameRate),
      ]);
      const reportStatus = (s) => onProgress({ message: capitalize(s), fraction: null });

      const refStream = request.refStream || null;
      const refStreamKind = request.refStreamKind || null;
      const isExplicitSub =
        Boolean(refStream) &&
        refStreamKind !== "pgs" &&
        refStreamKind !== "audio" &&
        (String(refStream).startsWith("s:") || String(refStream).startsWith("0:s:"));
      const isExplicitPgs = Boolean(refStream) && refStreamKind === "pgs";
      const isExplicitAudio =
        Boolean(refStream) &&
        (refStreamKind === "audio" ||
          String(refStream).startsWith("a:") ||
          String(refStream).startsWith("0:a:"));

      // 2a. Explicit PGS / VobSub timing stream selected
      if (isExplicitPgs) {
        onProgress({ message: `Reading PGS/VobSub timings (${refStream})…`, fraction: 0.05 });
        const timings = await mod.probePgsTimings(ff, refFile, refStream, { status: reportStatus });
        if (!timings || !timings.length) {
          return {
            ok: false,
            elapsed_seconds: (performance.now() - t0) / 1000,
            error: `Failed to read PGS/VobSub timings from stream ${refStream}.`,
          };
        }
        onProgress({ message: "Aligning PGS/VobSub timings…", fraction: 0.35 });
        return postWorkerSync(
          {
            type: "syncTimings",
            payload: { timings, inName: inFile.name, inBytes: inBytesRaw, options },
          },
          [inBytesRaw.buffer],
          onProgress,
          t0,
          [0.35, 1.0],
        );
      }

      // 2b. Explicit embedded text subtitle stream selected
      if (isExplicitSub) {
        onProgress({ message: `Extracting embedded subtitle (${refStream})…`, fraction: 0.05 });
        const subBytes = await mod.extractSubtitleStream(ff, refFile, refStream, {
          status: reportStatus,
          onProgress: (p) => {
            if (Number.isFinite(p) && p > 0 && p <= 1) {
              const pct = Math.round(p * 100);
              onProgress({
                message: `Extracting embedded subtitle (${pct}%)…`,
                fraction: 0.05 + p * 0.30,
              });
            }
          },
        });
        if (!subBytes || !subBytes.byteLength) {
          return {
            ok: false,
            elapsed_seconds: (performance.now() - t0) / 1000,
            error: `Failed to extract embedded subtitle stream ${refStream}.`,
          };
        }
        onProgress({ message: "Aligning against embedded subtitles…", fraction: 0.35 });
        return postWorkerSync(
          {
            type: "sync",
            payload: {
              refName: "embedded.srt",
              inName: inFile.name,
              refBytes: subBytes,
              inBytes: inBytesRaw,
              options,
            },
          },
          [subBytes.buffer, inBytesRaw.buffer],
          onProgress,
          t0,
          [0.35, 1.0],
        );
      }

      // 2c. Auto embedded subtitles -> PGS/VobSub timings fallback (subs_then_webrtc)
      const useEmbeddedFirst = !isExplicitAudio && String(request.vad || "").includes("subs");
      if (useEmbeddedFirst) {
        onProgress({ message: "Checking video for subtitles…", fraction: 0.04 });
        const probed = await probeStreamsCached(refFile, reportStatus);
        const streams = (probed && probed.streams) || [];

        // Try embedded text subtitle tracks (full dialogue first, forced/commentary last).
        const textTracks = sortByPriority(
          streams.filter((s) => s.kind === "subtitle" && !s.is_bitmap),
        );
        for (const track of textTracks) {
          const subBytes = await mod.extractSubtitleStream(ff, refFile, track.value, {
            status: reportStatus,
            onProgress: (p) => {
              if (Number.isFinite(p) && p > 0 && p <= 1) {
                const pct = Math.round(p * 100);
                onProgress({
                  message: `Extracting embedded subtitle (${pct}%)…`,
                  fraction: 0.05 + p * 0.30,
                });
              }
            },
          });
          if (subBytes && subBytes.byteLength > 0) {
            onProgress({ message: "Aligning against embedded subtitles…", fraction: 0.35 });
            const inCopy = inBytesRaw.slice();
            const res = await postWorkerSync(
              {
                type: "sync",
                payload: {
                  refName: "embedded.srt",
                  inName: inFile.name,
                  refBytes: subBytes,
                  inBytes: inCopy,
                  options,
                },
              },
              [subBytes.buffer, inCopy.buffer],
              onProgress,
              t0,
              [0.35, 1.0],
            );
            if (res && res.ok) return res;
          }
        }

        // Fall back to PGS / VobSub packet timings (up to 3 tracks, matching CLI/desktop).
        const pgsTracks = sortByPriority(streams.filter((s) => s.kind === "pgs")).slice(0, 3);
        for (const track of pgsTracks) {
          onProgress({ message: `Reading PGS/VobSub timings (${track.value})…`, fraction: 0.08 });
          const timings = await mod.probePgsTimings(ff, refFile, track.value, {
            status: reportStatus,
          });
          if (timings && timings.length >= 10 && activeSecondsForTimings(timings) >= 5.0) {
            const kindLabel =
              track.codec === "hdmv_pgs_subtitle" ? "PGS" : "VobSub";
            onProgress({
              message: `Using ${kindLabel} subtitle stream ${track.value}…`,
              fraction: 0.35,
            });
            const inCopy = inBytesRaw.slice();
            const res = await postWorkerSync(
              {
                type: "syncTimings",
                payload: { timings, inName: inFile.name, inBytes: inCopy, options },
              },
              [inCopy.buffer],
              onProgress,
              t0,
              [0.35, 1.0],
            );
            if (res && res.ok) return res;
          }
        }
      }

      // 2d. Audio VAD path (explicit audio track or fallback from embedded subs)
      if (!workerCaps.webrtcvad) {
        return {
          ok: false,
          elapsed_seconds: (performance.now() - t0) / 1000,
          error:
            "Video/audio sync needs the WebRTC VAD component, which isn't available " +
            "in this build. Use a subtitle reference, or build the webrtcvad wheel.",
        };
      }

      let pcm;
      try {
        onProgress({ message: "Loading ffmpeg.wasm…", fraction: 0.05 });
        pcm = await mod.decodeAudioToPcm(ff, refFile, {
          stream: isExplicitAudio ? refStream : undefined,
          status: reportStatus,
          onProgress: (p) => {
            if (Number.isFinite(p) && p > 0 && p <= 1) {
              const pct = Math.round(p * 100);
              onProgress({
                message: `Decoding audio (${pct}%)…`,
                fraction: 0.05 + p * 0.45,
              });
            }
          },
        });
      } catch (e) {
        console.error(e);
        return {
          ok: false,
          elapsed_seconds: (performance.now() - t0) / 1000,
          error: "Audio decode failed: " + ((e && e.message) || e),
        };
      }

      onProgress({ message: "Detecting speech + aligning…", fraction: 0.50 });
      return postWorkerSync(
        {
          type: "syncAudioPcm",
          payload: {
            pcm,
            frameRate: ff.frameRate,
            inName: inFile.name,
            inBytes: inBytesRaw,
            vad: "webrtc",
            options,
          },
        },
        [pcm.buffer, inBytesRaw.buffer],
        onProgress,
        t0,
        [0.50, 1.0],
      );
    },
  };
}
