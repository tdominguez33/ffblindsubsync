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

function subtitleMimeType(filename) {
  const low = String(filename || "").toLowerCase();
  if (low.endsWith(".vtt")) return "text/vtt;charset=utf-8";
  if (low.endsWith(".ass") || low.endsWith(".ssa")) return "text/plain;charset=utf-8";
  return "application/x-subrip;charset=utf-8";
}

function filterAndRankAutoTracks(streams) {
  const usable = streams.filter((s) => !s.deprioritized);
  if (usable.length <= 1) return usable;
  const maxFrames = usable.reduce((m, s) => Math.max(m, Number(s.num_frames) || 0), 0);
  const fullFloor = maxFrames > 0 ? Math.max(200, Math.floor(0.4 * maxFrames)) : 0;

  const isFull = (s) => {
    const n = Number(s.num_frames) || 0;
    return maxFrames === 0 || n === 0 || n >= fullFloor;
  };

  return usable
    .map((s, idx) => ({ s, idx }))
    .sort((a, b) => {
      const af = isFull(a.s) ? 1 : 0;
      const bf = isFull(b.s) ? 1 : 0;
      if (bf !== af) return bf - af;
      const aClean = a.s.is_sdh ? 0 : 1;
      const bClean = b.s.is_sdh ? 0 : 1;
      if (bClean !== aClean) return bClean - aClean;
      const aLang = a.s.matches_audio_lang ? 1 : 0;
      const bLang = b.s.matches_audio_lang ? 1 : 0;
      if (bLang !== aLang) return bLang - aLang;
      const aDef = a.s.is_default ? 1 : 0;
      const bDef = b.s.is_default ? 1 : 0;
      if (bDef !== aDef) return bDef - aDef;
      if (!af && !bf) {
        const diff = (Number(b.s.num_frames) || 0) - (Number(a.s.num_frames) || 0);
        if (diff !== 0) return diff;
      }
      return a.idx - b.idx;
    })
    .map((x) => x.s);
}

const WORKER_STATUS_KEY_MAP = new Map([
  ["fetching config…", "status.fetchingConfig"],
  ["loading python runtime…", "status.loadingPython"],
  ["loading runtime packages…", "status.loadingPackages"],
  ["installing libraries…", "status.installingLibs"],
  ["loading ffblindsubsync…", "status.loadingApp"],
  ["ready", "status.ready"],
  ["syncing…", "progress.syncing"],
  ["detecting speech + aligning…", "progress.detectingSpeechAligning"],
  ["aligning subtitle timings…", "progress.aligningSubTimings"],
  ["loading ffmpeg.wasm…", "progress.loadingFfmpegWasm"],
  ["decoding audio (this can take a bit on long videos)…", "progress.decodingAudioLong"],
]);

function mapWorkerMessage(raw) {
  if (!raw) return { messageKey: null, messageParams: null, message: undefined };
  const s = String(raw).trim();
  const low = s.toLowerCase();
  if (WORKER_STATUS_KEY_MAP.has(low)) {
    return { messageKey: WORKER_STATUS_KEY_MAP.get(low), messageParams: null, message: capitalize(s) };
  }
  const mAlign = /^computing alignment \((\d+)\/(\d+)\)/i.exec(s);
  if (mAlign) {
    return {
      messageKey: "progress.computingAlignmentStep",
      messageParams: { cur: Number(mAlign[1]), tot: Number(mAlign[2]) },
      message: capitalize(s),
    };
  }
  const mExtract = /^extracting embedded subtitle \(([^)]+)\)/i.exec(s);
  if (mExtract) {
    return {
      messageKey: "progress.extractingEmbeddedStream",
      messageParams: { stream: mExtract[1] },
      message: capitalize(s),
    };
  }
  const mPgs = /^reading pgs\/vobsub(?:\/dvb)? timings \(([^)]+)\)/i.exec(s);
  if (mPgs) {
    return {
      messageKey: "progress.readingPgsStream",
      messageParams: { stream: mPgs[1] },
      message: capitalize(s),
    };
  }
  return { messageKey: null, messageParams: null, message: capitalize(s) };
}

export function createAdapter({ version, withVersion, i18n, callbacks }) {
  const t = (key, params) => (i18n ? i18n.t(key, params) : key);
  const metaVersionEl = document.querySelector('meta[name="app-version"]');
  const metaVersion = (metaVersionEl && metaVersionEl.getAttribute("content") || "").trim();
  const appVersion = metaVersion || String(version || "").replace(/-[0-9a-f]{8}$/, "").trim();
  let worker = null;
  let engineReady = false;
  let workerCaps = { webrtcvad: false };
  let ffmpegConfig = null;
  let configPromise = null;
  let ffmpegModPromise = null;
  let pendingSync = null;
  let lastBlobUrl = null;
  let lastRuntimeState = null;
  const probeCache = new WeakMap();

  function emitRuntimeState(stateObj) {
    lastRuntimeState = stateObj;
    if (!stateObj) return;
    const text = stateObj.textKey ? t(stateObj.textKey, stateObj.textParams) : stateObj.text;
    callbacks.onRuntimeStatus({
      level: stateObj.level,
      text,
      visible: stateObj.visible,
      ready: stateObj.ready,
    });
  }

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
    const empty = { streams: [], has_text_subs: false, has_pgs_subs: false, video_fps: null };
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
      get pageTitle() {
        return t("app.titleWeb");
      },
      get subtitle() {
        return t("app.subtitleWeb");
      },
      badgeText: "",
      badgeVariant: "web",
      badgeTitle: "",
      get footerText() {
        return t("app.footerWeb");
      },
      appVersion,
    },
    capabilities: {
      editablePaths: false,
      outputPath: false,
      streamProbe: true,
      runtimeAction: false,
      topbarStatus: true,
      settings: true,
      ffmpegSettings: false,
      refAccept: REF_ACCEPT,
      subtitleAccept: SUBTITLE_ACCEPT,
    },

    retranslate() {
      if (lastRuntimeState) {
        emitRuntimeState(lastRuntimeState);
      }
    },

    formatFileDisplay(file) {
      return formatFileLabel(file);
    },

    async probeStreams(file) {
      return probeStreamsCached(file);
    },

    async init() {
      emitRuntimeState({
        level: "warn",
        textKey: "status.loadingPython",
        visible: false,
        ready: false,
      });

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
            const mapped = mapWorkerMessage(msg.message);
            pendingSync.onProgress({
              message: mapped.message,
              messageKey: mapped.messageKey,
              messageParams: mapped.messageParams,
              fraction: scaled,
            });
          }
        } else if (msg.type === "status") {
          const mapped = mapWorkerMessage(msg.status);
          if (pendingSync && pendingSync.onProgress) {
            pendingSync.onProgress({
              message: mapped.message,
              messageKey: mapped.messageKey,
              messageParams: mapped.messageParams,
              fraction: null,
            });
          } else if (!engineReady) {
            emitRuntimeState({
              level: "warn",
              textKey: mapped.messageKey,
              textParams: mapped.messageParams,
              text: mapped.message,
              visible: false,
              ready: false,
            });
          }
        } else if (msg.type === "ready") {
          engineReady = true;
          workerCaps = msg.capabilities || workerCaps;
          ffmpegConfig = msg.ffmpeg || ffmpegConfig;
          emitRuntimeState({
            level: "ok",
            textKey: "status.ready",
            visible: false,
            ready: true,
          });
        } else if (msg.type === "result") {
          const raw = msg.result || {};
          const elapsed = pendingSync ? (performance.now() - pendingSync.t0) / 1000 : 0;
          if (!raw.ok) {
            finishPending({
              ok: false,
              elapsed_seconds: elapsed,
              error: raw.error || t("result.syncFailed"),
              error_key: raw.error ? null : "result.syncFailed",
            });
            return;
          }
          if (lastBlobUrl) {
            URL.revokeObjectURL(lastBlobUrl);
            lastBlobUrl = null;
          }
          const outName = raw.output_name || "synced.srt";
          const blob = new Blob([raw.output_text || ""], {
            type: subtitleMimeType(outName),
          });
          lastBlobUrl = URL.createObjectURL(blob);
          finishPending({
            ok: true,
            offset_seconds: raw.offset_seconds,
            framerate_scale_factor: raw.framerate_scale_factor,
            split_segments: raw.split_segments,
            dropped_cues: raw.dropped_cues || 0,
            clipped_cues: raw.clipped_cues || 0,
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
              error: t("status.errorPrefix", { error: msg.error }),
              error_key: "status.errorPrefix",
              error_params: { error: msg.error },
            });
          } else {
            callbacks.onRuntimeProgress({ active: false, fraction: null });
            emitRuntimeState({
              level: "err",
              textKey: "status.errorPrefix",
              textParams: { error: msg.error },
              visible: true,
              ready: false,
            });
          }
        }
      };

      worker.onerror = (e) => {
        const errDetail = e.message || e.filename || e;
        const message = t("status.workerLoadFailed", { error: errDetail });
        if (pendingSync) {
          finishPending({
            ok: false,
            error: message,
            error_key: "status.workerLoadFailed",
            error_params: { error: errDetail },
          });
        } else {
          callbacks.onRuntimeProgress({ active: false, fraction: null });
          emitRuntimeState({
            level: "err",
            textKey: "status.errorPrefix",
            textParams: { error: message },
            visible: true,
            ready: false,
          });
        }
      };

      worker.onmessageerror = () => {
        const message = t("status.workerMessageError");
        if (pendingSync) {
          finishPending({
            ok: false,
            error: message,
            error_key: "status.workerMessageError",
          });
        } else {
          callbacks.onRuntimeProgress({ active: false, fraction: null });
          emitRuntimeState({
            level: "err",
            textKey: "status.errorPrefix",
            textParams: { error: message },
            visible: true,
            ready: false,
          });
        }
      };

      worker.postMessage({ type: "init", configUrl, engineUrl });
    },

    async sync(request, { onProgress }) {
      if (!worker || !engineReady) {
        return {
          ok: false,
          error: t("errors.runtimeNotReady"),
          error_key: "errors.runtimeNotReady",
        };
      }

      const refFile = request.ref;
      const inFile = request.input;
      if (!(refFile instanceof File) || !(inFile instanceof File)) {
        return {
          ok: false,
          error: t("errors.selectBothFiles"),
          error_key: "errors.selectBothFiles",
        };
      }

      const t0 = performance.now();
      const options = {
        output_encoding: "utf-8",
        no_fix_framerate: Boolean(request.noFixFramerate),
        gss: Boolean(request.gss),
        split_sync: request.splitMode === "force",
        auto_split: request.splitMode === "auto",
      };

      onProgress({ messageKey: "progress.readingInputSubs", fraction: 0.02 });
      const inBytesRaw = new Uint8Array(await inFile.arrayBuffer());

      // 1. Direct subtitle reference file (.srt, .ass, .vtt, etc.)
      if (isSubtitleFile(refFile.name)) {
        onProgress({ messageKey: "progress.readingRef", fraction: 0.05 });
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
      const reportStatus = (s) => {
        const mapped = mapWorkerMessage(s);
        onProgress({
          message: mapped.message,
          messageKey: mapped.messageKey,
          messageParams: mapped.messageParams,
          fraction: null,
        });
      };

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

      // 2a. Explicit PGS / VobSub / DVB timing stream selected
      if (isExplicitPgs) {
        onProgress({
          messageKey: "progress.readingPgsStream",
          messageParams: { stream: refStream },
          fraction: 0.05,
        });
        const probed = await probeStreamsCached(refFile, reportStatus);
        const matchedStream = ((probed && probed.streams) || []).find(
          (s) => s.value === refStream || `0:${s.value}` === refStream,
        );
        const timings = await mod.probePgsTimings(ff, refFile, refStream, {
          status: reportStatus,
          codec: matchedStream && matchedStream.codec,
        });
        if (!timings || !timings.length) {
          return {
            ok: false,
            elapsed_seconds: (performance.now() - t0) / 1000,
            error: t("errors.pgsReadFailed", { stream: refStream }),
            error_key: "errors.pgsReadFailed",
            error_params: { stream: refStream },
          };
        }
        onProgress({ messageKey: "progress.aligningPgs", fraction: 0.35 });
        const pgsOptions =
          probed && probed.video_fps
            ? { ...options, inferred_sub_fps: probed.video_fps }
            : options;
        return postWorkerSync(
          {
            type: "syncTimings",
            payload: { timings, inName: inFile.name, inBytes: inBytesRaw, options: pgsOptions },
          },
          [inBytesRaw.buffer],
          onProgress,
          t0,
          [0.35, 1.0],
        );
      }

      // 2b. Explicit embedded text subtitle stream selected
      if (isExplicitSub) {
        onProgress({
          messageKey: "progress.extractingEmbeddedStream",
          messageParams: { stream: refStream },
          fraction: 0.05,
        });
        const subBytes = await mod.extractSubtitleStream(ff, refFile, refStream, {
          status: reportStatus,
          onProgress: (p) => {
            if (Number.isFinite(p) && p > 0 && p <= 1) {
              const pct = Math.round(p * 100);
              onProgress({
                messageKey: "progress.extractingEmbeddedPct",
                messageParams: { pct },
                fraction: 0.05 + p * 0.30,
              });
            }
          },
        });
        if (!subBytes || !subBytes.byteLength) {
          return {
            ok: false,
            elapsed_seconds: (performance.now() - t0) / 1000,
            error: t("errors.embeddedExtractFailed", { stream: refStream }),
            error_key: "errors.embeddedExtractFailed",
            error_params: { stream: refStream },
          };
        }
        onProgress({ messageKey: "progress.aligningEmbedded", fraction: 0.35 });
        let subOptions = options;
        try {
          const probed = await probeStreamsCached(refFile, reportStatus);
          if (probed && probed.video_fps) {
            subOptions = { ...options, inferred_sub_fps: probed.video_fps };
          }
        } catch {
          /* ignore */
        }
        return postWorkerSync(
          {
            type: "sync",
            payload: {
              refName: "embedded.srt",
              inName: inFile.name,
              refBytes: subBytes,
              inBytes: inBytesRaw,
              options: subOptions,
            },
          },
          [subBytes.buffer, inBytesRaw.buffer],
          onProgress,
          t0,
          [0.35, 1.0],
        );
      }

      // 2c. Auto embedded subtitles -> PGS/VobSub/DVB timings fallback (subs_then_webrtc)
      const useEmbeddedFirst = !isExplicitAudio && String(request.vad || "").includes("subs");
      if (useEmbeddedFirst) {
        onProgress({ messageKey: "progress.checkingVideoSubs", fraction: 0.04 });
        const probed = await probeStreamsCached(refFile, reportStatus);
        const streams = (probed && probed.streams) || [];
        const probedOptions =
          probed && probed.video_fps
            ? { ...options, inferred_sub_fps: probed.video_fps }
            : options;

        // Try full embedded text subtitle tracks (skip forced/commentary/signs/songs
        // and sparse stubs with <50 cues so they fall through to bitmap/audio VAD).
        const textTracks = filterAndRankAutoTracks(
          streams.filter((s) => s.kind === "subtitle" && !s.is_bitmap),
        );
        for (const track of textTracks) {
          const subBytes = await mod.extractSubtitleStream(ff, refFile, track.value, {
            status: reportStatus,
            onProgress: (p) => {
              if (Number.isFinite(p) && p > 0 && p <= 1) {
                const pct = Math.round(p * 100);
                onProgress({
                  messageKey: "progress.extractingEmbeddedPct",
                  messageParams: { pct },
                  fraction: 0.05 + p * 0.30,
                });
              }
            },
          });
          if (subBytes && subBytes.byteLength > 0) {
            onProgress({ messageKey: "progress.aligningEmbedded", fraction: 0.35 });
            const inCopy = inBytesRaw.slice();
            const res = await postWorkerSync(
              {
                type: "sync",
                payload: {
                  refName: "embedded.srt",
                  inName: inFile.name,
                  refBytes: subBytes,
                  inBytes: inCopy,
                  options: { ...probedOptions, min_ref_cues: 50 },
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

        // Fall back to PGS / VobSub / DVB timings (non-forced/commentary, ranked by cue count).
        const pgsTracks = filterAndRankAutoTracks(
          streams.filter((s) => s.kind === "pgs"),
        ).slice(0, 3);
        for (const track of pgsTracks) {
          onProgress({
            messageKey: "progress.readingPgsStream",
            messageParams: { stream: track.value },
            fraction: 0.08,
          });
          const timings = await mod.probePgsTimings(ff, refFile, track.value, {
            status: reportStatus,
            codec: track.codec,
          });
          if (timings && timings.length >= 20 && activeSecondsForTimings(timings) >= 50.0) {
            const kindLabel =
              track.codec === "hdmv_pgs_subtitle"
                ? "PGS"
                : track.codec === "dvb_subtitle"
                  ? "DVB"
                  : "VobSub";
            onProgress({
              messageKey: "progress.usingStreamKind",
              messageParams: { kind: kindLabel, stream: track.value },
              fraction: 0.35,
            });
            const inCopy = inBytesRaw.slice();
            const res = await postWorkerSync(
              {
                type: "syncTimings",
                payload: { timings, inName: inFile.name, inBytes: inCopy, options: probedOptions },
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
          error: t("errors.missingWebrtcVad"),
          error_key: "errors.missingWebrtcVad",
        };
      }

      let probedForAudio = null;
      try {
        probedForAudio = await probeStreamsCached(refFile, reportStatus);
      } catch {
        probedForAudio = null;
      }
      const audioStreams = ((probedForAudio && probedForAudio.streams) || []).filter(
        (s) => s.kind === "audio",
      );
      let targetAudio = null;
      if (isExplicitAudio && refStream) {
        targetAudio =
          audioStreams.find(
            (s) => s.value === refStream || `0:${s.value}` === refStream,
          ) || null;
      } else if (audioStreams.length) {
        targetAudio = audioStreams.find((s) => s.is_default) || audioStreams[0];
      }
      const dualCenter = Boolean(targetAudio && targetAudio.has_center_channel);
      const audioOptions =
        probedForAudio && probedForAudio.video_fps
          ? { ...options, inferred_sub_fps: probedForAudio.video_fps }
          : options;

      let pcm;
      let channels = 1;
      try {
        onProgress({ messageKey: "progress.loadingFfmpegWasm", fraction: 0.05 });
        const decoded = await mod.decodeAudioToPcm(ff, refFile, {
          stream: isExplicitAudio ? refStream : undefined,
          dualCenter,
          status: reportStatus,
          onProgress: (p) => {
            if (Number.isFinite(p) && p > 0 && p <= 1) {
              const pct = Math.round(p * 100);
              onProgress({
                messageKey: "progress.decodingAudioPct",
                messageParams: { pct },
                fraction: 0.05 + p * 0.45,
              });
            }
          },
        });
        if (decoded && decoded.pcm) {
          pcm = decoded.pcm;
          channels = decoded.channels || 1;
        } else {
          pcm = decoded;
          channels = 1;
        }
      } catch (e) {
        console.error(e);
        const errDetail = (e && e.message) || e;
        return {
          ok: false,
          elapsed_seconds: (performance.now() - t0) / 1000,
          error: t("errors.audioDecodeFailed", { error: errDetail }),
          error_key: "errors.audioDecodeFailed",
          error_params: { error: errDetail },
        };
      }

      onProgress({ messageKey: "progress.detectingSpeechAligning", fraction: 0.50 });
      return postWorkerSync(
        {
          type: "syncAudioPcm",
          payload: {
            pcm,
            channels,
            frameRate: ff.frameRate,
            inName: inFile.name,
            inBytes: inBytesRaw,
            vad: "webrtc",
            options: audioOptions,
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
