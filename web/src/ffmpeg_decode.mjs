// Browser media helpers powered by ffmpeg.wasm (WORKERFS lazy mount, so multi-GB
// files are never read whole into memory). Supports:
//   - probeStreams: inspect audio / text subtitle / PGS / VobSub tracks via ffprobe
//   - extractSubtitleStream: extract an embedded text subtitle track to SRT
//   - probePgsTimings: extract PGS/VobSub packet timings via ffprobe
//   - decodeAudioToPcm: decode an audio track to mono s16le PCM for WebRTC VAD

let _ffmpegPromise = null;
let _opQueue = Promise.resolve();

const BITMAP_CODECS = new Set(["dvb_teletext", "xsub"]);
const PGS_CODECS = new Set(["hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"]);
const DEPRIORITIZED_TOKENS = ["forced", "commentary", "signs", "songs"];
const CENTER_LAYOUT_PREFIXES = [
  "3.0",
  "3.1",
  "4.0",
  "4.1",
  "5.0",
  "5.1",
  "6.0",
  "6.1",
  "7.0",
  "7.1",
];
const CENTER_LAYOUT_NAMES = new Set(["hexagonal", "octagonal"]);
const NO_CENTER_LAYOUTS = new Set([
  "mono",
  "stereo",
  "2.1",
  "2.2",
  "quad",
  "quad(side)",
  "2.1.2",
]);
const textDecoder = new TextDecoder("utf-8");

export function layoutHasCenterChannel(channelLayout) {
  const low = String(channelLayout || "").trim().toLowerCase();
  if (!low || low === "unknown" || NO_CENTER_LAYOUTS.has(low)) return false;
  if (CENTER_LAYOUT_NAMES.has(low)) return true;
  if (CENTER_LAYOUT_PREFIXES.some((p) => low.startsWith(p))) return true;
  return low.split(/[^a-z0-9]+/).includes("fc");
}

function parseFpsFraction(val) {
  const s = String(val || "").trim();
  if (!s || s === "0/0" || s === "N/A") return null;
  let fps;
  if (s.includes("/")) {
    const [numStr, denStr] = s.split("/", 2);
    const num = Number(numStr);
    const den = Number(denStr);
    if (!Number.isFinite(num) || !Number.isFinite(den) || den <= 0) return null;
    fps = num / den;
  } else {
    fps = Number(s);
  }
  return Number.isFinite(fps) && fps >= 1.0 && fps <= 240.0 ? fps : null;
}

function numberOfFramesFromTags(tags = {}) {
  for (const [k, v] of Object.entries(tags || {})) {
    if (String(k).toUpperCase().startsWith("NUMBER_OF_FRAMES")) {
      const n = Number(v);
      if (Number.isFinite(n) && n > 0) return Math.trunc(n);
    }
  }
  return 0;
}

function isDeprioritized(title = "", disp = {}) {
  if (disp && (disp.forced || disp.comment)) return true;
  const low = String(title || "").toLowerCase();
  return DEPRIORITIZED_TOKENS.some((tok) => low.includes(tok));
}

function isSdhTrack(title = "", disp = {}) {
  if (disp && disp.hearing_impaired) return true;
  const low = String(title || "").toLowerCase();
  return /\b(sdh|cc|hi)\b/.test(low) || low.includes("hearing impaired") || low.includes("closed caption");
}

function normalizeLang(lang) {
  const code = String(lang || "").trim().toLowerCase();
  return !code || code === "und" || code === "unk" || code === "unknown" ? "" : code;
}

function primaryAudioLanguage(rawStreams) {
  const audio = rawStreams.filter((s) => s.codec_type === "audio");
  if (!audio.length) return "";
  for (const s of audio) {
    const disp = s.disposition || {};
    if (disp.default && !disp.comment) return normalizeLang((s.tags || {}).language);
  }
  for (const s of audio) {
    const disp = s.disposition || {};
    const title = String((s.tags || {}).title || "").toLowerCase();
    if (!disp.comment && !title.includes("commentary")) {
      return normalizeLang((s.tags || {}).language);
    }
  }
  return normalizeLang((audio[0].tags || {}).language);
}

function normalizeMapSpec(stream) {
  if (!stream) return null;
  const s = String(stream).trim();
  if (!s) return null;
  return s.startsWith("0:") ? s : `0:${s}`;
}

function normalizeProbeStream(stream) {
  const s = String(stream || "").trim();
  return s.startsWith("0:") ? s.slice(2) : s;
}

async function getFfmpeg(cfg, status) {
  if (_ffmpegPromise) return _ffmpegPromise;
  _ffmpegPromise = (async () => {
    status && status("loading ffmpeg.wasm…");
    // @ffmpeg/ffmpeg is vendored same-origin (its worker uses relative imports),
    // so FFmpeg's default `new URL('./worker.js', import.meta.url)` resolves fine.
    const { FFmpeg } = await import(cfg.module);
    const { toBlobURL } = await import(cfg.util);
    const ffmpeg = new FFmpeg();
    // The core (JS + ~30MB wasm) stays on the CDN; fetch it as same-origin blobs.
    const [coreURL, wasmURL] = await Promise.all([
      toBlobURL(cfg.coreCdn + "ffmpeg-core.js", "text/javascript"),
      toBlobURL(cfg.coreCdn + "ffmpeg-core.wasm", "application/wasm"),
    ]);
    await ffmpeg.load({ coreURL, wasmURL });
    return ffmpeg;
  })();
  try {
    return await _ffmpegPromise;
  } catch (err) {
    _ffmpegPromise = null;
    throw err;
  }
}

function serialize(fn) {
  const next = _opQueue.then(fn, fn);
  _opQueue = next.catch(() => {});
  return next;
}

async function withMountedFile(cfg, file, status, fn) {
  return serialize(async () => {
    const ffmpeg = await getFfmpeg(cfg, status);
    const mountDir = "/mnt";
    const mounted = `${mountDir}/${file.name}`;
    try {
      await ffmpeg.createDir(mountDir);
    } catch {
      /* already exists */
    }
    await ffmpeg.mount("WORKERFS", { files: [file] }, mountDir);
    try {
      return await fn(ffmpeg, mounted);
    } finally {
      try {
        await ffmpeg.unmount(mountDir);
      } catch {
        /* ignore */
      }
    }
  });
}

async function safeDelete(ffmpeg, filePath) {
  try {
    await ffmpeg.deleteFile(filePath);
  } catch {
    /* ignore */
  }
}

export async function probeStreams(cfg, file, { status } = {}) {
  const empty = {
    streams: [],
    has_text_subs: false,
    has_pgs_subs: false,
    video_fps: null,
    duration: null,
  };
  if (!cfg || !file) return empty;
  return withMountedFile(cfg, file, status, async (ffmpeg, mounted) => {
    const outPath = "/probe_streams.json";
    await safeDelete(ffmpeg, outPath);
    try {
      // Note: @ffmpeg/core@0.12.x leaves Module.ret at -1 for ffprobe even on
      // success, so check whether outPath was written and parses as valid JSON.
      await ffmpeg.ffprobe([
        "-v", "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,channels,channel_layout,r_frame_rate,avg_frame_rate:stream_disposition=default,forced,comment,hearing_impaired,attached_pic:stream_tags",
        "-of", "json",
        mounted,
        "-o", outPath,
      ]);
      const raw = await ffmpeg.readFile(outPath);
      if (!raw || !raw.byteLength) return empty;
      const data = JSON.parse(textDecoder.decode(raw));
      const rawStreams = Array.isArray(data && data.streams) ? data.streams : [];
      const rawDuration = Number(data && data.format && data.format.duration);
      const duration = Number.isFinite(rawDuration) && rawDuration > 0 ? rawDuration : null;
      const audioLang = primaryAudioLanguage(rawStreams);

      let audioIdx = 0;
      let subIdx = 0;
      const items = [];
      let hasTextSubs = false;
      let hasPgsSubs = false;
      let videoFps = null;

      for (const s of rawStreams) {
        const ctype = s.codec_type;
        const codec = String(s.codec_name || "unknown").toLowerCase();
        const tags = s.tags || {};
        const disp = s.disposition || {};
        const lang = tags.language || "";
        const title = tags.title || "";
        const meta = [lang, title].filter(Boolean).join(" · ");
        const flags = [];
        if (disp.forced) flags.push("forced");
        if (disp.comment) flags.push("commentary");
        const flagNote = flags.length ? ` (${flags.join(", ")})` : "";

        if (ctype === "video") {
          if (videoFps === null && !disp.attached_pic) {
            videoFps = parseFpsFraction(s.avg_frame_rate) || parseFpsFraction(s.r_frame_rate);
          }
        } else if (ctype === "audio") {
          const spec = `a:${audioIdx}`;
          const channels = Number(s.channels) || 0;
          const channelLayout = String(s.channel_layout || "");
          const hasCenterChannel = channels >= 3 && layoutHasCenterChannel(channelLayout);
          const label = `Audio #${audioIdx + 1} (${spec}): ${meta || "default"}${flagNote} [${codec}]`;
          items.push({
            value: spec,
            label,
            kind: "audio",
            index: audioIdx + 1,
            spec,
            codec,
            lang,
            title,
            forced: Boolean(disp.forced),
            comment: Boolean(disp.comment),
            is_default: Boolean(disp.default),
            channels,
            channel_layout: channelLayout,
            has_center_channel: hasCenterChannel,
          });
          audioIdx += 1;
        } else if (ctype === "subtitle") {
          const spec = `s:${subIdx}`;
          const isPgs = PGS_CODECS.has(codec);
          let kind;
          let isBitmap;
          let note;
          let noteKind;
          if (isPgs) {
            kind = "pgs";
            isBitmap = false;
            hasPgsSubs = true;
            if (codec === "hdmv_pgs_subtitle") {
              noteKind = "pgs";
              note = "PGS timings";
            } else if (codec === "dvb_subtitle") {
              noteKind = "dvb";
              note = "DVB timings";
            } else {
              noteKind = "vobsub";
              note = "VobSub timings";
            }
          } else if (BITMAP_CODECS.has(codec)) {
            kind = "subtitle";
            isBitmap = true;
            noteKind = "image";
            note = `image: ${codec}`;
          } else {
            kind = "subtitle";
            isBitmap = false;
            hasTextSubs = true;
            noteKind = "text";
            note = `text: ${codec}`;
          }
          const label = `Subtitle #${subIdx + 1} (${spec}): ${meta || "default"}${flagNote} [${note}]`;
          items.push({
            value: spec,
            label,
            kind,
            index: subIdx + 1,
            spec,
            codec,
            lang,
            title,
            forced: Boolean(disp.forced),
            comment: Boolean(disp.comment),
            note_kind: noteKind,
            is_bitmap: isBitmap,
            is_pgs: isPgs,
            is_sdh: isSdhTrack(title, disp),
            is_default: Boolean(disp.default),
            matches_audio_lang: Boolean(audioLang && normalizeLang(lang) === audioLang),
            num_frames: numberOfFramesFromTags(tags),
            deprioritized: isDeprioritized(title, disp),
          });
          subIdx += 1;
        }
      }

      return {
        streams: items,
        has_text_subs: hasTextSubs,
        has_pgs_subs: hasPgsSubs,
        video_fps: videoFps,
        duration,
      };
    } catch {
      return empty;
    } finally {
      await safeDelete(ffmpeg, outPath);
    }
  });
}

export async function extractSubtitleStream(
  cfg,
  file,
  streamSpec,
  { status, onProgress } = {},
) {
  const mapSpec = normalizeMapSpec(streamSpec);
  if (!mapSpec) return null;
  return withMountedFile(cfg, file, status, async (ffmpeg, mounted) => {
    const outPath = "/embedded_sub.srt";
    await safeDelete(ffmpeg, outPath);
    status && status(`extracting embedded subtitle (${streamSpec})…`);
    const onP = onProgress ? ({ progress }) => onProgress(progress) : null;
    if (onP) ffmpeg.on("progress", onP);
    try {
      const code = await ffmpeg.exec([
        "-nostdin",
        "-i", mounted,
        "-map", mapSpec,
        "-f", "srt",
        outPath,
      ]);
      if (code !== 0) return null;
      const data = await ffmpeg.readFile(outPath);
      return data && data.byteLength > 0 ? data : null;
    } catch {
      return null;
    } finally {
      if (onP) ffmpeg.off("progress", onP);
      await safeDelete(ffmpeg, outPath);
    }
  });
}

function parseCompactLine(line) {
  const row = {};
  let hasField = false;
  for (const part of String(line || "").split("|")) {
    const eq = part.indexOf("=");
    if (eq <= 0) continue;
    row[part.slice(0, eq).trim()] = part.slice(eq + 1).trim();
    hasField = true;
  }
  return hasField ? row : null;
}

async function streamFfprobeCompact(ffmpeg, args, { duration = null, onProgress = null } = {}) {
  const rows = [];
  const hasDuration = Number.isFinite(duration) && duration > 0;
  let lastPts = 0;
  let lastEmitMs = 0;

  const onLog = ({ type, message }) => {
    if (type && type !== "stdout") return;
    if (!message || !message.includes("=")) return;
    for (const rawLine of String(message).split(/\r?\n/)) {
      const trimmed = rawLine.trim();
      if (!trimmed || !trimmed.includes("=")) continue;
      const row = parseCompactLine(trimmed);
      if (!row) continue;
      rows.push(row);
      if (onProgress && row.pts_time && row.pts_time !== "N/A") {
        const pts = Number(row.pts_time);
        if (Number.isFinite(pts) && pts > lastPts) {
          lastPts = pts;
          const now = typeof performance !== "undefined" ? performance.now() : Date.now();
          if (now - lastEmitMs >= 120) {
            lastEmitMs = now;
            const frac = hasDuration ? Math.max(0, Math.min(1, lastPts / duration)) : null;
            onProgress(frac, { ptsSeconds: lastPts, count: rows.length });
          }
        }
      }
    }
  };

  ffmpeg.on("log", onLog);
  try {
    if (onProgress && hasDuration) {
      onProgress(0, { ptsSeconds: 0, count: 0 });
    }
    await ffmpeg.ffprobe(args);
    if (onProgress && rows.length > 0) {
      onProgress(hasDuration ? 1 : null, {
        ptsSeconds: hasDuration ? duration : lastPts,
        count: rows.length,
      });
    }
    return rows;
  } finally {
    ffmpeg.off("log", onLog);
  }
}

function parseDvdFramesToTimings(frames) {
  const captions = [];
  for (const frame of frames || []) {
    const pts = Number(frame.pts_time);
    let startMs = frame.start_display_time != null ? Number(frame.start_display_time) : 0;
    let endMs = Number(frame.end_display_time);
    const numRects = frame.num_rects != null ? Number(frame.num_rects) : 1;
    if (!Number.isFinite(pts) || !Number.isFinite(startMs) || !Number.isFinite(endMs)) {
      continue;
    }
    if (numRects === 0) {
      startMs = 0;
      endMs = 0;
    }
    captions.push([pts, startMs / 1000.0, endMs / 1000.0]);
  }
  captions.sort((a, b) => a[0] - b[0]);
  const results = [];
  for (let i = 0; i < captions.length; i += 1) {
    const [pts, startOff, endOff] = captions[i];
    const start = pts + startOff;
    let end = pts + endOff;
    if (i + 1 < captions.length) {
      end = Math.min(end, captions[i + 1][0]);
    }
    if (end > start) results.push([start, end]);
  }
  return results.length ? results : null;
}

function parsePgsPacketsToTimings(packets) {
  const parsed = [];
  for (const pkt of packets || []) {
    if (pkt.pts_time == null || pkt.size == null) continue;
    const pts = Number(pkt.pts_time);
    const size = Number(pkt.size);
    if (!Number.isFinite(pts) || !Number.isFinite(size)) continue;
    let dur = null;
    if (pkt.duration_time != null && pkt.duration_time !== "N/A") {
      const parsedDur = Number(pkt.duration_time);
      if (!Number.isFinite(parsedDur)) continue;
      dur = parsedDur;
    }
    parsed.push({ pts, size, dur });
  }
  parsed.sort((a, b) => a.pts - b.pts);

  // Single-pass extraction supporting numeric durations, durationless
  // show/clear pairs, and streams mixing both.
  const MAX_CUE_S = 15.0;
  const results = [];
  let pendingShowPts = null;
  for (const { pts, size, dur } of parsed) {
    if (size <= 50) {
      if (pendingShowPts !== null && dur === null && pts > pendingShowPts) {
        results.push([pendingShowPts, Math.min(pts, pendingShowPts + MAX_CUE_S)]);
        pendingShowPts = null;
      }
      continue;
    }
    if (dur === null) {
      if (pendingShowPts !== null && pts > pendingShowPts) {
        results.push([pendingShowPts, Math.min(pts, pendingShowPts + MAX_CUE_S)]);
      }
      pendingShowPts = pts;
      continue;
    }
    pendingShowPts = null;
    if (dur > 0) {
      results.push([pts, pts + dur]);
    }
  }
  return results.length ? results : null;
}

async function probeDvdFramesInSession(
  ffmpeg,
  mounted,
  probeStream,
  { duration = null, onProgress = null } = {},
) {
  try {
    const streamed = await streamFfprobeCompact(
      ffmpeg,
      [
        "-v", "error",
        "-show_frames",
        "-select_streams", probeStream,
        "-show_entries", "subtitle=pts_time,start_display_time,end_display_time,num_rects",
        "-of", "compact=p=0:nk=0",
        mounted,
      ],
      { duration, onProgress },
    );
    return parseDvdFramesToTimings(streamed);
  } catch {
    /* fall through to JSON file output */
  }
  const outPath = "/dvd_frames.json";
  await safeDelete(ffmpeg, outPath);
  try {
    await ffmpeg.ffprobe([
      "-v", "error",
      "-show_frames",
      "-select_streams", probeStream,
      "-show_entries", "subtitle=pts_time,start_display_time,end_display_time,num_rects",
      "-of", "json",
      mounted,
      "-o", outPath,
    ]);
    const raw = await ffmpeg.readFile(outPath);
    if (!raw || !raw.byteLength) return null;
    const data = JSON.parse(textDecoder.decode(raw));
    const frames = Array.isArray(data && data.frames) ? data.frames : [];
    return parseDvdFramesToTimings(frames);
  } catch {
    return null;
  } finally {
    await safeDelete(ffmpeg, outPath);
  }
}

export async function probePgsTimings(
  cfg,
  file,
  streamSpec,
  { status, codec, duration = null, onProgress = null } = {},
) {
  const probeStream = normalizeProbeStream(streamSpec);
  if (!probeStream) return null;
  return withMountedFile(cfg, file, status, async (ffmpeg, mounted) => {
    status && status(`reading PGS/VobSub/DVB timings (${streamSpec})…`);
    if (codec === "dvd_subtitle") {
      const dvdTimings = await probeDvdFramesInSession(ffmpeg, mounted, probeStream, {
        duration,
        onProgress,
      });
      if (dvdTimings && dvdTimings.length) return dvdTimings;
    }
    try {
      const streamed = await streamFfprobeCompact(
        ffmpeg,
        [
          "-v", "error",
          "-show_packets",
          "-select_streams", probeStream,
          "-show_entries", "packet=pts_time,duration_time,size",
          "-of", "compact=p=0:nk=0",
          mounted,
        ],
        { duration, onProgress },
      );
      return parsePgsPacketsToTimings(streamed);
    } catch {
      /* fall through to JSON file output */
    }
    const outPath = "/pgs_packets.json";
    await safeDelete(ffmpeg, outPath);
    try {
      // @ffmpeg/core@0.12.x leaves Module.ret at -1 for ffprobe even on success.
      await ffmpeg.ffprobe([
        "-v", "error",
        "-show_packets",
        "-select_streams", probeStream,
        "-show_entries", "packet=pts_time,duration_time,size",
        "-of", "json",
        mounted,
        "-o", outPath,
      ]);
      const raw = await ffmpeg.readFile(outPath);
      if (!raw || !raw.byteLength) return null;
      const data = JSON.parse(textDecoder.decode(raw));
      const packets = Array.isArray(data && data.packets) ? data.packets : [];
      return parsePgsPacketsToTimings(packets);
    } catch {
      return null;
    } finally {
      await safeDelete(ffmpeg, outPath);
    }
  });
}

export async function decodeAudioToPcm(
  cfg,
  file,
  { status, stream, dualCenter = false, onProgress } = {},
) {
  return withMountedFile(cfg, file, status, async (ffmpeg, mounted) => {
    const frameRate = cfg.frameRate || 16000;
    const outPath = "/audio.pcm";
    await safeDelete(ffmpeg, outPath);

    status && status("decoding audio (this can take a bit on long videos)…");
    // ffmpeg reports decode progress as a 0–1 fraction. The instance is memoized in
    // getFfmpeg, so the listener MUST be removed after this decode.
    const onP = onProgress ? ({ progress }) => onProgress(progress) : null;
    if (onP) ffmpeg.on("progress", onP);

    const runDecode = async (useDualCenter) => {
      await safeDelete(ffmpeg, outPath);
      const args = ["-nostdin", "-i", mounted];
      const mapSpec = normalizeMapSpec(stream);
      if (mapSpec) args.push("-map", mapSpec);
      const afExpr = useDualCenter
        ? "aresample=async=1,asplit=2[a][b];[a]aresample=ochl=mono[m];[b]pan=mono|c0=FC[c];[m][c]amerge=inputs=2"
        : "aresample=async=1";
      args.push(
        "-vn",
        "-sn",
        "-dn",
        "-ac", useDualCenter ? "2" : "1",
        "-af", afExpr,
        "-ar", String(frameRate),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        outPath,
      );
      const code = await ffmpeg.exec(args);
      if (code !== 0) throw new Error(`ffmpeg exited with code ${code}`);
      const data = await ffmpeg.readFile(outPath);
      if (!data || data.byteLength < (useDualCenter ? 4 : 2)) {
        throw new Error("ffmpeg produced empty audio output");
      }
      return { pcm: data, channels: useDualCenter ? 2 : 1 };
    };

    try {
      if (dualCenter) {
        try {
          return await runDecode(true);
        } catch (err) {
          console.warn("dual-channel (mix + FC) decode failed; falling back to mono:", err);
        }
      }
      return await runDecode(false);
    } finally {
      if (onP) ffmpeg.off("progress", onP);
      await safeDelete(ffmpeg, outPath);
    }
  });
}
