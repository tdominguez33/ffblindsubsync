// Browser media helpers powered by ffmpeg.wasm (WORKERFS lazy mount, so multi-GB
// files are never read whole into memory). Supports:
//   - probeStreams: inspect audio / text subtitle / PGS / VobSub tracks via ffprobe
//   - extractSubtitleStream: extract an embedded text subtitle track to SRT
//   - probePgsTimings: extract PGS/VobSub packet timings via ffprobe
//   - decodeAudioToPcm: decode an audio track to mono s16le PCM for WebRTC VAD

let _ffmpegPromise = null;
let _opQueue = Promise.resolve();

const BITMAP_CODECS = new Set(["dvb_subtitle", "dvb_teletext", "xsub"]);
const PGS_CODECS = new Set(["hdmv_pgs_subtitle", "dvd_subtitle"]);
const DEPRIORITIZED_TOKENS = ["forced", "commentary", "signs", "songs"];
const textDecoder = new TextDecoder("utf-8");

function isDeprioritized(title = "", disp = {}) {
  if (disp && (disp.forced || disp.comment)) return true;
  const low = String(title || "").toLowerCase();
  return DEPRIORITIZED_TOKENS.some((tok) => low.includes(tok));
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
  const empty = { streams: [], has_text_subs: false, has_pgs_subs: false };
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
        "stream=index,codec_type,codec_name:stream_disposition=forced,comment:stream_tags=language,title",
        "-of", "json",
        mounted,
        "-o", outPath,
      ]);
      const raw = await ffmpeg.readFile(outPath);
      if (!raw || !raw.byteLength) return empty;
      const data = JSON.parse(textDecoder.decode(raw));
      const rawStreams = Array.isArray(data && data.streams) ? data.streams : [];

      let audioIdx = 0;
      let subIdx = 0;
      const items = [];
      let hasTextSubs = false;
      let hasPgsSubs = false;

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

        if (ctype === "audio") {
          const spec = `a:${audioIdx}`;
          const label = `Audio #${audioIdx + 1} (${spec}): ${meta || "default"}${flagNote} [${codec}]`;
          items.push({ value: spec, label, kind: "audio" });
          audioIdx += 1;
        } else if (ctype === "subtitle") {
          const spec = `s:${subIdx}`;
          const isPgs = PGS_CODECS.has(codec);
          let kind;
          let isBitmap;
          let note;
          if (isPgs) {
            kind = "pgs";
            isBitmap = false;
            hasPgsSubs = true;
            note = codec === "hdmv_pgs_subtitle" ? "PGS timings" : "VobSub timings";
          } else if (BITMAP_CODECS.has(codec)) {
            kind = "subtitle";
            isBitmap = true;
            note = `image: ${codec}`;
          } else {
            kind = "subtitle";
            isBitmap = false;
            hasTextSubs = true;
            note = `text: ${codec}`;
          }
          const label = `Subtitle #${subIdx + 1} (${spec}): ${meta || "default"}${flagNote} [${note}]`;
          items.push({
            value: spec,
            label,
            kind,
            codec,
            is_bitmap: isBitmap,
            is_pgs: isPgs,
            deprioritized: isDeprioritized(title, disp),
          });
          subIdx += 1;
        }
      }

      return {
        streams: items,
        has_text_subs: hasTextSubs,
        has_pgs_subs: hasPgsSubs,
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

export async function probePgsTimings(cfg, file, streamSpec, { status } = {}) {
  const probeStream = normalizeProbeStream(streamSpec);
  if (!probeStream) return null;
  return withMountedFile(cfg, file, status, async (ffmpeg, mounted) => {
    const outPath = "/pgs_packets.json";
    await safeDelete(ffmpeg, outPath);
    status && status(`reading PGS/VobSub timings (${streamSpec})…`);
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
      const explicitTimings = [];
      const parsed = [];
      for (const pkt of packets) {
        if (pkt.pts_time == null || pkt.size == null) continue;
        const pts = Number(pkt.pts_time);
        const size = Number(pkt.size);
        if (!Number.isFinite(pts) || !Number.isFinite(size)) continue;
        let hasExplicitDur = false;
        if (pkt.duration_time != null && pkt.duration_time !== "N/A") {
          const dur = Number(pkt.duration_time);
          if (Number.isFinite(dur)) {
            hasExplicitDur = true;
            if (size > 50 && dur > 0) {
              explicitTimings.push([pts, pts + dur]);
            }
          }
        }
        parsed.push({ pts, size, hasExplicitDur });
      }
      if (explicitTimings.length) return explicitTimings;

      // Fallback for MKV Blu-ray PGS streams where duration_time is omitted:
      // pair each large "show" packet (>50 B) with the next tiny "clear" packet.
      const MAX_CUE_S = 15.0;
      parsed.sort((a, b) => a.pts - b.pts);
      const inferred = [];
      let openStart = null;
      for (const { pts, size, hasExplicitDur } of parsed) {
        if (hasExplicitDur) continue;
        if (size > 50) {
          if (openStart !== null && pts > openStart) {
            inferred.push([openStart, Math.min(pts, openStart + MAX_CUE_S)]);
          }
          openStart = pts;
        } else if (openStart !== null && pts > openStart) {
          inferred.push([openStart, Math.min(pts, openStart + MAX_CUE_S)]);
          openStart = null;
        }
      }
      return inferred.length ? inferred : null;
    } catch {
      return null;
    } finally {
      await safeDelete(ffmpeg, outPath);
    }
  });
}

export async function decodeAudioToPcm(cfg, file, { status, stream, onProgress } = {}) {
  return withMountedFile(cfg, file, status, async (ffmpeg, mounted) => {
    const frameRate = cfg.frameRate || 16000;
    const outPath = "/audio.pcm";
    await safeDelete(ffmpeg, outPath);

    status && status("decoding audio (this can take a bit on long videos)…");
    // ffmpeg reports decode progress as a 0–1 fraction. The instance is memoized in
    // getFfmpeg, so the listener MUST be removed after this decode.
    const onP = onProgress ? ({ progress }) => onProgress(progress) : null;
    if (onP) ffmpeg.on("progress", onP);
    try {
      const args = ["-nostdin", "-i", mounted];
      const mapSpec = normalizeMapSpec(stream);
      if (mapSpec) args.push("-map", mapSpec);
      args.push(
        "-vn",
        "-ac", "1",
        "-af", "aresample=async=1",
        "-ar", String(frameRate),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        outPath,
      );
      const code = await ffmpeg.exec(args);
      if (code !== 0) throw new Error(`ffmpeg exited with code ${code}`);
      return await ffmpeg.readFile(outPath); // Uint8Array
    } finally {
      if (onP) ffmpeg.off("progress", onP);
      await safeDelete(ffmpeg, outPath);
    }
  });
}
