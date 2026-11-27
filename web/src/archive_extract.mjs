// Client-side archive extraction for Web (.zip, .rar, .7z, .tar.*, .gz).
// Lazy-loads the vendored libarchive.js WASM bundle on first archive upload so
// users who only sync plain subtitle files pay zero extra bandwidth.

const SUPPORTED_SUB_EXTS = new Set(["srt", "ass", "ssa", "sub", "vtt"]);
const COMPOUND_TAR_EXTS = [".tar.gz", ".tar.bz2", ".tar.xz"];
const ARCHIVE_EXTS = new Set([
  "zip",
  "rar",
  "7z",
  "tar",
  "gz",
  "tgz",
  "bz2",
  "tbz2",
  "xz",
  "txz",
]);

const MAX_SUBTITLE_BYTES = 25 * 1024 * 1024;
const MAX_ARCHIVE_SUBTITLES = 200;
const MAX_NESTING_DEPTH = 2;

let archiveModPromise = null;
let archiveInitialized = false;

export function isArchiveFilename(name) {
  const base = String(name || "").split(/[\\/]/).pop().toLowerCase();
  if (!base) return false;
  for (const ext of COMPOUND_TAR_EXTS) {
    if (base.endsWith(ext)) return true;
  }
  const dot = base.lastIndexOf(".");
  if (dot < 0) return false;
  return ARCHIVE_EXTS.has(base.slice(dot + 1));
}

function isJunkMember(normPath) {
  const parts = String(normPath || "").split("/").filter(Boolean);
  if (!parts.length) return true;
  if (parts.some((p) => p === "__MACOSX")) return true;
  const base = parts[parts.length - 1];
  return base.startsWith("._");
}

function isSyncedFilename(name) {
  const base = String(name || "").split(/[\\/]/).pop();
  const dot = base.lastIndexOf(".");
  const stem = dot >= 0 ? base.slice(0, dot) : base;
  return stem.toLowerCase().endsWith(".synced");
}

function isValidSubtitleMember(normPath) {
  if (isJunkMember(normPath)) return false;
  const base = normPath.split("/").pop();
  const dot = base.lastIndexOf(".");
  if (dot < 0) return false;
  if (!SUPPORTED_SUB_EXTS.has(base.slice(dot + 1).toLowerCase())) return false;
  return !isSyncedFilename(base);
}

function isNestedArchiveMember(normPath) {
  if (isJunkMember(normPath)) return false;
  return isArchiveFilename(normPath.split("/").pop());
}

function normalizeMemberPath(rawPath, fallbackName) {
  const joined = rawPath
    ? (rawPath.endsWith("/") ? rawPath + fallbackName : rawPath)
    : fallbackName;
  return String(joined || fallbackName || "")
    .replace(/\\/g, "/")
    .replace(/^\/+/, "");
}

async function getArchiveClass(archiveCfg) {
  if (!archiveModPromise) {
    archiveModPromise = import(archiveCfg.module);
  }
  const mod = await archiveModPromise;
  const Archive = mod.Archive || (mod.default && mod.default.Archive);
  if (!Archive) {
    throw new Error("libarchive.js failed to load Archive export");
  }
  if (!archiveInitialized) {
    Archive.init({ workerUrl: archiveCfg.worker });
    archiveInitialized = true;
  }
  return Archive;
}

async function tryDecompressSingleGzipSubtitle(file, prefix) {
  const low = String(file.name || "").toLowerCase();
  if (!low.endsWith(".gz") || low.endsWith(".tar.gz")) return null;
  const innerName = file.name.slice(0, -3);
  if (!isValidSubtitleMember(innerName)) return null;
  if (typeof DecompressionStream !== "function") return null;

  const ds = new DecompressionStream("gzip");
  const decompressed = file.stream().pipeThrough(ds);
  const buf = await new Response(decompressed).arrayBuffer();
  if (buf.byteLength > MAX_SUBTITLE_BYTES) return [];
  const extractedFile = new File([buf], innerName, {
    type: "text/plain;charset=utf-8",
    lastModified: file.lastModified || Date.now(),
  });
  return [
    {
      file: extractedFile,
      displayName: `${prefix} › ${innerName}`,
    },
  ];
}

export async function extractSubtitlesFromArchive(
  archiveFile,
  archiveCfg,
  { depth = 0, labelPrefix = null } = {},
) {
  const prefix = labelPrefix || archiveFile.name || "archive";

  // Fast zero-WASM path for standalone gzipped subtitle files (e.g. movie.srt.gz).
  const singleGz = await tryDecompressSingleGzipSubtitle(archiveFile, prefix);
  if (singleGz !== null) {
    return singleGz;
  }

  const Archive = await getArchiveClass(archiveCfg);
  const archive = await Archive.open(archiveFile);
  try {
    const encrypted = await archive.hasEncryptedData().catch(() => false);
    if (encrypted === true) {
      throw new Error(`Archive is password-protected: ${archiveFile.name}`);
    }

    const rawList = (await archive.getFilesArray()) || [];
    const subEntries = [];
    const nestedEntries = [];

    for (const item of rawList) {
      const cf = item && item.file;
      if (!cf || typeof cf.extract !== "function") continue;
      const normPath = normalizeMemberPath(item.path, cf.name);
      const size = Number(cf.size || 0);
      if (size > MAX_SUBTITLE_BYTES) continue;
      if (isValidSubtitleMember(normPath)) {
        subEntries.push({ normPath, cf });
      } else if (depth < MAX_NESTING_DEPTH && isNestedArchiveMember(normPath)) {
        nestedEntries.push({ normPath, cf });
      }
    }

    subEntries.sort((a, b) => a.normPath.localeCompare(b.normPath));
    nestedEntries.sort((a, b) => a.normPath.localeCompare(b.normPath));

    const baseCounts = new Map();
    for (const entry of subEntries) {
      const base = entry.normPath.split("/").pop().toLowerCase();
      baseCounts.set(base, (baseCounts.get(base) || 0) + 1);
    }

    const results = [];
    for (const entry of subEntries) {
      if (results.length >= MAX_ARCHIVE_SUBTITLES) break;
      const extracted = await entry.cf.extract();
      if (!extracted || extracted.size > MAX_SUBTITLE_BYTES) continue;
      const innerBase = entry.normPath.split("/").pop() || extracted.name;
      const cleanFile =
        extracted.name === innerBase
          ? extracted
          : new File([extracted], innerBase, {
              type: extracted.type || "text/plain;charset=utf-8",
              lastModified: extracted.lastModified || Date.now(),
            });
      const shownInner =
        (baseCounts.get(innerBase.toLowerCase()) || 0) > 1 ? entry.normPath : innerBase;
      results.push({
        file: cleanFile,
        displayName: `${prefix} › ${shownInner}`,
      });
    }

    for (const entry of nestedEntries) {
      if (results.length >= MAX_ARCHIVE_SUBTITLES) break;
      try {
        const nestedFile = await entry.cf.extract();
        if (!nestedFile) continue;
        const nestedBase = entry.normPath.split("/").pop() || nestedFile.name;
        const nestedSubs = await extractSubtitlesFromArchive(nestedFile, archiveCfg, {
          depth: depth + 1,
          labelPrefix: `${prefix} › ${nestedBase}`,
        });
        for (const sub of nestedSubs) {
          if (results.length >= MAX_ARCHIVE_SUBTITLES) break;
          results.push(sub);
        }
      } catch (err) {
        console.warn("Skipping nested archive:", entry.normPath, err);
      }
    }

    return results;
  } finally {
    try {
      await archive.close();
    } catch (_) {}
  }
}
