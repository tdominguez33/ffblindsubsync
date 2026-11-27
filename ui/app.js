// Unified UI controller shared by the native desktop app (pywebview) and the
// browser WebAssembly app (Pyodide + ffmpeg.wasm). Platform-specific behavior
// is isolated in ./adapters/desktop.js and ./adapters/web.js.

import { i18n, SYSTEM_LANGUAGE } from "./i18n/index.js";

const V = new URL(import.meta.url).searchParams.get("v") || "";
const withVersion = (url) => (V ? url + (url.includes("?") ? "&" : "?") + "v=" + V : url);

const els = {
  envSubtitle: document.getElementById("env-subtitle"),
  envBadge: document.getElementById("env-badge"),
  topbarActions: document.getElementById("topbar-actions"),
  topbarActionsDivider: document.getElementById("topbar-actions-divider"),
  statusRow: document.getElementById("status-row"),
  topbarStatusPill: document.getElementById("topbar-status-pill"),
  topbarStatusDot: document.getElementById("topbar-status-dot"),
  status: document.getElementById("status"),
  settingsBtn: document.getElementById("settings-btn"),
  themeToggle: document.getElementById("theme-toggle"),
  dragBrand: document.getElementById("drag-brand"),
  dragSpacer: document.getElementById("drag-spacer"),
  winControlsDivider: document.getElementById("win-controls-divider"),
  winControls: document.getElementById("win-controls"),
  winMinimize: document.getElementById("win-minimize"),
  winMaximize: document.getElementById("win-maximize"),
  winClose: document.getElementById("win-close"),
  resizeHandles: document.getElementById("resize-handles"),
  mainView: document.getElementById("main-view"),
  settingsView: document.getElementById("settings-view"),
  settingsCloseBtn: document.getElementById("settings-close-btn"),
  settingsLanguage: document.getElementById("settings-language"),
  settingsAlignmentProfile: document.getElementById("settings-alignment-profile"),
  settingsFfmpegCard: document.getElementById("settings-ffmpeg-card"),
  settingsFfPill: document.getElementById("settings-ff-pill"),
  settingsFfDot: document.getElementById("settings-ff-dot"),
  settingsFfStatus: document.getElementById("settings-ff-status"),
  settingsFfDir: document.getElementById("settings-ff-dir"),
  settingsFfChange: document.getElementById("settings-ff-change"),
  settingsFfOpen: document.getElementById("settings-ff-open"),
  settingsFfDownload: document.getElementById("settings-ff-download"),
  settingsFfDownloadLabel: document.getElementById("settings-ff-download-label"),
  settingsFfProg: document.getElementById("settings-ff-progress"),
  settingsFfFill: document.getElementById("settings-ff-fill"),
  settingsFfPct: document.getElementById("settings-ff-pct"),
  runtimeCard: document.getElementById("runtime-card"),
  runtimePill: document.getElementById("runtime-pill"),
  runtimeDot: document.getElementById("runtime-dot"),
  runtimeStatusText: document.getElementById("runtime-status-text"),
  runtimeActionBtn: document.getElementById("runtime-action-btn"),
  runtimeProg: document.getElementById("runtime-progress"),
  runtimeFill: document.getElementById("runtime-fill"),
  runtimePct: document.getElementById("runtime-pct"),
  dropRef: document.getElementById("drop-ref"),
  dropIn: document.getElementById("drop-in"),
  refPath: document.getElementById("ref-path"),
  refFile: document.getElementById("ref-file"),
  inPath: document.getElementById("in-path"),
  inputFile: document.getElementById("input-file"),
  inputFolder: document.getElementById("input-folder"),
  outField: document.getElementById("out-field"),
  outPath: document.getElementById("out-path"),
  browseRef: document.getElementById("browse-ref"),
  browseInSplit: document.getElementById("browse-in-split"),
  browseIn: document.getElementById("browse-in"),
  browseInMenuBtn: document.getElementById("browse-in-menu-btn"),
  browseInMenu: document.getElementById("browse-in-menu"),
  browseInOptAdd: document.getElementById("browse-in-opt-add"),
  browseInOptFiles: document.getElementById("browse-in-opt-files"),
  browseInOptFilesLabel: document.getElementById("browse-in-opt-files-label"),
  browseInOptFolder: document.getElementById("browse-in-opt-folder"),
  multiInPanel: document.getElementById("multi-in-panel"),
  multiInCount: document.getElementById("multi-in-count"),
  multiInList: document.getElementById("multi-in-list"),
  clearInBtn: document.getElementById("clear-in-btn"),
  browseOut: document.getElementById("browse-out"),
  streamWrap: document.getElementById("stream-wrap"),
  refStream: document.getElementById("ref-stream"),
  streamHint: document.getElementById("stream-hint"),
  vadMode: document.getElementById("vad-mode"),
  splitMode: document.getElementById("split-mode"),
  noFixFramerate: document.getElementById("no-fix-framerate"),
  gss: document.getElementById("gss"),
  syncBtn: document.getElementById("sync-btn"),
  syncProg: document.getElementById("sync-progress"),
  syncFill: document.getElementById("sync-fill"),
  syncPct: document.getElementById("sync-pct"),
  result: document.getElementById("result"),
  resultSingle: document.getElementById("result-single"),
  resultRanking: document.getElementById("result-ranking"),
  rankingTitle: document.getElementById("ranking-title"),
  rankingMeta: document.getElementById("ranking-meta"),
  rankingList: document.getElementById("ranking-list"),
  offset: document.getElementById("offset"),
  resultMeta: document.getElementById("result-meta"),
  resultQuality: document.getElementById("result-quality"),
  resultActionBtn: document.getElementById("result-action-btn"),
  downloadLink: document.getElementById("download"),
  footer: document.getElementById("app-footer"),
  footerText: document.getElementById("footer-text"),
  settingsFooter: document.getElementById("settings-footer"),
  settingsVersion: document.getElementById("settings-version"),
};

const ALIGNMENT_PROFILE_STORAGE_KEY = "ffs_alignment_profile";
const DEFAULT_ALIGNMENT_PROFILE = "balanced";
const ALIGNMENT_PROFILES = new Set(["balanced", "onset", "early"]);

function normalizeAlignmentProfile(raw) {
  const cleaned = String(raw || "").trim().toLowerCase();
  return ALIGNMENT_PROFILES.has(cleaned) ? cleaned : DEFAULT_ALIGNMENT_PROFILE;
}

function readSavedAlignmentProfile() {
  try {
    return normalizeAlignmentProfile(localStorage.getItem(ALIGNMENT_PROFILE_STORAGE_KEY));
  } catch (_) {
    return DEFAULT_ALIGNMENT_PROFILE;
  }
}

function writeSavedAlignmentProfile(profile) {
  try {
    localStorage.setItem(ALIGNMENT_PROFILE_STORAGE_KEY, profile);
  } catch (_) {}
}

let adapter = null;
let runtimeReady = false;
let busy = false;
let selectedRef = null; // string path (desktop) or File (web)
let selectedIns = []; // array of { value: string|File, display: string, name: string, key: string }
let appendNextWebPick = false;
let lastResult = null;
let lastStreamProbe = null;
let lastFfmpegInfo = null;
let lastSyncProgress = null;
let streamProbeSeq = 0;
let themeTransitionTimer = null;
let settingsOpen = false;
let openQualityPanel = null; // null | "sync" | "subtitle"
let expandedRankingId = null;
let rankingQualityTabs = new Map(); // candidateId -> "sync" | "subtitle" | null
let savedCandidatePaths = new Map(); // candidateId -> saved path (desktop)
let alignmentProfile = readSavedAlignmentProfile();

const SUPPORTED_SUB_EXTS = new Set(["srt", "ass", "ssa", "sub", "vtt"]);
const SUPPORTED_ARCHIVE_EXTS = new Set([
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

function basenameOf(str) {
  const s = String(str || "");
  const sep = Math.max(s.lastIndexOf("\\"), s.lastIndexOf("/"));
  return sep >= 0 ? s.slice(sep + 1) : s;
}

function isSupportedSubtitleName(name) {
  const base = basenameOf(name);
  const dot = base.lastIndexOf(".");
  if (dot < 0) return false;
  return SUPPORTED_SUB_EXTS.has(base.slice(dot + 1).toLowerCase());
}

function isSupportedArchiveName(name) {
  const base = basenameOf(name).toLowerCase();
  if (base.endsWith(".tar.gz") || base.endsWith(".tar.bz2") || base.endsWith(".tar.xz")) {
    return true;
  }
  const dot = base.lastIndexOf(".");
  if (dot < 0) return false;
  return SUPPORTED_ARCHIVE_EXTS.has(base.slice(dot + 1));
}

function isSupportedInputOrArchiveName(name) {
  return isSupportedSubtitleName(name) || isSupportedArchiveName(name);
}

function isSyncedFilename(name) {
  const base = basenameOf(name);
  const dot = base.lastIndexOf(".");
  const stem = dot >= 0 ? base.slice(0, dot) : base;
  return stem.toLowerCase().endsWith(".synced");
}

function normalizeSubtitleEntry(entry) {
  if (!entry || !entry.value) return null;
  const val = entry.value;
  if (typeof val === "string") {
    const trimmed = val.trim();
    if (!trimmed) return null;
    return {
      value: trimmed,
      display: entry.display || trimmed,
      name: entry.name || basenameOf(trimmed),
      default_output_path: entry.default_output_path || null,
      key: trimmed.toLowerCase(),
    };
  }
  const rawName = entry.name || val.name || entry.display || "subtitle.srt";
  const display =
    entry.display || (adapter && adapter.formatFileDisplay ? adapter.formatFileDisplay(val) : rawName);
  return {
    value: val,
    display,
    name: rawName,
    default_output_path: null,
    key: `${rawName.toLowerCase()}|${val.size || 0}|${val.lastModified || 0}`,
  };
}

function filterWebFolderFiles(fileList, refValue) {
  const refName = refValue && typeof refValue === "object" ? String(refValue.name || "").toLowerCase() : "";
  const refSize = refValue && typeof refValue === "object" ? Number(refValue.size || -1) : -1;
  const out = [];
  for (const file of Array.from(fileList || [])) {
    if (!file) continue;
    // Top-level only: webkitRelativePath for top-level files in a picked folder is "folder/file.srt" (2 parts).
    const rel = String(file.webkitRelativePath || "");
    if (rel) {
      const parts = rel.split("/").filter(Boolean);
      if (parts.length > 2) continue;
    }
    if (!isSupportedInputOrArchiveName(file.name)) continue;
    if (isSupportedSubtitleName(file.name) && isSyncedFilename(file.name)) continue;
    if (refName && file.name.toLowerCase() === refName && Number(file.size) === refSize) {
      continue;
    }
    out.push(file);
  }
  out.sort((a, b) => a.name.localeCompare(b.name));
  return out;
}

async function readDirectoryTopLevelFiles(dirEntry) {
  const reader = dirEntry.createReader();
  const entries = [];
  while (true) {
    const batch = await new Promise((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (!batch || !batch.length) break;
    entries.push(...batch);
  }
  const files = [];
  for (const ent of entries) {
    if (ent && ent.isFile) {
      try {
        const f = await new Promise((resolve, reject) => ent.file(resolve, reject));
        if (f) files.push(f);
      } catch (_) {}
    }
  }
  return files;
}

async function extractWebDroppedSubtitles(dataTransfer, refValue) {
  if (!dataTransfer) return { files: [], hadDir: false };
  const items = Array.from(dataTransfer.items || []);
  const entries = items
    .map((it) => (typeof it.webkitGetAsEntry === "function" ? it.webkitGetAsEntry() : null))
    .filter(Boolean);

  let hadDir = false;
  if (entries.some((e) => e.isDirectory)) {
    hadDir = true;
    const collected = [];
    for (const ent of entries) {
      if (ent.isDirectory) {
        const dirFiles = await readDirectoryTopLevelFiles(ent);
        collected.push(...filterWebFolderFiles(dirFiles, refValue));
      } else if (ent.isFile) {
        try {
          const f = await new Promise((resolve, reject) => ent.file(resolve, reject));
          if (f && isSupportedInputOrArchiveName(f.name)) collected.push(f);
        } catch (_) {}
      }
    }
    collected.sort((a, b) => a.name.localeCompare(b.name));
    return { files: collected, hadDir };
  }

  const dropped = Array.from(dataTransfer.files || []);
  if (dropped.length > 1) {
    const filtered = dropped.filter((f) => isSupportedInputOrArchiveName(f.name));
    return { files: filtered.length ? filtered : dropped, hadDir: false };
  }
  return { files: dropped, hadDir: false };
}

function renderAlignmentProfileSetting() {
  if (els.settingsAlignmentProfile) {
    els.settingsAlignmentProfile.value = alignmentProfile;
    observeSelect(els.settingsAlignmentProfile);
    fitSelectOptions(els.settingsAlignmentProfile);
  }
}

function applyAlignmentProfile(nextProfile, { persist = true, notifyAdapter = false } = {}) {
  alignmentProfile = normalizeAlignmentProfile(nextProfile);
  if (persist) {
    writeSavedAlignmentProfile(alignmentProfile);
  }
  renderAlignmentProfileSetting();
  if (notifyAdapter && adapter && typeof adapter.onAlignmentProfileChange === "function") {
    adapter.onAlignmentProfileChange(alignmentProfile);
  }
}

function setSettingsOpen(open) {
  if (!adapter || !adapter.capabilities.settings) return;
  closeAllCustomSelects();
  settingsOpen = Boolean(open);
  if (els.mainView) els.mainView.hidden = settingsOpen;
  if (els.settingsView) els.settingsView.hidden = !settingsOpen;
  if (els.settingsBtn) {
    els.settingsBtn.classList.toggle("active", settingsOpen);
    els.settingsBtn.setAttribute("aria-pressed", String(settingsOpen));
  }
  if (settingsOpen) {
    if (els.settingsLanguage) fitSelectOptions(els.settingsLanguage);
    if (els.settingsAlignmentProfile) fitSelectOptions(els.settingsAlignmentProfile);
  }
}

function isDarkTheme() {
  return document.documentElement.classList.contains("dark");
}

function syncThemeToggleLabel(dark) {
  if (els.themeToggle) {
    els.themeToggle.title = dark ? i18n.t("topbar.switchToLight") : i18n.t("topbar.switchToDark");
    els.themeToggle.setAttribute("aria-label", els.themeToggle.title);
  }
}

function syncMaximizedState(isMax) {
  const max = Boolean(isMax);
  document.body.classList.toggle("is-maximized", max);
  if (els.winMaximize) {
    els.winMaximize.classList.toggle("is-maximized", max);
    const label = max ? i18n.t("topbar.restore") : i18n.t("topbar.maximize");
    els.winMaximize.title = label;
    els.winMaximize.setAttribute("aria-label", label);
  }
}

function applyTheme(dark, { transition = false, persist = false } = {}) {
  const root = document.documentElement;
  if (transition) {
    root.classList.add("theme-transitioning");
    if (themeTransitionTimer) clearTimeout(themeTransitionTimer);
    themeTransitionTimer = setTimeout(() => {
      root.classList.remove("theme-transitioning");
      themeTransitionTimer = null;
    }, 250);
  }
  root.classList.toggle("dark", dark);
  syncThemeToggleLabel(dark);
  if (persist) {
    try {
      localStorage.setItem("ffs_theme", dark ? "dark" : "light");
    } catch (_) {}
  }
  if (adapter && typeof adapter.onThemeChange === "function") {
    adapter.onThemeChange(dark);
  }
}

syncThemeToggleLabel(isDarkTheme());
if (els.themeToggle) {
  els.themeToggle.addEventListener("click", () => {
    applyTheme(!isDarkTheme(), { transition: true, persist: true });
  });
}

function setProgressBar(container, fill, pctEl, { active, fraction }) {
  if (!active) {
    container.hidden = true;
    container.classList.remove("indeterminate");
    container.removeAttribute("aria-valuenow");
    return;
  }
  container.hidden = false;
  if (Number.isFinite(fraction) && fraction >= 0 && fraction <= 1) {
    const pct = Math.round(fraction * 100);
    container.classList.remove("indeterminate");
    fill.style.width = pct + "%";
    pctEl.textContent = pct + "%";
    container.setAttribute("aria-valuenow", String(pct));
  } else {
    container.classList.add("indeterminate");
    fill.style.width = "";
    pctEl.textContent = "";
    container.removeAttribute("aria-valuenow");
  }
}

function hasSelection(val) {
  if (!val) return false;
  if (typeof val === "string") return val.trim().length > 0;
  return true;
}

function refreshSyncBtn() {
  els.syncBtn.disabled =
    busy || !runtimeReady || !hasSelection(selectedRef) || selectedIns.length === 0;
  if (!busy) {
    els.syncBtn.textContent = i18n.t("form.syncBtn");
  } else if (lastSyncProgress && lastSyncProgress.messageKey) {
    els.syncBtn.textContent = i18n.t(lastSyncProgress.messageKey, lastSyncProgress.messageParams);
  }
}

function formatResultSummary(res) {
  const off = Number(res.offset_seconds);
  const scale = Number(res.framerate_scale_factor);
  const segs = Array.isArray(res.split_segments) ? res.split_segments : null;
  const dropped = Number(res.dropped_cues || 0);
  const clipped = Number(res.clipped_cues || 0);
  const dropNote = dropped > 0 ? i18n.t("result.orphanCues", { count: dropped }) : "";
  const clipNote = clipped > 0 ? i18n.t("result.clippedCues", { count: clipped }) : "";
  let splitNote = "";
  if (segs) {
    if (segs.length <= 1) {
      splitNote = i18n.t("result.noBreaks", { dropNote });
    } else {
      const fmt = (v) => `${v >= 0 ? "+" : ""}${Number(v).toFixed(2)}s`;
      const chain = segs.slice(0, 4).map((x) => fmt(x.offset_seconds)).join(" → ");
      const tail = segs.length > 4 ? i18n.t("result.segmentsTail", { count: segs.length }) : "";
      splitNote = i18n.t("result.breaksFixed", {
        count: segs.length - 1,
        chain,
        tail,
        dropNote,
      });
    }
  } else if (dropped > 0) {
    splitNote = dropNote;
  }
  const offsetStr = `${off >= 0 ? "+" : ""}${off.toFixed(3)}`;
  return (
    i18n.t("result.offset", { offset: offsetStr }) +
    (Math.abs(scale - 1) > 1e-6 ? i18n.t("result.framerate", { scale: scale.toFixed(3) }) : "") +
    splitNote +
    clipNote
  );
}

// Integrated custom dropdowns (replaces raw OS-native <select> popups while
// keeping the underlying <select> as the single source of truth). Preserves
// trailing bracketed metadata (e.g. "[eac3]", "[PGS timings]") on narrow widths.
const _measureCanvas = document.createElement("canvas");
const _measureCtx = _measureCanvas.getContext("2d");
const _TRAILING_BRACKETS_RE = /^(.*?)(\s*(?:\[[^\]]+\]\s*)+)$/;
const _customSelects = new WeakMap();
const _allCustomSelectInstances = new Set();
let _openCustomSelect = null;
let _selectResizeObserver = null;
let _customSelectGlobalListenersWired = false;

function truncateSelectLabel(fullText, maxPx, measure) {
  const text = String(fullText || "");
  if (!text || !Number.isFinite(maxPx) || maxPx <= 0 || measure(text) <= maxPx) {
    return text;
  }
  const ellipsis = "…";
  const match = _TRAILING_BRACKETS_RE.exec(text);
  if (match && match[1].trim().length > 0) {
    const head = match[1].trimEnd();
    const brackets = " " + match[2].trim();
    const tail = ellipsis + brackets;
    if (measure(tail) <= maxPx) {
      let lo = 0;
      let hi = head.length;
      while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        const prefix = head.slice(0, mid).replace(/[\s·,:;(\[-]+$/, "");
        if (measure(prefix + tail) <= maxPx) {
          lo = mid;
        } else {
          hi = mid - 1;
        }
      }
      const prefix = head.slice(0, lo).replace(/[\s·,:;(\[-]+$/, "");
      return prefix ? prefix + tail : brackets.trim();
    }
    const trimmedBrackets = brackets.trim();
    if (measure(trimmedBrackets) <= maxPx) {
      return trimmedBrackets;
    }
  }

  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    const prefix = text.slice(0, mid).replace(/[\s·,:;(\[-]+$/, "");
    if (measure(prefix + ellipsis) <= maxPx) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const prefix = text.slice(0, lo).replace(/[\s·,:;(\[-]+$/, "");
  return prefix ? prefix + ellipsis : ellipsis;
}

function measureSelectText(str, font) {
  if (_measureCtx) {
    _measureCtx.font = font;
    return _measureCtx.measureText(str).width;
  }
  return str.length * 6.5;
}

function createSelectChevronSvg() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "custom-select-chevron");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2.2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", "m6 9 6 6 6-6");
  svg.appendChild(path);
  return svg;
}

function createSelectCheckSvg() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2.4");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", "M20 6 9 17l-5-5");
  svg.appendChild(path);
  return svg;
}

function closeCustomSelect(inst, { restoreFocus = false } = {}) {
  if (!inst || !inst.isOpen) return;
  inst.isOpen = false;
  inst.activeIndex = -1;
  inst.wrap.classList.remove("is-open", "is-dropup");
  inst.trigger.setAttribute("aria-expanded", "false");
  inst.menu.hidden = true;
  if (_openCustomSelect === inst) {
    _openCustomSelect = null;
  }
  if (restoreFocus) {
    inst.trigger.focus();
  }
}

function closeAllCustomSelects() {
  if (_openCustomSelect) {
    closeCustomSelect(_openCustomSelect);
  }
}

function ensureCustomSelectGlobalListeners() {
  if (_customSelectGlobalListenersWired) return;
  _customSelectGlobalListenersWired = true;
  document.addEventListener("mousedown", (e) => {
    if (_openCustomSelect && !_openCustomSelect.wrap.contains(e.target)) {
      closeCustomSelect(_openCustomSelect);
    }
  });
  window.addEventListener("blur", () => closeAllCustomSelects());
  window.addEventListener("resize", () => {
    if (_openCustomSelect) positionCustomSelectMenu(_openCustomSelect);
  });
}

function positionCustomSelectMenu(inst) {
  if (!inst || !inst.isOpen) return;
  const rect = inst.trigger.getBoundingClientRect();
  const estHeight = Math.min(230, inst.menu.scrollHeight || 160);
  const spaceBelow = window.innerHeight - rect.bottom;
  const spaceAbove = rect.top;
  const dropUp = spaceBelow < estHeight + 14 && spaceAbove > spaceBelow;
  inst.wrap.classList.toggle("is-dropup", dropUp);
}

function setCustomSelectActiveIndex(inst, idx) {
  inst.activeIndex = idx;
  const rows = inst.menu.querySelectorAll(".custom-select-option");
  rows.forEach((row, i) => {
    row.classList.toggle("is-active", i === idx);
    if (i === idx && inst.menu.classList.contains("is-scrollable")) {
      row.scrollIntoView({ block: "nearest" });
    }
  });
}

function nextEnabledOptionIndex(selectEl, startIdx, step) {
  const opts = selectEl.options;
  const len = opts.length;
  if (!len) return -1;
  let i = startIdx;
  for (let tries = 0; tries < len; tries += 1) {
    i = (i + step + len) % len;
    if (!opts[i].disabled) return i;
  }
  return -1;
}

function openCustomSelect(inst) {
  if (!inst || inst.isOpen || inst.select.disabled || inst.select.options.length === 0) return;
  if (_openCustomSelect && _openCustomSelect !== inst) {
    closeCustomSelect(_openCustomSelect);
  }
  renderCustomSelectOptions(inst);
  inst.isOpen = true;
  inst.menu.hidden = false;
  inst.wrap.classList.add("is-open");
  inst.trigger.setAttribute("aria-expanded", "true");
  _openCustomSelect = inst;
  positionCustomSelectMenu(inst);
  const selIdx = inst.select.selectedIndex >= 0 ? inst.select.selectedIndex : 0;
  setCustomSelectActiveIndex(inst, selIdx);
}

function selectCustomOptionByIndex(inst, idx) {
  const opt = inst.select.options[idx];
  if (!opt || opt.disabled) return;
  const changed = inst.select.selectedIndex !== idx;
  inst.select.selectedIndex = idx;
  renderCustomSelectTrigger(inst);
  closeCustomSelect(inst, { restoreFocus: true });
  if (changed) {
    inst.select.dispatchEvent(new Event("change", { bubbles: true }));
  }
}

function renderCustomSelectTrigger(inst) {
  const { select, trigger, valueEl } = inst;
  const ariaLabel = select.getAttribute("aria-label");
  if (ariaLabel) {
    trigger.setAttribute("aria-label", ariaLabel);
  }
  const opt = select.options[select.selectedIndex];
  const full = opt ? (opt.dataset.fullLabel || opt.textContent || "") : "";
  const width = trigger.clientWidth;
  if (width > 0 && full) {
    const cs = window.getComputedStyle(trigger);
    const padL = parseFloat(cs.paddingLeft) || 10;
    const padR = parseFloat(cs.paddingRight) || 10;
    const maxPx = Math.max(40, width - padL - padR - 22);
    const font = `${cs.fontWeight || "500"} ${cs.fontSize || "12px"} ${cs.fontFamily || "sans-serif"}`;
    const fitted = truncateSelectLabel(full, maxPx, (s) => measureSelectText(s, font));
    valueEl.textContent = fitted;
    trigger.title = fitted !== full ? full : "";
  } else {
    valueEl.textContent = full;
    trigger.title = "";
  }
}

function renderCustomSelectOptions(inst) {
  const { select, menu, trigger } = inst;
  menu.innerHTML = "";
  const width = trigger.clientWidth;
  const cs = window.getComputedStyle(trigger);
  const optionFont = `500 ${cs.fontSize || "12px"} ${cs.fontFamily || "sans-serif"}`;
  const opts = Array.from(select.options);
  const hasTips = opts.some((o) => Boolean(o.dataset.tip));
  const scrollable = opts.length > 7 && !hasTips;
  menu.classList.toggle("is-scrollable", scrollable);

  opts.forEach((opt, idx) => {
    if (opt.dataset.fullLabel === undefined) {
      opt.dataset.fullLabel = opt.textContent || "";
    }
    const full = opt.dataset.fullLabel;
    const tipText = opt.dataset.tip || "";
    const row = document.createElement("div");
    row.className = "custom-select-option";
    row.setAttribute("role", "option");
    row.dataset.index = String(idx);
    const isSelected = idx === select.selectedIndex;
    row.setAttribute("aria-selected", String(isSelected));
    if (isSelected) row.classList.add("is-selected");
    if (opt.disabled) {
      row.classList.add("is-disabled");
      row.setAttribute("aria-disabled", "true");
    }

    const check = document.createElement("span");
    check.className = "custom-select-check";
    check.setAttribute("aria-hidden", "true");
    check.appendChild(createSelectCheckSvg());

    const label = document.createElement("span");
    label.className = "custom-select-option-label";
    let display = full;
    if (width > 0 && full) {
      const reserved = 44 + (tipText ? 22 : 0) + (scrollable ? 8 : 0);
      const maxPx = Math.max(40, width - reserved);
      display = truncateSelectLabel(full, maxPx, (s) => measureSelectText(s, optionFont));
    }
    label.textContent = display;
    if (display !== full) {
      row.title = full;
    }

    row.appendChild(check);
    row.appendChild(label);

    if (tipText) {
      const tip = createHelpTip(tipText);
      tip.classList.remove("tip-align-start");
      tip.addEventListener("mousedown", (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
      tip.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (document.activeElement === tip) {
          tip.blur();
        } else {
          tip.focus();
        }
      });
      row.appendChild(tip);
    }

    row.addEventListener("mouseenter", () => {
      if (
        document.activeElement &&
        menu.contains(document.activeElement) &&
        !row.contains(document.activeElement)
      ) {
        document.activeElement.blur();
      }
      if (!opt.disabled) setCustomSelectActiveIndex(inst, idx);
    });
    row.addEventListener("mousedown", (e) => {
      // Keep focus clean and select immediately on primary mouse click.
      if (e.button !== 0) return;
      e.preventDefault();
    });
    row.addEventListener("click", () => {
      if (!opt.disabled) selectCustomOptionByIndex(inst, idx);
    });

    menu.appendChild(row);
  });
}

function ensureCustomSelect(selectEl) {
  if (!selectEl) return null;
  if (_customSelects.has(selectEl)) return _customSelects.get(selectEl);
  ensureCustomSelectGlobalListeners();

  const wrap = document.createElement("div");
  wrap.className = "custom-select";
  if (selectEl.id) {
    wrap.dataset.for = selectEl.id;
  }

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "custom-select-trigger";
  trigger.setAttribute("role", "combobox");
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");

  const valueEl = document.createElement("span");
  valueEl.className = "custom-select-value";

  trigger.appendChild(valueEl);
  trigger.appendChild(createSelectChevronSvg());

  const menu = document.createElement("div");
  menu.className = "custom-select-menu";
  menu.setAttribute("role", "listbox");
  menu.hidden = true;

  const parent = selectEl.parentNode;
  if (parent) {
    parent.insertBefore(wrap, selectEl);
  }
  selectEl.classList.add("custom-select-native");
  selectEl.tabIndex = -1;
  selectEl.setAttribute("aria-hidden", "true");
  wrap.appendChild(selectEl);
  wrap.appendChild(trigger);
  wrap.appendChild(menu);

  const inst = {
    select: selectEl,
    wrap,
    trigger,
    valueEl,
    menu,
    isOpen: false,
    activeIndex: -1,
  };
  _customSelects.set(selectEl, inst);
  _allCustomSelectInstances.add(inst);

  trigger.addEventListener("click", () => {
    if (inst.isOpen) {
      closeCustomSelect(inst);
    } else {
      openCustomSelect(inst);
    }
  });

  trigger.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!inst.isOpen) {
        openCustomSelect(inst);
      } else {
        const next = nextEnabledOptionIndex(
          selectEl,
          inst.activeIndex >= 0 ? inst.activeIndex : selectEl.selectedIndex,
          1,
        );
        if (next >= 0) setCustomSelectActiveIndex(inst, next);
      }
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (!inst.isOpen) {
        openCustomSelect(inst);
      } else {
        const prev = nextEnabledOptionIndex(
          selectEl,
          inst.activeIndex >= 0 ? inst.activeIndex : selectEl.selectedIndex,
          -1,
        );
        if (prev >= 0) setCustomSelectActiveIndex(inst, prev);
      }
    } else if (e.key === "Home" && inst.isOpen) {
      e.preventDefault();
      const first = nextEnabledOptionIndex(selectEl, -1, 1);
      if (first >= 0) setCustomSelectActiveIndex(inst, first);
    } else if (e.key === "End" && inst.isOpen) {
      e.preventDefault();
      const last = nextEnabledOptionIndex(selectEl, 0, -1);
      if (last >= 0) setCustomSelectActiveIndex(inst, last);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (!inst.isOpen) {
        openCustomSelect(inst);
      } else if (inst.activeIndex >= 0) {
        selectCustomOptionByIndex(inst, inst.activeIndex);
      }
    } else if (e.key === "Escape" && inst.isOpen) {
      e.preventDefault();
      e.stopPropagation();
      closeCustomSelect(inst, { restoreFocus: true });
    } else if (e.key === "Tab" && inst.isOpen) {
      closeCustomSelect(inst);
    }
  });

  selectEl.addEventListener("change", () => {
    renderCustomSelectTrigger(inst);
    if (inst.isOpen) renderCustomSelectOptions(inst);
  });

  if (typeof ResizeObserver === "function") {
    if (!_selectResizeObserver) {
      _selectResizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          for (const candidate of _allCustomSelectInstances) {
            if (candidate.trigger === entry.target) {
              renderCustomSelectTrigger(candidate);
              if (candidate.isOpen) {
                renderCustomSelectOptions(candidate);
                positionCustomSelectMenu(candidate);
              }
              break;
            }
          }
        }
      });
    }
    _selectResizeObserver.observe(trigger);
  }

  renderCustomSelectTrigger(inst);
  return inst;
}

function fitSelectOptions(selectEl) {
  if (!selectEl) return;
  const inst = ensureCustomSelect(selectEl);
  if (!inst) return;
  renderCustomSelectTrigger(inst);
  if (inst.isOpen) {
    renderCustomSelectOptions(inst);
    positionCustomSelectMenu(inst);
  }
}

function observeSelect(selectEl) {
  ensureCustomSelect(selectEl);
}

function fitAllSelectOptions() {
  for (const sel of document.querySelectorAll("select")) {
    observeSelect(sel);
    fitSelectOptions(sel);
  }
}

if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(() => fitAllSelectOptions()).catch(() => {});
}

function formatStreamLabel(item) {
  if (!item || !item.index || !item.spec) {
    return (item && item.label) || "";
  }
  const metaParts = [item.lang, item.title].filter(Boolean);
  const meta = metaParts.length ? metaParts.join(" · ") : i18n.t("form.streamDefaultMeta");
  const flagParts = [];
  if (item.forced) flagParts.push(i18n.t("form.streamFlagForced"));
  if (item.comment) flagParts.push(i18n.t("form.streamFlagCommentary"));
  const flags = flagParts.length ? ` (${flagParts.join(", ")})` : "";

  if (item.kind === "audio") {
    return i18n.t("form.streamAudio", {
      index: item.index,
      spec: item.spec,
      meta,
      flags,
      codec: item.codec || "unknown",
    });
  }

  let note;
  if (item.note_kind === "pgs") {
    note = i18n.t("form.streamNotePgs");
  } else if (item.note_kind === "vobsub") {
    note = i18n.t("form.streamNoteVobSub");
  } else if (item.note_kind === "dvb") {
    note = i18n.t("form.streamNoteDvb");
  } else if (item.note_kind === "image" || item.is_bitmap) {
    note = i18n.t("form.streamNoteImage", { codec: item.codec || "unknown" });
  } else {
    note = i18n.t("form.streamNoteText", { codec: item.codec || "unknown" });
  }
  return i18n.t("form.streamSubtitle", {
    index: item.index,
    spec: item.spec,
    meta,
    flags,
    note,
  });
}

function resetStreamSelectToAuto() {
  els.refStream.innerHTML = "";
  const autoOpt = document.createElement("option");
  autoOpt.value = "";
  const autoLabel = i18n.t("form.streamAutoOption");
  autoOpt.dataset.fullLabel = autoLabel;
  autoOpt.textContent = autoLabel;
  els.refStream.appendChild(autoOpt);
  els.refStream.value = "";
  els.refStream.title = "";
  els.streamHint.textContent = i18n.t("form.streamHintAuto");
  fitSelectOptions(els.refStream);
}

function renderStreamProbe(res, { preserveSelection = false } = {}) {
  const prevVal = preserveSelection ? els.refStream.value : "";
  resetStreamSelectToAuto();
  const list = (res && res.streams) || [];
  if (list.length <= 1) {
    els.streamWrap.hidden = true;
    return;
  }
  for (const item of list) {
    const opt = document.createElement("option");
    opt.value = item.value;
    const label = formatStreamLabel(item);
    opt.dataset.fullLabel = label;
    opt.textContent = label;
    opt.dataset.kind = item.kind || "";
    if (item.is_bitmap) opt.disabled = true;
    els.refStream.appendChild(opt);
  }
  if (preserveSelection && prevVal) {
    els.refStream.value = prevVal;
  }
  const notes = [];
  if (res.has_text_subs) notes.push(i18n.t("form.streamTextSubsAvailable"));
  if (res.has_pgs_subs) notes.push(i18n.t("form.streamPgsAvailable"));
  els.streamHint.textContent = notes.length
    ? i18n.t("form.streamTracksNotes", { count: list.length, notes: notes.join(" · ") })
    : i18n.t("form.streamTracksAudioVad", { count: list.length });
  els.streamWrap.hidden = false;
  observeSelect(els.refStream);
  fitSelectOptions(els.refStream);
}

async function refreshStreams(refValue) {
  const seq = ++streamProbeSeq;
  lastStreamProbe = null;
  resetStreamSelectToAuto();
  els.streamWrap.hidden = true;
  if (!adapter || !adapter.capabilities.streamProbe || !refValue) return;
  try {
    const res = await adapter.probeStreams(refValue);
    if (seq !== streamProbeSeq) return;
    lastStreamProbe = res;
    renderStreamProbe(res);
  } catch (_) {
    if (seq === streamProbeSeq) {
      lastStreamProbe = null;
      els.streamWrap.hidden = true;
    }
  }
}

function setReference(entry) {
  if (!entry) return;
  selectedRef = entry.value;
  els.refPath.value = entry.display;
  refreshStreams(entry.value);
  refreshSyncBtn();
}

function createRemoveIconSvg() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2.2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  const p1 = document.createElementNS(ns, "path");
  p1.setAttribute("d", "M18 6 6 18");
  const p2 = document.createElementNS(ns, "path");
  p2.setAttribute("d", "m6 6 12 12");
  svg.appendChild(p1);
  svg.appendChild(p2);
  return svg;
}

function setBrowseSplitMenuOpen(open) {
  if (!els.browseInSplit || !els.browseInMenu || !els.browseInMenuBtn) return;
  const isOpen = Boolean(open);
  if (isOpen) {
    closeAllCustomSelects();
  }
  els.browseInSplit.classList.toggle("is-open", isOpen);
  els.browseInMenu.hidden = !isOpen;
  els.browseInMenuBtn.setAttribute("aria-expanded", String(isOpen));
}

function updateBrowseInSplitUi() {
  const hasFiles = selectedIns.length > 0;
  if (els.browseIn) {
    els.browseIn.textContent = hasFiles ? i18n.t("form.addMore") : i18n.t("form.browse");
  }
  if (els.browseInOptAdd) {
    els.browseInOptAdd.hidden = !hasFiles;
    els.browseInOptAdd.classList.toggle("is-default", hasFiles);
  }
  if (els.browseInOptFiles) {
    els.browseInOptFiles.classList.toggle("is-default", !hasFiles);
  }
  if (els.browseInOptFilesLabel) {
    els.browseInOptFilesLabel.textContent = hasFiles
      ? i18n.t("form.menuReplaceFiles")
      : i18n.t("form.menuSelectFiles");
  }
}

function renderInputSubtitlesUI({ forceAutoOut = false } = {}) {
  const count = selectedIns.length;
  const editable = Boolean(adapter && adapter.capabilities && adapter.capabilities.editablePaths);
  const supportsOut = Boolean(adapter && adapter.capabilities && adapter.capabilities.outputPath);

  updateBrowseInSplitUi();

  if (count === 0) {
    els.inPath.value = "";
    els.inPath.readOnly = !editable;
    if (els.multiInPanel) els.multiInPanel.hidden = true;
    if (els.outField) els.outField.hidden = !supportsOut;
  } else if (count === 1) {
    const single = selectedIns[0];
    els.inPath.value = single.display;
    els.inPath.readOnly = !editable || Boolean(single.default_output_path);
    if (els.multiInPanel) els.multiInPanel.hidden = true;
    if (els.outField) els.outField.hidden = !supportsOut;
    if (
      supportsOut &&
      typeof single.value === "string" &&
      (forceAutoOut || !els.outPath.value.trim() || els.outPath.dataset.auto === "1")
    ) {
      const suggested =
        single.default_output_path ||
        (adapter.suggestOutputPath ? adapter.suggestOutputPath(single.value) : "");
      els.outPath.value = suggested;
      els.outPath.dataset.auto = "1";
    }
  } else {
    els.inPath.value = i18n.t("form.multiSelectedSummary", { count });
    els.inPath.readOnly = true;
    if (els.outField) els.outField.hidden = true;
    if (els.multiInPanel) els.multiInPanel.hidden = false;
    if (els.multiInCount) {
      els.multiInCount.textContent = i18n.t("form.multiQueueCount", { count });
    }
    if (els.multiInList) {
      els.multiInList.innerHTML = "";
      selectedIns.forEach((item, idx) => {
        const li = document.createElement("li");
        li.className = "multi-in-item";

        const nameSpan = document.createElement("span");
        nameSpan.className = "multi-in-name";
        nameSpan.textContent = item.name;
        nameSpan.title = typeof item.value === "string" ? item.value : item.display;

        const rmBtn = document.createElement("button");
        rmBtn.type = "button";
        rmBtn.className = "multi-in-remove";
        const rmLabel = i18n.t("form.removeItem", { name: item.name });
        rmBtn.title = rmLabel;
        rmBtn.setAttribute("aria-label", rmLabel);
        rmBtn.appendChild(createRemoveIconSvg());
        rmBtn.addEventListener("click", () => {
          selectedIns.splice(idx, 1);
          renderInputSubtitlesUI({ forceAutoOut: selectedIns.length === 1 });
        });

        li.appendChild(nameSpan);
        li.appendChild(rmBtn);
        els.multiInList.appendChild(li);
      });
    }
  }
  refreshSyncBtn();
}

function setInputSubtitles(entries, { append = false, forceAutoOut = false } = {}) {
  const list = Array.isArray(entries) ? entries : (entries ? [entries] : []);
  const normalized = list.map(normalizeSubtitleEntry).filter(Boolean);
  if (!normalized.length) {
    if (!append) {
      selectedIns = [];
      renderInputSubtitlesUI({ forceAutoOut });
    }
    return;
  }
  if (append && selectedIns.length > 0) {
    const seen = new Set(selectedIns.map((x) => x.key));
    for (const item of normalized) {
      if (!seen.has(item.key)) {
        seen.add(item.key);
        selectedIns.push(item);
      }
    }
  } else {
    const seen = new Set();
    selectedIns = [];
    for (const item of normalized) {
      if (!seen.has(item.key)) {
        seen.add(item.key);
        selectedIns.push(item);
      }
    }
  }
  renderInputSubtitlesUI({ forceAutoOut });
}

function setInputSubtitle(entry, { forceAutoOut = false, append = false } = {}) {
  if (!entry) return;
  setInputSubtitles(Array.isArray(entry) ? entry : [entry], { append, forceAutoOut });
}

function showInputError(message) {
  if (!message) return;
  renderResult({ ok: false, error: message }, { scroll: false });
}

function renderFfmpegDir(info) {
  if (!els.settingsFfDir) return;
  let dir = (info && info.resource_dir) || "";
  if (!dir && info && info.ffmpeg_path) {
    const sep = Math.max(info.ffmpeg_path.lastIndexOf("\\"), info.ffmpeg_path.lastIndexOf("/"));
    dir = sep > 0 ? info.ffmpeg_path.slice(0, sep) : info.ffmpeg_path;
  }
  if (!dir) dir = (info && (info.custom_path || info.portable_dir)) || i18n.t("settings.ffmpegAutoDetect");
  els.settingsFfDir.textContent = dir;
  els.settingsFfDir.title = (info && info.ffmpeg_path) || dir;
}

function applyRuntimeStatus(update) {
  if (!update) return;
  const useTopbar = Boolean(adapter && adapter.capabilities.topbarStatus);
  if (update.visible !== undefined && els.runtimeCard) {
    els.runtimeCard.hidden = useTopbar || !update.visible;
  }
  if (update.level) {
    const levelClass = update.level === "muted" ? "" : ` ${update.level}`;
    els.runtimeDot.className = `dot${levelClass}`;
    if (els.runtimePill) {
      els.runtimePill.className = `status-pill${levelClass}`;
    }
    if (els.topbarStatusDot) els.topbarStatusDot.className = `dot${levelClass}`;
    if (els.topbarStatusPill) els.topbarStatusPill.className = `status-pill topbar-status${levelClass}`;
    if (els.settingsFfDot) els.settingsFfDot.className = `dot${levelClass}`;
    if (els.settingsFfPill) els.settingsFfPill.className = `status-pill${levelClass}`;
  }
  if (update.text !== undefined) {
    els.status.textContent = update.text;
    if (els.topbarStatusPill) els.topbarStatusPill.title = update.text;
    if (els.runtimeStatusText) els.runtimeStatusText.textContent = update.text;
    if (els.settingsFfStatus) els.settingsFfStatus.textContent = update.text;
  }
  if (update.badgeTitle !== undefined) {
    els.envBadge.title = update.badgeTitle;
  }
  if (adapter && adapter.capabilities.runtimeAction) {
    if (update.actionLabel !== undefined) {
      els.runtimeActionBtn.textContent = update.actionLabel;
      els.runtimeActionBtn.hidden = !update.actionLabel;
      if (els.settingsFfDownloadLabel) {
        els.settingsFfDownloadLabel.textContent =
          update.actionLabel || i18n.t("ffmpeg.downloadLatestBtn");
      }
    }
    if (update.actionDisabled !== undefined) {
      els.runtimeActionBtn.disabled = Boolean(update.actionDisabled);
      if (els.settingsFfDownload) {
        els.settingsFfDownload.disabled = Boolean(update.actionDisabled);
      }
    }
  }
  if (update.ffmpegInfo !== undefined) {
    lastFfmpegInfo = update.ffmpegInfo;
    if (update.ffmpegInfo) {
      renderFfmpegDir(update.ffmpegInfo);
    }
  }
  if (update.ready !== undefined) {
    runtimeReady = Boolean(update.ready);
    refreshSyncBtn();
  }
}

function scrollToResultTop() {
  const mainEl = document.querySelector("main");
  if (mainEl && mainEl.scrollTop > 0) {
    mainEl.scrollTo({ top: 0, behavior: "smooth" });
  }
  if (window.scrollY > 0) {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

const QUALITY_BADGE_SPECS = [
  {
    key: "sync",
    labelKey: "quality.syncLabel",
    metricKeys: [
      "temporal_consistency",
      "overlap_lift",
      "peak_confidence",
      "sync_integrity",
    ],
  },
  {
    key: "subtitle",
    labelKey: "quality.subtitleLabel",
    metricKeys: [
      "timeline_coverage",
      "reading_speed",
      "timing_hygiene",
      "text_integrity",
    ],
  },
];

function scoreTierClass(score) {
  const n = Number(score);
  if (!Number.isFinite(n)) return "tier-warn";
  if (n >= 75) return "tier-ok";
  if (n >= 45) return "tier-warn";
  return "tier-err";
}

function clampScore(value) {
  const n = Math.round(Number(value));
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, n));
}

function createHelpTip(text) {
  const tip = document.createElement("span");
  tip.className = "help-tip tip-align-start";
  tip.tabIndex = 0;
  tip.setAttribute("role", "note");
  tip.setAttribute("aria-label", text);

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");

  const circle = document.createElementNS(ns, "circle");
  circle.setAttribute("cx", "12");
  circle.setAttribute("cy", "12");
  circle.setAttribute("r", "10");
  svg.appendChild(circle);

  const qPath = document.createElementNS(ns, "path");
  qPath.setAttribute("d", "M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3");
  svg.appendChild(qPath);

  const dotPath = document.createElementNS(ns, "path");
  dotPath.setAttribute("d", "M12 17h.01");
  svg.appendChild(dotPath);

  const bubble = document.createElement("span");
  bubble.className = "help-tip-bubble";
  bubble.setAttribute("role", "tooltip");
  bubble.textContent = text;

  tip.appendChild(svg);
  tip.appendChild(bubble);
  return tip;
}

function createChevronSvg() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "quality-badge-chevron");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2.2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");

  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", "m6 9 6 6 6-6");
  svg.appendChild(path);
  return svg;
}

function groupHasLowMetric(spec, group) {
  const metricsObj = (group && group.metrics && typeof group.metrics === "object") ? group.metrics : {};
  for (const mKey of spec.metricKeys) {
    if (metricsObj[mKey] !== undefined && metricsObj[mKey] !== null) {
      if (clampScore(metricsObj[mKey]) < 50) return true;
    }
  }
  return false;
}

function hasQualityWarning(quality) {
  if (!quality || typeof quality !== "object") return false;
  for (const spec of QUALITY_BADGE_SPECS) {
    const group = quality[spec.key];
    if (!group || typeof group !== "object") continue;
    if (Number.isFinite(Number(group.score)) && clampScore(group.score) < 75) {
      return true;
    }
    if (groupHasLowMetric(spec, group)) {
      return true;
    }
  }
  return false;
}

function renderQualityBadgesInto(
  container,
  quality,
  { getOpenPanel, setOpenPanel, idPrefix = "quality-panel" } = {},
) {
  if (!container) return;
  if (!quality || typeof quality !== "object" || (!quality.sync && !quality.subtitle)) {
    container.innerHTML = "";
    container.hidden = true;
    return;
  }

  container.innerHTML = "";
  const badgesRow = document.createElement("div");
  badgesRow.className = "quality-badges";
  const panelsWrap = document.createElement("div");
  panelsWrap.className = "quality-panels";

  const items = [];

  function syncAccordionState() {
    const currentOpen = getOpenPanel ? getOpenPanel() : null;
    let anyOpen = false;
    for (const item of items) {
      const active = currentOpen === item.key;
      item.btn.setAttribute("aria-expanded", String(active));
      item.panel.hidden = !active;
      if (active) anyOpen = true;
    }
    panelsWrap.hidden = !anyOpen;
  }

  const currentOpen = getOpenPanel ? getOpenPanel() : null;
  for (const spec of QUALITY_BADGE_SPECS) {
    const group = quality[spec.key];
    if (!group || typeof group !== "object" || !Number.isFinite(Number(group.score))) {
      continue;
    }
    const score = clampScore(group.score);
    const hasLowMetric = groupHasLowMetric(spec, group);
    const tier = hasLowMetric && score >= 75 ? "tier-warn" : scoreTierClass(score);
    const isOpen = currentOpen === spec.key;
    const panelId = `${idPrefix}-${spec.key}`;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `quality-badge-btn ${tier}`;
    btn.setAttribute("aria-expanded", String(isOpen));
    btn.setAttribute("aria-controls", panelId);

    const dot = document.createElement("span");
    dot.className = "quality-badge-dot";
    dot.setAttribute("aria-hidden", "true");

    const label = document.createElement("span");
    label.textContent = i18n.t(spec.labelKey);

    const scoreEl = document.createElement("span");
    scoreEl.className = "quality-badge-score";
    scoreEl.textContent = `${score}/100`;

    btn.appendChild(dot);
    btn.appendChild(label);
    btn.appendChild(scoreEl);
    btn.appendChild(createChevronSvg());

    const panel = document.createElement("div");
    panel.id = panelId;
    panel.className = "quality-panel";
    panel.hidden = !isOpen;

    const head = document.createElement("div");
    head.className = "quality-panel-head";

    const headLeadSpacer = document.createElement("span");
    headLeadSpacer.setAttribute("aria-hidden", "true");

    const ruler = document.createElement("div");
    ruler.className = "quality-ruler";
    ruler.setAttribute("aria-hidden", "true");
    for (const mark of ["0", "25", "50", "75", "100"]) {
      const m = document.createElement("span");
      m.className = `m${mark}`;
      m.textContent = mark;
      ruler.appendChild(m);
    }

    const headSpacer = document.createElement("span");
    headSpacer.setAttribute("aria-hidden", "true");

    head.appendChild(headLeadSpacer);
    head.appendChild(ruler);
    head.appendChild(headSpacer);
    panel.appendChild(head);

    const list = document.createElement("div");
    list.className = "quality-metric-list";

    const isSubRef = quality.reference_kind === "subtitles";
    const metricsObj = (group.metrics && typeof group.metrics === "object") ? group.metrics : {};
    for (const mKey of spec.metricKeys) {
      if (metricsObj[mKey] === undefined || metricsObj[mKey] === null) continue;
      const mScore = clampScore(metricsObj[mKey]);
      const mTier = scoreTierClass(mScore);
      const i18nMetricKey =
        mKey === "overlap_lift" && isSubRef ? "overlap_lift_subtitles" : mKey;

      const row = document.createElement("div");
      row.className = "quality-metric-row";

      const labelWrap = document.createElement("div");
      labelWrap.className = "quality-metric-label";

      const nameSpan = document.createElement("span");
      nameSpan.className = "quality-metric-name";
      const metricName = i18n.t(`quality.metric.${i18nMetricKey}`);
      nameSpan.textContent = metricName;
      nameSpan.title = metricName;

      labelWrap.appendChild(nameSpan);
      labelWrap.appendChild(createHelpTip(i18n.t(`quality.help.${i18nMetricKey}`)));

      const axis = document.createElement("div");
      axis.className = `quality-axis ${mTier}`;
      axis.setAttribute("role", "meter");
      axis.setAttribute("aria-label", metricName);
      axis.setAttribute("aria-valuemin", "0");
      axis.setAttribute("aria-valuemax", "100");
      axis.setAttribute("aria-valuenow", String(mScore));

      const track = document.createElement("div");
      track.className = "quality-axis-track";

      for (const tickCls of ["t25", "t50", "t75"]) {
        const tick = document.createElement("span");
        tick.className = `quality-axis-tick ${tickCls}`;
        track.appendChild(tick);
      }

      const fill = document.createElement("div");
      fill.className = "quality-axis-fill";
      fill.style.width = `${mScore}%`;
      track.appendChild(fill);

      const node = document.createElement("span");
      node.className = "quality-axis-node";
      node.style.left = `${mScore}%`;
      track.appendChild(node);

      axis.appendChild(track);

      const val = document.createElement("span");
      val.className = `quality-metric-val ${mTier}`;
      val.textContent = String(mScore);

      row.appendChild(labelWrap);
      row.appendChild(axis);
      row.appendChild(val);
      list.appendChild(row);
    }

    panel.appendChild(list);

    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const next = (getOpenPanel ? getOpenPanel() : null) === spec.key ? null : spec.key;
      if (setOpenPanel) setOpenPanel(next);
      syncAccordionState();
    });

    items.push({ key: spec.key, btn, panel });
    badgesRow.appendChild(btn);
    panelsWrap.appendChild(panel);
  }

  if (items.length === 0) {
    container.hidden = true;
    return;
  }

  syncAccordionState();
  container.appendChild(badgesRow);
  container.appendChild(panelsWrap);
  container.hidden = false;
}

function renderQualityBadges(quality, { resetOpen = false } = {}) {
  if (resetOpen) {
    openQualityPanel = null;
  }
  renderQualityBadgesInto(els.resultQuality, quality, {
    getOpenPanel: () => openQualityPanel,
    setOpenPanel: (next) => {
      openQualityPanel = next;
    },
    idPrefix: "quality-panel",
  });
}

function createDownloadIconSvg() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(ns, "path");
  p.setAttribute("d", "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4");
  const poly = document.createElementNS(ns, "polyline");
  poly.setAttribute("points", "7 10 12 15 17 10");
  const line = document.createElementNS(ns, "line");
  line.setAttribute("x1", "12");
  line.setAttribute("x2", "12");
  line.setAttribute("y1", "15");
  line.setAttribute("y2", "3");
  svg.appendChild(p);
  svg.appendChild(poly);
  svg.appendChild(line);
  return svg;
}

function createFolderIconSvg() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(ns, "path");
  p.setAttribute(
    "d",
    "m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2",
  );
  svg.appendChild(p);
  return svg;
}

function renderRankingResult(res, { isNewResult = false, scroll = true } = {}) {
  if (isNewResult) {
    expandedRankingId = null;
    rankingQualityTabs.clear();
    savedCandidatePaths.clear();
  }
  if (els.resultSingle) els.resultSingle.hidden = true;
  if (els.resultRanking) els.resultRanking.hidden = false;

  const cands = Array.isArray(res.candidates) ? res.candidates : [];
  const winner = cands[0] || null;
  const anySavedCapable = cands.some((c) => c.can_save);
  const topWarn =
    winner &&
    (!winner.ok || hasQualityWarning(winner.quality) || clampScore(winner.overall_score) < 75);

  els.result.classList.toggle("error", !anySavedCapable);
  els.result.classList.toggle("warning", Boolean(anySavedCapable && topWarn));

  if (els.rankingTitle) {
    els.rankingTitle.textContent = i18n.t("result.rankingTitle", { count: cands.length });
  }
  if (els.rankingMeta) {
    const elapsed =
      typeof res.elapsed_seconds === "number" && Number.isFinite(res.elapsed_seconds)
        ? i18n.t("result.completedIn", { seconds: res.elapsed_seconds.toFixed(2) })
        : i18n.t("result.completed");
    els.rankingMeta.textContent = res.ref_cached
      ? `${elapsed} · ${i18n.t("result.reusedAudio")}`
      : elapsed;
  }

  const listEl = els.rankingList;
  if (!listEl) return;
  listEl.innerHTML = "";

  const isDesktopSave = Boolean(adapter && typeof adapter.saveCandidate === "function");

  for (const cand of cands) {
    const candId = String(cand.id || cand.rank);
    const hasErr = Boolean(cand.error);
    const isWinner = cand.rank === 1 && !hasErr && cand.can_save;
    const overall = clampScore(cand.overall_score);
    const overallTier = hasErr ? "tier-err" : scoreTierClass(overall);
    const isWinnerWarn = isWinner && (!cand.ok || overall < 75 || hasQualityWarning(cand.quality));
    const isExpanded = !hasErr && expandedRankingId === candId;

    const itemEl = document.createElement("div");
    itemEl.className = "ranking-item";
    if (isWinner) itemEl.classList.add("is-winner");
    if (isWinnerWarn) itemEl.classList.add("is-warn");
    if (hasErr) itemEl.classList.add("is-failed");
    if (isExpanded) itemEl.classList.add("is-expanded");

    const rowEl = document.createElement("div");
    rowEl.className = "ranking-row";
    if (!hasErr && cand.quality) {
      rowEl.setAttribute("role", "button");
      rowEl.tabIndex = 0;
      rowEl.setAttribute("aria-expanded", String(isExpanded));
      rowEl.title = i18n.t("result.expandDetails");
    }

    const rankEl = document.createElement("span");
    rankEl.className = "ranking-rank";
    rankEl.textContent = `#${cand.rank}`;

    const infoEl = document.createElement("div");
    infoEl.className = "ranking-info";

    const nameRow = document.createElement("div");
    nameRow.className = "ranking-name-row";

    const nameEl = document.createElement("span");
    nameEl.className = "ranking-name";
    nameEl.textContent = cand.input_name || basenameOf(cand.input_path) || "subtitle";
    nameEl.title = cand.input_path || cand.input_name || "";
    nameRow.appendChild(nameEl);

    if (isWinner && !isWinnerWarn) {
      const tag = document.createElement("span");
      tag.className = "ranking-tag";
      tag.textContent = i18n.t("result.bestMatch");
      nameRow.appendChild(tag);
    } else if (!hasErr && !cand.ok) {
      const tag = document.createElement("span");
      tag.className = "ranking-tag warn";
      tag.textContent = i18n.t("result.lowConfidence");
      nameRow.appendChild(tag);
    }

    const subMetaEl = document.createElement("div");
    subMetaEl.className = "ranking-submeta";
    if (hasErr) {
      subMetaEl.textContent = cand.error;
    } else if (cand.offset_seconds !== null && cand.offset_seconds !== undefined) {
      subMetaEl.textContent = formatResultSummary(cand);
    }

    infoEl.appendChild(nameRow);
    infoEl.appendChild(subMetaEl);

    const scoresEl = document.createElement("div");
    scoresEl.className = "ranking-scores";

    if (!hasErr && cand.quality) {
      const overallPill = document.createElement("span");
      overallPill.className = `score-pill ${overallTier}`;
      overallPill.textContent = `${overall}/100`;
      overallPill.title = i18n.t("result.overallLabel");
      scoresEl.appendChild(overallPill);
    }

    const actionsEl = document.createElement("div");
    actionsEl.className = "ranking-actions";

    if (cand.can_save) {
      if (isDesktopSave) {
        const savedPath = savedCandidatePaths.get(candId) || null;
        const saveBtn = document.createElement("button");
        saveBtn.type = "button";
        saveBtn.className = `${isWinner ? "primary" : "secondary"} ranking-save-btn`;
        saveBtn.appendChild(createDownloadIconSvg());
        const saveLabel = document.createElement("span");
        saveLabel.textContent = savedPath ? i18n.t("result.saved") : i18n.t("result.save");
        saveBtn.appendChild(saveLabel);

        const openBtn = document.createElement("button");
        openBtn.type = "button";
        openBtn.className = "secondary ranking-open-btn";
        openBtn.title = i18n.t("result.openFolder");
        openBtn.setAttribute("aria-label", i18n.t("result.openFolder"));
        openBtn.appendChild(createFolderIconSvg());
        openBtn.hidden = !savedPath;
        openBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const p = savedCandidatePaths.get(candId);
          if (p && adapter.openOutput) adapter.openOutput({ output_path: p });
        });

        saveBtn.addEventListener("click", async (e) => {
          e.stopPropagation();
          saveBtn.disabled = true;
          try {
            const saveRes = await adapter.saveCandidate(cand);
            if (saveRes && saveRes.saved && saveRes.path) {
              savedCandidatePaths.set(candId, saveRes.path);
              saveLabel.textContent = i18n.t("result.saved");
              openBtn.hidden = false;
            } else if (saveRes && saveRes.error) {
              alert(i18n.t("errors.saveFailed", { error: saveRes.error }));
            }
          } finally {
            saveBtn.disabled = false;
          }
        });

        actionsEl.appendChild(saveBtn);
        actionsEl.appendChild(openBtn);
      } else if (cand.download_url) {
        const dl = document.createElement("a");
        dl.className = `btn ${isWinner ? "primary" : "secondary"} ranking-save-btn`;
        dl.href = cand.download_url;
        dl.download = cand.download_name || cand.output_name || "synced.srt";
        dl.appendChild(createDownloadIconSvg());
        const dlLabel = document.createElement("span");
        dlLabel.textContent = i18n.t("result.download");
        dl.appendChild(dlLabel);
        dl.addEventListener("click", (e) => e.stopPropagation());
        actionsEl.appendChild(dl);
      }
    }

    rowEl.appendChild(rankEl);
    rowEl.appendChild(infoEl);
    rowEl.appendChild(scoresEl);
    rowEl.appendChild(actionsEl);
    if (!hasErr && cand.quality) {
      const chev = createChevronSvg();
      chev.setAttribute("class", "ranking-chevron");
      rowEl.appendChild(chev);
    } else {
      const spacer = document.createElement("span");
      spacer.className = "ranking-chevron-spacer";
      spacer.setAttribute("aria-hidden", "true");
      rowEl.appendChild(spacer);
    }
    itemEl.appendChild(rowEl);

    if (!hasErr && cand.quality) {
      const detailsEl = document.createElement("div");
      detailsEl.className = "ranking-details";
      detailsEl.hidden = !isExpanded;

      const qualityBox = document.createElement("div");
      qualityBox.className = "quality-section";
      detailsEl.appendChild(qualityBox);
      itemEl.appendChild(detailsEl);

      const renderDetails = () => {
        if (!rankingQualityTabs.has(candId)) {
          rankingQualityTabs.set(candId, "sync");
        }
        renderQualityBadgesInto(qualityBox, cand.quality, {
          getOpenPanel: () => rankingQualityTabs.get(candId) ?? null,
          setOpenPanel: (next) => rankingQualityTabs.set(candId, next),
          idPrefix: `ranking-q-${candId}`,
        });
      };

      if (isExpanded) {
        renderDetails();
      }

      const toggleExpand = () => {
        const willOpen = expandedRankingId !== candId;
        expandedRankingId = willOpen ? candId : null;
        if (willOpen && !rankingQualityTabs.has(candId)) {
          rankingQualityTabs.set(candId, "sync");
        }
        renderRankingResult(res, { isNewResult: false, scroll: false });
      };

      rowEl.addEventListener("click", toggleExpand);
      rowEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggleExpand();
        }
      });
    }

    listEl.appendChild(itemEl);
  }

  els.result.hidden = false;
  if (scroll) scrollToResultTop();
}

function renderResult(res, { scroll = true } = {}) {
  const isNewResult = res !== lastResult;
  lastResult = res;
  if (res && res.multi_mode && Array.isArray(res.candidates) && res.candidates.length > 0) {
    renderRankingResult(res, { isNewResult, scroll });
    return;
  }
  if (els.resultSingle) els.resultSingle.hidden = false;
  if (els.resultRanking) els.resultRanking.hidden = true;
  if (!res || !res.ok) {
    els.result.classList.add("error");
    els.result.classList.remove("warning");
    const errText =
      (res && res.error_key ? i18n.t(res.error_key, res.error_params) : null) ||
      (res && res.error) ||
      i18n.t("result.syncFailed");
    els.offset.textContent = errText;
    els.resultMeta.textContent = "";
    els.resultActionBtn.hidden = true;
    els.downloadLink.hidden = true;
    renderQualityBadges(null, { resetOpen: true });
    els.result.hidden = false;
    if (scroll) scrollToResultTop();
    return;
  }

  els.result.classList.remove("error");
  els.result.classList.toggle("warning", hasQualityWarning(res.quality));
  els.offset.textContent = formatResultSummary(res);

  const elapsed =
    typeof res.elapsed_seconds === "number" && Number.isFinite(res.elapsed_seconds)
      ? i18n.t("result.completedIn", { seconds: res.elapsed_seconds.toFixed(2) })
      : i18n.t("result.completed");
  const elapsedLabel = res.ref_cached ? `${elapsed} · ${i18n.t("result.reusedAudio")}` : elapsed;

  if (adapter.capabilities.outputPath) {
    els.resultMeta.textContent = res.output_label
      ? `${elapsedLabel} → ${res.output_label}`
      : elapsedLabel;
    els.resultActionBtn.hidden = !res.output_path;
    els.downloadLink.hidden = true;
  } else {
    els.resultMeta.textContent = res.output_label
      ? `${elapsedLabel} · ${res.output_label}`
      : elapsedLabel;
    els.resultActionBtn.hidden = true;
    if (res.download_url) {
      els.downloadLink.href = res.download_url;
      els.downloadLink.download = res.download_name || "synced.srt";
      els.downloadLink.hidden = false;
    } else {
      els.downloadLink.hidden = true;
    }
  }
  renderQualityBadges(res.quality, { resetOpen: isNewResult });
  els.result.hidden = false;
  if (scroll) scrollToResultTop();
}

function resetFormState() {
  selectedRef = null;
  selectedIns = [];
  lastStreamProbe = null;
  els.refPath.value = "";
  els.inPath.value = "";
  els.outPath.value = "";
  els.outPath.dataset.auto = "1";
  els.refFile.value = "";
  els.inputFile.value = "";
  if (els.inputFolder) els.inputFolder.value = "";
  resetStreamSelectToAuto();
  els.streamWrap.hidden = true;
  if (els.vadMode.options.length > 0) els.vadMode.selectedIndex = 0;
  els.splitMode.value = "auto";
  els.noFixFramerate.checked = false;
  els.gss.checked = false;
  renderInputSubtitlesUI();
  fitAllSelectOptions();
  refreshSyncBtn();
}

function applyAppVersion(rawVersion) {
  const clean = String(rawVersion || "").trim();
  if (!clean) return;
  const label = clean.startsWith("v") ? clean : `v${clean}`;
  if (els.settingsVersion && els.settingsFooter && adapter && adapter.capabilities && adapter.capabilities.settings) {
    els.settingsVersion.textContent = label;
    els.settingsFooter.hidden = false;
  }
}

function populateLanguageSelect() {
  if (!els.settingsLanguage) return;
  const resolvedSysCode = i18n.getSystemResolvedLocale();
  const resolvedSysMeta = i18n.getLocaleMeta(resolvedSysCode);
  const sysLangName = (resolvedSysMeta && resolvedSysMeta.nativeName) || resolvedSysCode;

  els.settingsLanguage.innerHTML = "";

  const sysOpt = document.createElement("option");
  sysOpt.value = SYSTEM_LANGUAGE;
  const sysLabel = i18n.t("settings.languageSystem", { language: sysLangName });
  sysOpt.dataset.fullLabel = sysLabel;
  sysOpt.textContent = sysLabel;
  els.settingsLanguage.appendChild(sysOpt);

  for (const loc of i18n.getAvailableLocales()) {
    const opt = document.createElement("option");
    opt.value = loc.code;
    opt.dataset.fullLabel = loc.nativeName;
    opt.textContent = loc.nativeName;
    els.settingsLanguage.appendChild(opt);
  }

  els.settingsLanguage.value = i18n.getPreference();
  observeSelect(els.settingsLanguage);
  fitSelectOptions(els.settingsLanguage);
}

function applyAdapterMeta(ad) {
  if (!ad) return;
  const { meta } = ad;
  document.title = meta.pageTitle || i18n.t("app.title");
  if (meta.subtitle) {
    els.envSubtitle.textContent = meta.subtitle;
    els.envSubtitle.hidden = false;
  } else {
    els.envSubtitle.textContent = "";
    els.envSubtitle.hidden = true;
  }
  if (meta.badgeText) {
    els.envBadge.textContent = meta.badgeText;
    els.envBadge.title = meta.badgeTitle || "";
    els.envBadge.className = meta.badgeVariant ? `badge ${meta.badgeVariant}` : "badge";
    els.envBadge.hidden = false;
  } else {
    els.envBadge.hidden = true;
  }
  if (meta.refPlaceholder) els.refPath.placeholder = meta.refPlaceholder;
  if (meta.inPlaceholder) els.inPath.placeholder = meta.inPlaceholder;
  if (meta.footerText) {
    if (els.footerText) {
      els.footerText.textContent = meta.footerText;
    } else {
      els.footer.textContent = meta.footerText;
    }
    els.footer.hidden = false;
  } else {
    els.footer.hidden = true;
  }
}

function applyLanguageToUi() {
  i18n.applyTranslations(document);
  populateLanguageSelect();
  renderAlignmentProfileSetting();
  syncThemeToggleLabel(isDarkTheme());
  syncMaximizedState(document.body.classList.contains("is-maximized"));

  if (adapter) {
    applyAdapterMeta(adapter);
    if (lastFfmpegInfo) {
      renderFfmpegDir(lastFfmpegInfo);
    }
    if (typeof adapter.retranslate === "function") {
      adapter.retranslate();
    }
  }

  if (lastStreamProbe) {
    renderStreamProbe(lastStreamProbe, { preserveSelection: true });
  } else {
    resetStreamSelectToAuto();
  }

  renderInputSubtitlesUI();
  refreshSyncBtn();

  if (lastResult && !els.result.hidden) {
    renderResult(lastResult, { scroll: false });
  }

  fitAllSelectOptions();
}

i18n.onChange(() => {
  applyLanguageToUi();
});

const narrowLayoutMq =
  typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia("(max-width: 600px)")
    : null;

function syncStatusPillPlacement() {
  if (!els.topbarStatusPill || !els.statusRow || !els.topbarActions) return;
  const hasTopbarStatus = Boolean(adapter && adapter.capabilities && adapter.capabilities.topbarStatus);
  const hasBadge = Boolean(adapter && adapter.meta && adapter.meta.badgeText);
  if (!hasTopbarStatus) {
    els.statusRow.hidden = true;
    els.topbarStatusPill.hidden = true;
    if (els.topbarActionsDivider) {
      els.topbarActionsDivider.hidden = !hasBadge;
    }
    return;
  }
  const isNarrow = Boolean(narrowLayoutMq && narrowLayoutMq.matches);
  if (isNarrow) {
    if (els.topbarStatusPill.parentElement !== els.statusRow) {
      els.statusRow.appendChild(els.topbarStatusPill);
    }
    els.statusRow.hidden = false;
    els.topbarStatusPill.hidden = false;
    if (els.topbarActionsDivider) {
      els.topbarActionsDivider.hidden = !hasBadge;
    }
  } else {
    if (els.topbarStatusPill.parentElement !== els.topbarActions) {
      els.topbarActions.prepend(els.topbarStatusPill);
    }
    els.statusRow.hidden = true;
    els.topbarStatusPill.hidden = false;
    if (els.topbarActionsDivider) {
      els.topbarActionsDivider.hidden = false;
    }
  }
}

if (narrowLayoutMq) {
  if (typeof narrowLayoutMq.addEventListener === "function") {
    narrowLayoutMq.addEventListener("change", syncStatusPillPlacement);
  } else if (typeof narrowLayoutMq.addListener === "function") {
    narrowLayoutMq.addListener(syncStatusPillPlacement);
  }
}

function configureUiForAdapter(ad) {
  const { meta, capabilities } = ad;
  resetFormState();
  applyAdapterMeta(ad);

  const hasTopbarStatus = Boolean(capabilities.topbarStatus);
  syncStatusPillPlacement();
  if (hasTopbarStatus && els.runtimeCard) {
    els.runtimeCard.hidden = true;
  }
  if (els.settingsBtn) {
    els.settingsBtn.hidden = !capabilities.settings;
  }
  if (els.settingsFfmpegCard) {
    els.settingsFfmpegCard.hidden = !capabilities.ffmpegSettings;
  }

  els.refPath.readOnly = !capabilities.editablePaths;
  els.inPath.readOnly = !capabilities.editablePaths;

  if (capabilities.refAccept) els.refFile.accept = capabilities.refAccept;
  if (capabilities.subtitleAccept) els.inputFile.accept = capabilities.subtitleAccept;

  els.outField.hidden = !capabilities.outputPath;
  els.runtimeActionBtn.hidden = !capabilities.runtimeAction;

  if (Array.isArray(capabilities.vadModes) && capabilities.vadModes.length) {
    els.vadMode.innerHTML = "";
    for (const mode of capabilities.vadModes) {
      const opt = document.createElement("option");
      opt.value = mode.value;
      const label = mode.labelKey ? i18n.t(mode.labelKey) : mode.label;
      if (mode.labelKey) opt.dataset.i18n = mode.labelKey;
      opt.dataset.fullLabel = label;
      opt.textContent = label;
      if (mode.selected) opt.selected = true;
      els.vadMode.appendChild(opt);
    }
  }

  applyLanguageToUi();

  if (meta.appVersion) {
    applyAppVersion(meta.appVersion);
  }

  const hasCustomChrome = Boolean(capabilities.customWindowControls);
  document.body.classList.toggle("is-desktop", hasCustomChrome);
  if (els.winControls) els.winControls.hidden = !hasCustomChrome;
  if (els.winControlsDivider) els.winControlsDivider.hidden = !hasCustomChrome;
  if (els.resizeHandles) els.resizeHandles.hidden = !hasCustomChrome;
}

function wireEvents(ad) {
  const { capabilities } = ad;

  if (capabilities.customWindowControls && ad.windowControls) {
    const wc = ad.windowControls;
    els.winMinimize.addEventListener("click", () => wc.minimize());
    els.winMaximize.addEventListener("click", async () => {
      const isMax = await wc.toggleMaximize();
      syncMaximizedState(isMax);
    });
    els.winClose.addEventListener("click", () => wc.close());

    let lastDragDownTime = 0;
    let lastDragDownX = 0;
    let lastDragDownY = 0;
    for (const dragEl of [els.dragBrand, els.dragSpacer]) {
      if (!dragEl) continue;
      dragEl.addEventListener("mousedown", async (e) => {
        if (e.button !== 0) return;
        const now = performance.now();
        const isDbl =
          now - lastDragDownTime < 350 &&
          Math.abs(e.clientX - lastDragDownX) < 6 &&
          Math.abs(e.clientY - lastDragDownY) < 6;
        if (isDbl) {
          lastDragDownTime = 0;
          const isMax = await wc.toggleMaximize();
          syncMaximizedState(isMax);
        } else {
          lastDragDownTime = now;
          lastDragDownX = e.clientX;
          lastDragDownY = e.clientY;
          wc.startDrag();
        }
      });
    }

    if (els.resizeHandles) {
      for (const edge of els.resizeHandles.querySelectorAll(".resize-edge")) {
        edge.addEventListener("mousedown", (e) => {
          if (e.button !== 0) return;
          e.preventDefault();
          wc.startResize(Number(edge.dataset.hit));
        });
      }
    }
  }

  // Prevent accidental navigation if a file is dropped anywhere outside the drop zones
  // (also enables HTML5 drop events in desktop WebView2 without Python dragover IPC).
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  const applyWebSubtitleFiles = async (
    files,
    { append = false, strictEmptyArchive = true, emptyFolderError = false } = {},
  ) => {
    if (!files || !files.length) {
      if (emptyFolderError) showInputError(i18n.t("errors.noSubsInFolder"));
      return;
    }
    if (typeof ad.expandSubtitleFiles === "function") {
      try {
        const expanded = await ad.expandSubtitleFiles(files, { strictEmptyArchive });
        if (expanded && expanded.error) {
          showInputError(expanded.error);
          return;
        }
        const entries = (expanded && expanded.entries) || [];
        if (!entries.length) {
          if (emptyFolderError) showInputError(i18n.t("errors.noSubsInFolder"));
          return;
        }
        setInputSubtitles(entries, { append });
      } catch (err) {
        showInputError(i18n.t("errors.archiveExtractFailed", { error: (err && err.message) || err }));
      }
      return;
    }
    setInputSubtitles(
      files.map((f) => ({
        value: f,
        display: ad.formatFileDisplay ? ad.formatFileDisplay(f) : f.name,
      })),
      { append },
    );
  };

  // Instant drag-enter / drag-over visual highlight for both desktop and web.
  for (const [zone, target] of [
    [els.dropRef, "ref"],
    [els.dropIn, "in"],
  ]) {
    zone.addEventListener("dragenter", (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    });
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
      zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", (e) => {
      if (!zone.contains(e.relatedTarget)) {
        zone.classList.remove("drag-over");
      }
    });
    zone.addEventListener("drop", async (e) => {
      zone.classList.remove("drag-over");
      if (!capabilities.editablePaths) {
        e.preventDefault();
        if (target === "ref") {
          const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
          if (!file) return;
          const display = ad.formatFileDisplay ? ad.formatFileDisplay(file) : file.name;
          setReference({ value: file, display });
        } else {
          const { files, hadDir } = await extractWebDroppedSubtitles(e.dataTransfer, selectedRef);
          await applyWebSubtitleFiles(files, {
            append: false,
            strictEmptyArchive: !hadDir,
            emptyFolderError: hadDir,
          });
        }
      }
    });
  }

  if (els.clearInBtn) {
    els.clearInBtn.addEventListener("click", () => {
      selectedIns = [];
      els.inputFile.value = "";
      if (els.inputFolder) els.inputFolder.value = "";
      renderInputSubtitlesUI();
    });
  }

  const pickSubtitleFiles = async ({ append = false } = {}) => {
    setBrowseSplitMenuOpen(false);
    if (!capabilities.editablePaths) {
      appendNextWebPick = append;
      els.inputFile.click();
      return;
    }
    const res = await ad.pickSubtitle(typeof selectedRef === "string" ? selectedRef : null);
    if (!res) return;
    if (Array.isArray(res)) {
      if (res.length) setInputSubtitles(res, { append, forceAutoOut: !append });
      return;
    }
    if (res.error) {
      showInputError(res.error);
      return;
    }
    if (res.entries && res.entries.length) {
      setInputSubtitles(res.entries, { append, forceAutoOut: !append });
    }
  };

  const pickSubtitleFolderAction = async () => {
    setBrowseSplitMenuOpen(false);
    if (!capabilities.editablePaths) {
      if (els.inputFolder) els.inputFolder.click();
      return;
    }
    if (typeof ad.pickSubtitleFolder === "function") {
      const res = await ad.pickSubtitleFolder(typeof selectedRef === "string" ? selectedRef : null);
      if (!res) return;
      if (res.error) {
        showInputError(res.error);
        return;
      }
      if (res.entries && res.entries.length) {
        setInputSubtitles(res.entries, { append: false, forceAutoOut: true });
      }
    }
  };

  // Unified split button: defaults to "Browse…" (replace) when empty and "+ Add" (append) when files exist.
  if (els.browseIn) {
    els.browseIn.addEventListener("click", () => {
      pickSubtitleFiles({ append: selectedIns.length > 0 });
    });
  }
  if (els.browseInMenuBtn && els.browseInMenu) {
    els.browseInMenuBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const willOpen = els.browseInMenu.hidden;
      setBrowseSplitMenuOpen(willOpen);
    });
    document.addEventListener("mousedown", (e) => {
      if (els.browseInSplit && !els.browseInSplit.contains(e.target)) {
        setBrowseSplitMenuOpen(false);
      }
    });
    window.addEventListener("blur", () => setBrowseSplitMenuOpen(false));
  }
  if (els.browseInOptAdd) {
    els.browseInOptAdd.addEventListener("click", () => pickSubtitleFiles({ append: true }));
  }
  if (els.browseInOptFiles) {
    els.browseInOptFiles.addEventListener("click", () => pickSubtitleFiles({ append: false }));
  }
  if (els.browseInOptFolder) {
    els.browseInOptFolder.addEventListener("click", () => pickSubtitleFolderAction());
  }

  if (!capabilities.editablePaths) {
    // Hidden <input type="file"> change handlers (also used by Playwright tests).
    els.refFile.addEventListener("change", () => {
      const file = els.refFile.files && els.refFile.files[0];
      if (!file) return;
      const display = ad.formatFileDisplay ? ad.formatFileDisplay(file) : file.name;
      setReference({ value: file, display });
    });
    els.inputFile.addEventListener("change", async () => {
      const files = Array.from(els.inputFile.files || []);
      const append = appendNextWebPick;
      appendNextWebPick = false;
      els.inputFile.value = "";
      if (!files.length) return;
      await applyWebSubtitleFiles(files, { append, strictEmptyArchive: true });
    });
    if (els.inputFolder) {
      els.inputFolder.addEventListener("change", async () => {
        const files = filterWebFolderFiles(els.inputFolder.files, selectedRef);
        els.inputFolder.value = "";
        await applyWebSubtitleFiles(files, {
          append: false,
          strictEmptyArchive: false,
          emptyFolderError: true,
        });
      });
    }

    // Clicking readonly display fields or Browse button opens the browser file picker.
    els.refPath.addEventListener("click", () => els.refFile.click());
    els.inPath.addEventListener("click", () => pickSubtitleFiles({ append: selectedIns.length > 0 }));
    els.browseRef.addEventListener("click", () => els.refFile.click());
  } else {
    // Desktop editable path fields + native dialogs.
    els.refPath.addEventListener("input", () => {
      const val = els.refPath.value.trim();
      selectedRef = val;
      if (!val) refreshStreams("");
      refreshSyncBtn();
    });
    els.refPath.addEventListener("change", () => refreshStreams(els.refPath.value.trim()));

    els.inPath.addEventListener("input", () => {
      if (selectedIns.length > 1) return;
      const val = els.inPath.value.trim();
      selectedIns = val ? [normalizeSubtitleEntry({ value: val, display: val })] : [];
      updateBrowseInSplitUi();
      if (
        capabilities.outputPath &&
        ad.suggestOutputPath &&
        (!els.outPath.value.trim() || els.outPath.dataset.auto === "1")
      ) {
        els.outPath.value = ad.suggestOutputPath(val);
        els.outPath.dataset.auto = "1";
      }
      refreshSyncBtn();
    });

    els.inPath.addEventListener("change", async () => {
      if (selectedIns.length > 1) return;
      const val = els.inPath.value.trim();
      if (!val || typeof ad.resolveSubtitleInputs !== "function") return;
      const res = await ad.resolveSubtitleInputs([val], typeof selectedRef === "string" ? selectedRef : null);
      if (res && res.error) {
        showInputError(res.error);
        return;
      }
      if (
        res &&
        Array.isArray(res.entries) &&
        (res.entries.length > 1 || (res.entries.length === 1 && res.entries[0].value !== val))
      ) {
        setInputSubtitles(res.entries, { append: false, forceAutoOut: true });
      }
    });

    els.outPath.addEventListener("input", () => {
      els.outPath.dataset.auto = "0";
    });

    els.browseRef.addEventListener("click", async () => {
      const entry = await ad.pickReference();
      if (entry) setReference(entry);
    });

    els.browseOut.addEventListener("click", async () => {
      const single = selectedIns.length === 1 ? selectedIns[0] : null;
      const singleIn = single ? single.value : null;
      const fallback =
        (single && single.default_output_path) ||
        (ad.suggestOutputPath && typeof singleIn === "string"
          ? ad.suggestOutputPath(singleIn)
          : "");
      const p = await ad.pickOutput(els.outPath.value.trim() || fallback);
      if (p) {
        els.outPath.value = p;
        els.outPath.dataset.auto = "0";
      }
    });
  }

  if (capabilities.runtimeAction && ad.runRuntimeAction) {
    els.runtimeActionBtn.addEventListener("click", () => ad.runRuntimeAction());
  }

  if (capabilities.settings) {
    if (els.settingsBtn) {
      els.settingsBtn.addEventListener("click", () => setSettingsOpen(!settingsOpen));
    }
    if (els.settingsCloseBtn) {
      els.settingsCloseBtn.addEventListener("click", () => setSettingsOpen(false));
    }
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (els.browseInMenu && !els.browseInMenu.hidden) {
          setBrowseSplitMenuOpen(false);
          if (els.browseInMenuBtn) els.browseInMenuBtn.focus();
          return;
        }
        if (_openCustomSelect) {
          closeCustomSelect(_openCustomSelect, { restoreFocus: true });
          return;
        }
        if (settingsOpen) {
          setSettingsOpen(false);
        }
      }
    });

    if (els.settingsLanguage) {
      els.settingsLanguage.addEventListener("change", () => {
        const nextPref = els.settingsLanguage.value || SYSTEM_LANGUAGE;
        i18n.setPreference(nextPref, { persist: true });
        if (typeof ad.onLanguageChange === "function") {
          ad.onLanguageChange(i18n.getPreference());
        }
      });
    }

    if (els.settingsAlignmentProfile) {
      els.settingsAlignmentProfile.addEventListener("change", () => {
        applyAlignmentProfile(els.settingsAlignmentProfile.value, {
          persist: true,
          notifyAdapter: true,
        });
      });
    }

    if (capabilities.ffmpegSettings && ad.settings) {
      if (els.settingsFfChange) {
        els.settingsFfChange.addEventListener("click", async () => {
          const picked = await ad.settings.pickFfmpegDir();
          if (picked) {
            await ad.settings.setFfmpegPath(picked);
            if (hasSelection(selectedRef)) {
              refreshStreams(selectedRef);
            }
          }
        });
      }
      if (els.settingsFfOpen) {
        els.settingsFfOpen.addEventListener("click", () => ad.settings.openFfmpegFolder());
      }
      if (els.settingsFfDownload && ad.runRuntimeAction) {
        els.settingsFfDownload.addEventListener("click", () => ad.runRuntimeAction());
      }
    }
  }

  if (capabilities.outputPath && ad.openOutput) {
    els.resultActionBtn.addEventListener("click", () => {
      if (lastResult) ad.openOutput(lastResult);
    });
  }

  els.syncBtn.addEventListener("click", async () => {
    if (busy || !runtimeReady || !hasSelection(selectedRef) || selectedIns.length === 0) return;
    busy = true;
    lastSyncProgress = { messageKey: "form.startingSync", messageParams: null };
    refreshSyncBtn();
    els.syncBtn.textContent = i18n.t("form.startingSync");
    els.result.hidden = true;
    els.result.classList.remove("error");
    setProgressBar(els.syncProg, els.syncFill, els.syncPct, { active: true, fraction: null });

    const [vad, fr] = (els.vadMode.value || "webrtc:16000").split(":");
    const isMulti = selectedIns.length > 1;
    const inputs = selectedIns.map((item) =>
      typeof item.value === "string" ? item.value.trim() : item.value,
    );
    let outPath = "";
    if (capabilities.outputPath && !isMulti) {
      const singleIn = inputs[0];
      const suggested =
        ad.suggestOutputPath && typeof singleIn === "string"
          ? ad.suggestOutputPath(singleIn)
          : "";
      outPath = els.outPath.value.trim() || suggested;
      els.outPath.value = outPath;
    }

    const selectedOpt = els.refStream.options[els.refStream.selectedIndex];
    const refStreamKind = (selectedOpt && selectedOpt.dataset && selectedOpt.dataset.kind) || null;

    let res;
    try {
      res = await ad.sync(
        {
          ref: typeof selectedRef === "string" ? selectedRef.trim() : selectedRef,
          input: inputs[0],
          inputs,
          outPath,
          vad,
          frameRate: Number(fr),
          refStream: els.refStream.value.trim() || null,
          refStreamKind,
          splitMode: els.splitMode.value,
          noFixFramerate: Boolean(els.noFixFramerate.checked),
          gss: Boolean(els.gss.checked),
          alignmentProfile,
        },
        {
          onProgress: ({ message, messageKey, messageParams, fraction }) => {
            if (messageKey) {
              lastSyncProgress = { messageKey, messageParams: messageParams || null };
              els.syncBtn.textContent = i18n.t(messageKey, messageParams);
            } else if (message) {
              lastSyncProgress = null;
              els.syncBtn.textContent = message;
            }
            setProgressBar(els.syncProg, els.syncFill, els.syncPct, {
              active: true,
              fraction,
            });
          },
        },
      );
    } catch (err) {
      res = { ok: false, error: String((err && err.stack) || err) };
    }

    busy = false;
    lastSyncProgress = null;
    setProgressBar(els.syncProg, els.syncFill, els.syncPct, { active: false, fraction: null });
    refreshSyncBtn();
    renderResult(res);
  });
}

async function boot() {
  const isDesktop = window.location.hash === "#desktop" || Boolean(window.pywebview);
  const adapterRel = isDesktop ? "./adapters/desktop.js" : "./adapters/web.js";
  const mod = await import(withVersion(new URL(adapterRel, import.meta.url).href));

  adapter = mod.createAdapter({
    version: V,
    withVersion,
    i18n,
    callbacks: {
      onRuntimeStatus: applyRuntimeStatus,
      onRuntimeProgress: (prog) => {
        setProgressBar(els.runtimeProg, els.runtimeFill, els.runtimePct, prog);
        if (els.settingsFfProg) {
          setProgressBar(els.settingsFfProg, els.settingsFfFill, els.settingsFfPct, prog);
        }
      },
      onExternalDrop: (target, entry) => {
        if (target === "ref") {
          setReference(entry);
        } else if (target === "in") {
          setInputSubtitles(entry, { append: false, forceAutoOut: true });
        }
      },
      onInputError: (msg) => showInputError(msg),
      getCurrentReference: () => (typeof selectedRef === "string" ? selectedRef : null),
      onWindowMaximizedChange: syncMaximizedState,
      onAppVersion: applyAppVersion,
      onAlignmentProfileInit: (profile) =>
        applyAlignmentProfile(profile, { persist: true, notifyAdapter: false }),
    },
  });

  configureUiForAdapter(adapter);
  wireEvents(adapter);
  await adapter.init();
  if (typeof adapter.onThemeChange === "function") {
    adapter.onThemeChange(isDarkTheme());
  }
}

boot().catch((err) => {
  console.error(err);
  const msg = i18n.t("status.errorPrefix", { error: (err && err.message) || err });
  if (els.statusRow && els.topbarStatusPill && els.topbarStatusPill.parentElement === els.statusRow) {
    els.statusRow.hidden = false;
  }
  if (els.topbarStatusDot) els.topbarStatusDot.className = "dot err";
  if (els.topbarStatusPill) {
    els.topbarStatusPill.className = "status-pill topbar-status err";
    els.topbarStatusPill.hidden = false;
  }
  els.runtimeDot.className = "dot err";
  if (els.runtimePill) els.runtimePill.className = "status-pill err";
  if (els.runtimeStatusText) els.runtimeStatusText.textContent = msg;
  els.status.textContent = msg;
});
