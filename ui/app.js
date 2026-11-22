// Unified UI controller shared by the native desktop app (pywebview) and the
// browser WebAssembly app (Pyodide + ffmpeg.wasm). Platform-specific behavior
// is isolated in ./adapters/desktop.js and ./adapters/web.js.

const V = new URL(import.meta.url).searchParams.get("v") || "";
const withVersion = (url) => (V ? url + (url.includes("?") ? "&" : "?") + "v=" + V : url);

const els = {
  envSubtitle: document.getElementById("env-subtitle"),
  envBadge: document.getElementById("env-badge"),
  topbarActionsDivider: document.getElementById("topbar-actions-divider"),
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
  outField: document.getElementById("out-field"),
  outPath: document.getElementById("out-path"),
  browseRef: document.getElementById("browse-ref"),
  browseIn: document.getElementById("browse-in"),
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
  offset: document.getElementById("offset"),
  resultMeta: document.getElementById("result-meta"),
  resultActionBtn: document.getElementById("result-action-btn"),
  downloadLink: document.getElementById("download"),
  footer: document.getElementById("app-footer"),
  footerText: document.getElementById("footer-text"),
  footerVersion: document.getElementById("footer-version"),
  settingsFooter: document.getElementById("settings-footer"),
  settingsVersion: document.getElementById("settings-version"),
};

let adapter = null;
let runtimeReady = false;
let busy = false;
let selectedRef = null; // string path (desktop) or File (web)
let selectedIn = null; // string path (desktop) or File (web)
let lastResult = null;
let streamProbeSeq = 0;
let themeTransitionTimer = null;
let settingsOpen = false;

function setSettingsOpen(open) {
  if (!adapter || !adapter.capabilities.settings) return;
  settingsOpen = Boolean(open);
  if (els.mainView) els.mainView.hidden = settingsOpen;
  if (els.settingsView) els.settingsView.hidden = !settingsOpen;
  if (els.settingsBtn) {
    els.settingsBtn.classList.toggle("active", settingsOpen);
    els.settingsBtn.setAttribute("aria-pressed", String(settingsOpen));
  }
}

function isDarkTheme() {
  return document.documentElement.classList.contains("dark");
}

function syncThemeToggleLabel(dark) {
  if (els.themeToggle) {
    els.themeToggle.title = dark ? "Switch to light theme" : "Switch to dark theme";
    els.themeToggle.setAttribute("aria-label", els.themeToggle.title);
  }
}

function syncMaximizedState(isMax) {
  const max = Boolean(isMax);
  document.body.classList.toggle("is-maximized", max);
  if (els.winMaximize) {
    els.winMaximize.classList.toggle("is-maximized", max);
    const label = max ? "Restore" : "Maximize";
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
    busy || !runtimeReady || !hasSelection(selectedRef) || !hasSelection(selectedIn);
  if (!busy) {
    els.syncBtn.textContent = "Sync subtitles";
  }
}

function formatResultSummary(res) {
  const off = Number(res.offset_seconds);
  const scale = Number(res.framerate_scale_factor);
  const segs = Array.isArray(res.split_segments) ? res.split_segments : null;
  const dropped = Number(res.dropped_cues || 0);
  const dropNote = dropped > 0 ? ` · ${dropped} orphan cue(s) removed` : "";
  let splitNote = "";
  if (segs) {
    if (segs.length <= 1) {
      splitNote = ` · no mid-file breaks${dropNote}`;
    } else {
      const fmt = (v) => `${v >= 0 ? "+" : ""}${Number(v).toFixed(2)}s`;
      const chain = segs.slice(0, 4).map((x) => fmt(x.offset_seconds)).join(" → ");
      const tail = segs.length > 4 ? ` → … (${segs.length} segments)` : "";
      splitNote = ` · ${segs.length - 1} break(s) fixed (${chain}${tail}${dropNote})`;
    }
  } else if (dropped > 0) {
    splitNote = dropNote;
  }
  return (
    `offset ${off >= 0 ? "+" : ""}${off.toFixed(3)}s` +
    (Math.abs(scale - 1) > 1e-6 ? ` · framerate ×${scale.toFixed(3)}` : "") +
    splitNote
  );
}

// Pixel-accurate truncation for all <select> dropdown options so native OS
// popups never overflow the window. Preserves trailing bracketed metadata
// (e.g. "[eac3]", "[PGS timings]", "[text: subrip]") whenever present.
const _measureCanvas = document.createElement("canvas");
const _measureCtx = _measureCanvas.getContext("2d");
const _TRAILING_BRACKETS_RE = /^(.*?)(\s*(?:\[[^\]]+\]\s*)+)$/;
const _selectObserved = new WeakSet();
let _selectResizeObserver = null;

function truncateSelectLabel(fullText, maxPx, measure) {
  const text = String(fullText || "");
  if (!text || !Number.isFinite(maxPx) || maxPx <= 0 || measure(text) <= maxPx) {
    return text;
  }
  const ellipsis = "...";
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
    // Extreme narrow width: still prioritize bracketed metadata over the prefix.
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

function updateSelectTooltip(selectEl) {
  if (!selectEl) return;
  const opt = selectEl.options[selectEl.selectedIndex];
  const full = opt ? (opt.dataset.fullLabel || opt.textContent || "") : "";
  selectEl.title = opt && opt.textContent !== full ? full : "";
}

function fitSelectOptions(selectEl) {
  if (!selectEl || !selectEl.options || selectEl.options.length === 0) return;
  const width = selectEl.clientWidth;
  if (width <= 0) return;
  const cs = window.getComputedStyle(selectEl);
  const padL = parseFloat(cs.paddingLeft) || 12;
  const padR = parseFloat(cs.paddingRight) || 36;
  const maxPx = Math.max(40, width - padL - padR - 8);
  const font = `${cs.fontWeight || "500"} ${cs.fontSize || "12px"} ${cs.fontFamily || "sans-serif"}`;
  const measure = (str) => {
    if (_measureCtx) {
      _measureCtx.font = font;
      return _measureCtx.measureText(str).width;
    }
    return str.length * 7;
  };

  for (const opt of selectEl.options) {
    if (opt.dataset.fullLabel === undefined) {
      opt.dataset.fullLabel = opt.textContent || "";
    }
    const full = opt.dataset.fullLabel;
    const fitted = truncateSelectLabel(full, maxPx, measure);
    if (opt.textContent !== fitted) {
      opt.textContent = fitted;
    }
    opt.title = fitted !== full ? full : "";
  }
  updateSelectTooltip(selectEl);
}

function observeSelect(selectEl) {
  if (!selectEl || _selectObserved.has(selectEl)) return;
  _selectObserved.add(selectEl);
  selectEl.addEventListener("change", () => updateSelectTooltip(selectEl));
  if (typeof ResizeObserver === "function") {
    if (!_selectResizeObserver) {
      _selectResizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          fitSelectOptions(entry.target);
        }
      });
    }
    _selectResizeObserver.observe(selectEl);
  }
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

async function refreshStreams(refValue) {
  const seq = ++streamProbeSeq;
  els.refStream.innerHTML = '<option value="">Auto (recommended)</option>';
  els.refStream.value = "";
  els.refStream.title = "";
  els.streamHint.textContent = "Auto";
  els.streamWrap.hidden = true;
  if (!adapter || !adapter.capabilities.streamProbe || !refValue) return;
  try {
    const res = await adapter.probeStreams(refValue);
    if (seq !== streamProbeSeq) return;
    const list = (res && res.streams) || [];
    if (list.length <= 1) return;
    for (const item of list) {
      const opt = document.createElement("option");
      opt.value = item.value;
      opt.dataset.fullLabel = item.label;
      opt.textContent = item.label;
      opt.dataset.kind = item.kind || "";
      if (item.is_bitmap) opt.disabled = true;
      els.refStream.appendChild(opt);
    }
    const notes = [];
    if (res.has_text_subs) notes.push("text subs available");
    if (res.has_pgs_subs) notes.push("PGS/VobSub timings available");
    els.streamHint.textContent = notes.length
      ? `${list.length} track(s) · ${notes.join(" · ")}`
      : `${list.length} track(s) · audio VAD`;
    els.streamWrap.hidden = false;
    observeSelect(els.refStream);
    fitSelectOptions(els.refStream);
  } catch (_) {
    if (seq === streamProbeSeq) els.streamWrap.hidden = true;
  }
}

function setReference(entry) {
  if (!entry) return;
  selectedRef = entry.value;
  els.refPath.value = entry.display;
  refreshStreams(entry.value);
  refreshSyncBtn();
}

function setInputSubtitle(entry, { forceAutoOut = false } = {}) {
  if (!entry) return;
  selectedIn = entry.value;
  els.inPath.value = entry.display;
  if (
    adapter.capabilities.outputPath &&
    adapter.suggestOutputPath &&
    (forceAutoOut || !els.outPath.value.trim() || els.outPath.dataset.auto === "1")
  ) {
    els.outPath.value = adapter.suggestOutputPath(entry.value);
    els.outPath.dataset.auto = "1";
  }
  refreshSyncBtn();
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
        els.settingsFfDownloadLabel.textContent = update.actionLabel || "Download latest FFmpeg";
      }
    }
    if (update.actionDisabled !== undefined) {
      els.runtimeActionBtn.disabled = Boolean(update.actionDisabled);
      if (els.settingsFfDownload) {
        els.settingsFfDownload.disabled = Boolean(update.actionDisabled);
      }
    }
  }
  if (update.ffmpegInfo) {
    const info = update.ffmpegInfo;
    if (els.settingsFfDir) {
      let dir = info.resource_dir || "";
      if (!dir && info.ffmpeg_path) {
        const sep = Math.max(info.ffmpeg_path.lastIndexOf("\\"), info.ffmpeg_path.lastIndexOf("/"));
        dir = sep > 0 ? info.ffmpeg_path.slice(0, sep) : info.ffmpeg_path;
      }
      if (!dir) dir = info.custom_path || info.portable_dir || "Auto-detect";
      els.settingsFfDir.textContent = dir;
      els.settingsFfDir.title = info.ffmpeg_path || dir;
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

function renderResult(res) {
  lastResult = res;
  if (!res || !res.ok) {
    els.result.classList.add("error");
    els.offset.textContent = (res && res.error) || "Sync failed.";
    els.resultMeta.textContent = "";
    els.resultActionBtn.hidden = true;
    els.downloadLink.hidden = true;
    els.result.hidden = false;
    scrollToResultTop();
    return;
  }

  els.result.classList.remove("error");
  els.offset.textContent = formatResultSummary(res);

  const elapsed =
    typeof res.elapsed_seconds === "number" && Number.isFinite(res.elapsed_seconds)
      ? `Completed in ${res.elapsed_seconds.toFixed(2)}s`
      : "Completed";

  if (adapter.capabilities.outputPath) {
    els.resultMeta.textContent = res.output_label ? `${elapsed} → ${res.output_label}` : elapsed;
    els.resultActionBtn.hidden = !res.output_path;
    els.downloadLink.hidden = true;
  } else {
    els.resultMeta.textContent = res.output_label ? `${elapsed} · ${res.output_label}` : elapsed;
    els.resultActionBtn.hidden = true;
    if (res.download_url) {
      els.downloadLink.href = res.download_url;
      els.downloadLink.download = res.download_name || "synced.srt";
      els.downloadLink.hidden = false;
    } else {
      els.downloadLink.hidden = true;
    }
  }
  els.result.hidden = false;
  scrollToResultTop();
}

function resetFormState() {
  selectedRef = null;
  selectedIn = null;
  els.refPath.value = "";
  els.inPath.value = "";
  els.outPath.value = "";
  els.outPath.dataset.auto = "1";
  els.refFile.value = "";
  els.inputFile.value = "";
  els.refStream.innerHTML = '<option value="">Auto (recommended)</option>';
  els.refStream.value = "";
  els.streamWrap.hidden = true;
  if (els.vadMode.options.length > 0) els.vadMode.selectedIndex = 0;
  els.splitMode.value = "auto";
  els.noFixFramerate.checked = false;
  els.gss.checked = false;
  refreshSyncBtn();
}

function applyAppVersion(rawVersion) {
  const clean = String(rawVersion || "").trim();
  if (!clean) return;
  const label = clean.startsWith("v") ? clean : `v${clean}`;
  if (els.footerVersion && adapter && adapter.meta && adapter.meta.footerText) {
    els.footerVersion.textContent = label;
    els.footerVersion.hidden = false;
  }
  if (els.settingsVersion && els.settingsFooter && adapter && adapter.capabilities && adapter.capabilities.settings) {
    els.settingsVersion.textContent = label;
    els.settingsFooter.hidden = false;
  }
}

function configureUiForAdapter(ad) {
  const { meta, capabilities } = ad;
  resetFormState();

  document.title = meta.pageTitle || "ffblindsubsync";
  if (meta.subtitle) {
    els.envSubtitle.textContent = meta.subtitle;
    els.envSubtitle.hidden = false;
  } else {
    els.envSubtitle.textContent = "";
    els.envSubtitle.hidden = true;
  }
  const hasTopbarStatus = Boolean(capabilities.topbarStatus);
  if (els.topbarStatusPill) {
    els.topbarStatusPill.hidden = !hasTopbarStatus;
  }
  if (hasTopbarStatus && els.runtimeCard) {
    els.runtimeCard.hidden = true;
  }
  if (meta.badgeText) {
    els.envBadge.textContent = meta.badgeText;
    els.envBadge.title = meta.badgeTitle || "";
    els.envBadge.className = meta.badgeVariant ? `badge ${meta.badgeVariant}` : "badge";
    els.envBadge.hidden = false;
  } else {
    els.envBadge.hidden = true;
  }
  if (els.topbarActionsDivider) {
    els.topbarActionsDivider.hidden = !(hasTopbarStatus || Boolean(meta.badgeText));
  }
  if (els.settingsBtn) {
    els.settingsBtn.hidden = !capabilities.settings;
  }

  els.refPath.placeholder = meta.refPlaceholder || els.refPath.placeholder;
  els.inPath.placeholder = meta.inPlaceholder || els.inPath.placeholder;

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
      opt.dataset.fullLabel = mode.label;
      opt.textContent = mode.label;
      if (mode.selected) opt.selected = true;
      els.vadMode.appendChild(opt);
    }
  }
  fitAllSelectOptions();

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
    zone.addEventListener("drop", (e) => {
      zone.classList.remove("drag-over");
      if (!capabilities.editablePaths) {
        e.preventDefault();
        const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (!file) return;
        const display = ad.formatFileDisplay ? ad.formatFileDisplay(file) : file.name;
        if (target === "ref") {
          setReference({ value: file, display });
        } else {
          setInputSubtitle({ value: file, display });
        }
      }
    });
  }

  if (!capabilities.editablePaths) {
    // Hidden <input type="file"> change handlers (also used by Playwright tests).
    els.refFile.addEventListener("change", () => {
      const file = els.refFile.files && els.refFile.files[0];
      if (!file) return;
      const display = ad.formatFileDisplay ? ad.formatFileDisplay(file) : file.name;
      setReference({ value: file, display });
    });
    els.inputFile.addEventListener("change", () => {
      const file = els.inputFile.files && els.inputFile.files[0];
      if (!file) return;
      const display = ad.formatFileDisplay ? ad.formatFileDisplay(file) : file.name;
      setInputSubtitle({ value: file, display });
    });

    // Clicking readonly display fields or Browse buttons opens the browser file picker.
    els.refPath.addEventListener("click", () => els.refFile.click());
    els.inPath.addEventListener("click", () => els.inputFile.click());
    els.browseRef.addEventListener("click", () => els.refFile.click());
    els.browseIn.addEventListener("click", () => els.inputFile.click());
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
      const val = els.inPath.value.trim();
      selectedIn = val;
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

    els.outPath.addEventListener("input", () => {
      els.outPath.dataset.auto = "0";
    });

    els.browseRef.addEventListener("click", async () => {
      const entry = await ad.pickReference();
      if (entry) setReference(entry);
    });

    els.browseIn.addEventListener("click", async () => {
      const entry = await ad.pickSubtitle();
      if (entry) setInputSubtitle(entry, { forceAutoOut: true });
    });

    els.browseOut.addEventListener("click", async () => {
      const fallback =
        ad.suggestOutputPath && typeof selectedIn === "string"
          ? ad.suggestOutputPath(selectedIn)
          : "";
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

  if (capabilities.settings && ad.settings) {
    if (els.settingsBtn) {
      els.settingsBtn.addEventListener("click", () => setSettingsOpen(!settingsOpen));
    }
    if (els.settingsCloseBtn) {
      els.settingsCloseBtn.addEventListener("click", () => setSettingsOpen(false));
    }
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && settingsOpen) {
        setSettingsOpen(false);
      }
    });

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

  if (capabilities.outputPath && ad.openOutput) {
    els.resultActionBtn.addEventListener("click", () => {
      if (lastResult) ad.openOutput(lastResult);
    });
  }

  els.syncBtn.addEventListener("click", async () => {
    if (busy || !runtimeReady || !hasSelection(selectedRef) || !hasSelection(selectedIn)) return;
    busy = true;
    refreshSyncBtn();
    els.syncBtn.textContent = "Starting sync…";
    els.result.hidden = true;
    els.result.classList.remove("error");
    setProgressBar(els.syncProg, els.syncFill, els.syncPct, { active: true, fraction: null });

    const [vad, fr] = (els.vadMode.value || "webrtc:16000").split(":");
    let outPath = "";
    if (capabilities.outputPath) {
      const suggested =
        ad.suggestOutputPath && typeof selectedIn === "string"
          ? ad.suggestOutputPath(selectedIn)
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
          input: typeof selectedIn === "string" ? selectedIn.trim() : selectedIn,
          outPath,
          vad,
          frameRate: Number(fr),
          refStream: els.refStream.value.trim() || null,
          refStreamKind,
          splitMode: els.splitMode.value,
          noFixFramerate: Boolean(els.noFixFramerate.checked),
          gss: Boolean(els.gss.checked),
        },
        {
          onProgress: ({ message, fraction }) => {
            if (message) els.syncBtn.textContent = message;
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
          setInputSubtitle(entry);
        }
      },
      onWindowMaximizedChange: syncMaximizedState,
      onAppVersion: applyAppVersion,
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
  const msg = "Error: " + ((err && err.message) || err);
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
