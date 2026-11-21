// Unified UI controller shared by the native desktop app (pywebview) and the
// browser WebAssembly app (Pyodide + ffmpeg.wasm). Platform-specific behavior
// is isolated in ./adapters/desktop.js and ./adapters/web.js.

const V = new URL(import.meta.url).searchParams.get("v") || "";
const withVersion = (url) => (V ? url + (url.includes("?") ? "&" : "?") + "v=" + V : url);

const els = {
  envSubtitle: document.getElementById("env-subtitle"),
  envBadge: document.getElementById("env-badge"),
  runtimeDot: document.getElementById("runtime-dot"),
  status: document.getElementById("status"),
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
};

let adapter = null;
let runtimeReady = false;
let busy = false;
let selectedRef = null; // string path (desktop) or File (web)
let selectedIn = null; // string path (desktop) or File (web)
let lastResult = null;
let streamProbeSeq = 0;

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

async function refreshStreams(refValue) {
  const seq = ++streamProbeSeq;
  els.refStream.innerHTML = '<option value="">Auto (recommended)</option>';
  els.refStream.value = "";
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
  if (update.level) {
    els.runtimeDot.className = update.level === "muted" ? "dot" : `dot ${update.level}`;
  }
  if (update.text !== undefined) {
    els.status.textContent = update.text;
  }
  if (update.badgeTitle !== undefined) {
    els.envBadge.title = update.badgeTitle;
  }
  if (adapter && adapter.capabilities.runtimeAction) {
    if (update.actionLabel !== undefined) {
      els.runtimeActionBtn.textContent = update.actionLabel;
      els.runtimeActionBtn.hidden = !update.actionLabel;
    }
    if (update.actionDisabled !== undefined) {
      els.runtimeActionBtn.disabled = Boolean(update.actionDisabled);
    }
  }
  if (update.ready !== undefined) {
    runtimeReady = Boolean(update.ready);
    refreshSyncBtn();
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
}

function configureUiForAdapter(ad) {
  const { meta, capabilities } = ad;

  document.title = meta.pageTitle || "ffblindsubsync";
  els.envSubtitle.textContent = meta.subtitle || "";
  if (meta.badgeText) {
    els.envBadge.textContent = meta.badgeText;
    els.envBadge.title = meta.badgeTitle || "";
    els.envBadge.hidden = false;
  } else {
    els.envBadge.hidden = true;
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
      opt.textContent = mode.label;
      if (mode.selected) opt.selected = true;
      els.vadMode.appendChild(opt);
    }
  }

  if (meta.footerText) {
    els.footer.textContent = meta.footerText;
    els.footer.hidden = false;
  } else {
    els.footer.hidden = true;
  }
}

function wireEvents(ad) {
  const { capabilities } = ad;

  // Drag-over visual highlight for both desktop and web.
  for (const [zone, target] of [
    [els.dropRef, "ref"],
    [els.dropIn, "in"],
  ]) {
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
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
    // Prevent accidental navigation if a user drops a media file outside the drop zones.
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => e.preventDefault());

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
      onRuntimeProgress: (prog) =>
        setProgressBar(els.runtimeProg, els.runtimeFill, els.runtimePct, prog),
      onExternalDrop: (target, entry) => {
        if (target === "ref") {
          setReference(entry);
        } else if (target === "in") {
          setInputSubtitle(entry);
        }
      },
    },
  });

  configureUiForAdapter(adapter);
  wireEvents(adapter);
  await adapter.init();
}

boot().catch((err) => {
  console.error(err);
  els.runtimeDot.className = "dot err";
  els.status.textContent = "Error: " + ((err && err.message) || err);
});
