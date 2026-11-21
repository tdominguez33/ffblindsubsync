// Native desktop adapter (pywebview + local Python / FFmpeg backend).

function suggestOutputPath(inPath) {
  if (!inPath) return "";
  const idx = inPath.lastIndexOf(".");
  if (idx > inPath.lastIndexOf("\\") && idx > inPath.lastIndexOf("/")) {
    const low = inPath.slice(idx).toLowerCase();
    if (low === ".srt" || low === ".ass" || low === ".ssa" || low === ".vtt") {
      return inPath.slice(0, idx) + ".synced" + low;
    }
    return inPath.slice(0, idx) + ".synced.srt";
  }
  return inPath + ".synced.srt";
}

function waitForPywebview() {
  if (window.pywebview && window.pywebview.api) {
    return Promise.resolve(window.pywebview.api);
  }
  return new Promise((resolve) => {
    window.addEventListener(
      "pywebviewready",
      () => resolve(window.pywebview.api),
      { once: true },
    );
  });
}

function buildFfmpegStatus(info) {
  if (info && info.available) {
    let text;
    let actionLabel;
    if (info.source === "custom") {
      text = `FFmpeg ready (custom · ${info.version || "local"})`;
      actionLabel = "Update FFmpeg";
    } else if (info.source === "portable") {
      text = `FFmpeg ready (portable · ${info.version || "local"})`;
      actionLabel = "Update FFmpeg";
    } else {
      text = `FFmpeg ready (system PATH · ${info.version || "system"})`;
      actionLabel = "Download portable FFmpeg";
    }
    return {
      level: "ok",
      text,
      actionLabel,
      actionDisabled: false,
      visible: false,
      ready: true,
      ffmpegInfo: info,
    };
  }
  const errText =
    (info && info.custom_error) ||
    "FFmpeg not found (will auto-download on Sync, or click to download now)";
  return {
    level: "err",
    text: errText,
    actionLabel: "Download latest FFmpeg",
    actionDisabled: false,
    visible: true,
    ready: true,
    ffmpegInfo: info || null,
  };
}

export function createAdapter({ callbacks }) {
  let api = null;
  let pollTimer = null;
  let lastDlSeq = 0;
  let lastSyncSeq = 0;
  let pendingSync = null;

  // Exposed for Python DOM drag-drop handler and window-state events (gui/desktop_app.py).
  window.setDroppedPath = function (target, fullPath) {
    if (!fullPath) return;
    callbacks.onExternalDrop(target, { value: fullPath, display: fullPath });
  };
  window.setWindowMaximized = function (isMax) {
    if (callbacks.onWindowMaximizedChange) {
      callbacks.onWindowMaximizedChange(Boolean(isMax));
    }
  };

  function ensurePolling() {
    if (pollTimer) return;
    pollTimer = setInterval(pollOnce, 150);
  }

  async function pollOnce() {
    if (!api) return;
    let st;
    try {
      st = await api.get_poll_state();
    } catch (err) {
      if (pendingSync) {
        const current = pendingSync;
        pendingSync = null;
        current.resolve({ ok: false, error: String(err) });
      }
      clearInterval(pollTimer);
      pollTimer = null;
      return;
    }

    const dl = st.ffmpeg_dl;
    if (dl && dl.active) {
      const hasFrac = dl.fraction !== null && dl.fraction !== undefined;
      let text;
      if (hasFrac) {
        const mb = (dl.downloaded / (1024 * 1024)).toFixed(1);
        const totMb = dl.total ? (dl.total / (1024 * 1024)).toFixed(1) : "?";
        text = `Downloading FFmpeg… ${mb} / ${totMb} MB`;
      } else {
        text = "Downloading FFmpeg…";
      }
      callbacks.onRuntimeStatus({
        level: "warn",
        text,
        actionDisabled: true,
        visible: true,
        ready: true,
      });
      callbacks.onRuntimeProgress({
        active: true,
        fraction: hasFrac ? dl.fraction : null,
      });
    } else if (dl && dl.done && dl.seq !== lastDlSeq) {
      lastDlSeq = dl.seq;
      callbacks.onRuntimeProgress({ active: false, fraction: null });
      if (dl.error) {
        callbacks.onRuntimeStatus({
          level: "err",
          text: "FFmpeg download failed: " + dl.error,
          actionDisabled: false,
          visible: true,
          ready: true,
        });
      } else if (dl.info) {
        callbacks.onRuntimeStatus(buildFfmpegStatus(dl.info));
      } else {
        callbacks.onRuntimeStatus({ actionDisabled: false, ready: true });
      }
    }

    const sy = st.sync;
    if (sy && sy.active) {
      if (pendingSync && pendingSync.onProgress) {
        pendingSync.onProgress({
          message: sy.message || "Syncing…",
          fraction: sy.fraction !== null && sy.fraction !== undefined ? sy.fraction : null,
        });
      }
    } else if (sy && sy.done && sy.seq !== lastSyncSeq) {
      lastSyncSeq = sy.seq;
      if (sy.ffmpeg_info) {
        callbacks.onRuntimeStatus(buildFfmpegStatus(sy.ffmpeg_info));
      }
      if (pendingSync) {
        const current = pendingSync;
        pendingSync = null;
        current.resolve({
          ok: Boolean(sy.ok),
          offset_seconds: sy.offset_seconds,
          framerate_scale_factor: sy.framerate_scale_factor,
          split_segments: sy.split_segments,
          dropped_cues: sy.dropped_cues || 0,
          elapsed_seconds: sy.elapsed_seconds,
          output_path: sy.output_path,
          output_label: sy.output_path || "",
          error: sy.error || null,
        });
      }
    }

    if ((!dl || !dl.active) && (!sy || !sy.active)) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  return {
    meta: {
      pageTitle: "ffblindsubsync",
      subtitle: "",
      badgeText: "",
      badgeVariant: "desktop",
      badgeTitle: "",
      footerText: null,
    },
    capabilities: {
      editablePaths: true,
      outputPath: true,
      streamProbe: true,
      runtimeAction: true,
      customWindowControls: true,
      settings: true,
    },

    windowControls: {
      minimize() {
        if (api) api.window_minimize();
      },
      async toggleMaximize() {
        if (!api) return false;
        return Boolean(await api.window_toggle_maximize());
      },
      close() {
        if (api) api.window_close();
      },
      startDrag() {
        if (api) api.window_start_drag();
      },
      startResize(hitCode) {
        if (api) api.window_start_resize(hitCode);
      },
    },

    settings: {
      async pickFfmpegDir() {
        if (!api) return null;
        return api.pick_ffmpeg_dir();
      },
      async setFfmpegPath(path) {
        if (!api) return null;
        const res = await api.set_ffmpeg_path(path || "");
        const ff =
          res && res.ffmpeg
            ? { ...res.ffmpeg, custom_path: res.custom_ffmpeg_path ?? res.ffmpeg.custom_path ?? "" }
            : null;
        callbacks.onRuntimeStatus(buildFfmpegStatus(ff));
        return ff;
      },
      openFfmpegFolder() {
        if (api) api.open_ffmpeg_folder();
      },
    },

    async init() {
      callbacks.onRuntimeStatus({
        level: "muted",
        text: "Checking FFmpeg…",
        actionLabel: "Download latest FFmpeg",
        actionDisabled: true,
        visible: false,
        ready: false,
      });
      api = await waitForPywebview();
      const state = await api.get_init_state();
      const ff =
        state && state.ffmpeg
          ? {
              ...state.ffmpeg,
              portable_dir: state.portable_dir,
              custom_path: state.custom_ffmpeg_path ?? state.ffmpeg.custom_path ?? "",
            }
          : null;
      callbacks.onRuntimeStatus(buildFfmpegStatus(ff));
    },

    async pickReference() {
      if (!api) return null;
      const p = await api.pick_reference();
      return p ? { value: p, display: p } : null;
    },

    async pickSubtitle() {
      if (!api) return null;
      const p = await api.pick_subtitle();
      return p ? { value: p, display: p } : null;
    },

    async pickOutput(defaultPath) {
      if (!api) return null;
      return api.pick_output(defaultPath || "");
    },

    suggestOutputPath,

    async probeStreams(refPath) {
      if (!api || !refPath) return { streams: [], has_text_subs: false, has_pgs_subs: false };
      return api.probe_streams(refPath);
    },

    async runRuntimeAction() {
      if (!api) return;
      callbacks.onRuntimeStatus({ actionDisabled: true, visible: true });
      callbacks.onRuntimeProgress({ active: true, fraction: null });
      await api.start_ffmpeg_download();
      ensurePolling();
    },

    async sync(request, { onProgress }) {
      if (!api) {
        return { ok: false, error: "Desktop API not ready." };
      }
      const res = await api.start_sync({
        ref_path: String(request.ref),
        in_path: String(request.input),
        out_path: String(request.outPath),
        vad: request.vad,
        frame_rate: Number(request.frameRate),
        ref_stream: request.refStream || null,
        ref_stream_kind: request.refStreamKind || null,
        split_mode: request.splitMode,
        no_fix_framerate: Boolean(request.noFixFramerate),
        gss: Boolean(request.gss),
      });
      if (!res || !res.started) {
        return { ok: false, error: "A sync operation is already running." };
      }
      return new Promise((resolve) => {
        pendingSync = { resolve, onProgress };
        ensurePolling();
      });
    },

    openOutput(result) {
      if (api && result && result.output_path) {
        api.open_output_folder(result.output_path);
      }
    },

    onThemeChange(isDark) {
      if (api && typeof api.set_theme === "function") {
        api.set_theme(Boolean(isDark)).catch(() => {});
      }
    },
  };
}
