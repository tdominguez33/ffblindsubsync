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

function buildFfmpegStatus(info, t) {
  if (info && info.available) {
    let text;
    let actionLabel;
    const verLocal = info.version || t("ffmpeg.versionLocal");
    const verSystem = info.version || t("ffmpeg.versionSystem");
    if (info.source === "custom") {
      text = t("ffmpeg.readyCustom", { version: verLocal });
      actionLabel = t("ffmpeg.updateBtn");
    } else if (info.source === "portable") {
      text = t("ffmpeg.readyPortable", { version: verLocal });
      actionLabel = t("ffmpeg.updateBtn");
    } else if (info.source === "appdata") {
      text = t("ffmpeg.readyAppdata", { version: verLocal });
      actionLabel = t("ffmpeg.updateBtn");
    } else {
      text = t("ffmpeg.readySystem", { version: verSystem });
      actionLabel = t("ffmpeg.downloadBtn");
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
  let errText;
  if (info && info.custom_path && info.custom_error) {
    errText = t("ffmpeg.notFoundInPath", { path: info.custom_path });
  } else {
    errText = (info && info.custom_error) || t("ffmpeg.notFound");
  }
  return {
    level: "err",
    text: errText,
    actionLabel: t("ffmpeg.downloadLatestBtn"),
    actionDisabled: false,
    visible: true,
    ready: true,
    ffmpegInfo: info || null,
  };
}

export function createAdapter({ i18n, callbacks }) {
  const t = (key, params) => (i18n ? i18n.t(key, params) : key);
  let api = null;
  let pollTimer = null;
  let lastDlSeq = 0;
  let lastSyncSeq = 0;
  let pendingSync = null;
  let lastStatusState = null;

  function emitRuntimeStatus(stateObj) {
    lastStatusState = stateObj;
    if (!stateObj) return;
    if (stateObj.kind === "checking") {
      callbacks.onRuntimeStatus({
        level: "muted",
        text: t("ffmpeg.checking"),
        actionLabel: t("ffmpeg.downloadLatestBtn"),
        actionDisabled: true,
        visible: false,
        ready: false,
      });
    } else if (stateObj.kind === "ffmpeg") {
      callbacks.onRuntimeStatus(buildFfmpegStatus(stateObj.info, t));
    } else if (stateObj.kind === "downloading") {
      const dl = stateObj.dl || {};
      const hasFrac = dl.fraction !== null && dl.fraction !== undefined;
      let text;
      if (hasFrac) {
        const mb = (dl.downloaded / (1024 * 1024)).toFixed(1);
        const totalMb = dl.total ? (dl.total / (1024 * 1024)).toFixed(1) : "?";
        text = t("ffmpeg.downloadingSize", { mb, totalMb });
      } else {
        text = t("ffmpeg.downloading");
      }
      callbacks.onRuntimeStatus({
        level: "warn",
        text,
        actionDisabled: true,
        visible: true,
        ready: true,
      });
    } else if (stateObj.kind === "download_error") {
      callbacks.onRuntimeStatus({
        level: "err",
        text: t("ffmpeg.downloadFailed", { error: stateObj.error }),
        actionDisabled: false,
        visible: true,
        ready: true,
      });
    }
  }

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
      emitRuntimeStatus({ kind: "downloading", dl });
      callbacks.onRuntimeProgress({
        active: true,
        fraction: hasFrac ? dl.fraction : null,
      });
    } else if (dl && dl.done && dl.seq !== lastDlSeq) {
      lastDlSeq = dl.seq;
      callbacks.onRuntimeProgress({ active: false, fraction: null });
      if (dl.error) {
        emitRuntimeStatus({ kind: "download_error", error: dl.error });
      } else if (dl.info) {
        emitRuntimeStatus({ kind: "ffmpeg", info: dl.info });
      } else {
        callbacks.onRuntimeStatus({ actionDisabled: false, ready: true });
      }
    }

    const sy = st.sync;
    if (sy && sy.active) {
      if (pendingSync && pendingSync.onProgress) {
        pendingSync.onProgress({
          message: sy.message || t("progress.syncing"),
          messageKey: sy.message_key || (!sy.message ? "progress.syncing" : null),
          messageParams: sy.message_params || null,
          fraction: sy.fraction !== null && sy.fraction !== undefined ? sy.fraction : null,
        });
      }
    } else if (sy && sy.done && sy.seq !== lastSyncSeq) {
      lastSyncSeq = sy.seq;
      if (sy.ffmpeg_info) {
        emitRuntimeStatus({ kind: "ffmpeg", info: sy.ffmpeg_info });
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
          error_key: sy.error_key || null,
          error_params: sy.error_params || null,
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
      get pageTitle() {
        return t("app.title");
      },
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
      ffmpegSettings: true,
    },

    retranslate() {
      if (lastStatusState) {
        emitRuntimeStatus(lastStatusState);
      }
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
        emitRuntimeStatus({ kind: "ffmpeg", info: ff });
        return ff;
      },
      openFfmpegFolder() {
        if (api) api.open_ffmpeg_folder();
      },
    },

    async init() {
      emitRuntimeStatus({ kind: "checking" });
      api = await waitForPywebview();
      const state = await api.get_init_state();
      if (i18n && state) {
        if (Array.isArray(state.system_languages) && state.system_languages.length) {
          i18n.setSystemLanguages(state.system_languages);
        }
        if (typeof state.language === "string" && state.language.trim()) {
          i18n.setPreference(state.language, { persist: true });
        }
      }
      if (state && state.version && callbacks.onAppVersion) {
        callbacks.onAppVersion(state.version);
      }
      const ff =
        state && state.ffmpeg
          ? {
            ...state.ffmpeg,
            portable_dir: state.portable_dir,
            custom_path: state.custom_ffmpeg_path ?? state.ffmpeg.custom_path ?? "",
          }
          : null;
      emitRuntimeStatus({ kind: "ffmpeg", info: ff });
    },

    async pickReference() {
      if (!api) return null;
      const fileTypes = [
        `${t("dialogs.mediaFiles")} (*.mkv;*.mp4;*.avi;*.mov;*.webm;*.m4v;*.mka;*.mp3;*.aac;*.flac;*.wav;*.srt;*.ass;*.ssa;*.sub;*.vtt)`,
        `${t("dialogs.videoFiles")} (*.mkv;*.mp4;*.avi;*.mov;*.webm;*.m4v)`,
        `${t("dialogs.subtitleFiles")} (*.srt;*.ass;*.ssa;*.sub;*.vtt)`,
        `${t("dialogs.allFiles")} (*.*)`,
      ];
      const p = await api.pick_reference(fileTypes);
      return p ? { value: p, display: p } : null;
    },

    async pickSubtitle() {
      if (!api) return null;
      const fileTypes = [
        `${t("dialogs.subtitles")} (*.srt;*.ass;*.ssa;*.sub;*.vtt)`,
        `${t("dialogs.allFiles")} (*.*)`,
      ];
      const p = await api.pick_subtitle(fileTypes);
      return p ? { value: p, display: p } : null;
    },

    async pickOutput(defaultPath) {
      if (!api) return null;
      const fileTypes = [
        `${t("dialogs.subRip")} (*.srt)`,
        `${t("dialogs.allFiles")} (*.*)`,
      ];
      return api.pick_output(defaultPath || "", fileTypes);
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
        return {
          ok: false,
          error: t("errors.desktopApiNotReady"),
          error_key: "errors.desktopApiNotReady",
        };
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
        return {
          ok: false,
          error: t("errors.syncAlreadyRunning"),
          error_key: "errors.syncAlreadyRunning",
        };
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
        api.set_theme(Boolean(isDark)).catch(() => { });
      }
    },

    onLanguageChange(pref) {
      if (api && typeof api.set_language === "function") {
        api.set_language(String(pref || "system")).catch(() => { });
      }
    },
  };
}
