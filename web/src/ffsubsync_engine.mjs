// Boots Pyodide and drives the vendored ffsubsync via the Python bridge.
// Kept free of DOM/worker specifics so it can be reused from a Web Worker (the
// browser app) or, in principle, a Node test harness.

export async function bootEngine({ configUrl, onStatus } = {}) {
  const status = (s) => onStatus && onStatus(s);

  // Cache-busting: configUrl carries ?v=<build>; append it to our other
  // same-origin fetches (but NOT wheel URLs — micropip parses the filename from
  // the URL, and those are already content-addressed by name+version).
  const v = new URL(configUrl).searchParams.get("v") || "";
  const withV = (u) => (v ? u + (u.includes("?") ? "&" : "?") + "v=" + v : u);

  status("fetching config…");
  const config = await (await fetch(configUrl)).json();

  status("loading Python runtime…");
  const { loadPyodide } = await import(config.pyodideCdn + "pyodide.mjs");
  const pyodide = await loadPyodide({ indexURL: config.pyodideCdn });

  status("loading runtime packages…");
  // corePackages (numpy, micropip, charset-normalizer, typing-extensions, tqdm,
  // setuptools) come from the Pyodide CDN, not PyPI.
  await pyodide.loadPackage(config.corePackages);

  // Install every vendored wheel from *our own origin* — never PyPI. This covers
  // the pure-Python deps (pysubs2, ffmpeg-python + future) and, when built, the
  // webrtcvad wasm wheel. deps=False guarantees micropip never contacts PyPI, so
  // the site works even where PyPI is blocked.
  status("installing libraries…");
  const capabilities = { webrtcvad: false, webrtcvadError: "", cchardet: false };
  let wheels = [];
  if (config.wheelManifest) {
    try {
      wheels = (await (await fetch(withV(new URL(config.wheelManifest, configUrl).href))).json()) || [];
    } catch (_) {
      wheels = [];
    }
  }
  if (wheels.length) {
    const urls = wheels.map((name) => new URL(config.wheelDir + name, configUrl).href);
    const pyUrls = pyodide.toPy(urls);
    pyodide.globals.set("_wheel_urls", pyUrls);
    await pyodide.runPythonAsync(
      "import micropip; await micropip.install(list(_wheel_urls), deps=False)",
    );
    if (pyUrls.destroy) pyUrls.destroy();
    // webrtcvad (if bundled) — report whether it actually imports (needs setuptools
    // for its pkg_resources import, which is in corePackages).
    const probe = pyodide.runPython(`
def _probe():
    try:
        import webrtcvad  # noqa: F401
        return (True, "")
    except Exception:
        import traceback
        return (False, traceback.format_exc())
_probe()
`);
    const [ok, err] = probe.toJs();
    probe.destroy();
    capabilities.webrtcvad = ok;
    if (!ok) {
      capabilities.webrtcvadError = err;
      console.warn("webrtcvad not available:\n" + err);
    }
    // cchardet (faust-cchardet wasm wheel, if bundled) — subtitle_parser prefers
    // it over charset-normalizer for exact CLI-parity encoding detection. Purely
    // diagnostic: charset-normalizer (a core package) is the fallback either way.
    capabilities.cchardet = !!pyodide.runPython(`
def _cc_probe():
    try:
        import cchardet  # noqa: F401
        return True
    except Exception:
        return False
_cc_probe()
`);
  }

  status("loading ffblindsubsync…");
  const sources = await (
    await fetch(withV(new URL(config.pySourcesManifest, configUrl).href))
  ).json();
  const FS = pyodide.FS;
  FS.mkdirTree("/lib");
  for (const [rel, text] of Object.entries(sources)) {
    const full = "/lib/" + rel;
    FS.mkdirTree(full.slice(0, full.lastIndexOf("/")));
    FS.writeFile(full, text);
  }
  pyodide.runPython('import sys; sys.path.insert(0, "/lib")');
  // Warm import so any load error surfaces here rather than mid-sync.
  pyodide.runPython("import ffsubsync_bridge");

  status("ready");
  return { pyodide, config, capabilities };
}

export function syncWithEngine(
  engine,
  { refName, refBytes, inName, inBytes, options = {}, onProgress = null },
) {
  const { pyodide } = engine;
  const opts = pyodide.toPy(options);
  pyodide.globals.set("_ref_name", refName);
  pyodide.globals.set("_in_name", inName);
  pyodide.globals.set("_ref_bytes", refBytes);
  pyodide.globals.set("_in_bytes", inBytes);
  pyodide.globals.set("_opts", opts);
  pyodide.globals.set("_on_progress", onProgress);
  let proxy;
  try {
    proxy = pyodide.runPython(`
import ffsubsync_bridge
ffsubsync_bridge.sync_subtitles(
    _ref_name, _ref_bytes, _in_name, _in_bytes,
    progress_cb=_on_progress,
    **dict(_opts),
)
`);
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    pyodide.globals.set("_on_progress", null);
    pyodide.globals.set("_ref_bytes", null);
    pyodide.globals.set("_in_bytes", null);
    pyodide.globals.set("_opts", null);
    if (proxy) proxy.destroy();
    if (opts && opts.destroy) opts.destroy();
  }
}

// Sync against decoded audio PCM (mono or dual-channel mix+FC s16le at frameRate)
// from a video/audio reference. `vad` is "webrtc" or "auditok".
export function syncAudioWithEngine(
  engine,
  { pcm, channels = 1, frameRate, inName, inBytes, vad, options = {}, onProgress = null },
) {
  const { pyodide } = engine;
  const opts = pyodide.toPy(options);
  pyodide.globals.set("_pcm", pcm);
  pyodide.globals.set("_channels", Number(channels) || 1);
  pyodide.globals.set("_frame_rate", frameRate);
  pyodide.globals.set("_in_name", inName);
  pyodide.globals.set("_in_bytes", inBytes);
  pyodide.globals.set("_vad", vad);
  pyodide.globals.set("_opts", opts);
  pyodide.globals.set("_on_progress", onProgress);
  let proxy;
  try {
    proxy = pyodide.runPython(`
import ffsubsync_bridge
ffsubsync_bridge.sync_with_audio(
    _pcm, _frame_rate, _in_name, _in_bytes, vad=_vad,
    channels=_channels,
    progress_cb=_on_progress,
    **dict(_opts),
)
`);
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    pyodide.globals.set("_on_progress", null);
    pyodide.globals.set("_pcm", null);
    pyodide.globals.set("_in_bytes", null);
    pyodide.globals.set("_opts", null);
    if (proxy) proxy.destroy();
    if (opts && opts.destroy) opts.destroy();
  }
}

// Sync against extracted (start_s, end_s) timing intervals (e.g. PGS/VobSub).
export function syncTimingsWithEngine(
  engine,
  { timings, inName, inBytes, options = {}, onProgress = null },
) {
  const { pyodide } = engine;
  const pyTimings = pyodide.toPy(timings);
  const opts = pyodide.toPy(options);
  pyodide.globals.set("_timings", pyTimings);
  pyodide.globals.set("_in_name", inName);
  pyodide.globals.set("_in_bytes", inBytes);
  pyodide.globals.set("_opts", opts);
  pyodide.globals.set("_on_progress", onProgress);
  let proxy;
  try {
    proxy = pyodide.runPython(`
import ffsubsync_bridge
ffsubsync_bridge.sync_with_timings(
    _timings, _in_name, _in_bytes,
    progress_cb=_on_progress,
    **dict(_opts),
)
`);
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    pyodide.globals.set("_on_progress", null);
    pyodide.globals.set("_timings", null);
    pyodide.globals.set("_in_bytes", null);
    pyodide.globals.set("_opts", null);
    if (proxy) proxy.destroy();
    if (pyTimings && pyTimings.destroy) pyTimings.destroy();
    if (opts && opts.destroy) opts.destroy();
  }
}

export function prepareRefSubsWithEngine(
  engine,
  { cacheKey, refName, refBytes, options = {} },
) {
  const { pyodide } = engine;
  const opts = pyodide.toPy(options);
  pyodide.globals.set("_cache_key", cacheKey);
  pyodide.globals.set("_ref_name", refName);
  pyodide.globals.set("_ref_bytes", refBytes);
  pyodide.globals.set("_opts", opts);
  let proxy;
  try {
    proxy = pyodide.runPython(`
import ffsubsync_bridge
ffsubsync_bridge.prepare_reference_subtitles(
    _cache_key, _ref_name, _ref_bytes, **dict(_opts)
)
`);
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    pyodide.globals.set("_ref_bytes", null);
    pyodide.globals.set("_opts", null);
    if (proxy) proxy.destroy();
    if (opts && opts.destroy) opts.destroy();
  }
}

export function prepareRefAudioWithEngine(
  engine,
  { cacheKey, pcm, channels = 1, frameRate, vad = "webrtc", nonSpeechLabel = 0.0, onProgress = null },
) {
  const { pyodide } = engine;
  pyodide.globals.set("_cache_key", cacheKey);
  pyodide.globals.set("_pcm", pcm);
  pyodide.globals.set("_channels", Number(channels) || 1);
  pyodide.globals.set("_frame_rate", frameRate);
  pyodide.globals.set("_vad", vad);
  pyodide.globals.set("_non_speech_label", nonSpeechLabel);
  pyodide.globals.set("_on_progress", onProgress);
  let proxy;
  try {
    proxy = pyodide.runPython(`
import ffsubsync_bridge
ffsubsync_bridge.prepare_reference_audio(
    _cache_key, _pcm, _frame_rate,
    vad=_vad, channels=_channels,
    non_speech_label=_non_speech_label,
    progress_cb=_on_progress,
)
`);
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    pyodide.globals.set("_on_progress", null);
    pyodide.globals.set("_pcm", null);
    if (proxy) proxy.destroy();
  }
}

export function prepareRefTimingsWithEngine(engine, { cacheKey, timings }) {
  const { pyodide } = engine;
  const pyTimings = pyodide.toPy(timings);
  pyodide.globals.set("_cache_key", cacheKey);
  pyodide.globals.set("_timings", pyTimings);
  let proxy;
  try {
    proxy = pyodide.runPython(`
import ffsubsync_bridge
ffsubsync_bridge.prepare_reference_timings(_cache_key, _timings)
`);
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    pyodide.globals.set("_timings", null);
    if (proxy) proxy.destroy();
    if (pyTimings && pyTimings.destroy) pyTimings.destroy();
  }
}

export function syncCachedWithEngine(
  engine,
  { cacheKey, inputs, options = {}, onProgress = null },
) {
  const { pyodide } = engine;
  const opts = pyodide.toPy(options);
  const pyInputs = pyodide.toPy(inputs);
  pyodide.globals.set("_cache_key", cacheKey);
  pyodide.globals.set("_inputs", pyInputs);
  pyodide.globals.set("_opts", opts);
  pyodide.globals.set("_on_progress", onProgress);
  let proxy;
  try {
    proxy = pyodide.runPython(`
import ffsubsync_bridge
ffsubsync_bridge.sync_with_cached_reference(
    _cache_key, _inputs,
    progress_cb=_on_progress,
    **dict(_opts),
)
`);
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    pyodide.globals.set("_on_progress", null);
    pyodide.globals.set("_inputs", null);
    pyodide.globals.set("_opts", null);
    if (proxy) proxy.destroy();
    if (pyInputs && pyInputs.destroy) pyInputs.destroy();
    if (opts && opts.destroy) opts.destroy();
  }
}
