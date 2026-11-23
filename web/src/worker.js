// Module Web Worker: runs Pyodide + ffsubsync off the main thread so the UI stays
// responsive. Audio decoding and media probing (ffmpeg.wasm) run on the main
// thread because nesting ffmpeg.wasm's worker inside this worker is unreliable.

let eng = null; // the engine module
let engine = null; // the booted engine

const onProgress = (fraction, message) => {
  post({ type: "progress", fraction, message });
};

self.onmessage = async (event) => {
  const msg = event.data || {};
  try {
    if (msg.type === "init") {
      eng = await import(msg.engineUrl);
      engine = await eng.bootEngine({
        configUrl: msg.configUrl,
        onStatus: (status) => post({ type: "status", status }),
      });
      post({
        type: "ready",
        capabilities: engine.capabilities,
        ffmpeg: engine.config.ffmpeg,
      });
    } else if (msg.type === "sync") {
      if (!engine) throw new Error("engine not initialized");
      post({ type: "progress", fraction: 0.02, message: "Syncing…" });
      post({
        type: "result",
        result: eng.syncWithEngine(engine, { ...msg.payload, onProgress }),
      });
    } else if (msg.type === "syncAudioPcm") {
      if (!engine) throw new Error("engine not initialized");
      post({ type: "progress", fraction: 0.0, message: "Detecting speech + aligning…" });
      const { pcm, channels = 1, frameRate, inName, inBytes, vad, options } = msg.payload;
      const result = eng.syncAudioWithEngine(engine, {
        pcm,
        channels,
        frameRate,
        inName,
        inBytes,
        vad,
        options,
        onProgress,
      });
      post({ type: "result", result });
    } else if (msg.type === "syncTimings") {
      if (!engine) throw new Error("engine not initialized");
      post({ type: "progress", fraction: 0.02, message: "Aligning subtitle timings…" });
      const { timings, inName, inBytes, options } = msg.payload;
      const result = eng.syncTimingsWithEngine(engine, {
        timings,
        inName,
        inBytes,
        options,
        onProgress,
      });
      post({ type: "result", result });
    } else {
      throw new Error("unknown worker message type: " + msg.type);
    }
  } catch (err) {
    post({ type: "error", error: formatError(err) });
  }
};

function formatError(err) {
  if (!err) return "Unknown error";
  const msg = err.message ? String(err.message) : "";
  const stack = err.stack ? String(err.stack) : "";
  if (msg && stack && !stack.includes(msg)) {
    return `${msg}\n${stack}`;
  }
  return stack || msg || String(err);
}

function post(message) {
  self.postMessage(message);
}
