// Assemble the deployable static site in web/dist/site/ from the shared ui/
// directory and the web/ runtime assets (src/, vendor/, build.config.json).
// Works cross-platform (Linux, macOS, Windows) with zero external dependencies.

import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(here, "..");
const repoRoot = path.resolve(webDir, "..");
const uiDir = path.join(repoRoot, "ui");
const guiDir = path.join(repoRoot, "gui");
const distDir = path.join(webDir, "dist", "site");

async function getAppVersion() {
  if (process.env.VERSION) return process.env.VERSION.trim();
  try {
    const text = await fs.readFile(path.join(repoRoot, "pyproject.toml"), "utf8");
    let inProject = false;
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      if (line.startsWith("[")) {
        inProject = line === "[project]";
        continue;
      }
      if (inProject) {
        const m = line.match(/^version\s*=\s*["']([^"']+)["']/);
        if (m) return m[1].trim();
      }
    }
  } catch {
    /* fall through */
  }
  return "0.0.0";
}

async function computeBuildHash(files) {
  const hash = createHash("sha256");
  for (const filePath of files) {
    hash.update(path.relative(repoRoot, filePath));
    hash.update(await fs.readFile(filePath));
  }
  return hash.digest("hex").slice(0, 8);
}

async function listFilesRecursive(dir) {
  const out = [];
  const entries = await fs.readdir(dir, { withFileTypes: true });
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...(await listFilesRecursive(full)));
    } else {
      out.push(full);
    }
  }
  return out;
}

async function copyDir(src, dest, { excludeNames = new Set() } = {}) {
  await fs.mkdir(dest, { recursive: true });
  const entries = await fs.readdir(src, { withFileTypes: true });
  for (const entry of entries) {
    if (excludeNames.has(entry.name)) continue;
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      await copyDir(srcPath, destPath, { excludeNames });
    } else {
      await fs.copyFile(srcPath, destPath);
    }
  }
}

async function main() {
  const appVersion = await getAppVersion();
  const i18nFiles = await listFilesRecursive(path.join(uiDir, "i18n"));
  const buildHash = await computeBuildHash([
    path.join(uiDir, "index.html"),
    path.join(uiDir, "styles.css"),
    path.join(uiDir, "app.js"),
    path.join(uiDir, "adapters", "web.js"),
    ...i18nFiles,
    path.join(guiDir, "icon.ico"),
    path.join(webDir, "build.config.json"),
    path.join(webDir, "src", "worker.js"),
    path.join(webDir, "src", "ffsubsync_engine.mjs"),
    path.join(webDir, "src", "ffmpeg_decode.mjs"),
    path.join(webDir, "src", "archive_extract.mjs"),
    path.join(webDir, "vendor", "py_sources.json"),
  ]);
  // Always include a content hash so local dirty rebuilds bust browser & Worker caches.
  const cacheVersion = `${appVersion}-${buildHash}`;

  await fs.rm(distDir, { recursive: true, force: true });
  await fs.mkdir(path.join(distDir, "adapters"), { recursive: true });

  // Shared UI assets + i18n locales + web backend adapter
  await fs.copyFile(path.join(uiDir, "styles.css"), path.join(distDir, "styles.css"));
  const rawAppJs = await fs.readFile(path.join(uiDir, "app.js"), "utf8");
  await fs.writeFile(
    path.join(distDir, "app.js"),
    rawAppJs.replace('from "./i18n/index.js"', `from "./i18n/index.js?v=${cacheVersion}"`),
    "utf8",
  );
  await copyDir(path.join(uiDir, "i18n"), path.join(distDir, "i18n"));
  const rawI18nIndex = await fs.readFile(path.join(distDir, "i18n", "index.js"), "utf8");
  await fs.writeFile(
    path.join(distDir, "i18n", "index.js"),
    rawI18nIndex.replace(/from\s+"(\.\/locales\/[^"]+\.js)"/g, `from "$1?v=${cacheVersion}"`),
    "utf8",
  );
  await fs.copyFile(
    path.join(uiDir, "adapters", "web.js"),
    path.join(distDir, "adapters", "web.js"),
  );
  await fs.copyFile(path.join(guiDir, "icon.ico"), path.join(distDir, "favicon.ico"));

  // Cache-busting + inject single-source-of-truth app version into HTML meta tag.
  // app.js threads the same ?v= query param to adapters/web.js and everything it loads.
  const rawHtml = await fs.readFile(path.join(uiDir, "index.html"), "utf8");
  const versionedHtml = rawHtml
    .replace(
      '<meta name="app-version" content="" />',
      `<meta name="app-version" content="${appVersion}" />`,
    )
    .replace('href="./favicon.ico"', `href="./favicon.ico?v=${cacheVersion}"`)
    .replace('href="./styles.css"', `href="./styles.css?v=${cacheVersion}"`)
    .replace('src="./app.js"', `src="./app.js?v=${cacheVersion}"`);
  await fs.writeFile(path.join(distDir, "index.html"), versionedHtml, "utf8");

  // Web runtime config, worker/engine JS, and vendored assets
  await fs.copyFile(
    path.join(webDir, "build.config.json"),
    path.join(distDir, "build.config.json"),
  );
  await copyDir(path.join(webDir, "src"), path.join(distDir, "src"), {
    excludeNames: new Set(["__pycache__", "ffsubsync_bridge.py"]),
  });
  await copyDir(path.join(webDir, "vendor"), path.join(distDir, "vendor"));
  const licensesPath = path.join(repoRoot, "THIRD_PARTY_LICENSES.md");
  try {
    await fs.copyFile(licensesPath, path.join(distDir, "THIRD_PARTY_LICENSES.md"));
  } catch {
    /* optional */
  }

  console.log(`assembled ${path.relative(repoRoot, distDir)} (v=${appVersion}, cache=${cacheVersion})`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
