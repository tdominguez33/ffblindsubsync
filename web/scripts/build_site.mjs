// Assemble the deployable static site in web/dist/site/ from the shared ui/
// directory and the web/ runtime assets (src/, vendor/, build.config.json).
// Works cross-platform (Linux, macOS, Windows) with zero external dependencies.

import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import { execSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(here, "..");
const repoRoot = path.resolve(webDir, "..");
const uiDir = path.join(repoRoot, "ui");
const distDir = path.join(webDir, "dist", "site");

function getGitVersion() {
  if (process.env.VERSION) return process.env.VERSION.trim();
  try {
    return execSync("git describe --tags --always --dirty", { cwd: repoRoot })
      .toString()
      .trim();
  } catch {
    return "dev";
  }
}

async function computeBuildHash(files) {
  const hash = createHash("sha256");
  for (const filePath of files) {
    hash.update(path.relative(repoRoot, filePath));
    hash.update(await fs.readFile(filePath));
  }
  return hash.digest("hex").slice(0, 8);
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
  const gitVer = getGitVersion();
  const buildHash = await computeBuildHash([
    path.join(uiDir, "index.html"),
    path.join(uiDir, "styles.css"),
    path.join(uiDir, "app.js"),
    path.join(uiDir, "adapters", "web.js"),
    path.join(webDir, "build.config.json"),
    path.join(webDir, "src", "worker.js"),
    path.join(webDir, "src", "ffsubsync_engine.mjs"),
    path.join(webDir, "src", "ffmpeg_decode.mjs"),
    path.join(webDir, "vendor", "py_sources.json"),
  ]);
  // Always include a content hash so local dirty rebuilds bust browser & Worker caches.
  const version = `${gitVer}-${buildHash}`;

  await fs.rm(distDir, { recursive: true, force: true });
  await fs.mkdir(path.join(distDir, "adapters"), { recursive: true });

  // Shared UI assets + web backend adapter
  for (const file of ["styles.css", "app.js"]) {
    await fs.copyFile(path.join(uiDir, file), path.join(distDir, file));
  }
  await fs.copyFile(
    path.join(uiDir, "adapters", "web.js"),
    path.join(distDir, "adapters", "web.js"),
  );

  // Cache-busting: version entry CSS/JS so deploys never mix fresh + stale files.
  // app.js threads the same ?v= query param to adapters/web.js and everything it loads.
  const rawHtml = await fs.readFile(path.join(uiDir, "index.html"), "utf8");
  const versionedHtml = rawHtml
    .replace('href="./styles.css"', `href="./styles.css?v=${version}"`)
    .replace('src="./app.js"', `src="./app.js?v=${version}"`);
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

  console.log(`assembled ${path.relative(repoRoot, distDir)} (v=${version})`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
