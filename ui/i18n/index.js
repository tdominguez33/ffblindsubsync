// Lightweight, extensible i18n engine for ffblindsubsync (desktop + web).
//
// Adding a new language:
//   1. Copy ./locales/en.js to ./locales/<code>.js and translate its `strings`.
//   2. Import it below and add it to REGISTERED_LOCALES.
//   Both desktop (PyInstaller) and web (build_site.mjs) auto-bundle ui/i18n/.

import en from "./locales/en.js";
import es from "./locales/es.js";

export const DEFAULT_LOCALE = "en";
export const SYSTEM_LANGUAGE = "system";
export const STORAGE_KEY = "ffs_lang";

export const REGISTERED_LOCALES = [en, es];

/**
 * Normalize a BCP-47 / POSIX locale tag to lowercase hyphenated subtags
 * (e.g. "es_AR.UTF-8" -> "es-ar", "EN_us@euro" -> "en-us").
 */
export function normalizeLocaleTag(tag) {
  if (typeof tag !== "string") return "";
  const cleaned = tag
    .trim()
    .split(/[.;@]/)[0]
    .replace(/_/g, "-")
    .toLowerCase();
  if (!cleaned || cleaned === "c" || cleaned === "posix") return "";
  return cleaned;
}

/**
 * Match an ordered list of preferred language tags against supported app locale
 * codes, ignoring region differences when no exact/prefix match exists:
 *   - Exact match (e.g. "es-AR" -> "es-ar" if registered)
 *   - Subtag prefix match (e.g. "es-419" -> "es", "zh-Hans-CN" -> "zh-hans")
 *   - Primary language match (e.g. "es-AR" -> "es-ES" if only "es-ES" is registered)
 * Falls back to `fallback` ("en") if nothing matches.
 */
export function matchLocale(preferredTags, supportedCodes, fallback = DEFAULT_LOCALE) {
  const codes = Array.isArray(supportedCodes) ? supportedCodes.filter(Boolean) : [];
  if (!codes.length) return fallback;

  const exactMap = new Map();
  const primaryMap = new Map();
  for (const code of codes) {
    const norm = normalizeLocaleTag(code);
    if (!norm) continue;
    if (!exactMap.has(norm)) exactMap.set(norm, code);
    const primary = norm.split("-")[0];
    if (primary && !primaryMap.has(primary)) primaryMap.set(primary, code);
  }

  const tags = Array.isArray(preferredTags) ? preferredTags : [preferredTags];
  for (const raw of tags) {
    const norm = normalizeLocaleTag(raw);
    if (!norm) continue;

    if (exactMap.has(norm)) return exactMap.get(norm);

    const parts = norm.split("-").filter(Boolean);
    for (let len = parts.length - 1; len >= 1; len -= 1) {
      const prefix = parts.slice(0, len).join("-");
      if (exactMap.has(prefix)) return exactMap.get(prefix);
    }

    const primary = parts[0];
    if (primary && primaryMap.has(primary)) return primaryMap.get(primary);
  }

  return exactMap.get(normalizeLocaleTag(fallback)) || fallback;
}

export function getNavigatorLanguages() {
  if (typeof navigator === "undefined") return [];
  const out = [];
  if (Array.isArray(navigator.languages)) {
    for (const lang of navigator.languages) {
      if (lang) out.push(lang);
    }
  }
  if (navigator.language) out.push(navigator.language);
  return out;
}

export function interpolate(template, params) {
  const str = String(template ?? "");
  if (!params || typeof params !== "object") return str;
  return str.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, key) => {
    if (Object.prototype.hasOwnProperty.call(params, key) && params[key] !== undefined && params[key] !== null) {
      return String(params[key]);
    }
    return match;
  });
}

function getDefaultStorage() {
  try {
    if (typeof window !== "undefined" && window.localStorage) {
      return window.localStorage;
    }
  } catch (_) {
    /* ignore storage access errors */
  }
  return null;
}

export function createI18n({
  locales = REGISTERED_LOCALES,
  fallbackLocale = DEFAULT_LOCALE,
  storage = getDefaultStorage(),
  systemLanguages = null,
} = {}) {
  const localeMap = new Map();
  const localeList = [];

  for (const entry of locales) {
    if (!entry || !entry.meta || !entry.meta.code) continue;
    const code = entry.meta.code;
    localeMap.set(code, {
      meta: { ...entry.meta },
      strings: { ...(entry.strings || {}) },
    });
    localeList.push(entry.meta);
  }

  const supportedCodes = localeList.map((m) => m.code);
  const listeners = new Set();

  let osLanguages = Array.isArray(systemLanguages) ? [...systemLanguages] : [];
  let preference = SYSTEM_LANGUAGE;
  let activeLocale = fallbackLocale;

  function readSavedPreference() {
    if (!storage) return SYSTEM_LANGUAGE;
    try {
      const saved = storage.getItem(STORAGE_KEY);
      if (saved && saved.trim()) return saved.trim();
    } catch (_) {
      /* ignore storage errors */
    }
    return SYSTEM_LANGUAGE;
  }

  function writeSavedPreference(pref) {
    if (!storage) return;
    try {
      storage.setItem(STORAGE_KEY, pref);
    } catch (_) {
      /* ignore storage errors */
    }
  }

  function getEffectiveSystemTags() {
    if (osLanguages.length > 0) {
      return [...osLanguages];
    }
    return getNavigatorLanguages();
  }

  function getSystemResolvedLocale() {
    return matchLocale(getEffectiveSystemTags(), supportedCodes, fallbackLocale);
  }

  function normalizePreference(rawPref) {
    const cleaned = String(rawPref || "").trim();
    if (!cleaned || cleaned.toLowerCase() === SYSTEM_LANGUAGE) {
      return SYSTEM_LANGUAGE;
    }
    const matched = matchLocale([cleaned], supportedCodes, "");
    return matched && localeMap.has(matched) ? matched : SYSTEM_LANGUAGE;
  }

  function resolveActiveLocale(pref) {
    if (pref === SYSTEM_LANGUAGE) {
      return getSystemResolvedLocale();
    }
    return matchLocale([pref], supportedCodes, fallbackLocale);
  }

  function syncDocumentLang() {
    if (typeof document !== "undefined" && document.documentElement) {
      document.documentElement.lang = activeLocale;
    }
  }

  function emitChange() {
    for (const fn of listeners) {
      try {
        fn({ locale: activeLocale, preference });
      } catch (err) {
        console.error("i18n listener error:", err);
      }
    }
  }

  function init(initialPref = null) {
    preference = normalizePreference(initialPref !== null ? initialPref : readSavedPreference());
    activeLocale = resolveActiveLocale(preference);
    syncDocumentLang();
    return activeLocale;
  }

  function setPreference(nextPref, { persist = true, notify = true } = {}) {
    const normalized = normalizePreference(nextPref);
    const nextLocale = resolveActiveLocale(normalized);
    const changed = normalized !== preference || nextLocale !== activeLocale;
    preference = normalized;
    activeLocale = nextLocale;
    if (persist) writeSavedPreference(preference);
    syncDocumentLang();
    if (notify && changed) {
      emitChange();
    }
    return activeLocale;
  }

  function setSystemLanguages(tags, { notify = true } = {}) {
    const prevSysResolved = getSystemResolvedLocale();
    osLanguages = Array.isArray(tags) ? tags.filter((x) => typeof x === "string" && x.trim()) : [];
    const nextSysResolved = getSystemResolvedLocale();
    const nextLocale = resolveActiveLocale(preference);
    const changed = nextLocale !== activeLocale || nextSysResolved !== prevSysResolved;
    activeLocale = nextLocale;
    syncDocumentLang();
    if (notify && changed) {
      emitChange();
    }
    return activeLocale;
  }

  function t(key, params = null) {
    const current = localeMap.get(activeLocale);
    const fallback = localeMap.get(fallbackLocale);
    let template =
      (current && Object.prototype.hasOwnProperty.call(current.strings, key)
        ? current.strings[key]
        : undefined);
    if (template === undefined && fallback && Object.prototype.hasOwnProperty.call(fallback.strings, key)) {
      template = fallback.strings[key];
    }
    if (template === undefined) return key;
    return interpolate(template, params);
  }

  function applyTranslations(root = typeof document !== "undefined" ? document : null) {
    if (!root) return;
    for (const el of root.querySelectorAll("[data-i18n]")) {
      const text = t(el.dataset.i18n);
      if (el.tagName === "OPTION") {
        el.dataset.fullLabel = text;
      }
      el.textContent = text;
    }
    for (const el of root.querySelectorAll("[data-i18n-html]")) {
      el.innerHTML = t(el.dataset.i18nHtml);
    }
    for (const el of root.querySelectorAll("[data-i18n-placeholder]")) {
      el.placeholder = t(el.dataset.i18nPlaceholder);
    }
    for (const el of root.querySelectorAll("[data-i18n-title]")) {
      el.title = t(el.dataset.i18nTitle);
    }
    for (const el of root.querySelectorAll("[data-i18n-aria-label]")) {
      el.setAttribute("aria-label", t(el.dataset.i18nAriaLabel));
    }
    for (const el of root.querySelectorAll("[data-i18n-tip]")) {
      el.dataset.tip = t(el.dataset.i18nTip);
    }
  }

  function onChange(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  init();

  return {
    t,
    init,
    getLocale: () => activeLocale,
    getPreference: () => preference,
    setPreference,
    setSystemLanguages,
    getSystemResolvedLocale,
    getAvailableLocales: () => localeList.map((m) => ({ ...m })),
    getLocaleMeta: (code) => {
      const entry = localeMap.get(code);
      return entry ? { ...entry.meta } : null;
    },
    applyTranslations,
    onChange,
  };
}

export const i18n = createI18n();
