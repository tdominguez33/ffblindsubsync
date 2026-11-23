// Unit tests for ui/i18n (locale matching, fallback, instant switching, key parity).
// Run: node web/tests/i18n_test.mjs

import assert from "node:assert/strict";
import en from "../../ui/i18n/locales/en.js";
import es from "../../ui/i18n/locales/es.js";
import {
  createI18n,
  matchLocale,
  normalizeLocaleTag,
  REGISTERED_LOCALES,
  SYSTEM_LANGUAGE,
} from "../../ui/i18n/index.js";

function makeMemoryStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
  };
}

function extractPlaceholders(str) {
  return [...String(str).matchAll(/\{([a-zA-Z0-9_]+)\}/g)].map((m) => m[1]).sort();
}

// 1. Key & placeholder parity between en and all registered locales
{
  const enKeys = Object.keys(en.strings).sort();
  assert.ok(enKeys.length > 50, "Expected rich en locale dictionary");
  for (const loc of REGISTERED_LOCALES) {
    assert.ok(loc.meta && loc.meta.code && loc.meta.nativeName, "Locale must define meta.code and meta.nativeName");
    const keys = Object.keys(loc.strings).sort();
    assert.deepEqual(keys, enKeys, `Locale '${loc.meta.code}' keys must match 'en'`);
    for (const key of enKeys) {
      assert.equal(typeof loc.strings[key], "string", `Locale '${loc.meta.code}' key '${key}' must be string`);
      assert.ok(loc.strings[key].trim().length > 0, `Locale '${loc.meta.code}' key '${key}' must not be empty`);
      assert.deepEqual(
        extractPlaceholders(loc.strings[key]),
        extractPlaceholders(en.strings[key]),
        `Locale '${loc.meta.code}' key '${key}' placeholders must match 'en'`,
      );
    }
  }
}

// 2. Normalization
{
  assert.equal(normalizeLocaleTag("es_AR.UTF-8"), "es-ar");
  assert.equal(normalizeLocaleTag("EN_US@euro"), "en-us");
  assert.equal(normalizeLocaleTag("  es-419 "), "es-419");
  assert.equal(normalizeLocaleTag("C"), "");
  assert.equal(normalizeLocaleTag("POSIX"), "");
  assert.equal(normalizeLocaleTag(""), "");
}

// 3. Locale matching across regions (base codes AND regional app codes)
{
  const supported = ["en", "es"];
  assert.equal(matchLocale(["es-AR"], supported), "es");
  assert.equal(matchLocale(["es-MX"], supported), "es");
  assert.equal(matchLocale(["es-ES"], supported), "es");
  assert.equal(matchLocale(["es-419"], supported), "es");
  assert.equal(matchLocale(["es_AR.UTF-8"], supported), "es");
  assert.equal(matchLocale(["es"], supported), "es");
  assert.equal(matchLocale(["en-GB"], supported), "en");
  assert.equal(matchLocale(["en-AU"], supported), "en");
  assert.equal(matchLocale(["en-US"], supported), "en");
  assert.equal(matchLocale(["en"], supported), "en");

  // Unsupported language falls back to English
  assert.equal(matchLocale(["fr-FR"], supported), "en");
  assert.equal(matchLocale(["ja-JP", "de-DE"], supported), "en");
  assert.equal(matchLocale([], supported), "en");

  // Ordered preference list picks first supported language
  assert.equal(matchLocale(["fr-FR", "es-AR", "en-US"], supported), "es");
  assert.equal(matchLocale(["ja-JP", "en-GB", "es-ES"], supported), "en");

  // Even when the app registers regional codes (e.g. es-ES / en-US),
  // es-AR still matches es-ES and en-GB still matches en-US.
  const regionalSupported = ["en-US", "es-ES"];
  assert.equal(matchLocale(["es-AR"], regionalSupported, "en-US"), "es-ES");
  assert.equal(matchLocale(["es"], regionalSupported, "en-US"), "es-ES");
  assert.equal(matchLocale(["en-GB"], regionalSupported, "en-US"), "en-US");
  assert.equal(matchLocale(["pt-BR"], regionalSupported, "en-US"), "en-US");

  // Exact regional match wins when both variants exist
  assert.equal(matchLocale(["pt-BR"], ["en", "pt-PT", "pt-BR"]), "pt-BR");
  assert.equal(matchLocale(["pt-AO"], ["en", "pt-PT", "pt-BR"]), "pt-PT");
}

// 4. Instant switching, persistence, system default, and easy extensibility
{
  const storage = makeMemoryStorage();
  const inst = createI18n({
    storage,
    systemLanguages: ["es-AR"],
  });

  // Defaults to system -> es-AR -> es
  assert.equal(inst.getPreference(), SYSTEM_LANGUAGE);
  assert.equal(inst.getLocale(), "es");
  assert.equal(inst.t("settings.title"), es.strings["settings.title"]);

  // Instant switch to English triggers listener synchronously and persists
  const events = [];
  const unsub = inst.onChange((ev) => events.push(ev));
  inst.setPreference("en");
  assert.equal(inst.getPreference(), "en");
  assert.equal(inst.getLocale(), "en");
  assert.equal(inst.t("settings.title"), "Settings");
  assert.equal(storage.getItem("ffs_lang"), "en");
  assert.equal(events.length, 1);
  assert.deepEqual(events[0], { locale: "en", preference: "en" });

  // Switch back to system -> resolves to Spanish again immediately
  inst.setPreference("system");
  assert.equal(inst.getPreference(), "system");
  assert.equal(inst.getLocale(), "es");
  assert.equal(storage.getItem("ffs_lang"), "system");
  assert.equal(events.length, 2);

  // Updating OS languages to unsupported language falls back to English immediately
  inst.setSystemLanguages(["ja-JP"]);
  assert.equal(inst.getLocale(), "en");
  assert.equal(events.length, 3);

  unsub();
}

// 5. Extensibility: adding a partial third locale falls back to English for missing keys
{
  const frPartial = {
    meta: { code: "fr", nativeName: "Français" },
    strings: {
      "settings.title": "Paramètres",
    },
  };
  const inst = createI18n({
    locales: [en, es, frPartial],
    storage: makeMemoryStorage(),
    systemLanguages: ["fr-CA"],
  });
  assert.equal(inst.getLocale(), "fr");
  assert.equal(inst.t("settings.title"), "Paramètres");
  assert.equal(inst.t("form.syncBtn"), en.strings["form.syncBtn"]);
}

console.log("PASS: all i18n unit tests passed");
