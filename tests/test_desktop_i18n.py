# -*- coding: utf-8 -*-
import importlib.util
import json
from pathlib import Path
import sys
import types


def _load_desktop_app_module():
    """Load gui/desktop_app.py without requiring a live pywebview GUI session."""
    fake_webview = types.ModuleType("webview")
    fake_webview.Window = object
    fake_webview.screens = []
    fake_dom = types.ModuleType("webview.dom")
    fake_dom.DOMEventHandler = object
    fake_webview.dom = fake_dom

    prev_webview = sys.modules.get("webview")
    prev_dom = sys.modules.get("webview.dom")
    sys.modules["webview"] = fake_webview
    sys.modules["webview.dom"] = fake_dom
    try:
        mod_path = Path(__file__).resolve().parent.parent / "gui" / "desktop_app.py"
        spec = importlib.util.spec_from_file_location("_test_desktop_app", str(mod_path))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod
    finally:
        if prev_webview is not None:
            sys.modules["webview"] = prev_webview
        else:
            sys.modules.pop("webview", None)
        if prev_dom is not None:
            sys.modules["webview.dom"] = prev_dom
        else:
            sys.modules.pop("webview.dom", None)


def test_get_system_languages_returns_clean_tags():
    mod = _load_desktop_app_module()
    langs = mod._get_system_languages()
    assert isinstance(langs, list)
    for tag in langs:
        assert isinstance(tag, str)
        assert tag.strip() == tag
        assert "_" not in tag
        assert "." not in tag


def test_desktop_api_language_default_and_persistence(tmp_path, monkeypatch):
    mod = _load_desktop_app_module()
    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr(mod, "_config_path", lambda: cfg_file)
    monkeypatch.setattr(mod, "get_portable_dir", lambda: tmp_path)
    monkeypatch.setattr(
        mod,
        "find_ffmpeg_binaries",
        lambda custom_path=None: {"available": False, "source": None},
    )

    api = mod.DesktopApi()
    init_state = api.get_init_state()
    assert init_state["language"] == "system"
    assert isinstance(init_state["system_languages"], list)

    res = api.set_language("es")
    assert res == {"language": "es"}
    saved = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert saved["language"] == "es"

    api_reloaded = mod.DesktopApi()
    assert api_reloaded.get_init_state()["language"] == "es"

    api_reloaded.set_language("system")
    saved2 = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert saved2["language"] == "system"
