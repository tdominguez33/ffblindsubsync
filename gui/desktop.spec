# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import copy_metadata

gui_dir = os.path.abspath(SPECPATH)
repo_root = os.path.abspath(os.path.join(gui_dir, ".."))

datas = [
    (os.path.join(repo_root, "ui", "index.html"), "ui"),
    (os.path.join(repo_root, "ui", "styles.css"), "ui"),
    (os.path.join(repo_root, "ui", "app.js"), "ui"),
    (
        os.path.join(repo_root, "ui", "adapters", "desktop.js"),
        os.path.join("ui", "adapters"),
    ),
    (os.path.join(repo_root, "resources", "img", "program_icon.png"), "."),
]
version_file = os.path.join(repo_root, "__version__")
if os.path.exists(version_file):
    datas.append((version_file, "."))
datas += copy_metadata("webrtcvad-wheels")

a = Analysis(
    [os.path.join(gui_dir, "ffsubsync-gui.py")],
    pathex=[repo_root, gui_dir],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "ffsubsync._frozen_version",
        "_webrtcvad",
        "webrtcvad",
        "cchardet",
        "charset_normalizer",
        "pysubs2",
        "srt",
        "auditok",
        "ffmpeg",
        "clr_loader",
        "pythonnet",
    ],
    hookspath=[os.path.join(gui_dir, "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

icon_path = os.path.join(repo_root, "resources", "img", "program_icon.ico")
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FFblindsubsync",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon_path if os.path.exists(icon_path) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="FFblindsubsync",
)
