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
    excludes=[
        # Unused GUI toolkits / non-Windows pywebview backends
        "tkinter",
        "_tkinter",
        "tcl",
        "tk",
        "webview.platforms.cocoa",
        "webview.platforms.gtk",
        "webview.platforms.qt",
        "webview.platforms.android",
        "webview.platforms.cef",
        # Redundant/CLI-only packages (cchardet + charset_normalizer cover encoding; console=False)
        "chardet",
        "rich",
        "pygments",
        "tzdata",
        # Dev / test modules not needed at runtime
        "pytest",
        "_pytest",
        "unittest",
        "pydoc",
        "doctest",
        "pdb",
        "lib2to3",
        "numpy.testing",
        "numpy.f2py",
        "numpy.distutils",
        # Optional heavy ML/scientific deps not used by desktop app
        "PIL",
        "matplotlib",
        "scipy",
        "torch",
        "sklearn",
        "IPython",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

icon_path = os.path.join(repo_root, "resources", "img", "program_icon.ico")
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FFblindsubsync",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=icon_path if os.path.exists(icon_path) else None,
)
