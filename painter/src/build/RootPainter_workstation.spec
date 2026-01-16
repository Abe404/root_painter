# -*- mode: python ; coding: utf-8 -*-
import os
from sys import platform
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# SPECPATH is provided by PyInstaller when executing the spec
ROOT = os.path.abspath(os.path.join(SPECPATH, "..", "..", ".."))

PAINTER_MAIN = os.path.join(ROOT, "painter", "src", "main", "python", "main.py")
TRAINER_MAIN = os.path.join(ROOT, "trainer", "src", "main.py")

# Your hidden imports (pyqtgraph templates)
hiddenimports = [
    "pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5",
    "pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt5",
    "pyqtgraph.imageview.ImageViewTemplate_pyqt5",
]

# Sometimes torch has submodules that PyInstaller misses; this is conservative.
# If it makes builds too big/slow, we can tighten later.
hiddenimports += collect_submodules("torch")

# Icon handling (PyInstaller applies it to the .app on mac)
ICON_ICO = os.path.join(ROOT, "painter", "src", "main", "icons", "Icon.ico")
ICON_ICNS = os.path.join(ROOT, "painter", "src", "main", "icons", "Icon.icns")
icon_file = ICON_ICNS if platform == "darwin" else ICON_ICO

# ---- Build trainer (console) ----
trainer_a = Analysis(
    [TRAINER_MAIN],
    pathex=[os.path.dirname(TRAINER_MAIN)],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

trainer_pyz = PYZ(trainer_a.pure, trainer_a.zipped_data, cipher=block_cipher)

trainer_exe = EXE(
    trainer_pyz,
    trainer_a.scripts,
    trainer_a.binaries,
    trainer_a.zipfiles,
    trainer_a.datas,
    [],
    name="RootPainterTrainer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,   # keep console for trainer
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ---- Build painter GUI (windowed on mac) ----
painter_a = Analysis(
    [PAINTER_MAIN],
    pathex=[os.path.dirname(PAINTER_MAIN)],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

painter_pyz = PYZ(painter_a.pure, painter_a.zipped_data, cipher=block_cipher)

painter_exe = EXE(
    painter_pyz,
    painter_a.scripts,
    painter_a.binaries,
    painter_a.zipfiles,
    painter_a.datas,
    [],
    name="RootPainter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app
    disable_windowed_traceback=False,
    argv_emulation=True if platform == "darwin" else False,
    icon=icon_file,
)

# On macOS this creates RootPainter.app.
# The "binaries" argument lets us drop RootPainterTrainer into Contents/MacOS,
# which matches find_bundled_trainer().
if platform == "darwin":
    app = BUNDLE(
        painter_exe,
        name="RootPainter.app",
        icon=icon_file,
        bundle_identifier="com.rootpainter",
        binaries=[(trainer_exe.name, "MacOS")],  # put trainer alongside RootPainter binary
    )
else:
    # Non-mac: just collect both executables into dist/
    coll = COLLECT(
        painter_exe,
        trainer_exe,
        a.binaries if False else [],
        a.zipfiles if False else [],
        a.datas if False else [],
        name="RootPainterWorkstation",
    )

