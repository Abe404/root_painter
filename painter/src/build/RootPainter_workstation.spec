# -*- mode: python ; coding: utf-8 -*-
import os
from sys import platform
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# SPECPATH is provided by PyInstaller when executing the spec
ROOT = os.path.abspath(os.path.join(SPECPATH, "..", "..", ".."))

PAINTER_MAIN = os.path.join(ROOT, "painter", "src", "main", "python", "main.py")
TRAINER_MAIN = os.path.join(ROOT, "trainer", "src", "main.py")

# Hidden imports
pyqtgraph_imports = [
    "pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5",
    "pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt5",
    "pyqtgraph.imageview.ImageViewTemplate_pyqt5",
]

torch_imports = collect_submodules("torch") + collect_submodules("torchvision")
torch_datas = collect_data_files("torch") + collect_data_files("torchvision")

# Icon handling (PyInstaller applies it to the .app on mac)
ICON_ICO = os.path.join(ROOT, "painter", "src", "main", "icons", "Icon.ico")
ICON_ICNS = os.path.join(ROOT, "painter", "src", "main", "icons", "Icon.icns")
icon_file = ICON_ICNS if platform == "darwin" else ICON_ICO

# Runtime hook to fix torch DLL loading on Windows
TORCH_DLL_HOOK = os.path.join(SPECPATH, "rthook_torch_dlls.py")

# ---- Trainer Analysis ----
trainer_a = Analysis(
    [TRAINER_MAIN],
    pathex=[os.path.dirname(TRAINER_MAIN)],
    binaries=[],
    datas=torch_datas,
    hiddenimports=torch_imports,
    hookspath=[],
    runtime_hooks=[TORCH_DLL_HOOK],
    excludes=[],
    noarchive=False,
)

trainer_pyz = PYZ(trainer_a.pure, trainer_a.zipped_data, cipher=block_cipher)

# ---- Painter Analysis ----
painter_a = Analysis(
    [PAINTER_MAIN],
    pathex=[os.path.dirname(PAINTER_MAIN)],
    binaries=[],
    datas=[],
    hiddenimports=pyqtgraph_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

painter_pyz = PYZ(painter_a.pure, painter_a.zipped_data, cipher=block_cipher)

if platform == "darwin":
    # macOS: onefile for both, trainer bundled into .app
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
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )

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
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=True,
        icon=icon_file,
    )

    app = BUNDLE(
        painter_exe,
        name="RootPainter.app",
        icon=icon_file,
        bundle_identifier="com.rootpainter",
        binaries=[(trainer_exe.name, "MacOS")],
    )
else:
    # Windows/Linux: onedir mode so torch DLLs are in a real directory
    # structure (onefile mode breaks torch's DLL loading on Windows)
    trainer_exe = EXE(
        trainer_pyz,
        trainer_a.scripts,
        [],
        name="RootPainterTrainer",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )

    painter_exe = EXE(
        painter_pyz,
        painter_a.scripts,
        [],
        name="RootPainter",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        icon=icon_file,
    )

    coll = COLLECT(
        painter_exe,
        painter_a.binaries,
        painter_a.zipfiles,
        painter_a.datas,
        trainer_exe,
        trainer_a.binaries,
        trainer_a.zipfiles,
        trainer_a.datas,
        name="RootPainterWorkstation",
    )
