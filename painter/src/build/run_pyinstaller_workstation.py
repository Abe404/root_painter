from pathlib import Path
import PyInstaller.__main__

HERE = Path(__file__).resolve().parent
SPEC = HERE / "RootPainter_workstation.spec"

PyInstaller.__main__.run([
    "--noconfirm",
    "--clean",
    str(SPEC),
])

