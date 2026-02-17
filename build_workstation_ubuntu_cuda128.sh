#!/usr/bin/env bash
# Build the Linux CUDA workstation bundle (painter + trainer) as an AppImage.
#
# Usage:
#   ./build_workstation_ubuntu_cuda128.sh              # RTX 5000 series (sm_120)
#   ./build_workstation_ubuntu_cuda128.sh rtx50        # same as above
#   ./build_workstation_ubuntu_cuda128.sh broad        # GTX 1660 through RTX 4090
#
# PyTorch wheels: uses local wheels from ./dist/ if present (built by
# ./build_custom_torch.sh), otherwise falls back to the URLs in
# trainer/requirements_torch_cu128_<variant>.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
VARIANT="${1:-rtx50}"

case "$VARIANT" in
  rtx50)
    REQUIREMENTS="requirements_torch_cu128.txt"
    APPIMAGE_NAME="RootPainterWorkstation_0.2.28_Ubuntu_CUDA128_RTX50.AppImage"
    ;;
  broad)
    REQUIREMENTS="requirements_torch_cu128_broad.txt"
    APPIMAGE_NAME="RootPainterWorkstation_0.2.28_Ubuntu_CUDA128_GTX1660_to_RTX4090.AppImage"
    ;;
  *)
    echo "ERROR: Unknown variant '$VARIANT'. Use 'rtx50' or 'broad'."
    exit 1
    ;;
esac

echo "Building variant: $VARIANT"
echo "  Requirements: $REQUIREMENTS"
echo "  AppImage:     $APPIMAGE_NAME"
echo ""

# Fail fast if interpreter isn't available
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "ERROR: $PYTHON not found. Set PYTHON=/path/to/python"
  exit 1
}

# Enforce Python 3.11 (without hardcoding binary name)
"$PYTHON" -c 'import sys; assert sys.version_info[:2]==(3,11), sys.version' || {
  echo "ERROR: Python 3.11 required. Current: $("$PYTHON" -V 2>&1)"
  exit 1
}

# --------------------
# Trainer
# --------------------
cd "$ROOT/trainer"
rm -rf env
"$PYTHON" -m venv env
source env/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements_base_no_torch.txt

# Use local wheels if available, otherwise fall back to requirements file
TORCH_WHEEL=$(ls "$ROOT"/dist/torch-*.whl 2>/dev/null | head -1 || true)
VISION_WHEEL=$(ls "$ROOT"/dist/torchvision-*.whl 2>/dev/null | head -1 || true)

if [ -n "$TORCH_WHEEL" ] && [ -n "$VISION_WHEEL" ]; then
  echo "Using local wheels:"
  echo "  $TORCH_WHEEL"
  echo "  $VISION_WHEEL"
  python -m pip install "$TORCH_WHEEL" "$VISION_WHEEL"
  # Install CUDA runtime libs (custom wheel doesn't declare pip deps on these)
  python -m pip install nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-curand-cu12 \
    nvidia-nvjitlink-cu12 nvidia-cuda-nvrtc-cu12
else
  echo "No local wheels in ./dist/, using $REQUIREMENTS"
  python -m pip install -r "$REQUIREMENTS"
fi

python -m pip install pyinstaller

python src/build/run_pyinstaller_trainer.py
deactivate

# --------------------
# Painter
# --------------------
cd "$ROOT/painter"
rm -rf env
"$PYTHON" -m venv env
source env/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

python src/build/run_pyinstaller_workstation.py
deactivate

# --------------------
# Bundle trainer
# --------------------
APP_DIR="$ROOT/painter/dist/RootPainterWorkstation"

rm -rf "$APP_DIR/RootPainterTrainerBundle"
cp -R "$ROOT/trainer/src/dist/RootPainterTrainer" \
      "$APP_DIR/RootPainterTrainerBundle"

chmod +x "$APP_DIR/RootPainterTrainerBundle/RootPainterTrainer"

# Remove the broken trainer EXE from the workstation spec (painter venv has no torch).
# The working trainer is in RootPainterTrainerBundle/.
rm -f "$APP_DIR/RootPainterTrainer"

# --------------------
# AppImage
# --------------------
APPDIR="$ROOT/painter/dist/RootPainter.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR"

# Move the workstation contents into the AppDir
mv "$APP_DIR"/* "$APPDIR"/
rmdir "$APP_DIR"

# AppRun entry point
cat > "$APPDIR/AppRun" << 'APPRUN'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/RootPainter" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

# Desktop file
cat > "$APPDIR/RootPainter.desktop" << 'DESKTOP'
[Desktop Entry]
Name=RootPainter
Comment=Corrective annotation for biological image segmentation
Exec=RootPainter
Icon=RootPainter
Type=Application
Categories=Science;Education;Graphics;
DESKTOP

# Icon
cp "$ROOT/painter/src/main/icons/linux/256.png" "$APPDIR/RootPainter.png"

# Download appimagetool if not present
APPIMAGETOOL="$ROOT/painter/dist/appimagetool"
if [ ! -x "$APPIMAGETOOL" ]; then
  echo "Downloading appimagetool..."
  curl -fSL -o "$APPIMAGETOOL" \
    "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
  chmod +x "$APPIMAGETOOL"
fi

# Build AppImage
export ARCH=x86_64
APPIMAGE_OUT="$ROOT/painter/dist/$APPIMAGE_NAME"
"$APPIMAGETOOL" --no-appstream "$APPDIR" "$APPIMAGE_OUT" \
  || "$APPIMAGETOOL" --appimage-extract-and-run --no-appstream "$APPDIR" "$APPIMAGE_OUT"

echo ""
echo "Built Ubuntu CUDA workstation AppImage ($VARIANT):"
echo "$APPIMAGE_OUT"
ls -lh "$APPIMAGE_OUT"
