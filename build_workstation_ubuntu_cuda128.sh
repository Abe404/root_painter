#!/usr/bin/env bash
# Build the Linux CUDA 12.8 workstation bundle (painter + trainer).
#
# PyTorch wheels: uses local wheels from ./dist/ if present (built by
# ./build_custom_torch.sh), otherwise falls back to the URLs in
# trainer/requirements_torch_cu128.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"

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
else
  echo "No local wheels in ./dist/, using requirements_torch_cu128.txt"
  python -m pip install -r requirements_torch_cu128.txt
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

echo "Built Ubuntu RTX50 CUDA128 workstation:"
echo "$APP_DIR"
