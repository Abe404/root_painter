#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3.11}"

# Fail fast if python3.11 isn't available
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "ERROR: $PYTHON not found. Install Python 3.11 or set PYTHON=/path/to/python3.11"
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
python -m pip install -r requirements_torch_cu128.txt
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
