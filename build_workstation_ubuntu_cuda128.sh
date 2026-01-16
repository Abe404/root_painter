#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# --------------------
# Trainer
# --------------------
cd "$ROOT/trainer"
python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install -r requirements_base_no_torch.txt
pip install -r requirements_torch_cu128.txt
pip install pyinstaller

python src/build/run_pyinstaller_trainer.py
deactivate

# --------------------
# Painter
# --------------------
cd "$ROOT/painter"
python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

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


