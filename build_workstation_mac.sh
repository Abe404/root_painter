#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Trainer: deps + build
cd "$ROOT/trainer"
source env/bin/activate
pip install -r requirements.txt pyinstaller
python src/build/run_pyinstaller_trainer.py
deactivate

# Painter: deps + build
cd "$ROOT/painter"
source env/bin/activate
pip install -r requirements.txt pyinstaller
pip install pyinstaller
python src/build/run_pyinstaller_workstation.py
deactivate

# Bundle trainer into app (use the *actual* PyInstaller output location)
APP="$ROOT/painter/dist/RootPainter.app"
APP_MACOS="$APP/Contents/MacOS"

# Sanity checks (fail fast, no mystery)
test -d "$APP" || { echo "ERROR: app not found at: $APP"; exit 1; }
test -d "$ROOT/trainer/src/dist/RootPainterTrainer" || { echo "ERROR: trainer onedir not found at: $ROOT/trainer/src/dist/RootPainterTrainer"; exit 1; }
test -f "$ROOT/trainer/src/dist/RootPainterTrainer/RootPainterTrainer" || { echo "ERROR: trainer executable missing inside onedir folder"; exit 1; }

# Ensure we don't accidentally launch a stale single-file helper
rm -f "$APP_MACOS/RootPainterTrainer"

# Copy the full onedir folder into the app as a bundle folder
rm -rf "$APP_MACOS/RootPainterTrainerBundle"
cp -R "$ROOT/trainer/src/dist/RootPainterTrainer" "$APP_MACOS/RootPainterTrainerBundle"

# Make sure main trainer binary is executable
chmod +x "$APP_MACOS/RootPainterTrainerBundle/RootPainterTrainer"

echo "OK: open \"$APP\""

