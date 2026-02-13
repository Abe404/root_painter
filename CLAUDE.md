# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RootPainter is a GUI-based tool for training deep neural networks on biological images using corrective annotation (human-in-the-loop). It uses a client-server architecture where the **painter** (PyQt5 GUI client) and **trainer** (PyTorch server) communicate via JSON instruction files in a shared filesystem directory (the "sync directory"). No network protocol is used—communication works over local filesystem, sshfs, Dropbox, or Google Drive.

## Architecture

**Two independent Python applications:**

- **`painter/`** — PyQt5 desktop GUI. Users annotate images with brush strokes, view model predictions as overlays, and manage projects/datasets. Entry point: `painter/src/main/python/main.py`. Main window class: `root_painter.py`.
- **`trainer/`** — PyTorch training server. Watches the sync directory for instructions, trains U-Net models, performs segmentation. Entry point: `trainer/src/main.py`. Core loop: `trainer.py` (`Trainer.main_loop()`). Installable as PyPI package `root-painter-trainer`.

**Filesystem-based IPC:** The client writes JSON instruction files to `<syncdir>/instructions/`. The trainer polls for these, processes them (train, segment, etc.), and writes segmentation results back to the project directory. The `instructions.py` module in the painter handles creating these files.

**Workstation mode:** `server_manager.py` in the painter can auto-launch a bundled trainer executable, or in dev mode, launch the trainer from `trainer/env/bin/python`.

**U-Net model** (`unet.py`): Uses Group Normalization (not Batch Norm). Default input patch size 572x572, output 500x500. Valid patch sizes: 572, 556, 540, ..., 28.

**Loss** (`loss.py`): Combined 0.7 Dice + 0.3 Cross-Entropy with softmax over 2 channels (foreground/background).

## Development Setup

```bash
# Trainer
cd trainer
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install -e .
pip install pytest

# Painter
cd painter
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Trainer (standalone)
cd trainer/src && python main.py --syncdir ~/root_painter_sync

# Or via pip entry point after install
start-trainer --syncdir ~/root_painter_sync

# Painter
cd painter && python src/main/python/main.py
```

## Testing

Tests are in `trainer/tests/`. Run from that directory:

```bash
# Unit tests (fast, no downloads)
cd trainer/tests
python -m pytest test_loss.py test_unet.py test_utils.py -v

# Single test
python -m pytest test_unet.py::TestUNet::test_forward_pass -v

# Training benchmarks (downloads datasets from Zenodo on first run, slow)
python -m pytest test_training.py -v -s
```

## Linting

Pylint config is at `painter/.pylint`. Many rules are intentionally disabled.

```bash
pylint painter/src/main/python/*.py
```

## Build

**Trainer PyPI package:**
```bash
cd trainer && python -m build
```

**Painter executable (PyInstaller):**
```bash
cd painter && python src/build/run_pyinstaller.py
```

**Workstation bundle (trainer + painter):**
```bash
./build_workstation_ubuntu_cuda128.sh   # Linux
./build_workstation_win.ps1             # Windows
./build_workstation_mac.sh              # macOS
```

**Custom PyTorch wheels (for minimal CUDA workstation builds):**

The Linux CUDA 12.8 workstation uses custom-built PyTorch wheels with unused CUDA libraries stripped out (~180MB vs ~1.5GB). Wheels are hosted as GitHub release assets and referenced in `trainer/requirements_torch_cu128.txt`.

```bash
# Build wheels locally (requires CUDA 12.8 toolkit + Python 3.11)
./build_custom_torch.sh                         # defaults: v2.7.1 / v0.22.1 / sm 12.0
./build_custom_torch.sh v2.8.0 v0.23.0 "12.0"  # specific versions
MAX_JOBS=16 ./build_custom_torch.sh             # override parallelism

# Or trigger via GitHub Actions: "Build Custom PyTorch Wheel" workflow
```

Outputs wheels to `./dist/`. After building, upload to a GitHub release and update the URLs in `trainer/requirements_torch_cu128.txt`.

## Key Constraints

- Python 3.11–3.12 required (`>=3.11,<3.13`)
- Trainer imports are relative (e.g., `from unet import ...`), not package-qualified—tests and entry points run from `trainer/src/`
- Batch size is auto-detected from GPU memory (CUDA/MPS/CPU fallback)
- Contributions require discussion with the maintainer before submitting PRs
