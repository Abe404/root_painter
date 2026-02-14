#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Build minimal PyTorch + torchvision wheels for RootPainter
#
# Produces wheels with only the CUDA features RootPainter needs,
# cutting ~3GB from the workstation bundle.
#
# Prerequisites:
#   - Python 3.11
#   - CUDA 12.8 toolkit (/usr/local/cuda-12.8)
#       Install with:
#         wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
#         sudo dpkg -i cuda-keyring_1.1-1_all.deb
#         sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8
#
# Usage:
#   ./build_custom_torch.sh                         # defaults: v2.7.1 / v0.22.1 / sm 12.0
#   ./build_custom_torch.sh v2.8.0 v0.23.0 "12.0"  # specific versions
#
# Outputs:  ./dist/torch-*.whl  ./dist/torchvision-*.whl
#
# Timings (32-core / 64GB RAM):
#   Fresh build: ~60 min    Incremental (cmake cache): ~11 min
# -----------------------------------------------------------------------
set -euo pipefail

PYTORCH_REF="${1:-v2.7.1}"
TORCHVISION_REF="${2:-v0.22.1}"
CUDA_ARCH="${3:-12.0}"

PYTHON="${PYTHON:-python3.11}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_ROOT="/tmp/rp-torch-build"
VENV_DIR="$BUILD_ROOT/venv"
PYTORCH_DIR="$BUILD_ROOT/pytorch"
VISION_DIR="$BUILD_ROOT/vision"
OUTPUT_DIR="$SCRIPT_DIR/dist"

# Auto-detect MAX_JOBS: 75% of cores, minimum 4
NPROC=$(nproc 2>/dev/null || echo 4)
MAX_JOBS="${MAX_JOBS:-$(( NPROC * 3 / 4 ))}"
[ "$MAX_JOBS" -lt 4 ] && MAX_JOBS=4

# -----------------------------------------------------------------------
# Preflight checks
# -----------------------------------------------------------------------
echo "=== Preflight checks ==="

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "ERROR: $PYTHON not found. Set PYTHON=/path/to/python3.11"
  exit 1
}

"$PYTHON" -c 'import sys; assert sys.version_info[:2]==(3,11), f"Need Python 3.11, got {sys.version}"' || exit 1

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc"
  echo "Install CUDA 12.8 toolkit first (see instructions at top of this script)"
  exit 1
fi
export CUDA_HOME
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

echo "Python:       $("$PYTHON" --version)"
echo "CUDA:         $(nvcc --version | grep release)"
echo "PyTorch ref:  $PYTORCH_REF"
echo "Torchvision:  $TORCHVISION_REF"
echo "CUDA arch:    $CUDA_ARCH"
echo "MAX_JOBS:     $MAX_JOBS"
echo "Build root:   $BUILD_ROOT"
echo ""

# -----------------------------------------------------------------------
# Build venv (setuptools<81 needed for pkg_resources; numpy needed at
# compile time for torch.from_numpy() support)
# -----------------------------------------------------------------------
echo "=== Setting up build venv ==="
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install cmake ninja wheel "setuptools<81" \
    typing_extensions pyyaml numpy
else
  echo "Reusing existing venv at $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# -----------------------------------------------------------------------
# Clone / update PyTorch
# -----------------------------------------------------------------------
echo "=== Cloning PyTorch ($PYTORCH_REF) ==="
CLONE_START=$(date +%s)
if [ -d "$PYTORCH_DIR/.git" ]; then
  echo "PyTorch source exists. Checking ref..."
  cd "$PYTORCH_DIR"
  CURRENT_REF=$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD)
  if [ "$CURRENT_REF" = "$PYTORCH_REF" ]; then
    echo "Already at $PYTORCH_REF, skipping clone"
  else
    echo "Different ref ($CURRENT_REF != $PYTORCH_REF), re-cloning"
    rm -rf "$PYTORCH_DIR"
    git clone --depth 1 --branch "$PYTORCH_REF" --recursive \
      https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
  fi
else
  git clone --depth 1 --branch "$PYTORCH_REF" --recursive \
    https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
fi
CLONE_END=$(date +%s)
echo "Clone: $(( CLONE_END - CLONE_START ))s"

# -----------------------------------------------------------------------
# Build PyTorch
# -----------------------------------------------------------------------
echo ""
echo "=== Building PyTorch wheel ==="
TORCH_START=$(date +%s)

# Enable only what RootPainter needs
export USE_CUDA=1
export USE_CUDNN=1
export USE_CUBLAS=1
export USE_CURAND=1
export USE_NUMPY=1
export USE_OPENMP=1
export USE_FBGEMM=1

# Disable unused CUDA libraries (~3GB savings)
export USE_NCCL=0
export USE_CUSPARSELT=0
export USE_CUSPARSE=0
export USE_CUFFT=0
export USE_CUSOLVER=0
export USE_CUFILE=0

# Disable distributed training (RootPainter uses DataParallel, not DDP)
export USE_DISTRIBUTED=0
export USE_TENSORPIPE=0
export USE_GLOO=0
export USE_MPI=0

# Disable unused backends
export BUILD_TEST=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0

# Target specific GPU architecture
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
export MAX_JOBS

cd "$PYTORCH_DIR"
python setup.py bdist_wheel

TORCH_END=$(date +%s)
TORCH_WHEEL=$(ls -1 "$PYTORCH_DIR/dist"/torch-*.whl | head -1)
echo ""
echo "PyTorch wheel: $TORCH_WHEEL"
echo "Size: $(du -h "$TORCH_WHEEL" | cut -f1)"
echo "Build time: $(( TORCH_END - TORCH_START ))s ($(( (TORCH_END - TORCH_START) / 60 ))m)"

# -----------------------------------------------------------------------
# Build torchvision
# -----------------------------------------------------------------------
echo ""
echo "=== Installing custom torch into build venv ==="
pip install "$TORCH_WHEEL"

echo ""
echo "=== Cloning torchvision ($TORCHVISION_REF) ==="
VISION_START=$(date +%s)
if [ -d "$VISION_DIR/.git" ]; then
  cd "$VISION_DIR"
  CURRENT_REF=$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD)
  if [ "$CURRENT_REF" = "$TORCHVISION_REF" ]; then
    echo "Already at $TORCHVISION_REF, skipping clone"
  else
    rm -rf "$VISION_DIR"
    git clone --depth 1 --branch "$TORCHVISION_REF" --recursive \
      https://github.com/pytorch/vision.git "$VISION_DIR"
  fi
else
  git clone --depth 1 --branch "$TORCHVISION_REF" --recursive \
    https://github.com/pytorch/vision.git "$VISION_DIR"
fi

echo ""
echo "=== Building torchvision wheel ==="
cd "$VISION_DIR"
python setup.py bdist_wheel

VISION_END=$(date +%s)
VISION_WHEEL=$(ls -1 "$VISION_DIR/dist"/torchvision-*.whl | head -1)
echo ""
echo "Torchvision wheel: $VISION_WHEEL"
echo "Size: $(du -h "$VISION_WHEEL" | cut -f1)"
echo "Build time: $(( VISION_END - VISION_START ))s"

# -----------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------
echo ""
echo "=== Smoke test ==="
pip install "$VISION_WHEEL"

python -c "
import torch
import numpy as np
import torchvision

print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('CUDA compiled:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())

# numpy interop
t = torch.from_numpy(np.ones((2, 3), dtype=np.float32))
assert t.shape == (2, 3), 'numpy interop failed'
print('numpy interop: OK')

# CPU forward + backward
conv = torch.nn.Conv2d(3, 16, 3, padding=1)
x = torch.randn(1, 3, 32, 32)
y = conv(x)
y.sum().backward()
print('CPU forward+backward: OK')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU:', torch.cuda.get_device_name(0))

    # GPU forward + backward
    conv_g = torch.nn.Conv2d(3, 16, 3, padding=1).to(device)
    y_g = conv_g(torch.randn(1, 3, 32, 32, device=device))
    y_g.sum().backward()
    print('GPU forward+backward: OK')

    # DataParallel + GroupNorm (what RootPainter uses)
    from torch.nn import Sequential, Conv2d, GroupNorm
    model = torch.nn.DataParallel(Sequential(Conv2d(3, 16, 3), GroupNorm(4, 16)))
    model.to(device)
    out = model(torch.randn(2, 3, 32, 32, device=device))
    print('DataParallel+GroupNorm: OK')

    # GPU <-> numpy roundtrip
    arr = torch.randn(4, 4, device=device).cpu().numpy()
    assert arr.shape == (4, 4)
    print('GPU->numpy roundtrip: OK')

print()
print('ALL SMOKE TESTS PASSED')
"

# -----------------------------------------------------------------------
# Collect output
# -----------------------------------------------------------------------
echo ""
echo "=== Collecting wheels ==="
mkdir -p "$OUTPUT_DIR"
cp "$TORCH_WHEEL" "$OUTPUT_DIR/"
cp "$VISION_WHEEL" "$OUTPUT_DIR/"

TOTAL_END=$(date +%s)
TOTAL_START=$CLONE_START

echo ""
echo "========================================"
echo "  Build complete!"
echo "========================================"
echo ""
echo "Wheels:"
ls -lh "$OUTPUT_DIR"/torch*.whl
echo ""
echo "Timings:"
echo "  Clone:        $(( CLONE_END - CLONE_START ))s"
echo "  PyTorch:      $(( TORCH_END - TORCH_START ))s ($(( (TORCH_END - TORCH_START) / 60 ))m)"
echo "  Torchvision:  $(( VISION_END - VISION_START ))s"
echo "  Total:        $(( TOTAL_END - TOTAL_START ))s ($(( (TOTAL_END - TOTAL_START) / 60 ))m)"
echo ""
echo "To install in a fresh venv:"
echo "  pip install $OUTPUT_DIR/$(basename "$TORCH_WHEEL") $OUTPUT_DIR/$(basename "$VISION_WHEEL")"
echo ""
echo "To upload to GitHub release:"
echo "  gh release create torch-cu128-sm120-v1 $OUTPUT_DIR/torch*.whl --prerelease"
