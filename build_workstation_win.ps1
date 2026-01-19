$ErrorActionPreference = "Stop"

function Run($cmd) {
  Write-Host ">>> $cmd"
  iex $cmd
}

# Always run from repo root
Set-Location $PSScriptRoot

# Helpful debug
Run "python --version"
Run "python -c `"import sys; print(sys.executable)`""

# Create venvs
Run "python -m venv trainer\env"
Run "python -m venv painter\env"

# Upgrade packaging tools in both
Run "trainer\env\Scripts\python -m pip install --upgrade pip wheel setuptools"
Run "painter\env\Scripts\python -m pip install --upgrade pip wheel setuptools"

# -----------------------
# Install deps
# -----------------------
# Trainer env (optional but useful for sanity / local trainer runs)
Run "trainer\env\Scripts\pip install -r trainer\requirements_base_no_torch.txt"
Run "trainer\env\Scripts\pip install -r trainer\requirements_torch_cu128.txt"

# Painter env (THIS is the env that runs PyInstaller, so it MUST include trainer deps too)
Run "painter\env\Scripts\pip install -r painter\requirements.txt"
Run "painter\env\Scripts\pip install -r trainer\requirements_base_no_torch.txt"
Run "painter\env\Scripts\pip install -r trainer\requirements_torch_cu128.txt"

# PyInstaller in painter env
Run "painter\env\Scripts\pip install --upgrade pyinstaller"

# Quick sanity checks
Run "painter\env\Scripts\python -c `"import PyQt5; print('PyQt5 OK')`""
Run "painter\env\Scripts\python -c `"import torch; print('torch OK:', torch.__version__)`""

# -----------------------
# Build (spec-driven)
# -----------------------
Run "painter\env\Scripts\python painter\src\build\run_pyinstaller_workstation.py"

# -----------------------
# Zip output if present
# -----------------------
$dist = "painter\src\build\dist"
if (Test-Path $dist) {
  $zipPath = "RootPainter-Workstation-Windows.zip"
  if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
  Run "powershell -Command `"Compress-Archive -Path '$dist\*' -DestinationPath '$zipPath'`""
  Write-Host "Created $zipPath"
} else {
  Write-Host "NOTE: '$dist' not found; PyInstaller may have written dist/ elsewhere depending on CWD."
}

