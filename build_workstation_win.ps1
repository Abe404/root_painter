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

# Install deps (adjust if you use requirements-dev or constraints)
Run "trainer\env\Scripts\pip install -r trainer\requirements.txt"
Run "painter\env\Scripts\pip install -r painter\requirements.txt"

# If your build relies on pyinstaller living in one env, pick ONE:
# Option A: install pyinstaller in painter env
Run "painter\env\Scripts\pip install pyinstaller"

# Optional: print PyQt sanity (if applicable)
Run "painter\env\Scripts\python -c `"import PyQt5; print('PyQt5 OK')`""

# Run your build entrypoint
# If you have a Windows script already, call it here:
#
Run "painter\env\Scripts\python painter\src\build\build_workstation.py"


# Zip output if present (nice for Actions downloads)
$dist = "painter\src\build\dist"
if (Test-Path $dist) {
  $zipPath = "RootPainter-Workstation-Windows.zip"
  if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
  Run "powershell -Command `"Compress-Archive -Path '$dist\*' -DestinationPath '$zipPath'`""
  Write-Host "Created $zipPath"
}

