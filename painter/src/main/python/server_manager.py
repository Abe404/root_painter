"""
server_manager.py

RootPainter Workstation MVP:
- Optionally launches a bundled trainer/server executable from the GUI.
- Clean dev mode: if running from a source checkout with ../trainer/env/bin/python,
  launch trainer via that venv (no wrapper scripts).

Contract:
- Trainer supports: --syncdir <path>
- Logs go to stdout/stderr
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from PyQt5 import QtCore, QtWidgets


def _exe_suffix() -> str:
    return ".exe" if sys.platform.startswith("win") else ""


@dataclass(frozen=True)
class LaunchSpec:
    """How to launch the trainer."""
    program: Path
    args_prefix: List[str]
    working_dir: Optional[Path] = None


def find_bundled_trainer() -> Optional[Path]:
    """
    Find the trainer executable bundled alongside the GUI executable.

    Supports:
      - macOS (PyInstaller .app):
          RootPainter.app/Contents/MacOS/RootPainterTrainerBundle/RootPainterTrainer
          (trainer built in onedir mode, bundled as a folder)
      - legacy single-file helper locations (optional):
          RootPainter.app/Contents/MacOS/RootPainterTrainer
          RootPainter.app/Contents/Resources/RootPainterTrainer
      - Windows/Linux:
          <install-dir>/RootPainterTrainer(.exe)
    """
    trainer_name = f"RootPainterTrainer{_exe_suffix()}"

    base_dir = Path(sys.executable).resolve().parent  # .../Contents/MacOS

    # 1) Workstation: trainer built as onedir, bundled as a folder
    bundle_candidate = base_dir / "RootPainterTrainerBundle" / trainer_name
    if bundle_candidate.is_file():
        return bundle_candidate

    # 2) Single-file helper next to GUI (if you ever do onefile trainer)
    candidate = base_dir / trainer_name
    if candidate.is_file():
        return candidate

    # 3) Extra fallback: sometimes helpers live in Resources
    contents_dir = base_dir.parent  # .../Contents
    resources_candidate = contents_dir / "Resources" / trainer_name
    if resources_candidate.is_file():
        return resources_candidate

    return None


def _find_repo_root_from_here() -> Optional[Path]:
    """
    Try to find a repo root by walking parents and looking for:
      <root>/painter and <root>/trainer
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "painter").is_dir() and (parent / "trainer").is_dir():
            return parent
    return None


def find_dev_trainer_launch() -> Optional[LaunchSpec]:
    """
    Dev mode: assumes repo layout:
      <repo>/painter/...
      <repo>/trainer/env/bin/python
      <repo>/trainer/src/main.py

    Returns a LaunchSpec that runs the trainer using its venv python.
    """
    repo_root = _find_repo_root_from_here()
    if repo_root is None:
        return None

    trainer_dir = repo_root / "trainer"
    py = trainer_dir / "env" / "bin" / "python"
    main_py = trainer_dir / "src" / "main.py"

    if py.is_file() and main_py.is_file():
        # Launch: <venv-python> -u <main.py> --syncdir <path>
        # -u => unbuffered stdout/stderr so logs appear promptly in the GUI
        return LaunchSpec(
            program=py,
            args_prefix=["-u", str(main_py), "--syncdir"],
            working_dir=trainer_dir,
        )

    return None


def find_trainer_launch() -> Optional[LaunchSpec]:
    """
    Preferred order:
      1) Explicit override via env var ROOTPAINTER_TRAINER (points to an executable)
      2) Bundled trainer next to GUI executable
      3) Dev trainer in repo checkout (trainer/env/bin/python + trainer/src/main.py)
    """
    override = os.environ.get("ROOTPAINTER_TRAINER")
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return LaunchSpec(program=p, args_prefix=["--syncdir"])

    bundled = find_bundled_trainer()
    if bundled:
        return LaunchSpec(program=bundled, args_prefix=["--syncdir"])

    return find_dev_trainer_launch()


class ServerManager(QtCore.QObject):
    """
    Manages a trainer/server process and streams its output.

    Usage:
        spec = find_trainer_launch()
        mgr = ServerManager(spec)
        mgr.start(sync_dir)
        mgr.stop()
    """

    log_line = QtCore.pyqtSignal(str)
    state_changed = QtCore.pyqtSignal(str)  # "running" / "stopped" / "starting"

    def __init__(self, launch: LaunchSpec, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.launch = launch
        self._proc: Optional[QtCore.QProcess] = None

        # NEW: track user-requested stop so we don't report it as a "crash"
        self._stopping: bool = False


        self._kill_timer = QtCore.QTimer()
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill_if_running)

    def is_available(self) -> bool:
        return self.launch.program.is_file()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() == QtCore.QProcess.Running

    def start(self, sync_dir: Path) -> None:
        if self.is_running():
            self.log_line.emit("Server already running.")
            return

        # NEW
        self._stopping = False

        if not self.is_available():
            self.log_line.emit(f"Trainer program not found: {self.launch.program}")
            return

        sync_dir = Path(sync_dir).expanduser().resolve()
        if not sync_dir.exists():
            self.log_line.emit(f"Sync dir does not exist: {sync_dir}")
            return

        proc = QtCore.QProcess(self)
        if self.launch.working_dir is not None:
            proc.setWorkingDirectory(str(self.launch.working_dir))

        proc.setProgram(str(self.launch.program))
        proc.setArguments(self.launch.args_prefix + [str(sync_dir)])

        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._read_output)
        proc.errorOccurred.connect(self._on_error)
        proc.started.connect(lambda: self.state_changed.emit("running"))
        proc.finished.connect(self._on_finished)

        self._proc = proc
        self.state_changed.emit("starting")
        self.log_line.emit(
            f"Starting server: {self.launch.program} "
            + " ".join(self.launch.args_prefix)
            + f" {sync_dir}"
        )
        proc.start()

    def stop(self) -> None:
        if not self.is_running():
            self.log_line.emit("Server is not running.")
            self.state_changed.emit("stopped")
            return

        # NEW
        self._stopping = True

        assert self._proc is not None
        self.log_line.emit("Stopping server...")
        self._proc.terminate()
        self._kill_timer.start(2000)

    def _kill_if_running(self) -> None:
        if self.is_running():
            assert self._proc is not None
            self.log_line.emit("Server did not exit; killing process.")
            self._proc.kill()

    def _read_output(self) -> None:
        if not self._proc:
            return
        data = bytes(self._proc.readAllStandardOutput())
        if not data:
            return
        text = data.decode("utf-8", errors="replace")

        # NEW: split into lines and ignore empties to reduce "gaps"
        for line in text.splitlines():
            if line.strip():
                self.log_line.emit(line)

    def _on_error(self, err: QtCore.QProcess.ProcessError) -> None:
        # NEW: during terminate/kill Qt may emit errorOccurred; ignore if we're stopping
        if self._stopping:
            return
        self.log_line.emit(f"Process error: {err}")
        self.state_changed.emit("stopped")

    def _on_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        # NEW: treat SIGTERM (15 on mac/linux) as normal stop
        if self._stopping or exit_code in (0, 15):
            self.log_line.emit(f"Server stopped (code={exit_code}).")
        else:
            status_str = "normal" if exit_status == QtCore.QProcess.NormalExit else "crash"
            self.log_line.emit(f"Server exited ({status_str}), code={exit_code}.")

        self.state_changed.emit("stopped")
        self._stopping = False

        if self._proc is not None:
            self._proc.deleteLater()
        self._proc = None


class ServerLogDialog(QtWidgets.QDialog):
    """Dialog that shows server logs and provides Start/Stop toggle."""

    def __init__(self, server_manager: ServerManager, sync_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RootPainter Server")
        self.resize(900, 500)


        self.server_manager = server_manager
        self.sync_dir = Path(sync_dir)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        layout.addWidget(self.log_box)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.toggle_btn = QtWidgets.QPushButton("Start server")
        self.toggle_btn.clicked.connect(self._toggle)
        controls.addWidget(self.toggle_btn)

        self.status_lbl = QtWidgets.QLabel("stopped")
        controls.addWidget(self.status_lbl)

        controls.addStretch(1)

        self.clear_btn = QtWidgets.QPushButton("Clear log")
        self.clear_btn.clicked.connect(self.log_box.clear)
        controls.addWidget(self.clear_btn)

        self.server_manager.log_line.connect(self._append_log)
        self.server_manager.state_changed.connect(self._on_state)

        self._on_state("running" if self.server_manager.is_running() else "stopped")


    def _append_log(self, line: str) -> None:
        if line.startswith("Training:"):
            cursor = self.log_box.textCursor()
            cursor.movePosition(cursor.End)

            # Move to start of the last line, then select it
            cursor.movePosition(cursor.StartOfLine, cursor.MoveAnchor)
            cursor.movePosition(cursor.EndOfLine, cursor.KeepAnchor)
            last_line = cursor.selectedText()

            if last_line.startswith("Training:"):
                cursor.removeSelectedText()
                cursor.insertText(line)
            else:
                self.log_box.appendPlainText(line)
        else:
            self.log_box.appendPlainText(line)


    def _on_state(self, state: str) -> None:
        self.status_lbl.setText(state)
        if state == "running":
            self.toggle_btn.setText("Stop server")
        elif state == "starting":
            self.toggle_btn.setText("Starting...")
        else:
            self.toggle_btn.setText("Start server")
        self.toggle_btn.setEnabled(state != "starting")

    def _toggle(self) -> None:
        if self.server_manager.is_running():
            self.server_manager.stop()
        else:
            self.server_manager.start(self.sync_dir)

