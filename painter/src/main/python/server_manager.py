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

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from PyQt5 import QtCore, QtWidgets

from instructions import send_instruction


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


class TrainerStatusDialog(QtWidgets.QDialog):
    """Dialog that polls for a trainer_status response file.

    Shows 'Opening project...' for at least 1s while polling.
    Happy path: auto-closes after 1s.
    Problems (no trainer, or training another project): switches
    to a warning after 3s and lets the user decide.
    """

    def __init__(self, instruction_path, response_path,
                 workstation=False, opening_project=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Opening Project')
        self.instruction_path = instruction_path
        self.response_path = response_path
        self.status_info = None
        self.workstation = workstation
        self.opening_project = opening_project
        self._open_time = time.time()
        self.setMinimumWidth(400)

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(12)
        self.setLayout(layout)

        self.label = QtWidgets.QLabel('Opening project...')
        self.label.setStyleSheet('font-size: 18px;')
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.label)

        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.hide()
        layout.addWidget(self.info_label)

        btn_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_layout)

        self.open_btn = QtWidgets.QPushButton('Open project anyway')
        self.open_btn.clicked.connect(self.accept)
        self.open_btn.hide()
        btn_layout.addWidget(self.open_btn)

        self.back_btn = QtWidgets.QPushButton('Go back')
        self.back_btn.clicked.connect(self.reject)
        self.back_btn.hide()
        btn_layout.addWidget(self.back_btn)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._poll)
        self.timer.start(500)

    def _elapsed(self):
        return time.time() - self._open_time

    def _poll(self):
        if os.path.isfile(self.response_path):
            self.timer.stop()
            try:
                with open(self.response_path, 'r') as f:
                    self.status_info = json.load(f)
            except Exception:
                pass

            training = self.status_info and self.status_info.get('training')
            training_project = self.status_info.get('project') if self.status_info else None
            same_project = (training and training_project
                            and training_project == self.opening_project)
            if training and not same_project:
                # Trainer is busy with a different project — tell the user.
                self._show_training_warning()
            else:
                # Happy path: auto-close after at least 1s total.
                remaining = max(0, 1.0 - self._elapsed())
                QtCore.QTimer.singleShot(int(remaining * 1000), self.accept)
            return

        # No response yet — after 3s, show buttons so user can decide.
        if self._elapsed() >= 3.0:
            self.timer.stop()
            self._show_waiting_warning()

    def _show_training_warning(self):
        project = self.status_info.get('project', 'unknown')
        self.label.setText(f'Trainer is currently training: {project}')
        self.info_label.setText(
            f'To train this project, open {project} and stop its '
            'training from Network \u2192 Stop training.')
        self.info_label.show()
        self.open_btn.setText('OK')
        self.open_btn.show()
        self.adjustSize()

    def _show_waiting_warning(self):
        self.label.setText('Waiting for trainer...')
        if self.workstation:
            self.info_label.setText(
                'The trainer is not running. Without the trainer, '
                'image segmentation and model training are disabled.\n\n'
                'To start the trainer go to Network \u2192 Open server, '
                'then click Start server.')
        else:
            self.info_label.setText(
                'No response from the trainer yet. This may take '
                'some time if syncing over a network.\n\n'
                'Possible reasons:\n\n'
                '\u2022 The trainer is not running\n'
                '\u2022 Sync is not set up between client and server\n'
                '\u2022 The trainer is an older version without status checks '
                '- everything might work fine anyway\n\n'
                'If training and segmentation do not work, see: '
                '<a href="https://github.com/Abe404/root_painter/blob/master/'
                'docs/FAQ.md#question---why-is-the-segmentation-not-loading">'
                'FAQ - Why is the segmentation not loading?</a>')
            self.info_label.setOpenExternalLinks(True)
        self.info_label.show()
        self.open_btn.show()
        self.back_btn.show()
        self.adjustSize()

    def _cleanup(self):
        """Remove instruction and response files."""
        for path in [self.instruction_path, self.response_path]:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass

    def done(self, result):
        self.timer.stop()
        self._cleanup()
        super().done(result)


def check_trainer_status(sync_dir, instruction_dir,
                         workstation=False, opening_project=None,
                         parent=None):
    """Send trainer_status instruction and show dialog while waiting.
       Returns (proceed: bool, status_info: dict or None)."""
    response_path = os.path.join(str(sync_dir), 'trainer_status.json')

    # Clean up any stale response from a previous check
    if os.path.isfile(response_path):
        os.remove(response_path)

    instruction_path = send_instruction(
        'trainer_status', {}, str(instruction_dir), str(sync_dir))

    dialog = TrainerStatusDialog(instruction_path, response_path,
                                 workstation=workstation,
                                 opening_project=opening_project,
                                 parent=parent)
    result = dialog.exec_()

    if result == QtWidgets.QDialog.Accepted:
        return True, dialog.status_info
    return False, None

