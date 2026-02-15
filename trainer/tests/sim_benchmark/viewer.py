"""Frame viewer for sim benchmark output.

Arrow keys to step, space to play/pause.
Watches the frames directory for new files — launch it before or
during a simulation run to watch live.
"""
import sys
import os
import glob

from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QSlider,
                              QVBoxLayout, QWidget)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer


HELP_TEXT = ("  ◀ ▶  step    Space  play/pause    ▲ ▼  speed    "
             "Home  start    End  live    Q  quit")


class Viewer(QMainWindow):
    def __init__(self, frames_dir, fps=6):
        super().__init__()
        self.frames_dir = frames_dir
        self.frames = []
        self.idx = 0
        self.playing = False
        self.live = False
        self.fps = fps

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.on_slider)

        # Help bar
        self.help_label = QLabel(HELP_TEXT)
        self.help_label.setStyleSheet(
            "background: #222; color: #aaa; padding: 4px; font-size: 11px;")

        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.image_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.help_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Playback timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Watch timer — poll for new frames
        self.watch_timer = QTimer()
        self.watch_timer.timeout.connect(self.scan_frames)
        self.watch_timer.start(200)

        self.scan_frames()
        self.idx = 0
        if self.frames:
            self.show_frame()
        self.setWindowTitle("sim benchmark viewer")

    def scan_frames(self):
        found = sorted(glob.glob(os.path.join(self.frames_dir, '*.png')))
        if len(found) != len(self.frames):
            was_at_end = self.idx >= len(self.frames) - 1
            self.frames = found
            self.slider.setMaximum(max(0, len(self.frames) - 1))
            if self.live and (self.playing or was_at_end):
                self.idx = len(self.frames) - 1
            self.show_frame()

    def show_frame(self):
        if not self.frames:
            self.setWindowTitle("waiting for frames...")
            return
        self.idx = max(0, min(self.idx, len(self.frames) - 1))
        pix = QPixmap(self.frames[self.idx])
        self.image_label.setPixmap(pix)
        n = len(self.frames)
        status = ""
        if self.playing:
            status = " ▶"
        elif self.live:
            status = " ◉"
        self.setWindowTitle(f"{self.idx + 1}/{n}  {self.fps}fps{status}")
        self.slider.blockSignals(True)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)

    def on_slider(self, value):
        self.live = False
        self.idx = value
        self.show_frame()

    def next_frame(self):
        if self.idx < len(self.frames) - 1:
            self.idx += 1
            self.show_frame()
        elif self.playing and not self.live:
            self.playing = False
            self.timer.stop()
            self.show_frame()

    def prev_frame(self):
        if self.idx > 0:
            self.idx -= 1
            self.show_frame()

    def toggle_play(self):
        self.playing = not self.playing
        self.live = False
        if self.playing:
            self.timer.start(int(1000 / self.fps))
        else:
            self.timer.stop()
        self.show_frame()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Right:
            self.live = False
            self.next_frame()
        elif key == Qt.Key_Left:
            self.live = False
            self.prev_frame()
        elif key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_Home:
            self.live = False
            self.idx = 0
            self.show_frame()
        elif key == Qt.Key_End:
            self.live = True
            self.idx = max(0, len(self.frames) - 1)
            self.show_frame()
        elif key == Qt.Key_Up:
            self.fps = min(60, self.fps + 2)
            if self.playing:
                self.timer.start(int(1000 / self.fps))
            self.show_frame()
        elif key == Qt.Key_Down:
            self.fps = max(1, self.fps - 2)
            if self.playing:
                self.timer.start(int(1000 / self.fps))
            self.show_frame()
        elif key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)


def main():
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'quick_output', 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    app = QApplication(sys.argv)
    viewer = Viewer(frames_dir)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
