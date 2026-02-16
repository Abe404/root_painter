"""Frame viewer for sim benchmark output.

Arrow keys to step, space to play/pause, N to focus note box.
Watches the frames directory for new files — launch it before or
during a simulation run to watch live.
"""
import csv
import json
import sys
import os
import glob

from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
                              QSizePolicy, QSlider, QStyle, QTextEdit,
                              QVBoxLayout, QWidget)
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import Qt, QRect, QTimer


HELP_TEXT = ("  \u25c0 \u25b6  step    Space  play/pause    \u25b2 \u25bc  speed    "
             "Home  start    End  live    N  note    Q  quit")


class NoteSlider(QSlider):
    """Slider with colored tick marks where notes exist."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.note_indices = set()

    def set_note_indices(self, indices):
        self.note_indices = set(indices)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.note_indices or self.maximum() == 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.yellow)

        opt = self.style().subControlRect(
            QStyle.CC_Slider, self._slider_option(), QStyle.SC_SliderGroove, self)
        groove_x = opt.x()
        groove_w = opt.width()

        for idx in self.note_indices:
            if idx > self.maximum():
                continue
            x = groove_x + int(idx / self.maximum() * groove_w)
            painter.drawRect(QRect(x - 1, 0, 3, self.height()))
        painter.end()

    def _slider_option(self):
        from PyQt5.QtWidgets import QStyleOptionSlider
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        return opt


class ClickableLabel(QLabel):
    """Image label that clears note focus when clicked."""

    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer

    def mousePressEvent(self, event):
        self._viewer.note_edit.clearFocus()
        self._viewer.setFocus()


def load_stats(frames_dir):
    """Load stats.csv into a dict keyed by filename."""
    stats_path = os.path.join(frames_dir, 'stats.csv')
    stats = {}
    if not os.path.exists(stats_path):
        return stats
    with open(stats_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats[row['filename']] = row
    return stats


class Viewer(QMainWindow):
    def __init__(self, frames_dir, fps=6):
        super().__init__()
        self.frames_dir = frames_dir
        self.frames = []
        self.idx = 0
        self.playing = False
        self.live = False
        self.fps = fps
        self._prev_fname = None

        # Notes stored as {filename: comment}
        self.notes_path = os.path.join(frames_dir, 'notes.json')
        self.notes = {}
        if os.path.exists(self.notes_path):
            with open(self.notes_path) as f:
                self.notes = json.load(f)

        # Frame stats from stats.csv
        self.stats = load_stats(frames_dir)

        # Image display — scales with window, fixed aspect ratio
        self.image_label = ClickableLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
        self.image_label.setMinimumSize(1, 1)
        self.current_pixmap = None

        # "Saved" indicator label above the note box
        self.saved_label = QLabel("Saved")
        self.saved_label.setFixedHeight(20)
        self.saved_label.setFixedWidth(200)
        self.saved_label.setAlignment(Qt.AlignCenter)
        self.saved_label.setStyleSheet(
            "background: #1a1a2e; color: #4a4; font-size: 12px; "
            "font-weight: bold;")
        self.saved_label.hide()
        self._save_flash_timer = QTimer()
        self._save_flash_timer.setSingleShot(True)
        self._save_flash_timer.timeout.connect(self.saved_label.hide)

        # Note text box — rectangle to the right of the image
        self.note_edit = QTextEdit()
        self.note_edit.setPlaceholderText("note (N to focus)")
        self.note_edit.setFixedWidth(200)
        self.note_edit.setStyleSheet(
            "background: #1a1a2e; color: #ddd; border: 1px solid #444; "
            "padding: 4px; font-size: 12px;")
        self.note_edit.setTabChangesFocus(True)
        self.note_edit.textChanged.connect(self._on_note_changed)

        # Note column: saved label + note box
        note_col = QVBoxLayout()
        note_col.setContentsMargins(0, 0, 0, 0)
        note_col.setSpacing(0)
        note_col.addWidget(self.saved_label, 0)
        note_col.addWidget(self.note_edit, 1)

        # Top row: image + note column
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(0)
        top_row.addWidget(self.image_label, 1)
        top_row.addLayout(note_col, 0)

        # Slider with note markers
        self.slider = NoteSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setFixedHeight(20)
        self.slider.valueChanged.connect(self.on_slider)

        # Stats info panel
        self.stats_label = QLabel()
        self.stats_label.setFixedHeight(28)
        self.stats_label.setStyleSheet(
            "background: #1a1a2e; color: #8cf; padding: 4px 8px; "
            "font-size: 14px; font-family: 'Menlo', 'Courier New', monospace;")

        # Help bar
        self.help_label = QLabel(HELP_TEXT)
        self.help_label.setFixedHeight(22)
        self.help_label.setStyleSheet(
            "background: #222; color: #aaa; padding: 4px; font-size: 11px;")

        # Layout — image row gets all the space
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(top_row, 1)
        layout.addWidget(self.slider, 0)
        layout.addWidget(self.stats_label, 0)
        layout.addWidget(self.help_label, 0)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setFocusPolicy(Qt.StrongFocus)

        # Playback timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Initial frame scan (before watch timer starts)
        found = sorted(glob.glob(os.path.join(self.frames_dir, '*.png')))
        self.frames = found
        self.slider.setMaximum(max(0, len(self.frames) - 1))
        self._sync_note_markers()
        self.idx = 0
        if self.frames:
            self.show_frame()
        self.setWindowTitle("sim benchmark viewer")

        # Watch timer — poll for new frames (started after initial display)
        self.watch_timer = QTimer()
        self.watch_timer.timeout.connect(self.scan_frames)
        self.watch_timer.start(200)

    def _sync_note_markers(self):
        """Update slider tick marks from current notes."""
        if not self.frames:
            self.slider.set_note_indices(set())
            return
        fname_to_idx = {}
        for i, path in enumerate(self.frames):
            fname_to_idx[os.path.basename(path)] = i
        indices = set()
        for fname in self.notes:
            if fname in fname_to_idx:
                indices.add(fname_to_idx[fname])
        self.slider.set_note_indices(indices)

    def _save_current_note(self):
        """Persist the note text for the current frame."""
        if not self.frames or self._prev_fname is None:
            return
        text = self.note_edit.toPlainText().strip()
        changed = False
        if text:
            if self.notes.get(self._prev_fname) != text:
                self.notes[self._prev_fname] = text
                changed = True
        elif self._prev_fname in self.notes:
            del self.notes[self._prev_fname]
            changed = True
        if changed:
            with open(self.notes_path, 'w') as f:
                json.dump(self.notes, f, indent=2)
            self._sync_note_markers()

    def scan_frames(self):
        found = sorted(glob.glob(os.path.join(self.frames_dir, '*.png')))
        if len(found) != len(self.frames):
            was_at_end = self.idx >= len(self.frames) - 1
            self.frames = found
            self.slider.setMaximum(max(0, len(self.frames) - 1))
            self._sync_note_markers()
            self.stats = load_stats(self.frames_dir)
            if self.live and (self.playing or was_at_end):
                self.idx = len(self.frames) - 1
            self.show_frame()

    def show_frame(self):
        if not self.frames:
            self.setWindowTitle("waiting for frames...")
            self.stats_label.setText("")
            return

        # Save note for previous frame before switching
        self._save_current_note()

        self.idx = max(0, min(self.idx, len(self.frames) - 1))
        self.current_pixmap = QPixmap(self.frames[self.idx])
        self._update_pixmap()
        n = len(self.frames)
        fname = os.path.basename(self.frames[self.idx])
        self._prev_fname = fname

        status = ""
        if self.playing:
            status = " \u25b6"
        elif self.live:
            status = " \u25c9"
        note_marker = " \u270e" if fname in self.notes else ""
        self.setWindowTitle(
            f"{self.idx + 1}/{n}  {fname}  "
            f"{self.fps}fps{status}{note_marker}")
        self.slider.blockSignals(True)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)

        # Load note for this frame into the text box
        self.note_edit.blockSignals(True)
        self.note_edit.setPlainText(self.notes.get(fname, ''))
        self.note_edit.blockSignals(False)

        # Update stats panel
        s = self.stats.get(fname)
        if s:
            seg = s.get('seg_f1', '?')
            val = s.get('val_f1', s.get('f1', '?'))
            cor = s.get('corrected_f1', '?')
            phase = 'corrective annot' if s['phase'] == 'corrective' else 'initial annot'
            self.stats_label.setText(
                f"  {s['image']}  {phase}    "
                f"seg {seg}  val {val}  corrected {cor}    "
                f"conf {s['confidence']}  ep {s['epochs']}  "
                f"t={s['sim_time']}s")
        else:
            self.stats_label.setText(f"  frame {self.idx+1}/{n}")

    def _update_pixmap(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            scaled = self.current_pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def _on_note_changed(self):
        """Live-save as the user types."""
        if not self.frames:
            return
        fname = os.path.basename(self.frames[self.idx])
        text = self.note_edit.toPlainText().strip()
        if text:
            self.notes[fname] = text
        elif fname in self.notes:
            del self.notes[fname]
        with open(self.notes_path, 'w') as f:
            json.dump(self.notes, f, indent=2)
        self._sync_note_markers()
        # Show "Saved" label briefly
        self.saved_label.show()
        self._save_flash_timer.start(1500)
        # Update title bar marker
        n = len(self.frames)
        status = ""
        if self.playing:
            status = " \u25b6"
        elif self.live:
            status = " \u25c9"
        note_marker = " \u270e" if fname in self.notes else ""
        self.setWindowTitle(
            f"{self.idx + 1}/{n}  {fname}  "
            f"{self.fps}fps{status}{note_marker}")

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
        # Let the note box handle its own keys when focused
        if self.note_edit.hasFocus():
            key = event.key()
            if key == Qt.Key_Escape:
                self.note_edit.clearFocus()
                self.setFocus()
                return
            return super().keyPressEvent(event)

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
        elif key == Qt.Key_N:
            self.note_edit.setFocus()
            self.note_edit.selectAll()
        elif key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._save_current_note()
        super().closeEvent(event)


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
