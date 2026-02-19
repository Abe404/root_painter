"""Event log viewer for sim benchmark output.

Arrow keys to step, space to play/pause, N to focus note box.
Watches the output directory for new events — launch it before or
during a simulation run to watch live.
"""
import csv
import json
import shutil
import sys
import os

from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
                              QSizePolicy, QTextEdit,
                              QVBoxLayout, QWidget)
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygon
from PyQt5.QtCore import Qt, QPoint, QTimer

from sim_benchmark.event_log import read_events, events_for_image
from sim_benchmark.replay import ImageReplay


HELP_TEXT = ("  \u25c0 \u25b6  step    Space  play/pause    \u25b2 \u25bc  speed    "
             "Home  start    End  live    N  note    S  save test case    Q  quit")

# Frames are rendered at this FPS of simulated time.
# Playing back at RENDER_FPS = real-time (1x human speed).
RENDER_FPS = 6
SPEED_PRESETS = [0.5, 1, 2, 3, 5, 10]


def load_images_csv(out_dir):
    """Load images.csv into a list of row dicts (ordered)."""
    path = os.path.join(out_dir, 'images.csv')
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


class ClickableLabel(QLabel):
    """Image label that clears note focus when clicked."""

    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer

    def mousePressEvent(self, event):
        self._viewer.note_edit.clearFocus()
        self._viewer.setFocus()


class ChartWidget(QWidget):
    """Mini line chart drawn with QPainter. No extra dependencies."""

    MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 40, 4, 4, 14

    def __init__(self, label, color, seek_callback=None, parent=None):
        super().__init__(parent)
        self.label = label
        self.color = QColor(color)
        self.values = []       # float values aligned to image list
        self.markers = []      # image indices for vertical marker lines
        self.phase_markers = [] # image indices for phase transitions
        self.note_positions = [] # image indices that have notes (top chart)
        self.cursor = -1       # current image index
        self.seek_callback = seek_callback  # callable(float fractional_idx)
        self.setFixedHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if seek_callback:
            self.setCursor(Qt.PointingHandCursor)

    def set_data(self, values, markers=None, phase_markers=None):
        self.values = values
        if markers is not None:
            self.markers = markers
        if phase_markers is not None:
            self.phase_markers = phase_markers
        self.update()

    def set_cursor(self, idx):
        """Set cursor position. Accepts float for smooth inter-image movement."""
        self.cursor = idx
        self.update()

    def paintEvent(self, event):
        if not self.values:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Dark background
        painter.fillRect(0, 0, w, h, QColor('#1a1a2e'))

        margin_l, margin_r, margin_t, margin_b = (
            self.MARGIN_L, self.MARGIN_R, self.MARGIN_T, self.MARGIN_B)
        cw = w - margin_l - margin_r
        ch = h - margin_t - margin_b
        if cw < 2 or ch < 2:
            painter.end()
            return

        n = len(self.values)
        valid = [v for v in self.values if v is not None]
        if not valid:
            painter.end()
            return
        y_min = min(valid)
        y_max = max(valid)
        if y_max - y_min < 1e-6:
            y_min -= 0.05
            y_max += 0.05

        def to_x(i):
            return margin_l + int(i / max(1, n - 1) * cw) if n > 1 else margin_l + cw // 2

        def to_y(v):
            return margin_t + ch - int((v - y_min) / (y_max - y_min) * ch)

        # Grid lines and Y-axis labels
        grid_pen = QPen(QColor('#333'), 1)
        painter.setPen(grid_pen)
        label_color = QColor('#888')
        for frac in [0.0, 0.5, 1.0]:
            yv = y_min + frac * (y_max - y_min)
            yp = to_y(yv)
            painter.drawLine(margin_l, yp, w - margin_r, yp)
            painter.setPen(label_color)
            painter.drawText(2, yp + 4, f"{yv:.2f}")
            painter.setPen(grid_pen)

        # Marker lines (best model saves)
        marker_pen = QPen(QColor('#555'), 1, Qt.DashLine)
        painter.setPen(marker_pen)
        for mi in self.markers:
            if 0 <= mi < n:
                mx = to_x(mi)
                painter.drawLine(mx, margin_t, mx, h - margin_b)

        # Phase transition lines
        phase_pen = QPen(QColor('#ff0'), 1, Qt.DashDotLine)
        painter.setPen(phase_pen)
        for mi in self.phase_markers:
            if 0 <= mi < n:
                mx = to_x(mi)
                painter.drawLine(mx, margin_t, mx, h - margin_b)

        # Data line
        data_pen = QPen(self.color, 1.5)
        painter.setPen(data_pen)
        prev = None
        for i, v in enumerate(self.values):
            if v is None:
                prev = None
                continue
            x1 = to_x(i)
            y1 = to_y(v)
            if prev is not None:
                painter.drawLine(prev[0], prev[1], x1, y1)
            prev = (x1, y1)

        # Note markers (yellow triangles at bottom of plot area)
        if self.note_positions:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor('#ff0'))
            tri_h = 6
            tri_half = 3
            for ni in self.note_positions:
                if 0 <= ni < n:
                    nx = to_x(ni)
                    by = margin_t + ch  # bottom of plot area
                    tri = QPolygon([
                        QPoint(nx, by - tri_h),
                        QPoint(nx - tri_half, by),
                        QPoint(nx + tri_half, by),
                    ])
                    painter.drawPolygon(tri)

        # Current image cursor
        if 0 <= self.cursor < n:
            cursor_pen = QPen(QColor('#fff'), 1)
            painter.setPen(cursor_pen)
            cx = to_x(self.cursor)
            painter.drawLine(cx, margin_t, cx, h - margin_b)

        # Label
        painter.setPen(self.color)
        painter.drawText(margin_l + 4, margin_t + 12, self.label)

        painter.end()

    def _x_to_fractional_idx(self, x):
        """Convert pixel x to fractional image index, or None if outside."""
        n = len(self.values)
        if n < 1:
            return None
        w = self.width()
        cw = w - self.MARGIN_L - self.MARGIN_R
        if cw < 2:
            return None
        frac = (x - self.MARGIN_L) / cw
        frac = max(0.0, min(1.0, frac))
        return frac * max(1, n - 1)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.seek_callback:
            idx = self._x_to_fractional_idx(event.x())
            if idx is not None:
                self.seek_callback(idx)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.seek_callback:
            idx = self._x_to_fractional_idx(event.x())
            if idx is not None:
                self.seek_callback(idx)
        else:
            super().mouseMoveEvent(event)


class Viewer(QMainWindow):
    def __init__(self, out_dir):
        super().__init__()
        self.out_dir = out_dir
        self.data_dir = os.path.join(out_dir, 'data')

        # Event log state
        self.all_events = []
        self.image_rows = []      # from images.csv (ordered)
        self.image_names = []     # e.g. ['img_00', 'img_01', ...]
        self.replays = {}         # image_name -> ImageReplay (lazy)
        self.total_duration = 0.0
        self.num_frames = 0       # total frames across all images

        # Per-image frame ranges: (start_frame, end_frame) indices
        self.image_frame_ranges = []

        self.idx = 0              # current frame index
        self.playing = False
        self.live = False
        self.speed_idx = SPEED_PRESETS.index(1)  # start at 1x
        self.fps = RENDER_FPS
        self._prev_image_name = None

        # Notes stored as {image_name: comment}
        self.notes_path = os.path.join(out_dir, 'notes.json')
        self.notes = {}
        if os.path.exists(self.notes_path):
            with open(self.notes_path) as f:
                self.notes = json.load(f)

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

        # Stats info panel
        self.stats_label = QLabel()
        self.stats_label.setFixedHeight(28)
        self.stats_label.setStyleSheet(
            "background: #1a1a2e; color: #8cf; padding: 4px 8px; "
            "font-size: 14px; font-family: 'Menlo', 'Courier New', monospace;")

        # Metric charts (per-image data points) — clickable for seeking
        self.confidence_chart = ChartWidget("confidence", '#0ff',
                                            self._on_chart_seek)
        self.val_f1_chart = ChartWidget("val F1", '#0f0',
                                        self._on_chart_seek)
        self.seg_f1_chart = ChartWidget("seg F1", '#88f',
                                        self._on_chart_seek)
        self.corrected_f1_chart = ChartWidget("corrected F1", '#f80',
                                              self._on_chart_seek)

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
        layout.addWidget(self.stats_label, 0)
        layout.addWidget(self.confidence_chart, 0)
        layout.addWidget(self.val_f1_chart, 0)
        layout.addWidget(self.seg_f1_chart, 0)
        layout.addWidget(self.corrected_f1_chart, 0)
        layout.addWidget(self.help_label, 0)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setFocusPolicy(Qt.StrongFocus)

        # Playback timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Initial data load
        self._reload_data()
        self.idx = 0
        if self.num_frames > 0:
            self.show_frame()
        self.setWindowTitle("sim benchmark viewer")

        # Watch timer — poll for new data (started after initial display)
        self._prev_event_count = len(self.all_events)
        self._prev_image_count = len(self.image_rows)
        self.watch_timer = QTimer()
        self.watch_timer.timeout.connect(self._poll_for_updates)
        self.watch_timer.start(200)

    def _reload_data(self):
        """Reload events.csv and images.csv, rebuild frame index."""
        events_path = os.path.join(self.out_dir, 'events.csv')
        if os.path.exists(events_path):
            self.all_events = read_events(events_path)
        else:
            self.all_events = []

        self.image_rows = load_images_csv(self.out_dir)
        self.image_names = [r['image'] for r in self.image_rows]

        # Rebuild frame ranges from image time spans
        self.image_frame_ranges = []
        total_frames = 0
        for row in self.image_rows:
            t_start = float(row['time_start'])
            t_end = float(row['time_end'])
            duration = max(0, t_end - t_start)
            n_frames = max(1, int(duration * RENDER_FPS))
            self.image_frame_ranges.append((total_frames, total_frames + n_frames))
            total_frames += n_frames

        self.num_frames = total_frames
        if self.image_rows:
            self.total_duration = float(self.image_rows[-1]['time_end'])
        else:
            self.total_duration = 0.0

        # Invalidate cached replays for images that may have changed
        self.replays = {}

        self._sync_chart_note_markers()
        self._update_charts()

    def _frame_to_image_and_time(self, frame_idx):
        """Map a global frame index to (image_index, sim_time)."""
        for img_idx, (start, end) in enumerate(self.image_frame_ranges):
            if frame_idx < end:
                row = self.image_rows[img_idx]
                t_start = float(row['time_start'])
                t_end = float(row['time_end'])
                n_frames = end - start
                local = frame_idx - start
                if n_frames > 1:
                    sim_time = t_start + local / RENDER_FPS
                else:
                    sim_time = t_end
                return img_idx, min(sim_time, t_end)
        # Past the end — return last image at its end time
        if self.image_rows:
            return len(self.image_rows) - 1, self.total_duration
        return 0, 0.0

    def _get_replay(self, image_name):
        """Get or create an ImageReplay for the given image."""
        if image_name in self.replays:
            return self.replays[image_name]

        # Find the row to get time_start
        for row in self.image_rows:
            if row['image'] == image_name:
                time_start = float(row['time_start'])
                break
        else:
            return None

        fname = f'{image_name}.png'
        img_events = events_for_image(self.all_events, fname)
        try:
            replay = ImageReplay(image_name, self.data_dir, img_events,
                                 time_start=time_start)
        except FileNotFoundError:
            return None
        self.replays[image_name] = replay
        return replay

    def _sync_chart_note_markers(self):
        """Update note markers on the confidence chart."""
        positions = []
        for name in self.notes:
            if name in self.image_names:
                positions.append(self.image_names.index(name))
        self.confidence_chart.note_positions = positions
        self.confidence_chart.update()

    def _save_current_note(self):
        """Persist the note text for the current image."""
        if self._prev_image_name is None:
            return
        text = self.note_edit.toPlainText().strip()
        changed = False
        if text:
            if self.notes.get(self._prev_image_name) != text:
                self.notes[self._prev_image_name] = text
                changed = True
        elif self._prev_image_name in self.notes:
            del self.notes[self._prev_image_name]
            changed = True
        if changed:
            with open(self.notes_path, 'w') as f:
                json.dump(self.notes, f, indent=2)
            self._sync_chart_note_markers()

    def _save_test_case(self):
        """Copy current image's data (image, gt, pred PNGs) to test_cases/."""
        if not self.image_rows:
            return
        img_idx, _ = self._frame_to_image_and_time(self.idx)
        image_name = self.image_names[img_idx]
        cases_dir = os.path.normpath(
            os.path.join(self.out_dir, '..', 'test_cases'))

        gt_path = os.path.join(self.data_dir, f'{image_name}_gt.png')
        pred_path = os.path.join(self.data_dir, f'{image_name}_pred.png')
        image_path = os.path.join(self.data_dir, f'{image_name}_image.png')
        if not os.path.exists(gt_path):
            self.saved_label.setText("No data/ for this image")
            self.saved_label.show()
            self._save_flash_timer.start(2000)
            return

        os.makedirs(cases_dir, exist_ok=True)
        for src in [gt_path, pred_path, image_path]:
            if os.path.exists(src):
                dst = os.path.join(cases_dir, os.path.basename(src))
                shutil.copy2(src, dst)

        self.saved_label.setText(f"Saved: {image_name}")
        self.saved_label.show()
        self._save_flash_timer.start(2000)
        print(f"Test case saved: {cases_dir}/{image_name}_*.png")

    def _update_charts(self):
        """Extract chart data from images.csv and push to chart widgets."""
        confidence = []
        val_f1 = []
        seg_f1 = []
        corrected_f1 = []
        best_markers = []
        phase_markers = []
        best_so_far = -1.0
        prev_phase = None

        for i, row in enumerate(self.image_rows):
            phase = row.get('phase')
            if phase != prev_phase and phase == 'corrective':
                phase_markers.append(i)
            prev_phase = phase

            try:
                confidence.append(float(row['confidence']))
            except (KeyError, ValueError):
                confidence.append(None)
            try:
                v = float(row['val_f1'])
                val_f1.append(v)
                if v > best_so_far:
                    best_so_far = v
                    best_markers.append(i)
            except (KeyError, ValueError):
                val_f1.append(None)
            try:
                seg_f1.append(float(row['seg_f1']))
            except (KeyError, ValueError):
                seg_f1.append(None)
            try:
                corrected_f1.append(float(row['corrected_f1']))
            except (KeyError, ValueError):
                corrected_f1.append(None)

        self.confidence_chart.set_data(confidence, phase_markers=phase_markers)
        self.val_f1_chart.set_data(val_f1, best_markers, phase_markers)
        self.seg_f1_chart.set_data(seg_f1, phase_markers=phase_markers)
        self.corrected_f1_chart.set_data(corrected_f1, phase_markers=phase_markers)

    def _poll_for_updates(self):
        """Check for new data in CSV files."""
        events_path = os.path.join(self.out_dir, 'events.csv')
        images_path = os.path.join(self.out_dir, 'images.csv')
        if not os.path.exists(events_path):
            return

        # Quick check: re-read images.csv and see if count changed
        new_rows = load_images_csv(self.out_dir)
        if len(new_rows) != self._prev_image_count:
            was_at_end = self.idx >= self.num_frames - 1
            self._reload_data()
            self._prev_event_count = len(self.all_events)
            self._prev_image_count = len(self.image_rows)
            if self.live and (self.playing or was_at_end):
                self.idx = max(0, self.num_frames - 1)
            self.show_frame()

    def show_frame(self):
        if self.num_frames == 0:
            self.setWindowTitle("waiting for events...")
            self.stats_label.setText("")
            return

        # Save note for previous image before switching
        self._save_current_note()

        self.idx = max(0, min(self.idx, self.num_frames - 1))
        img_idx, sim_time = self._frame_to_image_and_time(self.idx)
        image_name = self.image_names[img_idx]
        self._prev_image_name = image_name

        # Render the frame via replay
        replay = self._get_replay(image_name)
        if replay is not None:
            replay.advance_to(sim_time)
            frame_arr, _ = replay.render_frame()
            h, w, ch = frame_arr.shape
            qimg = QImage(frame_arr.data, w, h, w * ch, QImage.Format_RGB888)
            self.current_pixmap = QPixmap.fromImage(qimg.copy())
        else:
            self.current_pixmap = None
        self._update_pixmap()

        n = self.num_frames
        status = ""
        if self.playing:
            status = " \u25b6"
        elif self.live:
            status = " \u25c9"
        note_marker = " \u270e" if image_name in self.notes else ""
        self.setWindowTitle(
            f"{self.idx + 1}/{n}  {image_name}  t={sim_time:.1f}s  "
            f"{SPEED_PRESETS[self.speed_idx]}x ({self.fps}fps){status}{note_marker}")
        # Load note for this image into the text box
        self.note_edit.blockSignals(True)
        self.note_edit.setPlainText(self.notes.get(image_name, ''))
        self.note_edit.blockSignals(False)

        # Update stats panel
        speed = f"{SPEED_PRESETS[self.speed_idx]}x ({self.fps}fps)"
        row = self.image_rows[img_idx]
        phase = 'corrective annot' if row['phase'] == 'corrective' else 'initial annot'
        self.stats_label.setText(
            f"  {image_name}  {phase}    "
            f"seg {row['seg_f1']}  val {row['val_f1']}  "
            f"corrected {row['corrected_f1']}    "
            f"conf {row['confidence']}  ep {row['epochs']}  "
            f"t={sim_time:.1f}s    {speed}")

        # Update chart cursors — fractional position so cursor glides
        # smoothly through each image instead of jumping between them
        row = self.image_rows[img_idx]
        t_start_img = float(row['time_start'])
        t_end_img = float(row['time_end'])
        duration_img = t_end_img - t_start_img
        if duration_img > 0:
            frac = (sim_time - t_start_img) / duration_img
        else:
            frac = 1.0
        chart_cursor = img_idx + max(0.0, min(1.0, frac))
        self.confidence_chart.set_cursor(chart_cursor)
        self.val_f1_chart.set_cursor(chart_cursor)
        self.seg_f1_chart.set_cursor(chart_cursor)
        self.corrected_f1_chart.set_cursor(chart_cursor)

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
        if not self.image_rows:
            return
        img_idx, _ = self._frame_to_image_and_time(self.idx)
        image_name = self.image_names[img_idx]
        text = self.note_edit.toPlainText().strip()
        if text:
            self.notes[image_name] = text
        elif image_name in self.notes:
            del self.notes[image_name]
        with open(self.notes_path, 'w') as f:
            json.dump(self.notes, f, indent=2)
        self._sync_chart_note_markers()
        # Show "Saved" label briefly
        self.saved_label.show()
        self._save_flash_timer.start(1500)
        # Update title bar marker
        n = self.num_frames
        status = ""
        if self.playing:
            status = " \u25b6"
        elif self.live:
            status = " \u25c9"
        note_marker = " \u270e" if image_name in self.notes else ""
        self.setWindowTitle(
            f"{self.idx + 1}/{n}  {image_name}  "
            f"{SPEED_PRESETS[self.speed_idx]}x ({self.fps}fps){status}{note_marker}")

    def _on_chart_seek(self, fractional_idx):
        """Seek to a position from a chart click/drag."""
        self.live = False
        if not self.image_frame_ranges:
            return
        img_idx = int(fractional_idx)
        frac = fractional_idx - img_idx
        img_idx = max(0, min(img_idx, len(self.image_frame_ranges) - 1))
        start, end = self.image_frame_ranges[img_idx]
        n_frames = end - start
        self.idx = start + int(frac * max(1, n_frames - 1))
        self.show_frame()

    def next_frame(self):
        if self.idx < self.num_frames - 1:
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
            self.idx = max(0, self.num_frames - 1)
            self.show_frame()
        elif key == Qt.Key_Up:
            self.speed_idx = min(len(SPEED_PRESETS) - 1, self.speed_idx + 1)
            self.fps = int(RENDER_FPS * SPEED_PRESETS[self.speed_idx])
            if self.playing:
                self.timer.start(int(1000 / self.fps))
            self.show_frame()
        elif key == Qt.Key_Down:
            self.speed_idx = max(0, self.speed_idx - 1)
            self.fps = int(RENDER_FPS * SPEED_PRESETS[self.speed_idx])
            if self.playing:
                self.timer.start(int(1000 / self.fps))
            self.show_frame()
        elif key == Qt.Key_N:
            self.note_edit.setFocus()
            self.note_edit.selectAll()
        elif key == Qt.Key_S:
            self._save_test_case()
        elif key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._save_current_note()
        super().closeEvent(event)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'quick_output')
    os.makedirs(out_dir, exist_ok=True)

    app = QApplication(sys.argv)
    viewer = Viewer(out_dir)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
