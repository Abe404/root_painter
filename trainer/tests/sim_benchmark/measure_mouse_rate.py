"""Measure actual mouse event rate delivered by Qt on this system.

Drag the mouse across the canvas while holding the button down.
Displays the live event rate (Hz) and interval stats.

Usage:
    python -m sim_benchmark.measure_mouse_rate
"""
import sys
import time

from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt


class MouseRateWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mouse Event Rate Measurement")
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)

        self.timestamps = []
        self.drag_timestamps = []
        self.dragging = False

        self.info_label = QLabel("Drag the mouse across this window to measure event rate")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px; padding: 20px;")

        self.rate_label = QLabel("")
        self.rate_label.setAlignment(Qt.AlignCenter)
        self.rate_label.setStyleSheet(
            "font-size: 32px; font-weight: bold; color: #0af; padding: 10px;")

        self.stats_label = QLabel("")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet(
            "font-size: 14px; color: #aaa; padding: 10px; "
            "font-family: 'Menlo', 'Courier New', monospace;")

        self.history_label = QLabel("")
        self.history_label.setAlignment(Qt.AlignCenter)
        self.history_label.setStyleSheet(
            "font-size: 13px; color: #888; padding: 10px;")

        layout = QVBoxLayout()
        layout.addWidget(self.info_label)
        layout.addWidget(self.rate_label)
        layout.addWidget(self.stats_label)
        layout.addWidget(self.history_label)
        self.setLayout(layout)

        self.drag_results = []

    def mousePressEvent(self, event):
        self.dragging = True
        self.drag_timestamps = [time.perf_counter()]
        self.info_label.setText("Dragging... release to see results")

    def mouseMoveEvent(self, event):
        now = time.perf_counter()
        if self.dragging:
            self.drag_timestamps.append(now)
            n = len(self.drag_timestamps)
            if n >= 3:
                elapsed = now - self.drag_timestamps[0]
                rate = (n - 1) / elapsed if elapsed > 0 else 0
                self.rate_label.setText(f"{rate:.0f} Hz")

    def mouseReleaseEvent(self, event):
        if not self.dragging:
            return
        self.dragging = False

        ts = self.drag_timestamps
        n = len(ts)
        if n < 3:
            self.info_label.setText(
                "Too short — drag longer. Try again.")
            return

        elapsed = ts[-1] - ts[0]
        rate = (n - 1) / elapsed

        intervals_ms = [(ts[i+1] - ts[i]) * 1000 for i in range(n - 1)]
        intervals_ms.sort()
        min_ms = intervals_ms[0]
        max_ms = intervals_ms[-1]
        median_ms = intervals_ms[len(intervals_ms) // 2]
        mean_ms = sum(intervals_ms) / len(intervals_ms)

        self.info_label.setText("Drag again to measure more, or close")
        self.rate_label.setText(f"{rate:.0f} Hz")
        self.stats_label.setText(
            f"{n} events in {elapsed:.2f}s\n"
            f"interval:  min={min_ms:.1f}ms  median={median_ms:.1f}ms  "
            f"mean={mean_ms:.1f}ms  max={max_ms:.1f}ms")

        self.drag_results.append(rate)
        history = "  ".join(f"{r:.0f}" for r in self.drag_results)
        self.history_label.setText(f"All runs (Hz): {history}")

        print(f"  {n} events, {elapsed:.2f}s, {rate:.0f} Hz  "
              f"(min={min_ms:.1f} med={median_ms:.1f} max={max_ms:.1f} ms)")


def main():
    app = QApplication(sys.argv)
    w = MouseRateWidget()
    w.show()
    print("Mouse event rate measurement — drag across the window")
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
