"""
Copyright (C) 2024 Felipe Galindo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtCore

# class to create a new dialog window that shows the controls of the application
class ControlsDialog(QtWidgets.QDialog):
    closed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Controls")
        self.setMinimumSize(300, 200)

        # controls_text = """
        # Controls:
        # - Left Click: Draw
        # - Right Click: Erase
        # - Scroll: Zoom
        # - Arrow Keys: Move around
        # - Ctrl + Z: Undo
        # - Ctrl + Y: Redo
        # """

        controls_text = """
        Controls:
        - Show/Hide Predicted Segmentation: S
        - Show/Hide Human Annotations: A
        - Show/Hide Image: I

        - Foreground: Q
        - Background: W
        - Erase: E

        - Change Brush Size: Shift + Scroll

        - Undo: Z        
        - Redo: Shift + cmd + Z

        - Zoom In: Shift + '+'
        - Zoom Out: -
        - Zoom: Scroll

        - Pan: cmd + drag the on the image with right click
        
        """

        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(controls_text)
        layout.addWidget(label)
        self.setLayout(layout)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
