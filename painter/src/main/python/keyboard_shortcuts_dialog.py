"""
Copyright (C) 2025 Abraham George Smith

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

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class KeyboardShortcutsDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(370, 460)

        shortcuts = [
            ("Foreground", "Q"),
            ("Background", "W"),
            ("Eraser", "E"),
            ("Undo", "Z"),
            ("Redo", "Ctrl+Shift+Z"),
            ("Toggle segmentation", "S"),
            ("Toggle annotations", "A"),
            ("Toggle image", "I"),
            ("Brush size", "Shift+Scroll"),
            ("Zoom in", "+"),
            ("Zoom out", "-"),
            ("Zoom", "Scroll"),
            ("Fit to view", "Ctrl+F"),
            ("Actual size", "Ctrl+A"),
            ("Pan", "Right-click drag"),
        ]

        layout = QtWidgets.QVBoxLayout()
        table = QtWidgets.QTableWidget(len(shortcuts), 2)
        table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)

        for row, (action, shortcut) in enumerate(shortcuts):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(action))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(shortcut))

        layout.addWidget(table)
        self.setLayout(layout)
