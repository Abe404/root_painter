"""
Copyright (C) 2020 Abraham George Smith

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

# pylint: disable=C0111, I1101, E0611
import os

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import Qt


class NavWidget(QtWidgets.QWidget):
    """ Shows next and previous buttons as well as image position in folder.
    """
    file_change = QtCore.pyqtSignal(str)
    class_change = QtCore.pyqtSignal(str)

    def __init__(self, all_fnames, classes):
        super().__init__()
        self.image_path = None
        self.all_fnames = all_fnames
        self.classes = classes
        self.initUI()

    def initUI(self):
        # container goes full width to allow contents to be center aligned within it.
        nav = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout()

        # && to escape it and show single &
        self.prev_image_button = QtWidgets.QPushButton('< Previous')
        self.prev_image_button.setFocusPolicy(Qt.NoFocus)
        self.prev_image_button.clicked.connect(self.show_prev_image)
        nav_layout.addWidget(self.prev_image_button)
        self.nav_label = QtWidgets.QLabel()
        nav_layout.addWidget(self.nav_label)

        # && to escape it and show single &
        self.next_image_button = QtWidgets.QPushButton('Save && Next >')
        self.next_image_button.setFocusPolicy(Qt.NoFocus)
        self.next_image_button.clicked.connect(self.show_next_image)
        nav_layout.addWidget(self.next_image_button)

        # left, top, right, bottom
        nav_layout.setContentsMargins(0, 0, 0, 5)
        nav.setLayout(nav_layout)
        nav.setMaximumWidth(600)

        self.cb = QtWidgets.QComboBox()
        self.cb.addItems(self.classes)
        self.cb.currentIndexChanged.connect(self.selection_change)
        nav_layout.addWidget(self.cb)

        container_layout = QtWidgets.QHBoxLayout()
        container_layout.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(nav)
        self.setLayout(container_layout)
        container_layout.setContentsMargins(0, 0, 0, 0)

    def selection_change(self, _):
        self.class_change.emit(self.cb.currentText())

    def get_path_list(self, dir_path):
        all_files = self.all_fnames
        all_paths = [os.path.abspath(os.path.join(os.path.abspath(dir_path), a))
                     for a in all_files]
        return all_paths

    def show_next_image(self):
        self.next_image_button.setEnabled(False)
        self.next_image_button.setText('Loading..')
        self.next_image_button.setEnabled(False)
        QtWidgets.QApplication.processEvents()
        dir_path, _ = os.path.split(self.image_path)
        all_paths = self.get_path_list(dir_path)
        cur_idx = all_paths.index(self.image_path)
        next_idx = cur_idx + 1
        if next_idx >= len(all_paths):
            next_idx = 0
        self.image_path = all_paths[next_idx]
        self.file_change.emit(self.image_path)
        self.update_nav_label()

    def show_prev_image(self):
        dir_path, _ = os.path.split(self.image_path)
        all_paths = self.get_path_list(dir_path)
        cur_idx = all_paths.index(self.image_path)
        next_idx = cur_idx - 1
        if next_idx <= 0:
            next_idx = 0
        self.image_path = all_paths[next_idx]
        self.file_change.emit(self.image_path)
        self.update_nav_label()

    def update_nav_label(self):
        dir_path, _ = os.path.split(self.image_path)
        all_paths = self.get_path_list(dir_path)
        cur_idx = all_paths.index(os.path.abspath(self.image_path))
        self.nav_label.setText(f'{cur_idx + 1} / {len(all_paths)}')
