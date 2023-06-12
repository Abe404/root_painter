"""
Show visibility status of segmentation, image and annotation.

Copyright (C) 2020-2023 Abraham George Smith

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

# pylint: disable=E0611, C0111, C0111, R0903, I1101
from PyQt6 import QtWidgets

class VisibilityWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # container goes full width to allow contents to be center aligned within it.
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        # left, top, right, bottom
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        container_layout = QtWidgets.QVBoxLayout()
        self.setLayout(container_layout)
        container_layout.setContentsMargins(0, 0, 0, 0)

        seg_checkbox = QtWidgets.QCheckBox("Segmentation (S)")
        container_layout.addWidget(seg_checkbox)

        annot_checkbox = QtWidgets.QCheckBox("Annotation (A)")
        container_layout.addWidget(annot_checkbox)

        im_checkbox = QtWidgets.QCheckBox("Image (I)")
        container_layout.addWidget(im_checkbox)

        seg_checkbox.setChecked(False)
        annot_checkbox.setChecked(True)
        im_checkbox.setChecked(True)

        self.seg_checkbox = seg_checkbox
        self.annot_checkbox = annot_checkbox
        self.im_checkbox = im_checkbox
