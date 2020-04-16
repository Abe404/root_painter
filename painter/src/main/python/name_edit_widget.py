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

# disable docstring warnings
#pylint: disable=C0111, C0111

# Module has no member
#pylint: disable=I1101

import re
from PyQt5 import QtWidgets
from PyQt5 import QtCore

class NameEditWidget(QtWidgets.QWidget):

    changed = QtCore.pyqtSignal()

    def __init__(self, entity):
        super().__init__()
        self.name = None
        self.initUI(entity)

    def initUI(self, entity):
        name_widget_layout = QtWidgets.QHBoxLayout()
        self.setLayout(name_widget_layout)
        edit_name_label = QtWidgets.QLabel()
        edit_name_label.setText(f"{entity} name:")
        name_widget_layout.addWidget(edit_name_label)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.textChanged.connect(self.text_changed)
        name_widget_layout.addWidget(self.name_edit)

    def text_changed(self):
        new_text = re.sub(r'\W+', '', self.name_edit.text())
        self.name_edit.setText(new_text)
        self.name = new_text
        self.changed.emit()
