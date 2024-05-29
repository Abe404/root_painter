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

        self.settings = QtCore.QSettings("rp", "painter")
        self.language = self.settings.value("language", "English")

        self.controls_text_en = """
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

        self.controls_text_es = """
        Controles:
        - Mostrar/Ocultar Segmentación AI: S
        - Mostrar/Ocultar Anotacion Humana: A
        - Mostrar/Ocultar Imagen: I

        - Rojo(Si): Q
        - Verde(No): W
        - Borrar: E

        - Tamaño del Pincel: Shift + rueda de el mouse

        - Deshacer: Z        
        - Rehacer: Shift + cmd + Z

        - Acercar: Shift + '+'
        - Alejar: -
        - Zoom: rueda de el mouse

        - mover imagen: cmd + arrastrar sobre la imagen con clic derecho
        """

        self.layout = QtWidgets.QVBoxLayout()

        self.language_selector = QtWidgets.QComboBox()
        self.language_selector.addItem("English")
        self.language_selector.addItem("Spanish")
        self.language_selector.setCurrentText(self.language)
        self.language_selector.currentTextChanged.connect(self.update_text)

        self.layout.addWidget(self.language_selector)

        self.label = QtWidgets.QLabel()
        self.layout.addWidget(self.label)

        self.setLayout(self.layout)
        self.update_text(self.language)

    def update_text(self, language):
        if language == "Spanish":
            self.label.setText(self.controls_text_es)
        else:
            self.label.setText(self.controls_text_en)
        self.settings.setValue("language", language)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)