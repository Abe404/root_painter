import sys
from PyQt5 import QtWidgets
from PyQt5 import QtCore

# This class to create a new dialog window that shows the controls of the application
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