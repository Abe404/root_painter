"""
Container for canvas where image and lines are drawn.
Facilitates use of zoom and pan.

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

#pylint: disable=I1101,E0611,C0111
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6.QtCore import Qt

class CustomGraphicsView(QtWidgets.QGraphicsView):
    """
    Container for canvas where image and lines are drawn.
    Facilitates use of zoom and pan.
    """
    mouse_scroll_event = QtCore.pyqtSignal(QtGui.QWheelEvent)
    zoom_change = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.zoom = 1
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

    def update_zoom(self):
        """ Transform the view based on current zoom value """
        self.setTransform(QtGui.QTransform().scale(self.zoom, self.zoom))
        self.zoom_change.emit()

    def wheelEvent(self, event):
        self.mouse_scroll_event.emit(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

    def show_actual_size(self):
        self.zoom = 1
        self.update_zoom()

    def fit_to_view(self, _event=None):
        view_width = self.geometry().width()
        view_height = self.geometry().height()
        im_width = self.image.width()
        im_height = self.image.height()
        width_ratio = im_width / view_width
        height_ratio = im_height / view_height

        if width_ratio > height_ratio:
            self.zoom = view_width / im_width
        else:
            self.zoom = view_height / im_height

        scene_rect = self.sceneRect()
        def fitin():
            self.fitInView(scene_rect, Qt.KeepAspectRatio)
        fitin()
        QtCore.QTimer.singleShot(100, fitin)
        self.zoom_change.emit()
