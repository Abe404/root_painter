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

# pylint: disable=I1101, C0111, E0611, R0902
""" Canvas where image and annotations can be drawn """
from math import hypot
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt


class GraphicsScene(QtWidgets.QGraphicsScene):
    """
    Canvas where image and lines will be drawn
    """
    def __init__(self):
        super().__init__()
        self.drawing = False
        self.brush_size = 25
        # history is a list of pixmaps
        self.history = []
        self.redo_list = []

        self.last_x = None
        self.last_y = None
        self.annot_pixmap = None

        # These globals should eventually be loaded from a configuation file, which would
        # be created on project creation.
        self.foreground_color = QtGui.QColor(255, 0, 0, 180)
        self.background_color = QtGui.QColor(0, 255, 0, 180)
        self.eraser_color = QtGui.QColor(255, 105, 180, 0)
        self.brush_color = self.foreground_color


    def undo(self):
        if len(self.history) > 1:
            self.redo_list.append(self.history.pop().copy())
            # remove top item from history.
            new_state = self.history[-1].copy()
            self.annot_pixmap_holder.setPixmap(new_state)
            self.annot_pixmap = new_state


    def redo(self):
        if self.redo_list:
            new_state = self.redo_list.pop()
            self.history.append(new_state.copy())
            self.annot_pixmap_holder.setPixmap(new_state)
            self.annot_pixmap = new_state


    def mousePressEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if not modifiers & QtCore.Qt.ControlModifier and self.parent.annot_visible:
            self.drawing = True
            pos = event.scenePos()
            x, y = pos.x(), pos.y()
            if self.brush_size == 1:
                circle_x = x
                circle_y = y
            else:
                circle_x = x - (self.brush_size / 2) + 0.5
                circle_y = y - (self.brush_size / 2) + 0.5

            painter = QtGui.QPainter(self.annot_pixmap)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
            painter.drawPixmap(0, 0, self.annot_pixmap)
            painter.setPen(QtGui.QPen(self.brush_color, 0, Qt.SolidLine,
                                      Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(QtGui.QBrush(self.brush_color, Qt.SolidPattern))
            if self.brush_size == 1:
                painter.drawPoint(round(circle_x), round(circle_y))
            else:
                painter.drawEllipse(round(circle_x), round(circle_y),
                                    round(self.brush_size-1), round(self.brush_size-1))
            self.annot_pixmap_holder.setPixmap(self.annot_pixmap)
            painter.end()
            self.last_x = x
            self.last_y = y

    def mouseReleaseEvent(self, _event):
        if self.drawing:
            self.drawing = False
            # has to be some limit to history or RAM will run out
            if len(self.history) > 50:
                self.history = self.history[-50:]
            self.history.append(self.annot_pixmap.copy())
            self.redo_list = []

    def mouseMoveEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        shift_down = (modifiers & QtCore.Qt.ShiftModifier)
        pos = event.scenePos()
        x, y = pos.x(), pos.y()
        if shift_down:
            dist = self.last_y - y
            self.brush_size += dist
            self.brush_size = max(1, self.brush_size)
            # Warning: very tight coupling.
            self.parent.update_cursor()
        elif self.drawing:
            painter = QtGui.QPainter(self.annot_pixmap)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
            painter.drawPixmap(0, 0, self.annot_pixmap)
            pen = QtGui.QPen(self.brush_color, self.brush_size, Qt.SolidLine,
                             Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)

            # Based on empirical observation
            if self.brush_size % 2 == 0:
                painter.drawLine(round(self.last_x+0.5), round(self.last_y+0.5),
                                 round(x+0.5), round(y+0.5))
            else:
                painter.drawLine(round(self.last_x), round(self.last_y), round(x), round(y))

            self.annot_pixmap_holder.setPixmap(self.annot_pixmap)
            painter.end()
        self.last_x = x
        self.last_y = y
