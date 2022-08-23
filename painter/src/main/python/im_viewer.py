"""
Copyright (C) 2022 Abraham George Smith

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
import os
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
import numpy as np
from skimage.color import rgb2gray
import qimage2ndarray


class ContextViewer(QtWidgets.QWidget):

    def __init__(self, fpath, patch):
        super().__init__()
        self.title = 'Original Image: ' + os.path.basename(fpath)
        self.fpath = fpath 
        self.patch_np = np.array(qimage2ndarray.rgb_view(patch.toImage()))
        self.initUI()
        y = 0
        x = 0
        w = self.patch_np.shape[1]
        h = self.patch_np.shape[0]
        self.patch_np = rgb2gray(self.patch_np)
        self.patch_np -= np.min(self.patch_np)
        self.patch_np = self.patch_np / np.max(self.patch_np)
        lowest_diff = 1000*1000
        lowest_diff_x = 0
        lowest_diff_y = 0

        while x < self.full_im_np.shape[1]:
            y = 0
            while y < self.full_im_np.shape[0]:
                im_patch = self.full_im_np[y:y+h, x:x+w]
                im_patch = rgb2gray(im_patch)
                im_patch -= np.min(im_patch)
                im_patch = im_patch / np.max(im_patch)
                diff = np.sum(np.abs(im_patch - self.patch_np))
                if diff < lowest_diff:
                    lowest_diff = diff
                    lowest_diff_x = x
                    lowest_diff_y = y
                y += h
            x += w
        for p in [self.pixmap, self.orig_pixmap]:
            painter= QtGui.QPainter(p)
            pen = QtGui.QPen(QtCore.Qt.red)
            pen.setWidth(6)
            painter.setPen(pen)
            painter.drawRect(3+lowest_diff_x, lowest_diff_y, w, h)
            painter.end()
            self.resizeEvent(None)
                       

    def resizeEvent(self, event):
        size = self.size()
        width = size.width()
        height = size.height()
        self.pixmap = self.orig_pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)
        self.label.resize(self.pixmap.width(), self.pixmap.height())
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.label = QtWidgets.QLabel(self)
        self.orig_pixmap = QtGui.QPixmap(self.fpath)
        self.full_im_np = np.array(qimage2ndarray.rgb_view(self.orig_pixmap.toImage()))
        self.label.setPixmap(self.orig_pixmap)

        screen_size = QtWidgets.QApplication.primaryScreen().size()
        screen_width = screen_size.width()
        target_width = screen_width // 2
        target_height = (target_width / self.orig_pixmap.width()) * self.orig_pixmap.height()

        self.setGeometry(0, 0, round(target_width), round(target_height))


        self.show()
