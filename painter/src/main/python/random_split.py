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
#pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914, R0915, R0911
import os
import glob
import random
from pathlib import Path
import itertools
import json
from random import shuffle
import traceback
import shutil

import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from progress_widget import BaseProgressWidget
import im_utils

class CreationProgressWidget(BaseProgressWidget):
    def __init__(self):
        super().__init__('Creating split')

    def run(self, images, output_dir, split_percent):
        self.progress_bar.setMaximum(len(images))
        self.creation_thread = CreationThread(images, output_dir, split_percent)
        self.creation_thread.progress_change.connect(self.onCountChanged)
        self.creation_thread.done.connect(self.done)
        self.creation_thread.start()


class CreationThread(QtCore.QThread):
    """
    Runs another thread.
    """
    progress_change = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal(list)

    def __init__(self, images, output_dir, split_percent):
        super().__init__()
        self.images = images
        self.output_dir = output_dir
        self.split_percent = split_percent

    def run(self):
        error_messages = []
        random.shuffle(self.images)
        num_images_for_split_1 = round(((float(len(self.images)) * self.split_percent) / 100))
        images_for_split_1 = self.images[:num_images_for_split_1]
        images_for_split_2 = self.images[num_images_for_split_1:]

        split_1_name = 'split_1' 
        split_2_name = 'split_2'

        os.makedirs(os.path.join(self.output_dir, split_1_name))
        os.makedirs(os.path.join(self.output_dir, split_2_name))
        
        progress = 0

        for fpath in images_for_split_1:
            fname = os.path.basename(fpath)
            output_path = os.path.join(self.output_dir, split_1_name, fname)
            shutil.copyfile(fpath, output_path)
            progress += 1
            self.progress_change.emit(progress, len(self.images))

        for fpath in images_for_split_2:
            fname = os.path.basename(fpath)
            output_path = os.path.join(self.output_dir, split_2_name, fname)
            shutil.copyfile(fpath, output_path)
            progress += 1
            self.progress_change.emit(progress, len(self.images))

        self.done.emit(error_messages)


class RandomSplitWidget(QtWidgets.QWidget):

    submit = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.source_dir = None
        self.output_dir = None
        self.split_percent = 20 # default value
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Create Split")
        layout = QtWidgets.QVBoxLayout()
        self.layout = layout # to add progress bar later.
        self.setLayout(layout)

        # Add specify image directory button
        directory_label = QtWidgets.QLabel()
        directory_label.setText("Source image directory: Not yet specified")
        layout.addWidget(directory_label)
        self.directory_label = directory_label

        specify_image_dir_btn = QtWidgets.QPushButton('Specify source image directory')
        specify_image_dir_btn.clicked.connect(self.select_image_dir)
        layout.addWidget(specify_image_dir_btn)

        # Add specify output directory button
        out_directory_label = QtWidgets.QLabel()
        out_directory_label.setText("Output directory: Not yet specified")
        layout.addWidget(out_directory_label)
        self.out_directory_label = out_directory_label

        specify_out_dir_btn = QtWidgets.QPushButton('Specify output directory')
        specify_out_dir_btn.clicked.connect(self.select_output_dir)
        layout.addWidget(specify_out_dir_btn)

        # split percent input 
        split_percent_widget = QtWidgets.QWidget()
        layout.addWidget(split_percent_widget)
        split_percent_widget_layout = QtWidgets.QHBoxLayout()
        split_percent_widget.setLayout(split_percent_widget_layout)
        edit_split_percent_label = QtWidgets.QLabel()

        edit_split_percent_label.setText("Split percentage:")
        split_percent_widget_layout.addWidget(edit_split_percent_label)
        split_percent_edit_widget = QtWidgets.QSpinBox()
        split_percent_edit_widget.setMaximum(99)
        split_percent_edit_widget.setMinimum(1)
        split_percent_edit_widget.setValue(self.split_percent)
        split_percent_edit_widget.valueChanged.connect(self.validate)
        self.split_percent_edit_widget = split_percent_edit_widget
        split_percent_widget_layout.addWidget(split_percent_edit_widget)

        # info label for user feedback
        info_label = QtWidgets.QLabel()
        info_label.setText("Source directory, output directory and split percentage must"
                           " be specified in order to create split.")
        layout.addWidget(info_label)
        self.info_label = info_label

        # Add create button
        create_btn = QtWidgets.QPushButton('Create split')
        create_btn.clicked.connect(self.try_submit)
        layout.addWidget(create_btn)
        create_btn.setEnabled(False)
        self.create_btn = create_btn


    def validate(self):
        if not self.output_dir:
            self.info_label.setText("Output director must be specified to create split")
            self.create_btn.setEnabled(False)
            return
        if not self.source_dir:
            self.info_label.setText("source directory must be specified to create split")
            self.create_btn.setEnabled(False)
            return
        if not self.split_percent_edit_widget.value():
            self.info_label.setText("Split percent must be "
                                    "specified to create split")
            self.create_btn.setEnabled(False)
            return

        if not self.image_paths:
            message = ('Source image directory must contain image files')
            self.info_label.setText(message)
            self.create_btn.setEnabled(False)
            return

        # Sucess!
        self.info_label.setText("")
        self.create_btn.setEnabled(True)


    def try_submit(self):
        output_dir = Path(self.output_dir)
        split_percent = self.split_percent_edit_widget.value()
        all_images = self.image_paths
        self.progress_widget = CreationProgressWidget()
        self.progress_widget.run(all_images, output_dir, split_percent)
        self.close()
        self.progress_widget.show()


    def select_image_dir(self):
        self.image_dialog = QtWidgets.QFileDialog(self)
        self.image_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def output_selected():
            self.source_dir = self.image_dialog.selectedFiles()[0]
            self.directory_label.setText('Image directory: ' + self.source_dir)
            im_fnames = [f for f in os.listdir(self.source_dir) if im_utils.is_image(f)]
            self.image_paths = [os.path.join(self.source_dir, f) for f in im_fnames]
            self.validate()

        self.image_dialog.fileSelected.connect(output_selected)
        self.image_dialog.open()


    def select_output_dir(self):
        self.out_dialog = QtWidgets.QFileDialog(self)
        self.out_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def output_selected():
            self.output_dir = self.out_dialog.selectedFiles()[0]
            self.out_directory_label.setText('Output directory: ' + self.output_dir)
            self.validate()

        self.out_dialog.fileSelected.connect(output_selected)
        self.out_dialog.open()
