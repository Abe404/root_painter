"""
Copyright (C) 2023 Abraham George Smith

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
from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6 import QtCore
from skimage.io import imsave

from progress_widget import BaseProgressWidget
import im_utils

class CreationProgressWidget(BaseProgressWidget):
    def __init__(self):
        super().__init__('Resize Images')

    def run(self, all_image_paths, output_dir, resize_percent):
        self.progress_bar.setMaximum(len(all_image_paths))
        self.creation_thread = CreationThread(all_image_paths, output_dir, resize_percent)
        self.creation_thread.progress_change.connect(self.onCountChanged)
        self.creation_thread.done.connect(self.done)
        self.creation_thread.start()


class CreationThread(QtCore.QThread):
    """
    Runs another thread.
    """
    progress_change = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal(list)

    def __init__(self, all_image_paths, output_dir, resize_percent):
        super().__init__()
        self.all_image_paths = all_image_paths
        self.output_dir = output_dir
        self.resize_percent = resize_percent

    def run(self):
        error_messages = []
        progress = 0
        for fpath in self.all_image_paths:
            fname = os.path.basename(fpath)
            output_path = os.path.join(self.output_dir,
                                       os.path.splitext(fname)[0] + '.jpg')
            image = im_utils.load_image(fpath)
            resized = im_utils.resize_image(image, self.resize_percent)
            imsave(output_path, resized, quality=95)
            progress += 1
            self.progress_change.emit(progress, len(self.all_image_paths))
        self.done.emit(error_messages)


class ResizeWidget(QtWidgets.QWidget):

    submit = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.source_dir = None
        self.output_dir = None
        self.resize_percent = 50 # default value
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Resize Images")
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

        # resize percent input 
        resize_percent_widget = QtWidgets.QWidget()
        layout.addWidget(resize_percent_widget)
        resize_percent_widget_layout = QtWidgets.QHBoxLayout()
        resize_percent_widget.setLayout(resize_percent_widget_layout)
        edit_resize_percent_label = QtWidgets.QLabel()

        edit_resize_percent_label.setText("Resize percentage (target width and height):")
        resize_percent_widget_layout.addWidget(edit_resize_percent_label)
        resize_percent_edit_widget = QtWidgets.QSpinBox()
        resize_percent_edit_widget.setMaximum(800)
        resize_percent_edit_widget.setMinimum(1)
        resize_percent_edit_widget.setValue(self.resize_percent)
        resize_percent_edit_widget.valueChanged.connect(self.validate)
        self.resize_percent_edit_widget = resize_percent_edit_widget
        resize_percent_widget_layout.addWidget(resize_percent_edit_widget)

        # info label for user feedback
        info_label = QtWidgets.QLabel()
        info_label.setText("Source directory, output directory and resize percentage must"
                           " be specified in order to resize images.")
        layout.addWidget(info_label)
        self.info_label = info_label

        # Add create button
        create_btn = QtWidgets.QPushButton('Resize images')
        create_btn.clicked.connect(self.try_submit)
        layout.addWidget(create_btn)
        create_btn.setEnabled(False)
        self.create_btn = create_btn


    def validate(self):
        if not self.output_dir:
            self.info_label.setText("Output directory must be specified.")
            self.create_btn.setEnabled(False)
            return
        if not self.source_dir:
            self.info_label.setText("source directory must be specified.")
            self.create_btn.setEnabled(False)
            return
        if not self.resize_percent_edit_widget.value():
            self.info_label.setText("Resize percent must be "
                                    "specified to resize images")
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
        resize_percent = self.resize_percent_edit_widget.value()
        all_image_paths = self.image_paths
        self.progress_widget = CreationProgressWidget()
        self.progress_widget.run(all_image_paths, output_dir, resize_percent)
        self.close()
        self.progress_widget.show()


    def select_image_dir(self):
        self.image_dialog = QtWidgets.QFileDialog(self)
        self.image_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)

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
        self.out_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)

        def output_selected():
            self.output_dir = self.out_dialog.selectedFiles()[0]
            self.out_directory_label.setText('Output directory: ' + self.output_dir)
            self.validate()

        self.out_dialog.fileSelected.connect(output_selected)
        self.out_dialog.open()
