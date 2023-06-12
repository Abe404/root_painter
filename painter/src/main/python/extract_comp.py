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

#pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914
import os
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from im_utils import gen_composite
from progress_widget import BaseProgressWidget


class Thread(QtCore.QThread):
    progress_change = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal()

    def __init__(self, seg_dir, im_dir, comp_dir):
        super().__init__()
        self.seg_dir = seg_dir
        self.im_dir = im_dir
        self.comp_dir = comp_dir

    def run(self):
        seg_fnames = os.listdir(str(self.seg_dir))
        seg_fnames = [f for f in seg_fnames if os.path.splitext(f)[1] == '.png']
        for i, f in enumerate(seg_fnames):
            self.progress_change.emit(i+1, len(seg_fnames))
            if os.path.isfile(os.path.join(self.seg_dir, os.path.splitext(f)[0] + '.png')):
                gen_composite(self.seg_dir, self.im_dir, self.comp_dir, f, ext='.jpg')
        self.done.emit()


class CompProgressWidget(BaseProgressWidget):

    def __init__(self):
        super().__init__('Generating composites')

    def run(self, seg_dir, im_dir, comp_dir):
        self.seg_dir = seg_dir
        self.im_dir = im_dir
        self.comp_dir = comp_dir
        self.thread = Thread(seg_dir, im_dir, comp_dir)
        seg_fnames = os.listdir(str(self.seg_dir))
        seg_fnames = [f for f in seg_fnames if os.path.splitext(f)[1] == '.png']
        self.progress_bar.setMaximum(len(seg_fnames))
        self.thread.progress_change.connect(self.onCountChanged)
        self.thread.done.connect(self.done)
        self.thread.start()

    def done(self):
        QtWidgets.QMessageBox.about(self, 'Composites Extracted',
                                    f'Extracting composites from {self.seg_dir} '
                                    f'and {self.im_dir} to {self.comp_dir} '
                                    'is complete.')
        self.close()


class ExtractCompWidget(QtWidgets.QWidget):
    submit = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.seg_dir = None
        self.im_dir = None
        self.comp_dir = None
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("Extract Composites")

        # Add specify seg directory button
        seg_dir_label = QtWidgets.QLabel()
        seg_dir_label.setText("Segmentation directory: Not yet specified")
        layout.addWidget(seg_dir_label)
        self.seg_dir_label = seg_dir_label

        specify_seg_btn = QtWidgets.QPushButton('Specify segmentation directory')
        specify_seg_btn.clicked.connect(self.select_seg_dir)
        layout.addWidget(specify_seg_btn)

        # Add specify photo directory button
        im_dir_label = QtWidgets.QLabel()
        im_dir_label.setText("Image directory: Not yet specified")
        layout.addWidget(im_dir_label)
        self.im_dir_label = im_dir_label

        specify_im_dir_btn = QtWidgets.QPushButton('Specify image directory')
        specify_im_dir_btn.clicked.connect(self.select_im_dir)
        layout.addWidget(specify_im_dir_btn)

        # Add specify comp directory button
        comp_dir_label = QtWidgets.QLabel()
        comp_dir_label.setText("Composites directory: Not yet specified")
        layout.addWidget(comp_dir_label)
        self.comp_dir_label = comp_dir_label

        specify_comp_dir_btn = QtWidgets.QPushButton('Specify composites directory')
        specify_comp_dir_btn.clicked.connect(self.select_comp_dir)
        layout.addWidget(specify_comp_dir_btn)


        info_label = QtWidgets.QLabel()
        info_label.setText("Segmentation directory, image and composites directory"
                           "must be specified.")
        layout.addWidget(info_label)
        self.info_label = info_label

        submit_btn = QtWidgets.QPushButton('Extract Composites')
        submit_btn.clicked.connect(self.extract_composites)
        layout.addWidget(submit_btn)
        submit_btn.setEnabled(False)
        self.submit_btn = submit_btn

    def extract_composites(self):
        self.progress_widget = CompProgressWidget()
        self.progress_widget.run(self.seg_dir, self.im_dir, self.comp_dir)
        self.progress_widget.show()
        self.close()

    def validate(self):
        if not self.seg_dir:
            self.info_label.setText("Segmentation directory must be "
                                    "specified to extract composites.")
            self.submit_btn.setEnabled(False)
            return

        if not self.im_dir:
            self.info_label.setText("Image directory must be "
                                    "specified to extract composites.")
            self.submit_btn.setEnabled(False)
            return

        if not self.comp_dir:
            self.info_label.setText("Composites directory must be "
                                    "specified to extract composites.")
            self.submit_btn.setEnabled(False)
            return

        self.info_label.setText("")
        self.submit_btn.setEnabled(True)

    def select_seg_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        def input_selected():
            self.seg_dir = self.input_dialog.selectedFiles()[0]
            self.seg_dir_label.setText('Segmentation directory: ' + self.seg_dir)
            self.validate()
        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()

    def select_im_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def input_selected():
            self.im_dir = self.input_dialog.selectedFiles()[0]
            self.im_dir_label.setText('Image directory: ' + self.im_dir)
            self.validate()
        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()

    def select_comp_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        def input_selected():
            self.comp_dir = self.input_dialog.selectedFiles()[0]
            self.comp_dir_label.setText('Composites directory: ' + self.comp_dir)
            self.validate()
        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()
