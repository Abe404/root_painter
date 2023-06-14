"""
Copyright (C) 2020 Abraham George Smith
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

#pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914
import os
import numpy as np
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from progress_widget import BaseProgressWidget
from skimage.io import imread, imsave
from skimage import img_as_ubyte, img_as_float

class ConvertThread(QtCore.QThread):
    progress_change = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal()

    def __init__(self, conversion_function, seg_dir, out_dir):
        super().__init__()
        self.seg_dir = seg_dir
        self.out_dir = out_dir
        self.conversion_function = conversion_function

    def run(self):
        seg_fnames = os.listdir(str(self.seg_dir))
        seg_fnames = [f for f in seg_fnames if os.path.splitext(f)[1] == '.png']

        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        for i, f in enumerate(seg_fnames):
            self.progress_change.emit(i+1, len(seg_fnames))
            if os.path.isfile(os.path.join(self.seg_dir, os.path.splitext(f)[0] + '.png')):
                # Load RootPainter seg blue channel and invert.
                seg = imread(os.path.join(self.seg_dir, f))
                converted_seg = self.conversion_function(seg)
                imsave(os.path.join(self.out_dir, f),
                       converted_seg, check_contrast=False)
        self.done.emit()


def convert_seg_to_rve(seg):
    # Load RootPainter blue channel and invert.
    rve_seg = (seg[:, :, 2] == 0)
    return img_as_ubyte(rve_seg)

def convert_seg_to_annot(seg):
    """ segmention blue channel is foreground annotation
        everything else is background annotation """
    annot = np.zeros((seg.shape[0], seg.shape[1], 4))
    seg_fg = seg[:, :, 2] # assume blue is foreground prediction
    seg_fg = img_as_float(seg_fg)
    annot[:, :, 0] = (seg_fg >= 0.5) 
    annot[:, :, 1] = (seg_fg < 0.5) # bg channel
    annot[:, :, 3] = np.ones(seg_fg.shape) 
    return img_as_ubyte(annot)

class ConvertProgressWidget(BaseProgressWidget):

    def __init__(self):
        super().__init__('Converting Segmentations')

    def run(self, convert_function, seg_dir, out_dir):
        self.seg_dir = seg_dir
        self.out_dir = out_dir
        self.thread = ConvertThread(convert_function, seg_dir, out_dir)
        seg_fnames = os.listdir(str(self.seg_dir))
        seg_fnames = [f for f in seg_fnames if os.path.splitext(f)[1] == '.png']
        self.progress_bar.setMaximum(len(seg_fnames))
        self.thread.progress_change.connect(self.onCountChanged)
        self.thread.done.connect(self.done)
        self.thread.start()

    def done(self):
        QtWidgets.QMessageBox.about(self, 'Segmentations Converted',
                                    f'Converting segmentations from {self.seg_dir} '
                                    f'to {self.out_dir} '
                                    'is complete.')
        self.close()


class ConvertSegWidget(QtWidgets.QWidget):
    submit = QtCore.pyqtSignal()

    def __init__(self, convert_function, output_type):
        super().__init__()
        self.seg_dir = None
        self.out_dir = None
        self.convert_function = convert_function
        self.output_type_name = output_type
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle(f"Convert Segmentations to {self.output_type_name}")

        # Add specify seg directory button
        seg_dir_label = QtWidgets.QLabel()
        seg_dir_label.setText("Segmentation directory: Not yet specified")
        layout.addWidget(seg_dir_label)
        self.seg_dir_label = seg_dir_label

        self.specify_seg_btn = QtWidgets.QPushButton('Specify segmentation directory')
        self.specify_seg_btn.clicked.connect(self.select_seg_dir)
        layout.addWidget(self.specify_seg_btn)

        # Add specify comp directory button
        out_dir_label = QtWidgets.QLabel()
        out_dir_label.setText("Output directory: Not yet specified")
        layout.addWidget(out_dir_label)
        self.out_dir_label = out_dir_label

        specify_out_dir_btn = QtWidgets.QPushButton('Specify output directory')
        specify_out_dir_btn.clicked.connect(self.select_out_dir)
        layout.addWidget(specify_out_dir_btn)


        info_label = QtWidgets.QLabel()
        info_label.setText("Segmentation directory and output directory"
                           " must be specified.")
        layout.addWidget(info_label)
        self.info_label = info_label

        submit_btn = QtWidgets.QPushButton('Convert Segmentations')
        submit_btn.clicked.connect(self.convert_segmentations)
        layout.addWidget(submit_btn)
        submit_btn.setEnabled(False)
        self.submit_btn = submit_btn

    def convert_segmentations(self):
        self.progress_widget = ConvertProgressWidget()
        self.progress_widget.run(self.convert_function, self.seg_dir, self.out_dir)
        self.progress_widget.show()
        self.close()

    def validate(self):
        if not self.seg_dir:
            self.info_label.setText("Segmentation directory must be "
                                    "specified to convert files.")
            self.submit_btn.setEnabled(False)
            return

        if not self.out_dir:
            self.info_label.setText("Output directory must be "
                                    "specified to convert files.")
            self.submit_btn.setEnabled(False)
            return

        self.info_label.setText("")
        self.submit_btn.setEnabled(True)

    def select_seg_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        def input_selected():
            self.seg_dir = self.input_dialog.selectedFiles()[0]
            self.seg_dir_label.setText('Segmentation directory: ' + self.seg_dir)
            self.validate()
        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()

    def select_out_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        def input_selected():
            self.out_dir = self.input_dialog.selectedFiles()[0]
            self.out_dir_label.setText('Output directory: ' + self.out_dir)
            self.validate()
        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()
