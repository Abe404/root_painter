"""
Copyright (C) 2022 Abraham George Smith
Copyright (C) 2023 Rohan Howard Orton

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

# pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914
import os

from im_utils import save_corrected_segmentation, all_image_paths_in_dir
from progress_widget import BaseProgressWidget
from PyQt5 import QtCore, QtWidgets
import traceback


class Thread(QtCore.QThread):
    progress_change = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal()

    def __init__(self, annot_dir, seg_dir, out_dir):
        super().__init__()
        self.seg_dir = seg_dir
        self.annot_dir = annot_dir
        self.out_dir = out_dir

    def run(self):
        annot_fpaths = all_image_paths_in_dir(self.annot_dir)
        annot_fpaths = [f for f in annot_fpaths if os.path.splitext(f)[1] == ".png"]
        for i, f in enumerate(annot_fpaths):
            self.progress_change.emit(i + 1, len(annot_fpaths))
            if os.path.isfile(f):
                try:
                    save_corrected_segmentation(
                        f, self.seg_dir, self.out_dir,
                    )
                except Exception as e:
                    print('Exception handling', f)
                    print(e)
                    traceback.print_exc()
        self.done.emit()


class ProgressWidget(BaseProgressWidget):
    def __init__(self):
        super().__init__("Generating corrected segmentations")

    def run(self, annot_dir, seg_dir, out_dir):
        self.annot_dir = annot_dir
        self.seg_dir = seg_dir
        self.out_dir = out_dir
        self.thread = Thread(annot_dir, seg_dir, out_dir)
        annot_fpaths = all_image_paths_in_dir(self.annot_dir)
        annot_fpaths = [f for f in annot_fpaths if os.path.splitext(f)[1] == ".png"]
        self.progress_bar.setMaximum(len(annot_fpaths))
        self.thread.progress_change.connect(self.onCountChanged)
        self.thread.done.connect(self.done)
        self.thread.start()

    def done(self):
        QtWidgets.QMessageBox.about(
            self,
            "Corrected segmentations generated",
            f"Extracting corrected segmentations from {self.annot_dir} "
            f"and {self.seg_dir} to {self.out_dir} "
            "is complete.",
        )
        self.close()


class AssignCorrectionsWidget(QtWidgets.QWidget):
    submit = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.seg_dir = None
        self.annot_dir = None
        self.out_dir = None
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("Assign Corrections to Segmentation")

        # Add specify annotation directory button
        annot_dir_label = QtWidgets.QLabel()
        annot_dir_label.setText("Annotation (corrections) directory: Not yet specified")
        layout.addWidget(annot_dir_label)
        self.annot_dir_label = annot_dir_label

        specify_annot_btn = QtWidgets.QPushButton("Specify annotation directory")
        specify_annot_btn.clicked.connect(self.select_annot_dir)
        layout.addWidget(specify_annot_btn)

        # Add specify seg directory button
        seg_dir_label = QtWidgets.QLabel()
        seg_dir_label.setText("Segmentation (mask) directory: Not yet specified")
        layout.addWidget(seg_dir_label)
        self.seg_dir_label = seg_dir_label

        specify_seg_btn = QtWidgets.QPushButton("Specify segmentation directory")
        specify_seg_btn.clicked.connect(self.select_seg_dir)
        layout.addWidget(specify_seg_btn)

        # Add specify output directory button
        out_dir_label = QtWidgets.QLabel()
        out_dir_label.setText("Output directory: Not yet specified")
        layout.addWidget(out_dir_label)
        self.out_dir_label = out_dir_label

        specify_out_dir_btn = QtWidgets.QPushButton("Specify output directory")
        specify_out_dir_btn.clicked.connect(self.select_out_dir)
        layout.addWidget(specify_out_dir_btn)

        info_label = QtWidgets.QLabel()
        info_label.setText(
            "Annotation directory, segmentation directory and output directory must be specified."
        )
        layout.addWidget(info_label)
        self.info_label = info_label

        submit_btn = QtWidgets.QPushButton("Extract corrected segmentations")
        submit_btn.clicked.connect(self.extract)
        layout.addWidget(submit_btn)
        submit_btn.setEnabled(False)
        self.submit_btn = submit_btn

    def extract(self):
        self.progress_widget = ProgressWidget()
        self.progress_widget.run(self.annot_dir, self.seg_dir, self.out_dir)
        self.progress_widget.show()
        self.close()

    def validate(self):
        if not self.annot_dir:
            self.info_label.setText("Annotation directory must be specified.")
            self.submit_btn.setEnabled(False)
            return

        if not self.seg_dir:
            self.info_label.setText("Segmentation directory must be specified.")
            self.submit_btn.setEnabled(False)
            return

        if not self.out_dir:
            self.info_label.setText("Output directory must be specified.")
            self.submit_btn.setEnabled(False)
            return

        self.info_label.setText("")
        self.submit_btn.setEnabled(True)

    def select_annot_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def input_selected():
            self.annot_dir = self.input_dialog.selectedFiles()[0]
            self.annot_dir_label.setText("Annotation directory: " + self.annot_dir)
            self.validate()

        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()

    def select_seg_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def input_selected():
            self.seg_dir = self.input_dialog.selectedFiles()[0]
            self.seg_dir_label.setText("Segmentation directory: " + self.seg_dir)
            self.validate()

        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()

    def select_im_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def input_selected():
            self.im_dir = self.input_dialog.selectedFiles()[0]
            self.im_dir_label.setText("Image directory: " + self.im_dir)
            self.validate()

        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()

    def select_out_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def input_selected():
            self.out_dir = self.input_dialog.selectedFiles()[0]
            self.out_dir_label.setText("Masked image directory: " + self.out_dir)
            self.validate()

        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()
