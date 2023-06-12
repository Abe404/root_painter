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
import csv

from PyQt6 import QtWidgets
from PyQt6 import QtCore
from progress_widget import BaseProgressWidget


class Thread(QtCore.QThread):
    progress_change = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal()

    def __init__(self, segment_dir, csv_path, headers, extractor):
        super().__init__()
        self.segment_dir = segment_dir
        self.csv_path = csv_path
        self.headers = headers
        self.extractor = extractor

    def run(self):
        seg_fnames = os.listdir(str(self.segment_dir))
        seg_fnames = [f for f in seg_fnames if os.path.splitext(f)[1] == '.png']
        # if the file already exists then delete it.
        if os.path.isfile(self.csv_path):
            os.remove(self.csv_path)
        with open(self.csv_path, 'w+')  as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # Write the column headers
            writer.writerow(self.headers)
            for i, fname in enumerate(seg_fnames):
                self.progress_change.emit(i+1, len(seg_fnames))
                # headers allow the output options to be detected.
                self.extractor(self.segment_dir, fname, writer, self.headers)
            self.done.emit()


class ExtractProgressWidget(BaseProgressWidget):

    def __init__(self, feature):
        super().__init__(f'Extracting {feature}')

    def run(self, input_dir, csv_path, headers, extractor):
        self.input_dir = input_dir
        self.csv_path = csv_path
        self.progress_bar.setMaximum(len(os.listdir(input_dir)))
        self.thread = Thread(input_dir, csv_path, headers, extractor)
        self.thread.progress_change.connect(self.onCountChanged)
        self.thread.done.connect(self.done)
        self.thread.start()

class BaseExtractWidget(QtWidgets.QWidget):
    """ Extract measurements from csv """

    def __init__(self, feature, headers, extractor):
        super().__init__()
        self.input_dir = None
        self.output_csv = None
        self.feature = feature
        self.headers = headers
        self.extractor = extractor
        self.initUI()

    def initUI(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.setWindowTitle(f"Extract {self.feature}")
        self.add_seg_dir_btn()
        self.add_output_csv_btn()
        self.add_info_label()

    def add_seg_dir_btn(self):
        # Add specify image directory button
        in_dir_label = QtWidgets.QLabel()
        in_dir_label.setText("Segmentation directory: Not yet specified")
        self.layout.addWidget(in_dir_label)
        self.in_dir_label = in_dir_label

        specify_input_dir_btn = QtWidgets.QPushButton('Specify segmentation directory')
        specify_input_dir_btn.clicked.connect(self.select_input_dir)
        self.layout.addWidget(specify_input_dir_btn)

    def add_output_csv_btn(self):
        out_csv_label = QtWidgets.QLabel()
        out_csv_label.setText("Output CSV: Not yet specified")
        self.layout.addWidget(out_csv_label)
        self.out_csv_label = out_csv_label

        specify_output_csv_btn = QtWidgets.QPushButton('Specify output CSV')
        specify_output_csv_btn.clicked.connect(self.select_output_csv)
        self.layout.addWidget(specify_output_csv_btn)

    def add_info_label(self):
        info_label = QtWidgets.QLabel()
        info_label.setText("Segmentation directory and output CSV"
                           " must be specified.")
        self.layout.addWidget(info_label)
        self.info_label = info_label

        submit_btn = QtWidgets.QPushButton('Extract')
        submit_btn.clicked.connect(self.extract)
        self.layout.addWidget(submit_btn)
        submit_btn.setEnabled(False)
        self.submit_btn = submit_btn

    def extract(self):
        self.progress_widget = ExtractProgressWidget(self.feature)
        self.progress_widget.run(self.input_dir, self.output_csv,
                                 self.headers, self.extractor)
        self.progress_widget.show()
        self.close()

    def validate(self):
        if not self.input_dir:
            self.info_label.setText("Segmentation directory must be specified "
                                    "to extract {self.feature.lower()}")
            self.submit_btn.setEnabled(False)
            return

        if not self.output_csv:
            self.info_label.setText("Output CSV must be specified to extract "
                                    "region propertie.")
            self.submit_btn.setEnabled(False)
            return

        self.info_label.setText("")
        self.submit_btn.setEnabled(True)

    def select_input_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def input_selected():
            self.input_dir = self.input_dialog.selectedFiles()[0]
            self.in_dir_label.setText('Segmentation directory: ' + self.input_dir)
            self.validate()
        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()


    def select_output_csv(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Output CSV')
        if file_name:
            file_name = os.path.splitext(file_name)[0] + '.csv'
            self.output_csv = file_name
            self.out_csv_label.setText('Output CSV: ' + self.output_csv)
            self.validate()
