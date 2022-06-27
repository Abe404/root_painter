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

#pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914, C0303, C0103
import os
import csv
import math
import json
import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from skimage.io import imread
from progress_widget import BaseProgressWidget


def compute_metrics_from_masks(y_pred, y_true):
    """
    Compute TP, FP, TN, FN, dice and for y_pred vs y_true
    """
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    total = (tp + tn + fp + fn)
    print('tp', tp, 'tn', tn, 'total', total)
    accuracy = (tp + tn) / total
    print('accuracy', accuracy)
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
    else:
        precision = recall = f1 = float('NaN')
    return {
        "accuracy": accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compute_seg_metrics(seg_dir, annot_dir, fname):
    # annot and seg are both PNG
    fname = os.path.splitext(fname)[0] + '.png'
    seg_path = os.path.join(seg_dir, fname)
    if not os.path.isfile(seg_path):
        return None # no segmentation means no metrics.

    seg = imread(seg_path)
    annot_path = os.path.join(annot_dir, 'train', fname)
    if not os.path.isfile(annot_path):
        annot_path = os.path.join(annot_dir, 'val', fname)

    if os.path.isfile(annot_path):
        annot = imread(annot_path)
    else:
        annot = np.zeros(seg.shape) # no file implies empty annotation (no corrections)

    seg = np.array(seg[:, :, 3] > 0) # alpha channel to detect where seg exists.
    corrected = np.array(seg)
    # mask defines which pixels are defined in the annotation.
    foreground = annot[:, :, 0].astype(bool).astype(int)
    background = annot[:, :, 1].astype(bool).astype(int)
    corrected[foreground > 0] = 1
    corrected[background > 0] = 0
    corrected_segmentation_metrics = compute_metrics_from_masks(seg, corrected)

    return corrected_segmentation_metrics


class Thread(QtCore.QThread):
    progress_change = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal()

    def __init__(self, proj_dir, csv_path, fnames):
        super().__init__()
        self.proj_dir = proj_dir
        self.seg_dir = os.path.join(proj_dir, 'segmentations')
        self.annot_dir = os.path.join(proj_dir, 'annotations')
        self.csv_path = csv_path
        self.fnames = fnames

    def run(self):
        headers_written = False
        with open(self.csv_path, 'w+')  as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i, fname in enumerate(self.fnames):
                self.progress_change.emit(i+1, len(self.fnames))
                metrics = compute_seg_metrics(self.seg_dir, self.annot_dir, fname)
                if metrics: 
                    corrected_metrics = metrics
                    # Write the column headers
                    if not headers_written:
                        metric_keys = list(corrected_metrics.keys())
                        headers = ['file_name'] + metric_keys
                        writer.writerow(headers)
                        headers_written = True

                    # label_type corrected means full image is gt (with corrections assigned).
                    row = [fname]
                    for k in metric_keys:
                        row.append(corrected_metrics[k]) 
                    writer.writerow(row)
            self.done.emit()


class MetricsProgressWidget(BaseProgressWidget):

    def __init__(self):
        super().__init__('Computing metrics')

    def run(self, proj_file_path, csv_path):
        self.proj_file_path = proj_file_path
        self.csv_path = csv_path
        fnames = json.load(open(self.proj_file_path))['file_names']
        proj_dir = os.path.dirname(self.proj_file_path)
        self.thread = Thread(proj_dir, csv_path, fnames)
        self.progress_bar.setMaximum(len(fnames))
        self.thread.progress_change.connect(self.onCountChanged)
        self.thread.done.connect(self.done)
        self.thread.start()

    def done(self, errors=[]):
        QtWidgets.QMessageBox.about(self, 'Metrics Computed',
                                    f'Computing metrics for {self.proj_file_path} '
                                    f'to {self.csv_path} '
                                    'is complete.')
        self.close()


class ExtractMetricsWidget(QtWidgets.QWidget):
    submit = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.proj_file_path = None
        self.csv_path = None
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("Extract Metrics")

        proj_file_path_label = QtWidgets.QLabel()
        proj_file_path_label.setText("RootPainter project file (.seg_proj): Not yet specified")
        layout.addWidget(proj_file_path_label)
        self.proj_file_path_label = proj_file_path_label

        specify_proj_btn = QtWidgets.QPushButton('Specify project (.seg_proj) file')
        specify_proj_btn.clicked.connect(self.select_proj_file)
        layout.addWidget(specify_proj_btn)


        # Add output csv directory button
        out_csv_label = QtWidgets.QLabel()
        out_csv_label.setText("Output CSV: Not yet specified")
        layout.addWidget(out_csv_label)
        self.out_csv_label = out_csv_label

        specify_output_csv_btn = QtWidgets.QPushButton('Specify output CSV')
        specify_output_csv_btn.clicked.connect(self.select_output_csv)
        layout.addWidget(specify_output_csv_btn)

        info_label = QtWidgets.QLabel()
        info_label.setText("Project file and output csv path"
                           " must be specified.")
        layout.addWidget(info_label)
        self.info_label = info_label

        submit_btn = QtWidgets.QPushButton('Extract Metrics')
        submit_btn.clicked.connect(self.extract_metrics)
        layout.addWidget(submit_btn)
        submit_btn.setEnabled(False)
        self.submit_btn = submit_btn

    def extract_metrics(self):
        self.progress_widget = MetricsProgressWidget()
        self.progress_widget.run(self.proj_file_path, self.csv_path)
        self.progress_widget.show()
        self.close()

    def validate(self):
        if not self.proj_file_path:
            self.info_label.setText("RootPainter project file (.seg_proj) must be "
                                    "specified to compute metrics.")
            self.submit_btn.setEnabled(False)
            return

        if not self.csv_path:
            self.info_label.setText("Ouput csv path must be "
                                    "specified to compute metrics.")
            self.submit_btn.setEnabled(False)
            return

        self.info_label.setText("")
        self.submit_btn.setEnabled(True)


    def select_proj_file(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'RootPainter project file (.seg_proj)',
            None, "RootPainter project file (*.seg_proj)")
        if file_name:
            self.proj_file_path = file_name
            self.proj_file_path_label.setText('RootPainter project file: ' + self.proj_file_path)
            self.validate()

    def select_output_csv(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Output CSV')
        if file_name:
            file_name = os.path.splitext(file_name)[0] + '.csv'
            self.csv_path = file_name
            self.out_csv_label.setText('Output CSV: ' + self.csv_path)
            self.validate()
