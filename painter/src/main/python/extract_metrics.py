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
    accuracy = (tp + tn) / total
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
    seg = imread(os.path.join(seg_dir, fname))
    annot = imread(os.path.join(annot_dir, fname))
    seg = np.array(seg[:, :, 3] > 0) # alpha channel to detect where seg exists.
    corrected = np.array(seg)
    # mask defines which pixels are defined in the annotation.
    foreground = annot[:, :, 0].astype(bool).astype(int)
    background = annot[:, :, 1].astype(bool).astype(int)
    corrected[foreground > 0] = 1
    corrected[background > 0] = 0
    
    corrected_segmentation_metrics = compute_metrics_from_masks(seg, corrected)

    mask = foreground + background
    mask = mask.astype(bool).astype(int)
    seg = seg * mask
    seg = seg.astype(bool).astype(int)
    y_defined = mask.reshape(-1)
    y_pred = seg.reshape(-1)[y_defined > 0]
    y_true = foreground.reshape(-1)[y_defined > 0]
    correction_segmentation_metrics = compute_metrics_from_masks(y_pred, y_true)
    return corrected_segmentation_metrics, correction_segmentation_metrics


class Thread(QtCore.QThread):
    progress_change = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal()

    def __init__(self, seg_dir, annot_dir, csv_path):
        super().__init__()
        self.seg_dir = seg_dir
        self.annot_dir = annot_dir
        self.csv_path = csv_path

    def run(self):
        annot_fnames = os.listdir(str(self.annot_dir))

        all_corrected_metrics = []
        all_correction_metrics = []

        for i, fname in enumerate(annot_fnames):
            self.progress_change.emit(i+1, len(annot_fnames))
            metrics = compute_seg_metrics(self.seg_dir, self.annot_dir, fname)
            corrected_metrics, correction_metrics = metrics
            all_corrected_metrics.append(corrected_metrics)
            all_correction_metrics.append(correction_metrics)
        # if any metrics were computed. 
        if len(all_corrected_metrics): 
            with open(self.csv_path, 'w+')  as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                # Write the column headers
                metric_keys = list(all_corrected_metrics[0].keys())
                headers = ['label_type'] + metric_keys + ['num_images']
                writer.writerow(headers)
                # label_type corrected means full image is gt (with corrections assigned).
                row = ['corrected']
                num_images = len(all_corrected_metrics)
                for k in metric_keys:
                    avg = np.mean([m[k] for m in all_corrected_metrics if not math.isnan(m[k])])
                    row.append(avg)
                row.append(num_images)
                writer.writerow(row)
                row = ['correction']
                for k in metric_keys:
                    avg = np.mean([m[k] for m in all_correction_metrics if not math.isnan(m[k])])
                    row.append(avg)
                row.append(num_images)
                writer.writerow(row)
            self.done.emit()


class MetricsProgressWidget(BaseProgressWidget):

    def __init__(self):
        super().__init__('Computing metrics')

    def run(self, seg_dir, annot_dir, csv_path):
        self.seg_dir = seg_dir
        self.annot_dir = annot_dir
        self.csv_path = csv_path
        self.thread = Thread(seg_dir, annot_dir, csv_path)
        seg_fnames = os.listdir(str(self.seg_dir))
        seg_fnames = [f for f in seg_fnames if os.path.splitext(f)[1] == '.png']
        self.progress_bar.setMaximum(len(seg_fnames))
        self.thread.progress_change.connect(self.onCountChanged)
        self.thread.done.connect(self.done)
        self.thread.start()

    def done(self, errors=[]):
        QtWidgets.QMessageBox.about(self, 'Metrics Computed',
                                    f'Computing metrics from {self.seg_dir} '
                                    f'and {self.annot_dir} to {self.csv_path} '
                                    'is complete.')
        self.close()


class ExtractMetricsWidget(QtWidgets.QWidget):
    submit = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.seg_dir = None
        self.annot_dir = None
        self.comp_dir = None
        self.csv_path = None
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("Extract Metrics")

        # Add specify seg directory button
        seg_dir_label = QtWidgets.QLabel()
        seg_dir_label.setText("Segmentation directory: Not yet specified")
        layout.addWidget(seg_dir_label)
        self.seg_dir_label = seg_dir_label

        specify_seg_btn = QtWidgets.QPushButton('Specify segmentation directory')
        specify_seg_btn.clicked.connect(self.select_seg_dir)
        layout.addWidget(specify_seg_btn)

        # Add specify photo directory button
        self.annot_dir_label = QtWidgets.QLabel()
        self.annot_dir_label.setText("Annotation directory: Not yet specified")
        layout.addWidget(self.annot_dir_label)

        specify_annot_dir_btn = QtWidgets.QPushButton('Specify anotation directory')
        specify_annot_dir_btn.clicked.connect(self.select_annot_dir)
        layout.addWidget(specify_annot_dir_btn)

        # Add output csv directory button
        out_csv_label = QtWidgets.QLabel()
        out_csv_label.setText("Output CSV: Not yet specified")
        layout.addWidget(out_csv_label)
        self.out_csv_label = out_csv_label

        specify_output_csv_btn = QtWidgets.QPushButton('Specify output CSV')
        specify_output_csv_btn.clicked.connect(self.select_output_csv)
        layout.addWidget(specify_output_csv_btn)

        info_label = QtWidgets.QLabel()
        info_label.setText("Segmentation directory, annotation "
                           "directory and output csv path"
                           "must be specified.")
        layout.addWidget(info_label)
        self.info_label = info_label

        submit_btn = QtWidgets.QPushButton('Extract Metrics')
        submit_btn.clicked.connect(self.extract_metrics)
        layout.addWidget(submit_btn)
        submit_btn.setEnabled(False)
        self.submit_btn = submit_btn

    def extract_metrics(self):
        self.progress_widget = MetricsProgressWidget()
        self.progress_widget.run(self.seg_dir, self.annot_dir, self.csv_path)
        self.progress_widget.show()
        self.close()

    def validate(self):
        if not self.seg_dir:
            self.info_label.setText("Segmentation directory must be "
                                    "specified to compute metrics.")
            self.submit_btn.setEnabled(False)
            return

        if not self.annot_dir:
            self.info_label.setText("Annotation directory must be "
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

    def select_seg_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        def input_selected():
            self.seg_dir = self.input_dialog.selectedFiles()[0]
            self.seg_dir_label.setText('Segmentation directory: ' + self.seg_dir)
            self.validate()
        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()

    def select_annot_dir(self):
        self.input_dialog = QtWidgets.QFileDialog(self)
        self.input_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def input_selected():
            self.annot_dir = self.input_dialog.selectedFiles()[0]
            self.annot_dir_label.setText('Annotation directory: ' + self.annot_dir)
            self.validate()
        self.input_dialog.fileSelected.connect(input_selected)
        self.input_dialog.open()

    def select_output_csv(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Output CSV')
        if file_name:
            file_name = os.path.splitext(file_name)[0] + '.csv'
            self.csv_path = file_name
            self.out_csv_label.setText('Output CSV: ' + self.csv_path)
            self.validate()
