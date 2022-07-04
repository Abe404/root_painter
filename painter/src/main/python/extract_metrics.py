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
import time
import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from plot_seg_metrics import plot_dice_metric
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

    def __init__(self, proj_dir, csv_path, plot_path, fnames, rolling_average_size):
        super().__init__()
        self.proj_dir = proj_dir
        self.seg_dir = os.path.join(proj_dir, 'segmentations')
        self.annot_dir = os.path.join(proj_dir, 'annotations')
        self.csv_path = csv_path
        self.plot_path = plot_path
        self.rolling_average_size = rollling_average_size
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

                    row = [fname]
                    for k in metric_keys:
                        row.append(corrected_metrics[k]) 
                    writer.writerow(row)
        plot_dice_metric(self.csv_path, self.plot_path, self.rolling_average_size)
        self.done.emit()


class MetricsProgressWidget(BaseProgressWidget):

    def __init__(self):
        super().__init__('Computing metrics')

    def run(self, proj_file_path, csv_path, plot_path, rolling_average_size):
        self.proj_file_path = proj_file_path
        self.csv_path = csv_path
        fnames = json.load(open(self.proj_file_path))['file_names']
        proj_dir = os.path.dirname(self.proj_file_path)
        self.thread = Thread(proj_dir, csv_path, plot_path, fnames, rolling_average_size)
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

    def __init__(self, proj_file_path):
        super().__init__()
        self.proj_file_path = proj_file_path
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("Extract Metrics")

        info_label = QtWidgets.QLabel()
        info_label.setText("")
        layout.addWidget(info_label)
        self.info_label = info_label

        edit_rolling_average_label = QtWidgets.QLabel()
        edit_rolling_average_label = QtWidgets.QLabel()
        edit_rolling_average_label.setText("Plot rolling average size")
        layout.addWidget(edit_rolling_average_label)
        rolling_average_edit_widget = QtWidgets.QSpinBox()
        rolling_average_edit_widget.setMaximum(10000)
        rolling_average_edit_widget.setMinimum(1)
        rolling_average_edit_widget.setValue(30)
        rolling_average_edit_widget.valueChanged.connect(self.validate)
        self.rolling_average_edit_widget = rolling_average_edit_widget
        layout.addWidget(rolling_average_edit_widget)

        submit_btn = QtWidgets.QPushButton('Extract Metrics')
        submit_btn.clicked.connect(self.extract_metrics)
        layout.addWidget(submit_btn)
        self.submit_btn = submit_btn

    def extract_metrics(self):
        self.progress_widget = MetricsProgressWidget()
        metrics_dir = os.path.join(os.path.dirname(self.proj_file_path), 'metrics')
        rolling_average_n = self.rolling_average_edit_widget.value()
        if not os.path.isdir(metrics_dir):
            os.makedirs(metrics_dir)
        prefix = str(round(time.time())) + '_'
        csv_path = os.path.join(metrics_dir, f'{prefix}metrics.csv')
        plot_path = os.path.join(metrics_dir, f'{prefix}dice.png')
        self.progress_widget.run(self.proj_file_path, csv_path, plot_path, rolling_average_n)
        self.progress_widget.show()
        self.close()

    def validate(self):
        self.info_label.setText("")
        self.submit_btn.setEnabled(True)
