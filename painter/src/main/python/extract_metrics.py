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
import json
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from plot_seg_metrics import plot_dice_metric, plot_dice_metric_qtgraph
from skimage.io import imread
from progress_widget import BaseProgressWidget

def compute_metrics_from_masks(y_pred, y_true, fg_labels, bg_labels):
    """
    Compute TP, FP, TN, FN, dice and for y_pred vs y_true
    """
    tp = int(np.sum(np.logical_and(y_pred == 1, y_true == 1)))
    tn = int(np.sum(np.logical_and(y_pred == 0, y_true == 0)))
    fp = int(np.sum(np.logical_and(y_pred == 1, y_true == 0)))
    fn = int(np.sum(np.logical_and(y_pred == 0, y_true == 1)))
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
        "f1": f1,
        # how many actually manually annotated.
        "annot_fg": int(fg_labels),
        "annot_bg": int(bg_labels)
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
    corrected_segmentation_metrics = compute_metrics_from_masks(
        seg, corrected, np.sum(foreground > 0), np.sum(background > 0))
    return corrected_segmentation_metrics


class Thread(QtCore.QThread):
    progress_change = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal(str)

    def __init__(self, proj_dir, csv_path, plot_path, fnames):
        super().__init__()
        self.proj_dir = proj_dir
        self.seg_dir = os.path.join(proj_dir, 'segmentations')
        self.annot_dir = os.path.join(proj_dir, 'annotations')
        self.csv_path = csv_path
        self.plot_path = plot_path
        self.fnames = fnames

    def run(self):
        headers_written = False
        all_metrics = []
        all_fnames = []
        with open(self.csv_path, 'w+', newline='')  as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i, fname in enumerate(self.fnames):
                self.progress_change.emit(i+1, len(self.fnames))
                metrics = compute_seg_metrics(self.seg_dir, self.annot_dir, fname)
                if metrics: 
                    all_metrics.append(metrics)
                    all_fnames.append(fname)
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
        self.done.emit(json.dumps(all_metrics))


class MetricsProgressWidget(BaseProgressWidget):

    def __init__(self):
        super().__init__('Computing metrics')

    def run(self, proj_file_path, csv_path, plot_path, rolling_average_size):
        self.proj_file_path = proj_file_path
        self.csv_path = csv_path
        self.plot_path = plot_path
        fnames = json.load(open(self.proj_file_path))['file_names']
        proj_dir = os.path.dirname(self.proj_file_path)
        self.rolling_average_size = rolling_average_size
        self.thread = Thread(proj_dir, csv_path, plot_path, fnames)
        self.progress_bar.setMaximum(len(fnames))
        self.thread.progress_change.connect(self.onCountChanged)
        self.thread.done.connect(self.done)
        self.thread.start()

    def done(self, all_metrics_str):
        all_metrics = json.loads(all_metrics_str)
        plot_dice_metric_qtgraph(all_metrics, self.plot_path, self.rolling_average_size)
        plot_dice_metric(all_metrics, self.plot_path, self.rolling_average_size)
        QtWidgets.QMessageBox.about(self, 'Metrics Computed',
                                    f'Metrics computed for {os.path.dirname(self.proj_file_path)}. '
                                    f'The CSV file has been saved to {self.csv_path}')
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

        text_label = QtWidgets.QLabel()
        text_label.setText("")
        layout.addWidget(text_label)
        text_label.setText("""
Metrics for each of the project segmentations and annotations will be output to a CSV file and a     
plot will be generated. The metrics measure agreement between the initial segmentation predicted     
by the model and the corrected segmentation, which includes the initial segmentation with the     
corrections assigned.

The metrics saved to the CSV file include accuracy, tn (true negatives), fp (false positives),     
tp (true positives), precision, recall, f1 (also known as dice score), annot_fg (number of pixels      
annotated as foreground) and annot_bg (number of pixels annotated as background).

A dice score plot will also be generated. The dice plot excludes the 
initial images until there are at least 6 annotations, as it is assumed these initial images were 
annotated to include clear examples and were were not annotated correctively.

        """)

        edit_rolling_average_label = QtWidgets.QLabel()
        edit_rolling_average_label = QtWidgets.QLabel()
        edit_rolling_average_label.setText("Plot rolling average size:")
        layout.addWidget(edit_rolling_average_label)
        rolling_average_edit_widget = QtWidgets.QSpinBox()
        rolling_average_edit_widget.setMaximum(10000)
        rolling_average_edit_widget.setMinimum(1)
        rolling_average_edit_widget.setValue(30)
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
