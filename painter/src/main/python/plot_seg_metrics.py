#pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914, C0303, C0103
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

import os
import csv
import math
import json
import time
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import mkQApp
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from skimage.io import imread
from progress_widget import BaseProgressWidget

def moving_average(original, w):
    averages = []
    for i, x in enumerate(original):
        if i >= (w//2) and i <= (len(original) - (w-(w//2))):
            elements = original[i-(w//2):i+(w-(w//2))]
            elements = [e for e in elements if not math.isnan(e)]
            if elements:
                avg = np.nanmean(elements)
                averages.append(avg)
            else:
                averages.append(float('NaN'))
        else:
            averages.append(float('NaN'))
     
    x_pos = list(range(len(averages)))

    if w % 2 == 0:
        # shift left by 0.5 as each point is the average of itself and the previous point.
        x_pos = [a - 0.5 for a in x_pos]

    return x_pos, averages


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
        self.done.emit(json.dumps([all_fnames, all_metrics]))


class MetricsProgressWidget(BaseProgressWidget):

    done = QtCore.pyqtSignal(str)

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
        self.thread.done.connect(self.metrics_computed)
        self.thread.start()

    def metrics_computed(self, all_metrics_str):
        QtWidgets.QMessageBox.about(self, 'Metrics Computed',
                                    f'Metrics computed for {os.path.dirname(self.proj_file_path)}. '

                                    f'The CSV file has been saved to {self.csv_path}')
        self.done.emit(all_metrics_str)
        self.close()


class ExtractMetricsWidget(QtWidgets.QWidget):
    submit = QtCore.pyqtSignal()
    done = QtCore.pyqtSignal(str)

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
        def metrics_computed(metric_str):
            self.done.emit(metric_str) 
        self.progress_widget.done.connect(metrics_computed) 
        self.progress_widget.run(self.proj_file_path, csv_path, plot_path, rolling_average_n)
        self.progress_widget.show()
        self.close()


class MetricsPlot:

    def __init__(self):
        self.graph_plot = None
        self.proj_file_path = None
        self.proj_dir = None

    def add_file_metrics(self, fname):
        if self.graph_plot is not None:
            seg_dir = os.path.join(self.proj_dir, 'segmentations')
            annot_dir = os.path.join(self.proj_dir, 'annotations')
            f_metrics = compute_seg_metrics(seg_dir, annot_dir, fname)
            if f_metrics: 
                self.graph_plot.add_point(fname, f_metrics)

    def show_extract_metrics(self, proj_file_path): 
        self.proj_file_path = proj_file_path
        self.proj_dir = os.path.dirname(self.proj_file_path)
        self.extract_metrics_widget = ExtractMetricsWidget(proj_file_path)

        def extract_done(metric_str):
            fnames, metrics_list = json.loads(metric_str)
            self.graph_plot = QtGraphMetricsPlot(fnames, metrics_list, rolling_n=30)
            self.graph_plot.show()

        self.extract_metrics_widget.done.connect(extract_done)
        self.extract_metrics_widget.show()


class QtGraphMetricsPlot(QtWidgets.QMainWindow):

    def __init__(self, fnames, metrics_list, rolling_n):
        super().__init__()

        self.setWindowTitle('RootPainter: Metrics Plot')

        self.central_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.fnames = fnames
        self.metrics_list = metrics_list
        self.rolling_n = rolling_n
        self.graph_plot = None
        self.create_plot()
        self.render_data()
        self.add_average_control()


    def avg_changed(self, sb):
        self.rolling_n = sb.value()
        self.render_data()

    def add_average_control(self):
        spin_widget = QtWidgets.QWidget()
        spin_widget_layout = QtWidgets.QHBoxLayout()
        spin_widget.setLayout(spin_widget_layout)
        spin_widget_layout.addWidget(QtWidgets.QLabel("Rolling average:"))
        self.avg_spin = pg.SpinBox(
            value=self.rolling_n, bounds=[1, 1000],
            int=True, minStep=1, step=1, wrapping=False)

        spin_widget_layout.addWidget(self.avg_spin)
        self.avg_spin.sigValueChanged.connect(self.avg_changed)
        self.layout.addWidget(spin_widget)
        spin_widget.setMaximumWidth(200)


    def add_point(self, fname, metrics):
        if fname in self.fnames:
            idx = self.fnames.index(fname)
            self.metrics_list[idx] = metrics
        else:
            self.fnames.append(fname)
            self.metrics_list.append(metrics) 
        self.render_data()

    def render_data(self):
        assert self.graph_plot is not None, 'plot should be created before rendering data'
        corrected_dice = self.get_corrected_dice()
        x = list(range(len(corrected_dice)))
        y = corrected_dice
        self.graph_plot.plot(x, y, pen=None, symbol='x', clear=True)

        x, y = moving_average(corrected_dice, self.rolling_n)
        self.graph_plot.plot(x, y, pen = pg.mkPen('r', width=3),
                             symbol=None, name=f'average (n={self.rolling_n})')


    def get_corrected_dice(self):
        # should not consider first annotated images as these are likely
        # not done correctively.
        annots_found = 0
        corrected_dice = [] # dice scores between predicted and corrected for corrected images.
        for i, m in enumerate(self.metrics_list):
            if (m['annot_fg'] + m['annot_bg']) > 0:
                annots_found += 1
            # once 6 annotations found then start recording disagreement
            if annots_found > 6:
                corrected_dice.append(m['f1'])
        return corrected_dice


    def create_plot(self):
        mkQApp() # create or get the QtApp
        ## Switch to using white background and black foreground
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'w')
        view = pg.GraphicsView()
        l = pg.GraphicsLayout(border=None)
        l.setContentsMargins(10, 10, 10, 10)
        view.setCentralItem(l)
        view.setWindowTitle('RootPainter: Segmentation Metrics')
        view.resize(800,600)
        l2 = l.addLayout()
        l2.setContentsMargins(0, 0, 0, 0)
        pg.setConfigOption('foreground', 'k')
        p21 = l2.addPlot()
        p21.showGrid(x = True, y = True, alpha = 0.4)
        p21.addLegend(offset=(-90, -90))
        p21.setLabel('left', 'Dice')
        p21.setLabel('bottom', 'Image')
        self.graph_plot = p21
        hide_weird_options(self.graph_plot)
        self.view = view # avoid errors with view being deleted
        self.layout.addWidget(self.view)


def hide_weird_options(graph_plot):
    # there are some plot options that I will remove to make things simpler.
    # hide option to invert axis (its useless)
    graph_plot.vb.menu.ctrl[0].invertCheck.hide()
    graph_plot.vb.menu.ctrl[1].invertCheck.hide()
    # hide option to show visible data only. Who knows what this does.
    graph_plot.vb.menu.ctrl[0].visibleOnlyCheck.hide()
    graph_plot.vb.menu.ctrl[1].visibleOnlyCheck.hide()
    # hide option to autoPan
    graph_plot.vb.menu.ctrl[0].autoPanCheck.hide()
    graph_plot.vb.menu.ctrl[1].autoPanCheck.hide()
    # hide link combo
    graph_plot.vb.menu.ctrl[0].linkCombo.hide()
    graph_plot.vb.menu.ctrl[1].linkCombo.hide()
    # hide link combo
    graph_plot.vb.menu.ctrl[0].label.hide()
    graph_plot.vb.menu.ctrl[1].label.hide()
    # hide option to disable mouse. why would you want this?
    graph_plot.vb.menu.ctrl[0].mouseCheck.hide()
    graph_plot.vb.menu.ctrl[1].mouseCheck.hide()
    graph_plot.ctrlMenu = None


if __name__ == '__main__':
    # a quick test to demo the plot. Useful for testing a debugging.
    app = QtWidgets.QApplication([])
    corrected_dice = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    fnames = ['1', '2', '3', '4', '5', '6']
    corrected_dice += corrected_dice 
    fnames += fnames 
    fnames += ['7']
    corrected_dice += [0.7]
    metrics_list = [{'f1': a, 'annot_fg': 100, 'annot_bg': 100} for a in corrected_dice]
    rolling_n = 3
    plot = QtGraphMetricsPlot(fnames, metrics_list, rolling_n)
    plot.show()
    import random
    def mouseMoved(evt):
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        plot.add_point(str(len(plot.metrics_list)),
                       {'f1': random.random(),
                       'annot_fg': 100,
                       'annot_bg': 100})
    proxy = pg.SignalProxy(plot.view.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
    app.exec_()
