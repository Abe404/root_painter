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
import math
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import mkQApp
from PyQt5 import QtWidgets

# def plot_seg_accuracy_over_time(ax, metrics_list, rolling_n):
#     annots_found = 0
#     corrected_dice = [] # dice scores between predicted and corrected for corrected images.
#     for i, m in enumerate(metrics_list):
#         if (m['annot_fg'] + m['annot_bg']) > 0:
#             annots_found += 1
#         # once 6 annotations found then start recording disagreement
#         if annots_found > 6:
#             corrected_dice.append(m['f1'])
# 
#     ax.scatter(list(range(len(corrected_dice))), corrected_dice,
#                 c='b', s=4, marker='x', label='image')
# 
#     avg_corrected = moving_average(corrected_dice, rolling_n)
#     ax.plot(avg_corrected, c='r', label=f'average (n={rolling_n})')
#     ax.legend()
#     ax.grid()
#
# def plot_dice_metric(metrics_list, output_plot_path, rolling_n):
#     fig, ax = plt.subplots() # fig : figure object, ax : Axes object
#     ax.set_xlabel('image')
#     ax.set_ylabel('dice')
#     ax.set_yticks(list(np.arange(0.0, 1.1, 0.05)))
#     ax.set_ylim([0.0, 1])
#     plot_seg_accuracy_over_time(ax, metrics_list, rolling_n)
#     plt.tight_layout()
#     plt.savefig(output_plot_path)


# def test_plot():
#     from PyQt5 import QtWidgets
#     ## Always start by initializing Qt (only once per application)
#     app = QtWidgets.QApplication([])
# 
#     corrected_dice = [1,2,3,4]
#     x = list(range(len(corrected_dice)))
#     y = corrected_dice
#     args = [x, y]
#     mkQApp()
# 
#     ## Switch to using white background and black foreground
#     pg.setConfigOption('background', 'w')
#     pg.setConfigOption('foreground', 'w')
# 
#     view = pg.GraphicsView()
# 
#     l = pg.GraphicsLayout(border=None)
#     l.setContentsMargins(10, 10, 10, 10)
#     view.setCentralItem(l)
#     view.setWindowTitle('RootPainter: Segmentation Metrics')
#     view.resize(800,600)
#     #l.nextRow()
#     l2 = l.addLayout()
#     l2.setContentsMargins(0, 0, 0, 0)
#     #l2.nextRow()
#     pg.setConfigOption('foreground', 'k')
#     p21 = l2.addPlot()
# 
#     p21.showGrid(x = True, y = True, alpha = 0.4)
#     p21.addLegend(offset=(-90, -90))
#     p21.plot(x, y, pen=None, symbol='x', name='image')  
# 
#     p21.setLabel('left', 'Dice')
#     p21.setLabel('bottom', 'Image')
#     plots.append(p21)
# 
#     proxy = pg.SignalProxy(plots[0].scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
#     view.show()
#     ## Start the Qt event loop
#     app.exec_()
# 
# 
# from PyQt5.QtCore import QTimer
# timer = QTimer()
# 
# 
# def add_point(seg_dir, annot_dir, fname):
#     
#     # TODO: Call this function from the code that saves the annotation.
#     #       Maybe have it pass the file name so the file name or
#     #       path to the annotation and segmentation so the metrics can be
#     #       computed for the image and then added to the plot.
#     #       if the file name is already in the plot then we would need to
#     #       update the metric value (corrected_dice) for the existing
#     #       file metrics entry.
# 
# 
#     metrics = compute_seg_metrics(seg_dir, annot_dir, fname)
# 
#     corrected_dice.append(y)
#     #x = list(range(len(corrected_dice)))
#     #y = corrected_dice
#     #print('set data ', y)
#     #plots[0].setData(len(corrected_dice), len(corrected_dice))        
#     x = list(range(len(corrected_dice)))
#     y = corrected_dice
#     plots[0].plot(x, y, pen=None, symbol='x', clear=True)
#def mouseMoved(evt):
#    pos = evt[0]  ## using signal proxy turns original arguments into a tuple
#    #print('pos', pos, dir(pos))
#    add_point(pos.y())



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


