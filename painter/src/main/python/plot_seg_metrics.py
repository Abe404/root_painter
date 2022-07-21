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

def moving_average(original, w):
    averages = []
    for i, x in enumerate(original):
        if i >= (w//2) and i < (len(original) - (w//2)):
            elements = original[i-(w//2):i+(w//2)]
            elements = [e for e in elements if not math.isnan(e)]
            if elements:
                avg = np.nanmean(elements)
                averages.append(avg)
            else:
                averages.append(float('NaN'))
        else:
            averages.append(float('NaN'))
    return averages


def plot_seg_accuracy_over_time(ax, metrics_list, rolling_n):
    annots_found = 0
    corrected_dice = [] # dice scores between predicted and corrected for corrected images.
    for i, m in enumerate(metrics_list):
        if (m['annot_fg'] + m['annot_bg']) > 0:
            annots_found += 1
        # once 6 annotations found then start recording disagreement
        if annots_found > 6:
            corrected_dice.append(m['f1'])

    ax.scatter(list(range(len(corrected_dice))), corrected_dice,
                c='b', s=4, marker='x', label='image')

    avg_corrected = moving_average(corrected_dice, rolling_n)
    ax.plot(avg_corrected, c='r', label=f'average (n={rolling_n})')
    ax.legend()
    ax.grid()


def plot_dice_metric(metrics_list, output_plot_path, rolling_n):
    fig, ax = plt.subplots() # fig : figure object, ax : Axes object
    ax.set_xlabel('image')
    ax.set_ylabel('dice')
    ax.set_yticks(list(np.arange(0.0, 1.1, 0.05)))
    ax.set_ylim([0.0, 1])
    plot_seg_accuracy_over_time(ax, metrics_list, rolling_n)
    plt.tight_layout()
    plt.savefig(output_plot_path)




def plot_dice_metric_qtgraph(metrics_list, output_plot_path, rolling_n):
    annots_found = 0
    corrected_dice = [] # dice scores between predicted and corrected for corrected images.
    for i, m in enumerate(metrics_list):
        if (m['annot_fg'] + m['annot_bg']) > 0:
            annots_found += 1
        # once 6 annotations found then start recording disagreement
        if annots_found > 6:
            corrected_dice.append(m['f1'])

    avg_corrected = moving_average(corrected_dice, rolling_n)
    x = list(range(len(corrected_dice)))
    y = corrected_dice
    ## Switch to using white background and black foreground
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    # setting pen=None disables line drawing
    window = pg.plot()  
    window.showGrid(x = True, y = True, alpha = 0.4)
    window.addLegend(offset=(-90, -90))
    window.plot(x, y, pen=None, symbol='x', name='image')  
    window.plot(avg_corrected, pen = pg.mkPen('r', width=3), symbol=None, name=f'average (n={rolling_n})')

    window.setLabel('left', "dice")
    window.setLabel('bottom', "image")
    window.show()
