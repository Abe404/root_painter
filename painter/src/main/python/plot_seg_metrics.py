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

def get_dice(fname, lines):
    for l in lines:
        (f, accuracy, tn,
         fp, fn, tp, precision, recall, f1) = l.split(',')
        if f == fname:
            return float(f1)
    return 1

def plot_seg_accuracy_over_time(
    input_csv, rolling_n, show_legend,
    get_metric):

    lines = open(input_csv).readlines()
    headers = lines[0].split(',')
    fnames = []    
    corrected_vals = []
    for l in lines:
        # file_name,accuracy,tn,fp,fn,tp,precision,recall,f1
        (fname, accuracy, tn,
         fp, fn, tp, precision, recall, f1) = l.split(',')
        if fname not in fnames: # dont add twice, makes a mess.
            fnames.append(fname)
    annots_found = 0
    corrected_fnames = []
    for i, f in enumerate(fnames):
        # once 6 annotations found then start recording disagreement
        if annots_found > 6:
            corrected_fnames.append(f) 
            corrected_vals.append(get_metric(f, lines))

    fnames = corrected_fnames

    plt.scatter(list(range(len(fnames))), corrected_vals,
                c='b', s=4, marker='x', label='image')

    avg_corrected = moving_average(corrected_vals, rolling_n)
    plt.plot(avg_corrected, c='r', label=f'average (n={rolling_n})')

    if show_legend:
        plt.legend()
    plt.grid()


def plot_dice_metric(input_metrics_csv_path, output_plot_path, rolling_n):
    figsize=(10, 6)
    plt.figure(figsize=figsize)
    plt.yticks(list(np.arange(0.5, 1.1, 0.05)))
    plt.ylim([0.0, 1])
    plot_seg_accuracy_over_time(input_metrics_csv_path, rolling_n,
                                show_legend=True, get_metric=get_dice)
    plt.tight_layout()
    plt.savefig(output_plot_path)
