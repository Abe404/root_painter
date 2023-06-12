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


from PyQt6 import QtWidgets
from PyQt6 import QtCore
import numpy as np
from skimage import measure
from PIL import Image
from eccentricity import eccentricity2
from base_extract import BaseExtractWidget

region_props_headers = ['file_name', 'x', 'y', 'diameter',
                        'area', 'perimeter', 'eccentricity']


def get_region_props(seg_dir, fname, writer, headers):
    """ input headers to allow 'eccentricity' output to be detected. """
    seg_im = Image.open(os.path.join(seg_dir, fname))
    seg_im = np.array(seg_im)
    seg_im = seg_im[:, :, 2].astype(bool).astype(int)
    seg_im = measure.label(seg_im > 0, connectivity=seg_im.ndim)
    name = fname.replace('.png', '')
    regions = measure.regionprops(seg_im)
    for region in regions:
        row, column = region.centroid
        x = column
        y = row
        diameter = region.equivalent_diameter
        area = region.area
        perimeter = region.perimeter
        # eccentricity can be disabled as it can cause memory errors (seg fault)
        # for very large images (larger than 5000x5000)
        if 'eccentricity' in headers:
            eccentricity = eccentricity2(region)
            writer.writerow([name, x, y, diameter, area, perimeter, eccentricity])
        else:
            writer.writerow([name, x, y, diameter, area, perimeter])

class ExtractRegionsWidget(BaseExtractWidget):
    def __init__(self):
        super().__init__(
            "Region Properites",
            region_props_headers,
            get_region_props)


        # Add option to disable eccentricity
        info_label = QtWidgets.QLabel()
        info_label.setFixedWidth(600)
        info_label.setText("Extracted region properties include x, y, diameter, area,"
                           " perimeter and eccentricity. For extremely large images (over"
                           " 5000x5000) eccentricity calculation may crash RootPainter.")

        info_label.setWordWrap(True)
        self.layout.addWidget(info_label)
        self.eccentricity_checkbox = QtWidgets.QCheckBox("Output eccentricity")
        self.layout.addWidget(self.eccentricity_checkbox)
        self.eccentricity_checkbox.setChecked(True)
        self.eccentricity_checkbox.stateChanged.connect(self.output_eccentricity_changed)

    def output_eccentricity_changed(self, state):
        checked = (state == QtCore.Qt.Checked)
        if checked:
            self.headers = region_props_headers
        else:
            self.headers = [h for h in region_props_headers if h != 'eccentricity']
