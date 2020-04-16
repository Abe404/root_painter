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
import numpy as np
from skimage import measure
from PIL import Image
from base_extract import BaseExtractWidget


def save_count_to_csv(seg_dir, fname, writer):
    fpath = os.path.join(seg_dir, fname)
    seg_im = Image.open(fpath)
    seg_im = np.array(seg_im)[:, :, 2].astype(bool)
    count = measure.label(seg_im).max()
    writer.writerow([os.path.basename(fpath), count])

class ExtractCountWidget(BaseExtractWidget):
    def __init__(self):
        super().__init__(
            "Count",
            ['file_name', 'count'],
            save_count_to_csv,
        )
