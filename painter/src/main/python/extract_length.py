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
from skimage.morphology import skeletonize
from PIL import Image
from base_extract import BaseExtractWidget

def save_length_to_csv(seg_dir, fname, writer, headers):
    seg_im = Image.open(os.path.join(seg_dir, fname))
    seg_im = np.array(seg_im)
    seg_im = seg_im[:, :, 2].astype(bool).astype(int)
    skel = skeletonize(seg_im)
    skel = skel.astype(int)
    skel_pixels = np.sum(skel)
    name = fname.replace('.png', '')
    writer.writerow([name, skel_pixels])

class ExtractLengthWidget(BaseExtractWidget):
    def __init__(self):
        super().__init__(
            "Length",
            ['file_name', 'length_pixels'],
            save_length_to_csv,
        )
