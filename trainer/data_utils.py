"""
Form a batch

Copyright (C) 2021 Abraham George Smith

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
import numpy as np

def collate_fn(batch):
    num_items = len(batch)
    im_tiles = []
    batch_fgs = []
    batch_bgs = []
    batch_classes = []

    for i in range(num_items):
        item = batch[i]

        class_data = {}

        im_tiles.append(item[0])

        batch_fgs.append(item[1])
        batch_bgs.append(item[2])
        batch_classes.append(item[3])

    im_tiles = np.array(im_tiles)
    return im_tiles, batch_fgs, batch_bgs, batch_classes
