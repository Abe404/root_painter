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
along with this program.  If not, see <https://www.gnu.o
"""

import os
import numpy as np
from cairosvg import svg2png
from PIL import Image
import skimage.util as skim_util
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import img_as_ubyte
svg_text = open('icon.svg', 'r').read()
base_sizes = [16, 24, 32, 48, 64]
linux_sizes = [128, 256, 512, 1024]
mac_sizes = [128, 256, 512, 1024]

def pad(image, width: int, constant_values=0):
    # only pad the first two dimensions
    pad_width = [(width, width), (width, width)]
    if len(image.shape) >= 3:
        # don't pad channels
        pad_width.append((0, 0))
    return skim_util.pad(image, pad_width, mode='constant',
                     constant_values=constant_values)

# base and linux png icons don't need extra border.
for w in base_sizes:
    fpath = os.path.join('base', str(w) + '.png')
    svg2png(bytestring=svg_text, write_to=fpath, output_width=w, output_height=w)

for w in linux_sizes:
    fpath = os.path.join('linux', str(w) + '.png')
    svg2png(bytestring=svg_text, write_to=fpath, output_width=w, output_height=w)

# Right now mac icon is done manually as cairosvg fails to add dropshadow properly.
if False:
    for w in mac_sizes:
        fpath = os.path.join('mac', str(w) + '.png')
        # make it a bit smaller
        svg2png(bytestring=svg_text, write_to=fpath,
                output_width=w, output_height=w)
        # Max icons contain ~5% transparent margin, so they don't look too big
        # in the Dock or app switcher
        im = imread(fpath)
        pad_w = round((im.shape[1]/100) * 5) # pad using 5% margin
        padded_im = pad(im, pad_w)
        im = resize(padded_im, (w, w), mode='reflect', anti_aliasing=True)
        # im[:, :, 3][im[:, :, 3] < 1.0] = 0
        # print('unique alpha = ', print(np.unique(im[:, :, 3])))
        imsave(fpath, im)

# Take the biggest png and generate some variations in an ico file
filename = r'linux/1024.png'
img = Image.open(filename)
icon_sizes = [(16,16), (24, 24), (32, 32), (48, 48), (64,64),
              (128, 128), (256, 256), (512, 512), (1024, 1024)]
img.save('Icon.ico', sizes=icon_sizes)
