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

# pylint: disable=C0111,E1102,C0103,W0703,W0511,E1136
import os
import time
import glob
import shutil
from math import ceil
import random
import numpy as np
import skimage.util as skim_util

from PIL import Image, ImageOps
from skimage import color
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from file_utils import ls


def is_photo(fname):
    """ extensions that have been tested with so far """
    extensions = {".jpg", ".png", ".jpeg", '.tif', '.tiff'}
    fname_ext = os.path.splitext(fname)[1].lower()
    return fname_ext in extensions

def normalize_tile(tile):
    if np.min(tile) < np.max(tile):
        tile = rescale_intensity(tile, out_range=(0, 1))
    assert np.min(tile) >= 0, f"tile min {np.min(tile)}"
    assert np.max(tile) <= 1, f"tile max {np.max(tile)}"
    return tile

def load_train_image_and_annot(dataset_dir, train_annot_dir):
    max_attempts = 60
    attempts = 0
    # used for logging which file caused the problem.
    latest_annot_path = None
    latest_im_path = None
    latest_error = None
    while attempts < max_attempts:
        attempts += 1
        # file systems are unpredictable.
        # We may have problems reading the file.
        # try-catch to avoid this.
        # (just try again)
        try:
            # set to None each time.
            latest_annot_path = None
            latest_im_path = None

            # This might take ages, profile and optimize
            fnames = ls(train_annot_dir)
            fnames = [a for a in fnames if is_photo(a)]
            fname = random.sample(fnames, 1)[0]
            annot_path = os.path.join(train_annot_dir, fname)
            image_path_part = os.path.join(dataset_dir,
                                           os.path.splitext(fname)[0])
            # it's possible the image has a different extenstion
            # so use glob to get it
            image_path = glob.glob(image_path_part + '.*')[0]
            latest_im_path = image_path
            image = load_image(image_path)
            latest_annot_path = annot_path
            annot = imread(annot_path).astype(bool)
            assert np.sum(annot) > 0
            assert image.shape[2] == 3 # should be RGB
            # also return fname for debugging purposes.
            return image, annot, fname
        except Exception as e:
            latest_error = e
            # This could be due to an empty annotation saved by the user.
            # Which happens rarely due to deleting all labels in an 
            # existing annotation and is not a problem.
            # give it some time and try again.
            time.sleep(0.1)

    if attempts == max_attempts:
        if latest_annot_path is None: # if annot path still None we know it failed on the photo
            raise Exception(f'Could not load photo {latest_im_path}, {latest_error}')
        else:
            # otherwise it must have failed on the annotation
            raise Exception(f'Could not load annotation {latest_annot_path}, {e}')


def pad(image, width: int, mode='reflect', constant_values=0):
    # only pad the first two dimensions
    pad_width = [(width, width), (width, width)]
    if len(image.shape) == 3:
        # don't pad channels
        pad_width.append((0, 0))
    if mode == 'reflect':
        return np.pad(image, pad_width, mode)
    return np.pad(image, pad_width, mode=mode,
                         constant_values=constant_values)


def crop_from_pad_settings(image, pad_settings):
    """ Crop image back to what it was before padding using
        the pad_settings. See pad_to_min for how 
        pad_settings are generated and used """
    h_pad_before, h_pad_after = pad_settings[0]
    w_pad_before, w_pad_after = pad_settings[1]
    h_start = h_pad_before
    h_stop = image.shape[0] - h_pad_after
    w_start = w_pad_before
    w_stop = image.shape[1] - w_pad_after
    return image[h_start:h_stop, w_start:w_stop]


def pad_to_min(im, min_w, min_h):
    h, w, _ = im.shape
    h_pad = 0
    w_pad = 0
    if h < min_h:
        h_pad = min_h - h
    if w < min_w:
        w_pad = min_w - w
    h_pad_before = h_pad // 2
    h_pad_after = h_pad - h_pad_before
    w_pad_before = w_pad // 2
    w_pad_after = w_pad - w_pad_before
    pad_settings = ((h_pad_before, h_pad_after), (w_pad_before, w_pad_after), (0, 0))
    if h_pad or w_pad:
        im = np.pad(im, pad_settings, mode='reflect')
    return im, pad_settings

def add_salt_pepper(image, intensity):
    image = np.array(image)
    white = [1, 1, 1]
    black = [0, 0, 0]
    if len(image.shape) == 2 or image.shape[-1] == 1:
        white = 1
        black = 0
    num = np.ceil(intensity * image.size).astype(int)
    x_coords = np.floor(np.random.rand(num) * image.shape[1])
    x_coords = x_coords.astype(int)
    y_coords = np.floor(np.random.rand(num) * image.shape[0]).astype(int)
    image[x_coords, y_coords] = white
    x_coords = np.floor(np.random.rand(num) * image.shape[1]).astype(int)
    y_coords = np.floor(np.random.rand(num) * image.shape[0]).astype(int)
    image[y_coords, x_coords] = black
    return image

def add_gaussian_noise(image, sigma):
    assert np.min(image) >= 0, f"should be at least 0, min {np.min(image)}"
    assert np.max(image) <= 1, f"can't exceed 1, max {np.max(image)}"
    gaussian_noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
    gaussian_noise = gaussian_noise.reshape(image.shape)
    return image + gaussian_noise

def get_tiles(image, in_tile_shape, out_tile_shape):
    width_diff = in_tile_shape[1] - out_tile_shape[1]
    pad_width = width_diff // 2
    padded_photo = pad(image, pad_width)

    horizontal_count = ceil(image.shape[1] / out_tile_shape[1])
    vertical_count = ceil(image.shape[0] / out_tile_shape[0])

    # first split the image based on the tiles that fit
    x_coords = [h*out_tile_shape[1] for h in range(horizontal_count-1)]
    y_coords = [v*out_tile_shape[0] for v in range(vertical_count-1)]

    # The last row and column of tiles might not fit
    # (Might go outside the image)
    # so get the tile positiion by subtracting tile size from the
    # edge of the image.
    right_x = padded_photo.shape[1] - in_tile_shape[1]
    bottom_y = padded_photo.shape[0] - in_tile_shape[0]

    y_coords.append(bottom_y)
    x_coords.append(right_x)

    # because its a rectangle get all combinations of x and y
    tile_coords = [(x, y) for x in x_coords for y in y_coords]
    tiles = tiles_from_coords(padded_photo, tile_coords, in_tile_shape)
    return tiles, tile_coords


def reconstruct_from_tiles(tiles, coords, output_shape):
    image = np.zeros(output_shape)
    for tile, (x, y) in zip(tiles, coords):
        image[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
    return image


def tiles_from_coords(image, coords, tile_shape):
    tiles = []
    for x, y in coords:
        tile = image[y:y+tile_shape[0],
                     x:x+tile_shape[1]]
        tiles.append(tile)
    return tiles

def save_then_move(out_path, seg_output, npy=False):
    """ need to save as a tmp file first (.tmp.fname) and
        then rename after saving.
        this is because scripts are monitoring the segmentation folder
        and the file saving takes time..
        We don't want the scripts that monitor the segmentation
        folder to try loading the file half way through saving
        as this causes errors. Thus we save and then rename.
    """
    fname = os.path.basename(out_path)
    temp_path = os.path.join(os.path.dirname(out_path), '.tmp.' + fname)

    if npy:
        # numpy can be easier than PNG to work with in some scripts.
        np.savez_compressed(temp_path, seg=seg_output)
    else:
        imsave(temp_path, seg_output, check_contrast=False)

    attempts = 0
    max_attempts = 50
    # we found on google colab that writing a file doesn't mean
    # that it will immediately exist. so we will retry 50 times in case
    # it takes some time (5 seconds).
    while attempts < max_attempts:
        attempts += 1
        if os.path.isfile(temp_path):
            shutil.copy(temp_path, out_path)
            os.remove(temp_path)
            return
        time.sleep(0.1)
    raise Exception(f'Could not find image at {temp_path} '
                    f'after trying {max_attempts} times')

def load_image(photo_path):
    photo = Image.open(photo_path)
    photo = ImageOps.exif_transpose(photo)
    photo = np.array(photo)

    # sometimes photo is a list where first element is the photo
    if len(photo.shape) == 1:
        photo = photo[0]
    # if 4 channels then convert to rgb
    # (presuming 4th channel is alpha channel)
    if len(photo.shape) > 2 and photo.shape[2] == 4:
        photo = color.rgba2rgb(photo)

    # if image is black and white then change it to rgb
    # TODO: train directly on B/W instead of doing this conversion.
    if len(photo.shape) == 2:
        photo = color.gray2rgb(photo)
    return photo
