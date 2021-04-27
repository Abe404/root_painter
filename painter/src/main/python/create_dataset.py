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
#pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914, R0915, R0911
import os
import glob
import random
from pathlib import Path
import itertools
import json
from random import shuffle

import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from skimage.io import imread, imsave
from skimage.color import rgba2rgb
from im_utils import is_image

# Avoiding bug with truncated images,
# "Reason: "broken data stream when reading image file"
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from progress_widget import BaseProgressWidget
from name_edit_widget import NameEditWidget
import im_utils

def all_image_paths_in_dir(dir_path):
    root_dir = os.path.abspath(dir_path)
    all_paths = glob.iglob(root_dir + '/**/*', recursive=True)
    image_paths = []
    for p in all_paths:
        name = os.path.basename(p)
        if name[0] != '.':
            ext = os.path.splitext(name)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                image_paths.append(p)
    return image_paths


def get_dupes(a):
    seen = {}
    dupes = []
    for x in a:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return dupes, seen


def get_file_pieces(im, target_size):
    im_h = im.shape[0]
    im_w = im.shape[1]

    # first get all possible ways to slice the images.
    # The minimum width or height possible is 600 (CNN input size is 572)
    max_h_pieces = im_h // 600
    max_w_pieces = im_w // 600

    poss_h_pieces = list(range(1, max_h_pieces + 1))
    poss_w_pieces = list(range(1, max_w_pieces + 1))
    possible_pieces = list(itertools.product(poss_h_pieces, poss_w_pieces))

    # for each of the possible pieces.
    # get the width and height of the proposed piece
    widths = [im_w // p[1] for p in possible_pieces]
    heights = [im_h // p[0] for p in possible_pieces]

    # and then get the ratio between width and height
    ratios = [width/height for (width, height) in zip(widths, heights)]

    squareness = [abs(r - 1) for r in ratios]
    squareness = np.array(squareness) / (np.max(squareness) + 1e-5)
    pix_counts = [width*height for (width, height) in zip(widths, heights)]

    # how close is the size to the target size
    size_dists = [abs(p - (target_size * target_size)) for p in pix_counts]

    # scale 0-1
    size_dists = np.array(size_dists) / (np.max(size_dists) + 1e-5)

    assert min(size_dists) >= 0
    assert max(size_dists) <= 1
    assert min(squareness) >= 0, f"{squareness}"
    assert max(squareness) <= 1, f"{squareness}"
    assert len(size_dists) == len(squareness)
    combined_dists = squareness + size_dists
    best_idx = sorted(zip(combined_dists, range(len(pix_counts))))[0][1]
    h_pieces, w_pieces = possible_pieces[best_idx]
    piece_w = widths[best_idx]
    piece_h = heights[best_idx]
    assert h_pieces and w_pieces

    if h_pieces == 1 and w_pieces == 1:
        return [im]

    # now get the actual pieces from the image.
    pieces = []
    for hi in range(h_pieces):
        for wi in range(w_pieces):
            h_start = hi * piece_h
            h_end = h_start + piece_h
            w_start = wi * piece_w
            w_end = w_start + piece_w
            pieces.append(im[h_start:h_end, w_start:w_end])
    return pieces


def save_im_pieces(im_path, target_dir, pieces_from_each_image, target_size):
    pieces = get_file_pieces(im_utils.load_image(im_path), target_size)
    pieces = random.sample(pieces, min(pieces_from_each_image, len(pieces)))
    fname = os.path.basename(im_path)
    fname = os.path.splitext(fname)[0]
    for i, p in enumerate(pieces):
        piece_fname = f"{fname}_{str(i).zfill(3)}.jpg"
        if p.shape[-1] == 4:
            p = rgba2rgb(p)
        imsave(os.path.join(target_dir, piece_fname), p, check_contrast=False)

class CreationProgressWidget(BaseProgressWidget):
    """
    Once the dataset creation process starts the CreateDatasetWidget
    is closed and the CreationProgressWidget will display the progress.
    """
    def __init__(self):
        super().__init__('Creating dataset')

    def run(self, ims_to_sample_from, target_dir,
            tiles_per_image, target_size):
        self.progress_bar.setMaximum(len(ims_to_sample_from))
        self.creation_thread = CreationThread(ims_to_sample_from, target_dir,
                                              tiles_per_image,
                                              target_size)
        self.creation_thread.progress_change.connect(self.onCountChanged)
        self.creation_thread.done.connect(self.done)
        self.creation_thread.start()


class CreationThread(QtCore.QThread):
    """
    Runs another thread.
    """
    progress_change = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal()

    def __init__(self, images_for_dataset, target_dir,
                 pieces_from_each_image, target_size):
        super().__init__()
        self.images_for_dataset = images_for_dataset
        self.target_dir = target_dir
        self.pieces_from_each_image = pieces_from_each_image
        self.target_size = target_size

    def run(self):
        for i, fpath in enumerate(self.images_for_dataset):
            save_im_pieces(fpath, self.target_dir,
                           self.pieces_from_each_image, self.target_size)
            self.progress_change.emit(i+1, len(self.images_for_dataset))
        self.done.emit()


class CreateDatasetWidget(QtWidgets.QWidget):

    submit = QtCore.pyqtSignal()

    def __init__(self, sync_dir):
        super().__init__()
        self.use_random = False
        self.source_dir = None
        self.im_size_default = 900
        self.sync_dir = sync_dir
        self.initUI()

    def on_radio_clicked(self):
        radio = self.sender()
        if radio.isChecked():
            use_random = (radio.name == 'random')
            self.num_ims_widget.setVisible(use_random)
            self.use_random = use_random
            self.validate()

    def initUI(self):
        self.setWindowTitle("Create Dataset")
        layout = QtWidgets.QVBoxLayout()
        self.layout = layout # to add progress bar later.
        self.setLayout(layout)

        self.name_edit_widget = NameEditWidget('Dataset')
        self.name_edit_widget.changed.connect(self.validate)
        self.layout.addWidget(self.name_edit_widget)

        # Add specify image directory button
        directory_label = QtWidgets.QLabel()
        directory_label.setText("Source image directory: Not yet specified")
        layout.addWidget(directory_label)
        self.directory_label = directory_label

        specify_image_dir_btn = QtWidgets.QPushButton('Specify source image directory')
        specify_image_dir_btn.clicked.connect(self.select_image_dir)
        layout.addWidget(specify_image_dir_btn)


        # User can select all images or sample randomly
        radio_widget = QtWidgets.QWidget()
        radio_layout = QtWidgets.QHBoxLayout()
        radio_widget.setLayout(radio_layout)
        layout.addWidget(radio_widget)

        # Add radio, use random weight or specify model file.
        radio = QtWidgets.QRadioButton("All Images")
        radio.setChecked(True)
        radio.name = "all"
        radio.toggled.connect(self.on_radio_clicked)
        radio_layout.addWidget(radio)

        radio = QtWidgets.QRadioButton("Random sample")
        radio.name = "random"
        radio.toggled.connect(self.on_radio_clicked)
        radio_layout.addWidget(radio)

        # num ims input
        num_ims_widget = QtWidgets.QWidget()
        layout.addWidget(num_ims_widget)
        num_ims_widget_layout = QtWidgets.QHBoxLayout()
        num_ims_widget.setLayout(num_ims_widget_layout)
        self.num_ims_widget = num_ims_widget
        edit_num_ims_label = QtWidgets.QLabel()
        self.num_ims_widget.setVisible(self.use_random)

        edit_num_ims_label.setText("Images to sample from")
        num_ims_widget_layout.addWidget(edit_num_ims_label)
        num_ims_edit_widget = QtWidgets.QSpinBox()
        num_ims_edit_widget.setMaximum(999999)
        num_ims_edit_widget.setMinimum(1)
        num_ims_edit_widget.valueChanged.connect(self.validate)
        self.num_ims_edit_widget = num_ims_edit_widget
        num_ims_widget_layout.addWidget(num_ims_edit_widget)

        # max tiles per image input
        tiles_per_im_widget = QtWidgets.QWidget()
        layout.addWidget(tiles_per_im_widget)
        tiles_per_im_widget_layout = QtWidgets.QHBoxLayout()
        tiles_per_im_widget.setLayout(tiles_per_im_widget_layout)
        edit_tiles_per_im_label = QtWidgets.QLabel()

        edit_tiles_per_im_label.setText("Maximum tiles per image:")
        tiles_per_im_widget_layout.addWidget(edit_tiles_per_im_label)
        tiles_per_im_edit_widget = QtWidgets.QSpinBox()
        tiles_per_im_edit_widget.setMaximum(999999)
        tiles_per_im_edit_widget.setMinimum(1)
        tiles_per_im_edit_widget.valueChanged.connect(self.validate)
        self.tiles_per_im_edit_widget = tiles_per_im_edit_widget
        tiles_per_im_widget_layout.addWidget(tiles_per_im_edit_widget)

        # Image size (used for width and height)
        im_size_widget = QtWidgets.QWidget()
        layout.addWidget(im_size_widget)
        im_size_widget_layout = QtWidgets.QHBoxLayout()
        im_size_widget.setLayout(im_size_widget_layout)
        edit_im_size_label = QtWidgets.QLabel()

        edit_im_size_label.setText("Target width and height (pixels):")
        im_size_widget_layout.addWidget(edit_im_size_label)
        im_size_edit_widget = QtWidgets.QSpinBox()
        im_size_edit_widget.setMaximum(999999)
        im_size_edit_widget.setMinimum(600)

        im_size_edit_widget.valueChanged.connect(self.validate)
        self.im_size_edit_widget = im_size_edit_widget
        im_size_widget_layout.addWidget(im_size_edit_widget)

        # info label for user feedback
        info_label = QtWidgets.QLabel()
        info_label.setText("Name, directory and maximum images must"
                           " be specified in order to create a dataset.")
        layout.addWidget(info_label)
        self.info_label = info_label

        # Add create button
        create_btn = QtWidgets.QPushButton('Create dataset')
        create_btn.clicked.connect(self.try_submit)
        layout.addWidget(create_btn)
        create_btn.setEnabled(False)
        self.create_btn = create_btn

        # call validation error
        im_size_edit_widget.setValue(self.im_size_default)

    def validate(self):
        name = self.name_edit_widget.name

        if name:
            self.target_dir = os.path.join(self.sync_dir, 'datasets', name)

        if not name:
            self.info_label.setText("Name must be specified to create dataset")
            self.create_btn.setEnabled(False)
            return
        if not self.source_dir:
            self.info_label.setText("Directory must be specified to create dataset")
            self.create_btn.setEnabled(False)
            return

        if not self.tiles_per_im_edit_widget.value():
            self.info_label.setText("Maximum tiles per image must be "
                                    "specified to create dataset")
            self.create_btn.setEnabled(False)
            return

        if not self.im_size_edit_widget.value():
            self.info_label.setText("Target width and height must be "
                                    "specified to create dataset")
            self.create_btn.setEnabled(False)
            return

        if not self.image_paths:
            message = ('Source image directory must contain image files '
                       ' with png, jpg, jpeg, tif'
                       'or tiff extension.')
            self.info_label.setText(message)
            self.create_btn.setEnabled(False)
            return

        # check no duplicates (based on file name)
        fnames = [os.path.basename(i) for i in self.image_paths]
        if len(set(fnames)) != len(fnames):
            dupes, seen = get_dupes(fnames)
            self.info_label.setText(f"{len(dupes)} duplicates found including "
                                    f"{dupes[0]} which was found {seen[dupes[0]]} times.")
            self.create_btn.setEnabled(False)
            return
        if os.path.exists(self.target_dir):
            self.info_label.setText(f"Dataset with name {name} already exists")
            self.create_btn.setEnabled(False)
            return

        # Sucess!
        self.info_label.setText("")
        self.create_btn.setEnabled(True)


    def try_submit(self):
        target_dir = Path(self.target_dir)
        tiles_per_image = self.tiles_per_im_edit_widget.value()
        num_ims_to_sample_from = self.num_ims_edit_widget.value()
        target_size = self.im_size_edit_widget.value()
        all_images = self.image_paths

        if self.use_random:
            ims_to_sample_from = random.sample(all_images,
                                               num_ims_to_sample_from)
        else:
            ims_to_sample_from = all_images

        os.makedirs(target_dir)

        self.progress_widget = CreationProgressWidget()
        self.progress_widget.run(ims_to_sample_from, target_dir,
                                 tiles_per_image, target_size)
        self.close()
        self.progress_widget.show()


    def select_image_dir(self):
        self.image_dialog = QtWidgets.QFileDialog(self)
        self.image_dialog.setFileMode(QtWidgets.QFileDialog.Directory)

        def output_selected():
            self.source_dir = self.image_dialog.selectedFiles()[0]
            self.directory_label.setText('Image directory: ' + self.source_dir)
            self.image_paths = all_image_paths_in_dir(self.source_dir)
            self.validate()

        self.image_dialog.fileSelected.connect(output_selected)
        self.image_dialog.open()


def check_extend_dataset(main_window, dataset_dir, prev_fnames, proj_file_path):

    all_image_names = [f for f in os.listdir(dataset_dir) if is_image(f)]

    new_image_names = [f for f in all_image_names if f not in prev_fnames]

    button_reply = QtWidgets.QMessageBox.question(main_window,
        'Confirm',
        f"There are {len(new_image_names)} new images in the dataset."
        " Are you sure you want to extend the project to include these new images?",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
        QtWidgets.QMessageBox.No)

    if button_reply == QtWidgets.QMessageBox.Yes:
        # shuffle the new file names
        shuffle(new_image_names)
        # load the project json for reading and writing
        settings = json.load(open(proj_file_path, 'r'))
        # read the file_names
        all_file_names = settings['file_names'] + new_image_names
        settings['file_names'] = all_file_names

        # Add the new_files to the list
        # then save the json again
        json.dump(settings, open(proj_file_path, 'w'), indent=4)
        return True, all_file_names
    else:
        return False, all_image_names