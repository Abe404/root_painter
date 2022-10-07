"""
Copyright (C) 2020 Abraham Goerge Smith

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

# pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914
# pylint: disable=W0703 # Too broad an exception

import os
from os.path import splitext
import filecmp
import numpy as np
from skimage.io import imread

def ls(dir_path):
    # Don't show hidden files
    # These can happen due to issues like file system 
    # synchonisation technology. RootPainter doesn't use them anywhere
    fnames = os.listdir(dir_path)
    fnames = [f for f in fnames if f[0] != '.']
    return fnames

def last_fname_with_annotations(fnames, train_annot_dirs, val_annot_dirs):
    """
    Go through fnames and return the one after
    the last in the list with an annotation.
    If no annotations are found return None.
    """
    annot_fnames = []
    for annot_dir in train_annot_dirs + val_annot_dirs:
        annot_fnames += os.listdir(annot_dir)

    last_fname = None
    # remove the extensions as annotations will always be PNG
    # but fnames could be JPG or other.
    annot_fnames = [os.path.splitext(f)[0] for f in annot_fnames]

    for i, fname in enumerate(fnames):
        if os.path.splitext(fname)[0] in annot_fnames:
            if i+1 < len(fnames):
                last_fname = fnames[i+1]
            else:
                return fnames[0]
    return last_fname


def get_annot_path(fname, train_dir, val_dir):
    """
    return path to annot if it is found in
    train or val annot dirs.
    Otherwise return None
    """
    train_path = os.path.join(train_dir, fname)
    val_path = os.path.join(val_dir, fname)
    if os.path.isfile(train_path):
        return train_path
    if os.path.isfile(val_path):
        return val_path
    return None


def get_new_annot_target_dir(train_annot_dir, val_annot_dir):
    """ Should we add new annotations to train or validation data? """
    train_annots = os.listdir(train_annot_dir)
    val_annots = os.listdir(val_annot_dir)
    train_annots = [f for f in train_annots if splitext(f)[1] == '.png']
    val_annots = [f for f in val_annots if splitext(f)[1] == '.png']
    num_train_annots = len(train_annots)
    num_val_annots = len(val_annots)
    # first aim to get at least one annotation in train and validation.
    if num_train_annots == 0 and num_val_annots > 0:
        return train_annot_dir
    if num_train_annots > 0 and num_val_annots == 0:
        return val_annot_dir
    # then only add files to validation if there is at least 5x in train
    if num_train_annots >= (num_val_annots * 5):
        return val_annot_dir
    return train_annot_dir


#pylint: disable=R0913 # Too many arguments
def maybe_save_annotation(proj_location, annot_pixmap, annot_path, png_fname,
                          train_annot_dir, val_annot_dir):
    # First save to project folder as temp file.
    temp_out = os.path.join(proj_location, 'temp_annot.png')
    annot_pixmap.save(temp_out, 'PNG')

    # if there is an existing annotation.
    if annot_path:
        # and the annot we are saving is different.
        if not filecmp.cmp(temp_out, annot_path):
            # Then we must over-write the previously saved annoation.
            # The user is performing an edit, possibly correcting an error.
            annot_pixmap.save(annot_path, 'PNG')
    else:
        # if there is not an existing annotation
        # and the annotation has some content
        if np.sum(imread(temp_out)):
            # then find the best place to put it based on current counts.
            annot_dir = get_new_annot_target_dir(train_annot_dir, val_annot_dir)
            annot_path = os.path.join(annot_dir, png_fname)
            annot_pixmap.save(annot_path, 'PNG')
        else:
            # if the annotation did not have content.
            # and there was not an existing annotation
            # then don't save anything, this data is useless for
            # training.
            print('not saving as annotation empty')

    # clean up the temp file
    while os.path.isfile(temp_out):
        try:
            # Added try catch because this error happened (very rarely)
            # PermissionError: [WinError 32]
            # The process cannot access the file becausegc
            # it is being used by another process
            os.remove(temp_out)
        except Exception as e:
            print('Caught exception when trying to detele temp annot', e)
    return annot_path
