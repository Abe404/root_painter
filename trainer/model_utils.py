"""
Utilities for working with the U-Net models

Copyright (C) 2020 Abraham George Smith
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

# pylint: disable=C0111, R0913, R0914, W0511

# bs doesn't conform to naming style
# pylint: disable=C0103

# too general exception
# pylint: disable=W0703

# too many statements
# pylint: disable=R0915

# torch has no 'from_numpy' member
# pylint: disable=E1101

import os
import time
import glob
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import softmax
from skimage.io import imread
from skimage import img_as_float32
import im_utils
from unet import UNetGNRes
from metrics import get_metrics
from file_utils import ls

def get_latest_model_paths(model_dir, k):
    fnames = ls(model_dir)
    fnames = sorted(fnames)[-k:]
    fpaths = [os.path.join(model_dir, f) for f in fnames]
    return fpaths

def load_model(model_path, classes):
    state_dict = torch.load(model_path)
    if 'classes' not in state_dict:
        print(f'adding classes {classes} to model and resaving')
        # this fixes older multiclass models that did not have classes saved.
        # classes should be saved wih the model to enable segment folder to function
        state_dict['classes'] = classes
        torch.save(state_dict, model_path)
    classes = state_dict['classes']
    del state_dict['classes'] # pytorch cant handle this key.
    model = UNetGNRes(classes)
    try:
        model.load_state_dict(state_dict)
        model = torch.nn.DataParallel(model)
    # bare except
    # pylint: disable=W0702
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
    model.cuda()
    return model, classes

def create_first_model_with_random_weights(model_dir, classes):
    # used when no model was specified on project creation.
    model_num = 1
    model_name = str(model_num).zfill(6)
    model_name += '_' + str(int(round(time.time()))) + '.pkl'
    model = UNetGNRes(classes)
    model = torch.nn.DataParallel(model)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    model.cuda()
    return model


def get_prev_model(model_dir, classes):
    prev_path = get_latest_model_paths(model_dir, k=1)[0]
    prev_model = load_model(prev_path, classes)[0]
    return prev_model, prev_path

def get_val_metrics(cnn, val_annot_dirs, dataset_dir,
        in_w, out_w, bs, project_classes):
    """
    Return the TP, FP, TN, FN, defined_sum, duration
    for the {cnn} on the validation set

    TODO - This is too similar to the train loop. Merge both and use flags.
    """
    assert isinstance(val_annot_dirs, list), 'val dir should be list'
    start = time.time()
    fnames = []
    dirnames = []
    # get all the annotations for validation
    # including their file name and class (containing dir name)
    for val_annot_dir in val_annot_dirs:
        class_annots = ls(val_annot_dir)
        for fname in class_annots:
            if im_utils.is_photo(fname):
                fnames.append(fname)
                dirnames.append(val_annot_dir)

    # TODO: In order to speed things up, be a bit smarter here
    # by only segmenting the parts of the image where we have
    # some annotation defined.
    # implement a 'partial segment' which exlcudes tiles with no
    # annotation defined.

    tps = 0
    fps = 0
    tns = 0
    fns = 0
    defined_sum = 0

    foregrounds = []
    backgrounds = []
    classes = []
    image_paths = [] # image path for each associated fg,bg and class
    image_class_pred_maps = {} 

    # for each val_dirname and annotation in it
    for dirname, fname in zip(dirnames, fnames):
        annot_path = os.path.join(dirname,
                                  os.path.splitext(fname)[0] + '.png')
    
        # reading the image may throw an exception.
        # I suspect this is due to it being only partially written to disk
        # simply retry if this happens.
        try:
            annot = imread(annot_path)
        except Exception as ex:
            print(f'Exception reading annotation {annot_path} inside validation method.'
                  'Will retry in 0.1 seconds')
            print(fname, ex)
            time.sleep(0.1)
            annot = imread(annot_path)

        annot = np.array(annot)
        foreground = annot[:, :, 0].astype(bool).astype(int)
        background = annot[:, :, 1].astype(bool).astype(int)
        foregrounds.append(foreground)
        backgrounds.append(background)

        # Assuming class name is in annotation path
        # i.e annotations/{class_name}/train/annot1.png,annot2.png..
        class_name = Path(dirname).parts[-2]
        classes.append(class_name)

        # Prediction should include channels for each class.
        image_path_part = os.path.join(dataset_dir, os.path.splitext(fname)[0])
        image_path = glob.glob(image_path_part + '.*')[0]
        image_paths.append(image_path)
        
        # ensure image has a prediction.
        if not image_path in image_class_pred_maps:
            image = im_utils.load_image(image_path)
            image, pad_settings = im_utils.pad_to_min(image, min_w=572, min_h=572)
            # predictions for all classes
            class_pred_maps = unet_segment(cnn, image, bs, in_w,
                                           out_w, threshold=0.5)
            # crop those that were padded
            cropped_pred_maps = []
            for class_pred_map in class_pred_maps:
                class_pred_map = im_utils.crop_from_pad_settings(class_pred_map, pad_settings)
                cropped_pred_maps.append(class_pred_map)
            image_class_pred_maps[image_path] = cropped_pred_maps

    # the annotation is 729 by 729, so how is it possible for the
    # foregrounds and backgrounds to have dimensions of 786by786????
    assert len(image_paths) == len(backgrounds) == len(classes), (
        f"{len(foregrounds)},{len(backgrounds)},{len(classes)}")

    for (image_path, foreground,
         background, class_name) in zip(image_paths, foregrounds,
                                        backgrounds, classes):
        class_pred_maps = image_class_pred_maps[image_path] 
        pred = class_pred_maps[project_classes.index(class_name)]

        # for each individual annotation, associated with a specific class.
        # mask defines which pixels are defined in the annotation.
        mask = foreground + background
        mask = mask.astype(bool).astype(int)
        pred *= mask
        pred = pred.astype(bool).astype(int)
        y_defined = mask.reshape(-1)
        y_pred = pred.reshape(-1)[y_defined > 0]
        y_true = foreground.reshape(-1)[y_defined > 0]
        tps += np.sum(np.logical_and(y_pred == 1, y_true == 1))
        tns += np.sum(np.logical_and(y_pred == 0, y_true == 0))
        fps += np.sum(np.logical_and(y_pred == 1, y_true == 0))
        fns += np.sum(np.logical_and(y_pred == 0, y_true == 1))
        defined_sum += np.sum(y_defined > 0)

    duration = round(time.time() - start, 3)
    metrics = get_metrics(tps, fps, tns, fns, defined_sum, duration)
    return metrics

def save_if_better(model_dir, cur_model, prev_model_path,
                   cur_f1, prev_f1, model_classes):
    print('prev f1', str(round(prev_f1, 5)).ljust(7, '0'),
          'cur f1', str(round(cur_f1, 5)).ljust(7, '0'))
    if cur_f1 > prev_f1:
        prev_model_fname = os.path.basename(prev_model_path)
        prev_model_num = int(prev_model_fname.split('_')[0])
        model_num = prev_model_num + 1
        now = int(round(time.time()))
        model_name = str(model_num).zfill(6) + '_' + str(now) + '.pkl'
        model_path = os.path.join(model_dir, model_name)
        print('saving', model_path, time.strftime('%H:%M:%S', time.localtime(now)))
        state_dict = cur_model.state_dict()
        state_dict['classes'] = model_classes # save classes with model to enable segmentation without loading the project.
        torch.save(state_dict, model_path)
        return True
    return False

def ensemble_segment(model_paths, image, bs, in_w, out_w, classes,
                     threshold=0.5):
    """ Average predictions from each model specified in model_paths """
    pred_count = 0
    class_pred_sums = [None] * len(classes)
    image, pad_settings = im_utils.pad_to_min(image, min_w=in_w, min_h=in_w)
    # then add predictions from the previous models to form an ensemble
    for model_path in model_paths:
        cnn, classes = load_model(model_path, classes)
        pred_maps = unet_segment(cnn, image,
                                bs, in_w, out_w, threshold=None)
        for i, pred_map in enumerate(pred_maps):
            if class_pred_sums[i] is not None:
                class_pred_sums[i] += pred_map
            else:
                class_pred_sums[i] = pred_map
        pred_count += 1
        # get flipped version too (test time augmentation)
        flipped_im = np.fliplr(image)
        flipped_pred_maps = unet_segment(cnn, flipped_im, bs, in_w,
                                         out_w, threshold=None)

        for i, flipped_pred in enumerate(flipped_pred_maps):
            pred_map = np.fliplr(flipped_pred)
            class_pred_sums[i] += pred_map
        pred_count += 1

    class_pred_maps = []
    for pred_sum in class_pred_sums:
        pred_sum = im_utils.crop_from_pad_settings(pred_sum, pad_settings)
        mean_pred_map = pred_sum / pred_count
        predicted = mean_pred_map > threshold
        predicted = predicted.astype(int)
        class_pred_maps.append(predicted)
    return class_pred_maps

def unet_segment(cnn, image, bs, in_w, out_w, threshold=0.5):

    # dont need classes input here.
    # we can see how many classes are output by dividing num output channels by 2
    """
    Threshold set to None means probabilities returned without thresholding.
    """
    assert image.shape[0] >= in_w, str(image.shape[0])
    assert image.shape[1] >= in_w, str(image.shape[1])

    tiles, coords = im_utils.get_tiles(image,
                                       in_tile_shape=(in_w, in_w, 3),
                                       out_tile_shape=(out_w, out_w))
    tile_idx = 0
    batches = []
    while tile_idx < len(tiles):
        tiles_to_process = []
        for _ in range(bs):
            if tile_idx < len(tiles):
                tile = tiles[tile_idx]
                tile = img_as_float32(tile)
                tile = im_utils.normalize_tile(tile)
                tile = np.moveaxis(tile, -1, 0)
                tile_idx += 1
                tiles_to_process.append(tile)
        tiles_for_gpu = torch.from_numpy(np.array(tiles_to_process))
        tiles_for_gpu.cuda()
        tiles_for_gpu = tiles_for_gpu.float()
        batches.append(tiles_for_gpu)
    
    class_output_tiles = None # list of tiles for each class
    for gpu_tiles in batches:
        # gpu tiles shape torch.Size([3, 3, 572, 572])
        outputs = cnn(gpu_tiles)
        # outputs shape torch.Size([3, 2, 500, 500])
        # bg channel index for each class in network output.
        class_idxs = [x * 2 for x in range(outputs.shape[1] // 2)]

        if class_output_tiles is None:
            class_output_tiles = [[] for _ in class_idxs]

        # output: (batch_size, bg/fg...bg/fg, height, width)

        for i, class_idx in enumerate(class_idxs):
            class_output = outputs[:, class_idx:class_idx+2]
            # class_output : (batch_size, bg/fg, height, width)

            softmaxed = softmax(class_output, 1) 
            # softmaxed: (4, 2, 500, 500)

            foreground_probs = softmaxed[:, 1]  # just the foreground probability.
            # foreground_probs: (batch_size, 500, 500)
            if threshold is not None:
                predicted = foreground_probs > threshold
                predicted = predicted.int()
            else:
                predicted = foreground_probs
            pred_np = predicted.data.cpu().numpy()
            for out_tile in pred_np:
                class_output_tiles[i].append(out_tile)

    # why could this be?
    assert len(class_output_tiles[0]) == len(coords), (
        f'{len(class_output_tiles[0])} {len(coords)}')
    class_pred_maps = []

    for i in range(len(class_output_tiles)):
        # reconstruct for each class
        reconstructed = im_utils.reconstruct_from_tiles(class_output_tiles[i],
                                                        coords, image.shape[:-1])
        class_pred_maps.append(reconstructed)
    return class_pred_maps
