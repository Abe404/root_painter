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
import os
import time
import glob
import numpy as np
from pathlib import Path
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
    model = UNetGNRes(classes)
    try:
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model)
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
    model.cuda()
    return model

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
    prev_model = load_model(prev_path, classes)
    return prev_model, prev_path

def get_val_metrics(cnn, val_annot_dirs, dataset_dir, in_w, out_w, bs):
    """
    Return the TP, FP, TN, FN, defined_sum, duration
    for the {cnn} on the validation set

    TODO - This is too similar to the train loop. Merge both and use flags.
    """
    assert type(val_annot_dirs) == list
    start = time.time()
    fnames = []
    dirnames = []
    for val_annot_dir in val_annot_dirs:
        class_annots = ls(val_annot_dir)
        for fname in class_annots:
            if im_utils.is_photo(fname):
                fnames.append(fname)
                dirnames.append(val_annot_dir)
        
    cnn.half()
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
    
    for dirname, fname in zip(dirnames, fnames):
        annot_path = os.path.join(dirname,
                                  os.path.splitext(fname)[0] + '.png')
    
        # reading the image may throw an exception.
        # I suspect this is due to it being only partially written to disk
        # simply retry if this happens.
        try:
            annot = imread(annot_path)
        except Exception as ex:
            print('Exception reading annotation inside validation method.'
                  'Will retry in 0.1 seconsds')
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
        class_name = Path(train_annot_dir).parts[-2]
        classes.append(class_name)

    # Prediction should include channels for each class.
    image_path_part = os.path.join(dataset_dir, os.path.splitext(fname)[0])
    image_path = glob.glob(image_path_part + '.*')[0]
    image = im_utils.load_image(image_path)

    # predictions for all classes
    class_pred_maps = unet_segment(cnn, image, bs, in_w,
                                         out_w, classes, threshold=0.5)

    for pred, foreground, background, class_name in zip(class_pred_maps, foregrounds,
                                                        backgrounds, classes):
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
                   cur_f1, prev_f1):
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
        torch.save(cur_model.state_dict(), model_path)
        return True
    return False

def ensemble_segment(model_paths, image, bs, in_w, out_w, classes
                     threshold=0.5):
    """ Average predictions from each model specified in model_paths """
    pred_count = 0
    class_pred_sums = [None] * len(classes)
    class_idx = range(classes)
    # then add predictions from the previous models to form an ensemble
    for model_path in model_paths:
        cnn = load_model(model_path, classes)
        cnn.half()
        pred_maps = unet_segment(cnn, image,
                                bs, in_w, out_w, classes, threshold=None)
        for i, pred_map in enumerate(pred_maps): 
            if class_pred_sums[i] is not None:
                class_pred_sums[i] += preds
            else:
                class_pred_sums[i] = preds
        pred_count += 1
        # get flipped version too (test time augmentation)
        flipped_im = np.fliplr(image)
        flipped_pred_maps = unet_segment(cnn, flipped_im, bs, in_w,
                                         out_w, classes, threshold=None)

        for i, flipped_pred in enumerate(flipped_pred_maps): 
            pred_map = np.fliplr(flipped_pred)
            class_pred_sums[i] += pred_map
        pred_count += 1

    class_pred_maps = []
    for pred_sum in class_pred_sums:
        mean_pred_map = pred_sum / pred_count
        predicted = mean_pred_map > threshold
        predicted = predicted.astype(int)
        class_pred_maps.append(predicted)
    return class_pred_maps

def unet_segment(cnn, image, bs, in_w, out_w, classes, threshold=0.5):
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
        tiles_for_gpu = tiles_for_gpu.half()
        batches.append(tiles_for_gpu)

    class_output_tiles = [[]] * len(classes)
    for gpu_tiles in batches:
        outputs = cnn(gpu_tiles)
        for i in range(len(classes)):
            class_channel_idx = i * 2 # output channel index for this class
            # softmax each pair of foreground, background channels.
            class_output = outputs[class_channel_idx:class_channel_idx+2]
            softmaxed = softmax(softmaxed, 1)
            foreground_probs = softmaxed[:, 1, :]  # just the foreground probability.
            if threshold is not None:
                predicted = foreground_probs > threshold
                predicted = predicted.view(-1).int()
            else:
                predicted = foreground_probs
            pred_np = predicted.data.cpu().numpy()
            out_tiles = pred_np.reshape((len(gpu_tiles), out_w, out_w))
            for out_tile in out_tiles:
                class_output_tiles[i].append(out_tile)

        assert len(output_tiles) == len(coords), (
            f'{len(output_tiles)} {len(coords)}')
    class_pred_maps = []
    for i in range(len(classes)):
        # reconstruct for each class
        reconstructed = im_utils.reconstruct_from_tiles(class_output_tiles[i],
                                                        coords, image.shape[:-1])
        class_pred_maps.append(reconstructed)
    return class_pred_maps
