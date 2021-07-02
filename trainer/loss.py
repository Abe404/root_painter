"""
Copyright (C) 2019 Abraham George Smith

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
# torch has no sum member
# pylint: disable=E1101

import torch
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy


def dice_loss(predictions, labels):
    """ based on loss function from V-Net paper """
    softmaxed = softmax(predictions, 1)
    predictions = softmaxed[:, 1, :]  # just the root probability.
    labels = labels.float()
    preds = predictions.contiguous().view(-1)
    labels = labels.view(-1)
    intersection = torch.sum(torch.mul(preds, labels))
    union = torch.sum(preds) + torch.sum(labels)
    return 1 - ((2 * intersection) / (union))


def combined_loss(predictions, labels):
    """ mix of dice and cross entropy """
    # if they are bigger than 1 you get a strange gpu error
    # without a stack track so you will have no idea why.
    assert torch.max(labels) <= 1
    if torch.sum(labels) > 0:
        return (dice_loss(predictions, labels) +
                (0.3 * cross_entropy(predictions, labels)))
    # When no roots use only cross entropy
    # as dice is undefined.
    return 0.3 * cross_entropy(predictions, labels)




def get_batch_loss_instance_wise(outputs, batch_fg_tiles, batch_bg_tiles, batch_classes, project_classes):
    """
        This loss is OK but computes dice on each individual instance, instead of on the entire batch.
        So it's not consistent with the previous single class loss.

        outputs - predictions from neural network (not softmaxed)
        batch_fg_tiles - list of tiles, each tile is binary map of foreground annotation
        batch_bg_tiles - list of tiles, each tile is binary map of background annotation

        returns
            batch_loss - loss used to update the network
            tps - true positives for batch
            tns - true negatives for batch
            fps - false positives for batch
            fns - false negatives for batch
            defined_total - number of pixels with annotation defined.
    """

    tps = 0
    fps = 0
    tns = 0
    fns = 0
    defined_total = 0

    batch_loss = None
    # TODO get to the individual instance level for computing loss.
    for im_idx in range(outputs.shape[0]):
        output = outputs[im_idx]
        for i, classname in enumerate(batch_classes[im_idx]):
            classname = batch_classes[im_idx][i]
            fg_tile = batch_fg_tiles[im_idx][i]
            bg_tile = batch_bg_tiles[im_idx][i]

            # used for determining which channel of the network output to work with.
            class_idx = project_classes.index(classname)
            """
            Shapes:
                output: (2, 500, 500) (bg/fg, height, width)
                class_name: string
                fg_tile: (500, 500)
                bg_tile: (500, 500)
            """

            fg_tile = fg_tile.cuda()
            bg_tile = bg_tile.cuda()
            mask = fg_tile + bg_tile
            class_idx *= 2 # as each class has two channels 
            class_output = output[class_idx:class_idx+2]
            softmaxed = softmax(class_output, 1)

            # just the foreground probability.
            foreground_probs = softmaxed[0]
            # remove any of the predictions for which we don't have ground truth
            # Set outputs to 0 where annotation undefined so that
            # The network can predict whatever it wants without any penalty.

            # add batch dimension to allow loss computation.
            class_output = torch.unsqueeze(class_output, 0)
            fg_tile = torch.unsqueeze(fg_tile, 0)
            masked_output = class_output * mask

            class_loss = combined_loss(masked_output, fg_tile)
            if batch_loss is None:
                batch_loss = class_loss
            else:
                batch_loss += class_loss

            foreground_probs *= mask
            predicted = foreground_probs > 0.5

            # we only want to calculate metrics on the
            # part of the predictions for which annotations are defined
            # so remove all predictions and foreground labels where
            # we didn't have any annotation.
            defined_list = mask.view(-1)
            preds_list = predicted.view(-1)[defined_list > 0]
            foregrounds_list = fg_tile.view(-1)[defined_list > 0]

            # # calculate all the false positives, false negatives etc
            tps += torch.sum((foregrounds_list == 1) * (preds_list == 1)).cpu().numpy()
            tns += torch.sum((foregrounds_list == 0) * (preds_list == 0)).cpu().numpy()
            fps += torch.sum((foregrounds_list == 0) * (preds_list == 1)).cpu().numpy()
            fns += torch.sum((foregrounds_list == 1) * (preds_list == 0)).cpu().numpy()
            defined_total += torch.sum(defined_list > 0).cpu().numpy()
    return batch_loss, tps, tns, fps, fns, defined_total 


def get_batch_loss(outputs, batch_fg_tiles, batch_bg_tiles, batch_classes, project_classes):
    """

        outputs - predictions from neural network (not softmaxed)
        batch_fg_tiles - list of tiles, each tile is binary map of foreground annotation
        batch_bg_tiles - list of tiles, each tile is binary map of background annotation

        returns
            batch_loss - loss used to update the network
            tps - true positives for batch
            tns - true negatives for batch
            fps - false positives for batch
            fns - false negatives for batch
            defined_total - number of pixels with annotation defined.
    """

    tps = 0
    fps = 0
    tns = 0
    fns = 0
    defined_total = 0
    class_losses = [] # loss for each class
    for unique_class in project_classes:

        # for each class we need to get a tensor with shape
        # [batch_size, 2 (bg,fg), h, w]

        # fg,bg pairs related to this class for each im_tile in the batch
        class_outputs = []

        # The fg tiles (ground truth)
        fg_tiles = []
        masks = [] # and regions of the image that were annotated.

        # go through each instance in the batch.
        for im_idx in range(outputs.shape[0]):

            # go through each class for this instance.
            for i, classname in enumerate(batch_classes[im_idx]):
                
                # if the classname is the class we are interested in
                if classname == unique_class:

                    # foregorund and background channels
                    fg_tile = batch_fg_tiles[im_idx][i]
                    bg_tile = batch_bg_tiles[im_idx][i]
                    mask = fg_tile + bg_tile
                    masks.append(mask)
                    fg_tiles.append(fg_tile)

                    class_idx = project_classes.index(classname) * 2 # posiion in output.
                    class_outputs.append(outputs[im_idx][class_idx:class_idx+2])

        if not len(fg_tiles):
            continue

        fg_tiles = torch.stack(fg_tiles).cuda()
        masks = torch.stack(masks).cuda()
        class_outputs = torch.stack(class_outputs)
        softmaxed = softmax(class_outputs, 1)
        # just the foreground probability.
        foreground_probs = softmaxed[:, 1]
        # remove any of the predictions for which we don't have ground truth
        # Set outputs to 0 where annotation undefined so that
        # The network can predict whatever it wants without any penalty.
        class_outputs[:, 0] *= masks
        class_outputs[:, 1] *= masks
        class_loss = combined_loss(class_outputs, fg_tiles)

        class_losses.append(class_loss)
        foreground_probs *= masks
        class_predicted = foreground_probs > 0.5
        # we only want to calculate metrics on the
        # part of the predictions for which annotations are defined
        # so remove all predictions and foreground labels where
        # we didn't have any annotation.
        defined_list = masks.view(-1)
        preds_list = class_predicted.view(-1)[defined_list > 0]
        foregrounds_list = fg_tiles.view(-1)[defined_list > 0]

        # # calculate all the false positives, false negatives etc
        tps += torch.sum((foregrounds_list == 1) * (preds_list == 1)).cpu().numpy()
        tns += torch.sum((foregrounds_list == 0) * (preds_list == 0)).cpu().numpy()
        fps += torch.sum((foregrounds_list == 0) * (preds_list == 1)).cpu().numpy()
        fns += torch.sum((foregrounds_list == 1) * (preds_list == 0)).cpu().numpy()
        defined_total += torch.sum(defined_list > 0).cpu().numpy()

    return torch.mean(torch.stack(class_losses)), tps, tns, fps, fns, defined_total 
