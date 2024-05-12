
import sys; import os; import numpy as np
# Add the parent directory to sys.path
parent_dir = os.path.abspath('../')
sys.path.insert(0, parent_dir)

import torch
from torch.nn.functional import softmax, binary_cross_entropy 
from torch.nn.functional import cross_entropy
from loss import combined_loss as criterion
from skimage.io import imsave
from skimage import img_as_uint
from loss import dice_loss, dice_loss2




def get_acc(pred, true):
    if hasattr(pred, 'numpy'):
        pred = pred.numpy()
    if hasattr(true, 'numpy'):
        true = true.numpy()

    assert np.min(pred) >= 0
    assert np.max(pred) <= 1

    assert np.min(true) >= 0
    assert np.max(true) <= 1

    output_bool = pred.reshape(-1) >= 0.5
    target_bool = true.reshape(-1) >= 0.5
    return (np.sum(output_bool == target_bool) / true.size)
 
def test_dice_loss_goes_to_zero_no_mask():
    test_input = np.zeros((1, 3, 572, 572))
    test_input[:, :, 100:-100,100:-100] = 1.0
    test_input = torch.from_numpy(test_input)
    target = (test_input[:, 0, 36:-36, 36:-36] > 0.5).long()

    # batch, class, height, width
    preds = torch.zeros(1, 2, 500, 500)
    print('target shape', target.shape)
    print('preds shape', preds.shape)
    preds[0, 1] = torch.clone(target[0]) * (2**60)# this is the 'foreground probability'
    preds[0, 0] = (1 - torch.clone(target[0])) * (2**60)  # this is the 'background probability'

    print('preds shape', preds.shape)
    print('target shape', target.shape)
    print('dice loss 1', dice_loss(preds, target).item())
    print('cross_entropy', cross_entropy(preds, target).item())
    cbl = criterion(preds, target).item()
    print('combiend loss', cbl)

    output = softmax(preds, 1)[:, 1] # just fg probability
    output = output.detach().cpu().numpy()
    output = output[0] # singke image.

    print('output mean', output.mean())
    print('output unique', np.unique(output))
    print('target mean', target.reshape(-1).float().mean())
    print('target unique', np.unique(target))
    print('abs difference', np.absolute(output - target[0].numpy()))
    print('output shape', output.shape) 
    print('target shape', target.shape) 

    assert np.mean(np.absolute(output - target[0].numpy())) < 0.000001

    acc =  get_acc(output, target)
    print('accuracy', acc)
    assert acc == 1.0, acc

    assert cbl < 0.0001


def test_dice_loss_goes_to_zero_with_mask():
    test_input = np.zeros((1, 3, 572, 572))
    test_input[:, :, 100:-100,100:-100] = 1.0
    test_input = torch.from_numpy(test_input)
    target = (test_input[:, 0, 36:-36, 36:-36] > 0.5).long()
    print('target mean', target.reshape(-1).float().mean())
    print('target unique', np.unique(target))

    defined = np.zeros((1, 500, 500))
    defined[:, :250] = 1
    defined = torch.from_numpy(defined).float()
   
    # batch, class, height, width
    preds = torch.zeros(1, 2, 500, 500)
    print('target shape', target.shape)
    print('preds shape', preds.shape)

    preds[0, 1] = torch.clone(target[0]) * (2**60) # this is the 'foreground probability'
    preds[0, 0] = (1 - torch.clone(target[0])) * (2**60)  # this is the 'background probability'

    preds_for_loss = torch.clone(preds)
    preds_for_loss[:, 0] *= defined
    preds_for_loss[:, 1] *= defined

    print('dice loss 1', dice_loss(preds_for_loss, target).item())
    print('cross_entropy', cross_entropy(preds_for_loss, target).item())
    cbl = criterion(preds_for_loss, target).item()
    print('combiend loss', cbl)

    output = softmax(preds, 1)[:, 1] # just fg probability

    print('dice loss 2', dice_loss2(output, target).item())

    output = output.detach().cpu().numpy()

    output = output[0] # singke image.
    print('output mean', output.mean())
    print('output unique', np.unique(output))
    print('target mean', target.reshape(-1).float().mean())
    print('target unique', np.unique(target))
    print('abs difference', np.absolute(output - target[0].numpy()))
    print('output shape', output.shape) 
    print('target shape', target.shape) 

    acc =  get_acc(output, target)
    print('accuracy', acc)
    assert acc == 1.0, acc

    # Cannot assert loss is 0 here because the masked regions of the output lead to non-zero loss. 
    # assert cbl < 0.00001
