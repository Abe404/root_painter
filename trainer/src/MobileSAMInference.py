"""
MobileSAM inference for RootPainter.

Segments images using a pre-trained MobileSAM ViT-Tiny model.
Inference only, no training.

Copyright (C) 2026 Abraham George Smith

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
import torch
import torch.nn.functional as F

from mobile_sam import build_sam_vit_t
from model_utils import get_device
import im_utils

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TILE_SIZE = 1024

_cached_model = None
_cached_path = None


def _load_model(checkpoint_path, device):
    model = build_sam_vit_t(checkpoint=None)
    weights = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Strip torch.compile prefix if present (e.g. "model._orig_mod.")
    prefix = 'model._orig_mod.'
    if any(k.startswith(prefix) for k in weights):
        weights = {k[len(prefix):]: v for k, v in weights.items()}
    if not any(k.startswith('image_encoder') for k in weights):
        raise ValueError(
            f"{checkpoint_path} does not appear to be a MobileSAM checkpoint"
        )
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model


def _get_model(checkpoint_path, device):
    global _cached_model, _cached_path
    if _cached_path != checkpoint_path:
        _cached_model = _load_model(checkpoint_path, device)
        _cached_path = checkpoint_path
    return _cached_model


def _forward(model, x):
    """
    MobileSAM forward pass without prompts.
    x: (1, 3, 1024, 1024) normalized tensor.
    Returns (1, 1, 1024, 1024) logits.
    Matches seg_mb_sam.py Model.forward for the no-padding case.
    """
    img_embed = model.image_encoder(x)
    prompt_embeds = model.prompt_encoder(None, None, None)
    masks, _ = model.mask_decoder(
        img_embed,
        model.prompt_encoder.get_dense_pe(),
        *prompt_embeds,
        False
    )
    return F.interpolate(masks, size=x.shape[2:],
                         mode='bilinear', align_corners=False)


def mobilesam_segment_image(checkpoint_path, image):
    """
    Segment an image using MobileSAM.

    Args:
        checkpoint_path: path to .pth checkpoint
        image: numpy array HxWx3 uint8 (from im_utils.load_image)

    Returns:
        binary mask, numpy array HxW, int (0 or 1)
    """
    device = get_device()
    model = _get_model(checkpoint_path, device)

    image = image.astype(np.float32) / 255.0
    orig_h, orig_w = image.shape[:2]

    # Pad to at least 1024x1024 with zeros.
    # After ImageNet normalization, zero pixels become IMAGENET_MIN,
    # matching the padding used in the seg repo's training pipeline.
    pad_h = max(0, TILE_SIZE - orig_h)
    pad_w = max(0, TILE_SIZE - orig_w)
    if pad_h or pad_w:
        h_before = pad_h // 2
        h_after = pad_h - h_before
        w_before = pad_w // 2
        w_after = pad_w - w_before
        image = np.pad(image,
                       ((h_before, h_after), (w_before, w_after), (0, 0)),
                       mode='constant', constant_values=0)
        pad_settings = ((h_before, h_after), (w_before, w_after), (0, 0))
    else:
        pad_settings = ((0, 0), (0, 0), (0, 0))

    tiles, coords = im_utils.get_tiles(image,
                                        in_tile_shape=(TILE_SIZE, TILE_SIZE, 3),
                                        out_tile_shape=(TILE_SIZE, TILE_SIZE))

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    output_tiles = []
    with torch.no_grad():
        for tile in tiles:
            t = torch.from_numpy(np.moveaxis(tile, -1, 0)).unsqueeze(0).to(device)
            t = (t - mean) / std
            logits = _forward(model, t)
            predicted = (logits >= 0).squeeze(0).squeeze(0)
            output_tiles.append(predicted.cpu().numpy())

    reconstructed = im_utils.reconstruct_from_tiles(
        output_tiles, coords, image.shape[:2])

    reconstructed = im_utils.crop_from_pad_settings(reconstructed, pad_settings)
    return reconstructed.astype(int)
