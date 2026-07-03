"""
Tests for Extract Composites (gen_composite).

Extract Composites overlays a segmentation onto the original photo. It must
work for both segmentation formats produced by Segment Folder:

  - RootPainter Default: RGBA, cyan foreground on a transparent background.
  - RhizoVision Explorer: single channel, black foreground on white.

A RhizoVision Explorer segmentation is a 2-D (H, W) image, which used to crash
gen_composite when it read annot.shape[2], freezing Extract Composites on the
first file. These tests cover both formats and both the darwin (subsample) and
non-darwin (resize) downscaling paths.
"""
import os
import sys
import tempfile

import numpy as np
import pytest
from skimage.io import imsave

test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src', 'main', 'python')
sys.path.insert(0, src_dir)

import im_utils
from im_utils import gen_composite

# A constant, non-red photo so any red pixel in the output is from the overlay.
PHOTO_COLOR = [100, 150, 200]
HEIGHT, WIDTH = 40, 30
# A small foreground block so the foreground is a clear minority of the image.
# This also catches inverted polarity (which would paint most of the image).
FG_ROWS = slice(8, 16)
FG_COLS = slice(6, 12)


def write_photo(im_dir, name):
    photo = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    photo[:, :] = PHOTO_COLOR
    imsave(os.path.join(im_dir, name + '.jpg'), photo, check_contrast=False)


def write_rootpainter_seg(seg_dir, name):
    """ RGBA, cyan foreground with alpha, transparent background. """
    seg = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    seg[FG_ROWS, FG_COLS] = [0, 255, 255, 178]
    imsave(os.path.join(seg_dir, name + '.png'), seg, check_contrast=False)


def write_rhizovision_seg(seg_dir, name):
    """ Single channel, black foreground (0) on white background (255). """
    seg = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    seg[FG_ROWS, FG_COLS] = 0
    imsave(os.path.join(seg_dir, name + '.png'), seg, check_contrast=False)


@pytest.mark.parametrize('platform', ['darwin', 'linux'])
@pytest.mark.parametrize('write_seg', [write_rootpainter_seg, write_rhizovision_seg])
def test_gen_composite_overlays_foreground(monkeypatch, platform, write_seg):
    monkeypatch.setattr(im_utils.sys, 'platform', platform)
    with tempfile.TemporaryDirectory() as tmpdir:
        seg_dir = os.path.join(tmpdir, 'seg')
        im_dir = os.path.join(tmpdir, 'im')
        comp_dir = os.path.join(tmpdir, 'comp')
        for d in (seg_dir, im_dir, comp_dir):
            os.makedirs(d)

        write_photo(im_dir, 'root1')
        write_seg(seg_dir, 'root1')

        gen_composite(seg_dir, im_dir, comp_dir, 'root1.png')

        out_path = os.path.join(comp_dir, 'root1.jpg')
        assert os.path.isfile(out_path)

        from skimage.io import imread
        comp = imread(out_path)
        assert comp.dtype == np.uint8

        # The composite is the photo beside a copy with the foreground in red.
        # Width < height * 1.2 here, so the two halves are stacked horizontally.
        # jpg is lossy, so match red with a tolerance rather than exactly.
        red = (comp[:, :, 0] > 180) & (comp[:, :, 1] < 80) & (comp[:, :, 2] < 80)
        half = comp.shape[1] // 2
        left, right = red[:, :half], red[:, half:]

        # The untouched original is not red; the foreground lands in the overlay.
        assert left.sum() < 5
        assert right.sum() > 0
        # Foreground is a small block, so it must stay a minority of the half.
        # Inverted polarity would paint most of the half instead.
        assert right.sum() < right.size / 2
