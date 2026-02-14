"""
Benchmark orchestrator for simulated corrective annotation.

Runs a loop: annotate → train → segment → evaluate → correct → repeat,
using ground truth masks to simulate a human annotator.
"""
import sys
import os
import time

import numpy as np
import torch
from skimage.io import imsave

# Ensure trainer/src is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
_tests_dir = os.path.dirname(_this_dir)
_src_dir = os.path.join(os.path.dirname(_tests_dir), 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import model_utils
import im_utils
from unet import UNetGNRes
from datasets import TrainDataset
from multi_epoch.multi_epoch_loader import MultiEpochsDataLoader
from metrics import get_metrics

from sim_benchmark.sim_user import initial_annotation, corrective_annotation


def _load_ground_truths(gt_dir, fnames):
    """Load ground truth .npy masks, keyed by stem name."""
    gts = {}
    for fname in fnames:
        stem = os.path.splitext(fname)[0]
        gt_path = os.path.join(gt_dir, stem + '.npy')
        gts[stem] = np.load(gt_path)
    return gts


def _save_annot(annot, path):
    """Save an RGBA annotation array as a PNG."""
    imsave(path, annot, check_contrast=False)


def run_benchmark(dataset_dir, ground_truth_dir, output_dir,
                  num_initial_points=20, num_corrective_points=10,
                  brush_radius=15, epochs_per_round=3, num_rounds=3,
                  batch_size=6, in_w=572, out_w=500, lr=0.01, seed=1,
                  min_epoch_tiles=612):
    """Run a simulated corrective annotation benchmark.

    Args:
        dataset_dir: directory containing RGB images (.jpg/.png)
        ground_truth_dir: directory containing .npy binary masks (same stems)
        output_dir: directory for saving annotations and results
        num_initial_points: annotation points for round 0
        num_corrective_points: correction points for rounds 1+
        brush_radius: radius of simulated brush
        epochs_per_round: training epochs per round
        num_rounds: total rounds (1 initial + N-1 corrective)
        batch_size: training batch size
        in_w: UNet input tile width
        out_w: UNet output tile width
        lr: learning rate
        seed: random seed
        min_epoch_tiles: minimum number of samples per epoch (default 612)

    Returns:
        list of per-round metric dicts (with f1, precision, recall, etc.)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    train_annot_dir = os.path.join(output_dir, 'annotations', 'train')
    val_annot_dir = os.path.join(output_dir, 'annotations', 'val')
    os.makedirs(train_annot_dir, exist_ok=True)
    os.makedirs(val_annot_dir, exist_ok=True)

    # list dataset images
    from im_utils import is_photo
    fnames = sorted(f for f in os.listdir(dataset_dir) if is_photo(f))
    assert len(fnames) >= 2, f"Need at least 2 images, got {len(fnames)}"

    # split: first image is val, rest are train
    val_fnames = fnames[:1]
    train_fnames = fnames[1:]

    # load ground truths
    gts = _load_ground_truths(ground_truth_dir, fnames)

    # init model
    model = UNetGNRes()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.99, nesterov=True)

    # track annotations per image (keyed by stem)
    annots = {}
    results = []

    for round_idx in range(num_rounds):
        round_seed = seed + round_idx
        print(f"\n=== Round {round_idx} ===")

        if round_idx == 0:
            # initial annotation
            for fname in fnames:
                stem = os.path.splitext(fname)[0]
                gt = gts[stem]
                annot = initial_annotation(gt, num_initial_points,
                                           brush_radius, seed=round_seed)
                annots[stem] = annot
        else:
            # corrective annotation: segment then correct
            model.eval()
            with torch.no_grad():
                for fname in fnames:
                    stem = os.path.splitext(fname)[0]
                    gt = gts[stem]
                    image = im_utils.load_image(
                        os.path.join(dataset_dir, fname))
                    image, pad_settings = im_utils.pad_to_min(
                        image, min_w=in_w, min_h=in_w)
                    pred = model_utils.unet_segment(
                        model, image, batch_size, in_w, out_w, threshold=0.5)
                    pred = im_utils.crop_from_pad_settings(pred, pad_settings)
                    annots[stem] = corrective_annotation(
                        gt, pred, annots[stem],
                        num_corrective_points, brush_radius, seed=round_seed)

        # save annotations to disk
        for fname in train_fnames:
            stem = os.path.splitext(fname)[0]
            _save_annot(annots[stem],
                        os.path.join(train_annot_dir, stem + '.png'))
        for fname in val_fnames:
            stem = os.path.splitext(fname)[0]
            _save_annot(annots[stem],
                        os.path.join(val_annot_dir, stem + '.png'))

        # train
        train_set = TrainDataset(train_annot_dir, dataset_dir, in_w, out_w,
                                 min_epoch_tiles=min_epoch_tiles)
        train_loader = MultiEpochsDataLoader(
            train_set, batch_size, shuffle=True,
            num_workers=0, drop_last=False, pin_memory=False)

        for ep in range(epochs_per_round):
            start = time.time()
            train_result = model_utils.epoch(
                model, train_loader, batch_size,
                optimizer=optimizer, step_callback=None, stop_fn=None)
            duration = time.time() - start
            if train_result is None:
                break
            tps, fps, tns, fns, defined_sum, avg_loss = train_result
            train_metrics = get_metrics(tps, fps, tns, fns,
                                        defined_sum, duration, avg_loss)
            print(f"  epoch {ep}: train f1={train_metrics['f1']:.4f} "
                  f"loss={avg_loss:.4f}")

        # validate
        model.eval()
        val_metrics = model_utils.get_val_metrics(
            model, val_annot_dir, dataset_dir, in_w, out_w, bs=batch_size)
        print(f"  Round {round_idx} val: f1={val_metrics['f1']:.4f} "
              f"precision={val_metrics['precision']:.4f} "
              f"recall={val_metrics['recall']:.4f}")
        results.append(val_metrics)

    return results
