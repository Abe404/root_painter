"""
Benchmark orchestrator for simulated corrective annotation.

Follows the RootPainter protocol:
- Annotate 2+ clear examples before starting training
- Switch to corrective when model is approximately predicting the object
- Corrective phase: correct all clear errors, skip good images
- Train/val split follows the painter's logic
- Best model tracked by validation F1, used for segmentation
"""
import sys
import os
import copy

import numpy as np
import torch
from skimage.io import imsave

# Ensure trainer/src is importable
this_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(this_dir)
src_dir = os.path.join(os.path.dirname(tests_dir), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import model_utils
import im_utils
from unet import UNetGNRes
from datasets import TrainDataset
from multi_epoch.multi_epoch_loader import MultiEpochsDataLoader
from metrics import get_metrics

from sim_benchmark.sim_user import initial_annotation, corrective_annotation
from sim_benchmark.video import make_frame, save_frames, render_trajectory_frames


def load_ground_truths(gt_dir, fnames):
    """Load ground truth .npy masks, keyed by stem name."""
    gts = {}
    for fname in fnames:
        stem = os.path.splitext(fname)[0]
        gt_path = os.path.join(gt_dir, stem + '.npy')
        gts[stem] = np.load(gt_path)
    return gts


def save_annot(annot, path):
    """Save an RGBA annotation array as a PNG."""
    imsave(path, annot, check_contrast=False)


def get_annot_target_dir(train_annot_dir, val_annot_dir):
    """Decide whether new annotation goes to train or val.

    Replicates painter's file_utils.get_new_annot_target_dir logic.
    """
    train_pngs = [f for f in os.listdir(train_annot_dir)
                  if os.path.splitext(f)[1] == '.png']
    val_pngs = [f for f in os.listdir(val_annot_dir)
                if os.path.splitext(f)[1] == '.png']
    num_train = len(train_pngs)
    num_val = len(val_pngs)
    if num_train == 0 and num_val > 0:
        return train_annot_dir
    if num_train > 0 and num_val == 0:
        return val_annot_dir
    if num_train >= num_val * 5:
        return val_annot_dir
    return train_annot_dir


def run_benchmark(dataset_dir, ground_truth_dir, output_dir,
                  min_initial_images=2, initial_coverage=0.05,
                  corrective_f1_threshold=0.2,
                  epochs_between_images=2,
                  batch_size=6, in_w=572, out_w=500, lr=0.01, seed=1,
                  min_epoch_tiles=20, save_video=False):
    """Run a simulated corrective annotation benchmark.

    Follows the RootPainter protocol:
    1. Annotate min_initial_images clear examples, then start training
    2. Continue initial annotation until model is useful (val F1 > threshold)
    3. Switch to corrective: correct all clear errors, skip good images

    Args:
        dataset_dir: directory containing RGB images (.jpg/.png)
        ground_truth_dir: directory containing .npy binary masks (same stems)
        output_dir: directory for saving annotations and results
        min_initial_images: images annotated before training starts
        initial_coverage: fraction of each class region to annotate initially
        corrective_f1_threshold: val F1 needed to switch to corrective
        epochs_between_images: training epochs after each annotation
        batch_size: training batch size
        in_w: UNet input tile width
        out_w: UNet output tile width
        lr: learning rate
        seed: random seed
        min_epoch_tiles: minimum tiles per training epoch
        save_video: if True, save verification frames to output_dir

    Returns:
        list of per-image metric dicts
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

    # load ground truths
    gts = load_ground_truths(ground_truth_dir, fnames)

    # init model
    model = UNetGNRes()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.99, nesterov=True)
    best_model_state = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    num_annotated = 0
    corrective_mode = False

    results = []
    frames = []
    video_data = []  # collected per-image for deferred rendering

    for i, fname in enumerate(fnames):
        stem = os.path.splitext(fname)[0]
        gt = gts[stem]
        image_seed = seed + i
        pred = None
        print(f"\n--- Image {i}: {stem} ---")

        # Decide phase: initial until model is useful
        if num_annotated < min_initial_images:
            phase = 'initial'
        elif not corrective_mode and best_f1 < corrective_f1_threshold:
            phase = 'initial'
        else:
            if not corrective_mode:
                print(f"  Switching to corrective (val F1={best_f1:.3f})")
            corrective_mode = True
            phase = 'corrective'

        if phase == 'initial':
            annot, traj = initial_annotation(gt, coverage=initial_coverage,
                                             seed=image_seed)
            print(f"  Initial annotation (coverage={initial_coverage})")
        else:
            # Segment with best model
            seg_model = UNetGNRes()
            seg_model.load_state_dict(best_model_state)
            seg_model.to(model_utils.device)
            seg_model.eval()

            image = im_utils.load_image(os.path.join(dataset_dir, fname))
            image, pad_settings = im_utils.pad_to_min(
                image, min_w=in_w, min_h=in_w)
            with torch.no_grad():
                pred = model_utils.unet_segment(
                    seg_model, image, batch_size, in_w, out_w, threshold=0.5)
            pred = im_utils.crop_from_pad_settings(pred, pad_settings)

            annot, traj = corrective_annotation(gt, pred)
            if annot is None:
                print("  Skipping (no clear errors)")
                if save_video:
                    rgb = im_utils.load_image(os.path.join(dataset_dir, fname))
                    frames.append(make_frame(
                        rgb, gt, None, pred, stem, 'skipped', best_f1))
                results.append({
                    'image': stem, 'phase': 'skipped',
                    'val_f1': best_f1,
                })
                continue
            print("  Corrective annotation")

        # Report annotation time
        annot_time = sum(e.get('dt', 0.0) for e in traj)
        print(f"  Annotation time: {annot_time:.1f}s "
              f"({len(traj)} events, {sum(1 for e in traj if e['painting'])} dabs)")

        # Save annotation using painter's train/val split logic
        target_dir = get_annot_target_dir(train_annot_dir, val_annot_dir)
        save_annot(annot, os.path.join(target_dir, stem + '.png'))
        split = 'val' if target_dir == val_annot_dir else 'train'
        num_annotated += 1
        print(f"  Saved to {split} (total annotated: {num_annotated})")

        if save_video:
            rgb = im_utils.load_image(os.path.join(dataset_dir, fname))
            video_data.append({
                'rgb': rgb, 'gt': gt, 'traj': traj, 'pred': pred,
                'annot': annot, 'stem': stem, 'phase': phase,
                'image_index': i, 'val_f1': best_f1,
            })

        # Training starts after min_initial_images are annotated
        if num_annotated < min_initial_images:
            results.append({
                'image': stem, 'phase': phase, 'val_f1': best_f1,
            })
            continue

        # Train epochs on all annotations currently on disk
        has_train = any(f.endswith('.png') for f in os.listdir(train_annot_dir))
        has_val = any(f.endswith('.png') for f in os.listdir(val_annot_dir))

        if has_train:
            train_set = TrainDataset(train_annot_dir, dataset_dir, in_w, out_w,
                                     min_epoch_tiles=min_epoch_tiles)
            train_loader = MultiEpochsDataLoader(
                train_set, batch_size, shuffle=True,
                num_workers=0, drop_last=False, pin_memory=False)

            for ep in range(epochs_between_images):
                train_result = model_utils.epoch(
                    model, train_loader, batch_size,
                    optimizer=optimizer, step_callback=None, stop_fn=None)
                if train_result is None:
                    break
                tps, fps, tns, fns, defined_sum, avg_loss = train_result
                train_metrics = get_metrics(tps, fps, tns, fns,
                                            defined_sum, 0, avg_loss)
                print(f"\n  Epoch {ep}: train f1={train_metrics['f1']:.4f} "
                      f"loss={avg_loss:.4f}")

                # Check if model improved on validation
                if has_val:
                    model.eval()
                    val_metrics = model_utils.get_val_metrics(
                        model, val_annot_dir, dataset_dir, in_w, out_w,
                        bs=batch_size)
                    val_f1 = val_metrics['f1']
                    if not np.isnan(val_f1) and val_f1 > best_f1:
                        best_f1 = val_f1
                        best_model_state = copy.deepcopy(model.state_dict())
                        print(f"  New best model: val f1={best_f1:.4f}")

        results.append({
            'image': stem, 'phase': phase,
            'val_f1': best_f1,
        })

    if save_video and video_data:
        from sim_benchmark.video import trajectory_duration
        video_fps = 6
        video_duration = 60.0  # render at most 60s of simulated time
        sim_time = 0.0
        for d in video_data:
            traj_dur = trajectory_duration(d['traj'])
            if sim_time + traj_dur > video_duration:
                # render partial trajectory up to the time limit
                traj_frames, sim_time = render_trajectory_frames(
                    d['rgb'], d['gt'], d['traj'], d['pred'],
                    d['stem'], d['phase'], d['val_f1'],
                    image_index=d['image_index'], time_offset=sim_time,
                    fps=video_fps, max_time=video_duration)
                frames.extend(traj_frames)
                break
            traj_frames, sim_time = render_trajectory_frames(
                d['rgb'], d['gt'], d['traj'], d['pred'],
                d['stem'], d['phase'], d['val_f1'],
                image_index=d['image_index'], time_offset=sim_time,
                fps=video_fps)
            frames.extend(traj_frames)
            frames.append(make_frame(
                d['rgb'], d['gt'], d['annot'], d['pred'],
                d['stem'], d['phase'], d['val_f1']))

        if frames:
            paths = save_frames(frames, output_dir)
            print(f"\nVideo: {sim_time:.1f}s simulated, "
                  f"{len(paths)} frames at {video_fps} FPS")
            print(f"  Saved to {os.path.dirname(paths[0])}/")
            print(f"  Open in Finder and arrow through")

    return results
