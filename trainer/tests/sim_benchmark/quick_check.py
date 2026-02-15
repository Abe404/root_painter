"""Quick visual check — 3 synthetic images, saves frames only."""
import sys, os
import shutil
import numpy as np
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(this_dir)
src_dir = os.path.join(os.path.dirname(tests_dir), 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, tests_dir)

import model_utils
import im_utils
from unet import UNetGNRes
from sim_benchmark.sim_user import initial_annotation
from sim_benchmark.video import render_trajectory_frames, save_frames


IN_W = 172
OUT_W = 100


def make_images(size=200):
    """Three varied shapes on noisy backgrounds."""
    rng = np.random.RandomState(42)
    y, x = np.ogrid[:size, :size]
    images = []

    # 1: circle
    mask = ((x - 100)**2 + (y - 100)**2 <= 40**2).astype(np.uint8)
    rgb = rng.randint(20, 50, (size, size, 3), dtype=np.uint8)
    rgb[mask == 1] = rng.randint(180, 230, (np.sum(mask), 3), dtype=np.uint8)
    images.append((rgb, mask, 'circle'))

    # 2: wide ellipse, off-center
    ex = (((x - 60.0) / 50)**2 + ((y - 130.0) / 25)**2 <= 1).astype(np.uint8)
    rgb2 = rng.randint(15, 45, (size, size, 3), dtype=np.uint8)
    rgb2[ex == 1] = rng.randint(160, 240, (np.sum(ex), 3), dtype=np.uint8)
    images.append((rgb2, ex, 'ellipse'))

    # 3: two small blobs
    b1 = ((x - 60)**2 + (y - 60)**2 <= 25**2)
    b2 = ((x - 140)**2 + (y - 140)**2 <= 30**2)
    blobs = (b1 | b2).astype(np.uint8)
    rgb3 = rng.randint(25, 55, (size, size, 3), dtype=np.uint8)
    rgb3[blobs == 1] = rng.randint(170, 235, (np.sum(blobs), 3), dtype=np.uint8)
    images.append((rgb3, blobs, 'blobs'))

    return images


def segment(model, rgb):
    """Segment an image with the current model, like the trainer does."""
    image, pad_settings = im_utils.pad_to_min(rgb, min_w=IN_W, min_h=IN_W)
    with torch.no_grad():
        pred = model_utils.unet_segment(
            model, image, 2, IN_W, OUT_W, threshold=0.5)
    return im_utils.crop_from_pad_settings(pred, pad_settings)


def main():
    out_dir = os.path.join(this_dir, 'quick_output')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Fresh model with random weights, same as real flow
    model = UNetGNRes()
    model.to(model_utils.device)
    model.eval()

    all_frames = []
    sim_time = 0.0

    for i, (rgb, gt, name) in enumerate(make_images()):
        pred = segment(model, rgb)
        annot, traj = initial_annotation(gt, seed=i + 1)

        total_time = sum(e.get('dt', 0) for e in traj)
        n_strokes = sum(1 for j, e in enumerate(traj)
                        if e['painting'] and (j == 0 or not traj[j-1]['painting']))
        print(f"{name}: {total_time:.1f}s, {len(traj)} events, {n_strokes} strokes")

        frames, sim_time = render_trajectory_frames(
            rgb, gt, traj, pred, name, 'initial', 0.0,
            image_index=i, time_offset=sim_time, fps=6)
        all_frames.extend(frames)

    if all_frames:
        paths = save_frames(all_frames, out_dir)
        print(f"\nSaved {len(paths)} frames to {out_dir}/frames/")
    else:
        print("No frames rendered")


if __name__ == '__main__':
    main()
