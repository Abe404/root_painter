"""Generate a demo video of the simulated corrective annotation benchmark.

Creates synthetic images with varied shapes, runs the full benchmark loop,
and outputs an animated GIF showing the sim user annotating in real time
while the model learns.

Usage:
    cd trainer/tests
    python sim_benchmark/make_demo.py
"""
import sys
import os
import tempfile

import numpy as np
from PIL import Image

this_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(this_dir)
src_dir = os.path.join(os.path.dirname(tests_dir), 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, tests_dir)

from sim_benchmark.benchmark import run_benchmark


def make_demo_dataset(tmpdir, num_images=8, size=200):
    """Create varied synthetic images — circles, ellipses, blobs."""
    dataset_dir = os.path.join(tmpdir, 'dataset')
    gt_dir = os.path.join(tmpdir, 'ground_truth')
    os.makedirs(dataset_dir)
    os.makedirs(gt_dir)

    rng = np.random.RandomState(42)
    y, x = np.ogrid[:size, :size]

    shapes = [
        # (cx, cy, rx, ry, angle) — ellipse params
        (100, 100, 40, 40, 0),      # centered circle
        (60, 140, 35, 25, 0),       # wide ellipse, off-center
        (140, 60, 25, 45, 0),       # tall ellipse, off-center
        (100, 100, 55, 30, 30),     # tilted ellipse
        (70, 70, 30, 30, 0),        # small circle, corner
        (130, 130, 45, 45, 0),      # large circle, corner
        (100, 100, 20, 60, 60),     # very elongated
        (100, 100, 50, 50, 0),      # big centered circle
    ]

    for i in range(num_images):
        cx, cy, rx, ry, angle = shapes[i % len(shapes)]
        # Add some randomness
        cx += rng.randint(-15, 15)
        cy += rng.randint(-15, 15)

        # Rotated ellipse mask
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        dx = x - cx
        dy = y - cy
        rx_term = (cos_a * dx + sin_a * dy) / max(1, rx)
        ry_term = (-sin_a * dx + cos_a * dy) / max(1, ry)
        mask = (rx_term ** 2 + ry_term ** 2 <= 1).astype(np.uint8)

        # RGB: dark noisy BG, bright FG with color variation
        bg_color = rng.randint(15, 50, 3)
        fg_color = rng.randint(160, 240, 3)
        image = np.full((size, size, 3), bg_color, dtype=np.uint8)
        image += rng.randint(0, 25, (size, size, 3), dtype=np.uint8)
        fg_noise = rng.randint(-20, 20, (np.sum(mask), 3))
        image[mask == 1] = np.clip(fg_color + fg_noise, 0, 255).astype(np.uint8)

        fname = f'demo_{i:03d}'
        Image.fromarray(image).save(os.path.join(dataset_dir, fname + '.jpg'))
        np.save(os.path.join(gt_dir, fname + '.npy'), mask)

    return dataset_dir, gt_dir


def main():
    output_dir = os.path.join(this_dir, 'demo_output')
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Generating synthetic dataset...")
        dataset_dir, gt_dir = make_demo_dataset(tmpdir, num_images=8, size=200)

        print("Running benchmark with video recording...\n")
        results = run_benchmark(
            dataset_dir=dataset_dir,
            ground_truth_dir=gt_dir,
            output_dir=output_dir,
            min_initial_images=2,
            corrective_f1_threshold=0.2,
            epochs_between_images=1,
            batch_size=2,
            in_w=172,
            out_w=100,
            lr=0.01,
            seed=1,
            min_epoch_tiles=10,
            save_video=True,
        )

    first_f1 = results[0]['val_f1']
    last_f1 = results[-1]['val_f1']
    print(f"\nF1: {first_f1:.3f} -> {last_f1:.3f}")

    gif_path = os.path.join(output_dir, 'demo.gif')
    if os.path.exists(gif_path):
        size_mb = os.path.getsize(gif_path) / 1024 / 1024
        print(f"GIF: {gif_path} ({size_mb:.1f} MB)")
        os.system(f'open "{gif_path}"')
    else:
        print("GIF not created — opening frames instead")
        os.system(f'open "{os.path.join(output_dir, "frames")}"')


if __name__ == '__main__':
    main()
