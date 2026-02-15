"""
Integration test for the simulated corrective annotation benchmark.

Uses synthetic data (programmatically generated images with known ground truth)
so no Zenodo download is needed.
"""
import sys
import os
import tempfile

import numpy as np
from PIL import Image

test_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(test_dir)
src_dir = os.path.join(os.path.dirname(tests_dir), 'src')
sys.path.insert(0, src_dir)

from sim_benchmark.benchmark import run_benchmark


def make_synthetic_dataset(tmpdir, num_images=4, size=200):
    """Create synthetic RGB images with circle foregrounds and matching GT masks."""
    dataset_dir = os.path.join(tmpdir, 'dataset')
    gt_dir = os.path.join(tmpdir, 'ground_truth')
    os.makedirs(dataset_dir)
    os.makedirs(gt_dir)

    rng = np.random.RandomState(42)
    y, x = np.ogrid[:size, :size]

    for i in range(num_images):
        # random circle position and radius
        cx = rng.randint(size // 4, 3 * size // 4)
        cy = rng.randint(size // 4, 3 * size // 4)
        r = rng.randint(30, size // 4)

        mask = ((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2).astype(np.uint8)

        # RGB image: foreground is bright, background is dark with noise
        image = rng.randint(20, 60, (size, size, 3), dtype=np.uint8)
        image[mask == 1] = rng.randint(180, 240, (np.sum(mask), 3), dtype=np.uint8)

        fname = f'synth_{i:03d}'
        Image.fromarray(image).save(os.path.join(dataset_dir, fname + '.jpg'))
        np.save(os.path.join(gt_dir, fname + '.npy'), mask)

    return dataset_dir, gt_dir


def test_benchmark_improves():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir, gt_dir = make_synthetic_dataset(tmpdir)
        output_dir = os.path.join(tmpdir, 'output')

        results = run_benchmark(
            dataset_dir=dataset_dir,
            ground_truth_dir=gt_dir,
            output_dir=output_dir,
            min_initial_images=2,
            initial_coverage=0.05,
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

        assert len(results) >= 2
        first_f1 = results[0]['val_f1']
        last_f1 = results[-1]['val_f1']
        print(f"\nF1 first image: {first_f1:.4f}, F1 last image: {last_f1:.4f}")
        assert last_f1 > first_f1, (
            f"Expected improvement: first F1={first_f1:.4f}, "
            f"last F1={last_f1:.4f}"
        )

        # Verify frames were produced and copy to persistent location
        frames_dir = os.path.join(output_dir, 'frames')
        assert os.path.isdir(frames_dir), "verification frames not created"
        import shutil
        persistent_frames = os.path.join(test_dir, 'frames')
        if os.path.exists(persistent_frames):
            shutil.rmtree(persistent_frames)
        shutil.copytree(frames_dir, persistent_frames)
        print(f"\nVerification frames copied to {persistent_frames}/")
