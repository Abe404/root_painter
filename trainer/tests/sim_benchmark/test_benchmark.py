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


def _make_synthetic_dataset(tmpdir, num_images=4, size=750):
    """Create synthetic RGB images with circle foregrounds and matching GT masks."""
    dataset_dir = os.path.join(tmpdir, 'dataset')
    gt_dir = os.path.join(tmpdir, 'ground_truth')
    os.makedirs(dataset_dir)
    os.makedirs(gt_dir)

    rng = np.random.RandomState(42)
    y, x = np.ogrid[:size, :size]

    for i in range(num_images):
        # random circle position and radius
        cx = rng.randint(200, size - 200)
        cy = rng.randint(200, size - 200)
        r = rng.randint(80, 150)

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
        dataset_dir, gt_dir = _make_synthetic_dataset(tmpdir)
        output_dir = os.path.join(tmpdir, 'output')

        results = run_benchmark(
            dataset_dir=dataset_dir,
            ground_truth_dir=gt_dir,
            output_dir=output_dir,
            num_initial_points=15,
            num_corrective_points=10,
            brush_radius=10,
            epochs_per_round=2,
            num_rounds=2,
            batch_size=2,
            in_w=92,
            out_w=20,
            lr=0.01,
            seed=1,
            min_epoch_tiles=20,
        )

        assert len(results) == 2
        first_f1 = results[0]['f1']
        last_f1 = results[-1]['f1']
        print(f"\nF1 round 0: {first_f1:.4f}, F1 round 1: {last_f1:.4f}")
        assert last_f1 > first_f1, (
            f"Expected improvement: round 0 F1={first_f1:.4f}, "
            f"round 1 F1={last_f1:.4f}"
        )
