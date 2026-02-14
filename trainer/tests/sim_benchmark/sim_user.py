"""
Simulated annotator for corrective annotation benchmarking.

Creates initial and corrective annotations by comparing ground truth
with model predictions, mimicking a human annotator painting brush
strokes on error regions.
"""
import numpy as np
from sim_benchmark.brush import paint, new_annot


def initial_annotation(ground_truth, num_points=20, brush_radius=5, seed=None):
    """Create initial annotations by sampling points from FG and BG regions.

    Args:
        ground_truth: 2D binary array (1=FG, 0=BG)
        num_points: total number of points to sample (split evenly FG/BG)
        brush_radius: radius of painted circles
        seed: random seed for reproducibility

    Returns:
        (H, W, 4) uint8 annotation array
    """
    rng = np.random.RandomState(seed)
    h, w = ground_truth.shape
    annot = new_annot(h, w)

    fg_points = num_points // 2
    bg_points = num_points - fg_points

    fg_rows, fg_cols = np.where(ground_truth == 1)
    bg_rows, bg_cols = np.where(ground_truth == 0)

    if len(fg_rows) > 0 and fg_points > 0:
        idx = rng.choice(len(fg_rows), size=min(fg_points, len(fg_rows)), replace=False)
        for i in idx:
            paint(annot, (fg_rows[i], fg_cols[i]), brush_radius, channel=0)

    if len(bg_rows) > 0 and bg_points > 0:
        idx = rng.choice(len(bg_rows), size=min(bg_points, len(bg_rows)), replace=False)
        for i in idx:
            paint(annot, (bg_rows[i], bg_cols[i]), brush_radius, channel=1)

    return annot


def corrective_annotation(ground_truth, prediction, existing_annot,
                          num_points=10, brush_radius=5, seed=None):
    """Add corrective annotations targeting model errors.

    Finds false positive and false negative regions, samples points from
    them, and paints corrections onto a copy of the existing annotation.

    Args:
        ground_truth: 2D binary array (1=FG, 0=BG)
        prediction: 2D binary array from model segmentation
        existing_annot: (H, W, 4) uint8 annotation to build upon
        num_points: total correction points (split between FP and FN)
        brush_radius: radius of painted circles
        seed: random seed for reproducibility

    Returns:
        (H, W, 4) uint8 updated annotation array
    """
    rng = np.random.RandomState(seed)
    annot = existing_annot.copy()

    # false negatives: gt=1 but pred=0 → need FG annotation
    fn_rows, fn_cols = np.where((ground_truth == 1) & (prediction == 0))
    # false positives: gt=0 but pred=1 → need BG annotation
    fp_rows, fp_cols = np.where((ground_truth == 0) & (prediction == 1))

    fn_points = num_points // 2
    fp_points = num_points - fn_points

    # if one error type is empty, give all points to the other
    if len(fn_rows) == 0:
        fp_points = num_points
        fn_points = 0
    elif len(fp_rows) == 0:
        fn_points = num_points
        fp_points = 0

    if len(fn_rows) > 0 and fn_points > 0:
        idx = rng.choice(len(fn_rows), size=min(fn_points, len(fn_rows)), replace=False)
        for i in idx:
            paint(annot, (fn_rows[i], fn_cols[i]), brush_radius, channel=0)

    if len(fp_rows) > 0 and fp_points > 0:
        idx = rng.choice(len(fp_rows), size=min(fp_points, len(fp_rows)), replace=False)
        for i in idx:
            paint(annot, (fp_rows[i], fp_cols[i]), brush_radius, channel=1)

    return annot
