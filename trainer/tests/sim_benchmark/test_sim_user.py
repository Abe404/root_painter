"""
Unit tests for simulated annotator using synthetic ground truth.
"""
import numpy as np
from sim_benchmark.sim_user import initial_annotation, corrective_annotation


def _circle_gt(size=200, radius=40):
    """Synthetic ground truth: filled circle centered in a square image."""
    gt = np.zeros((size, size), dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    c = size // 2
    gt[(x - c) ** 2 + (y - c) ** 2 <= radius ** 2] = 1
    return gt


def test_initial_annotation_covers_both_classes():
    gt = _circle_gt()
    annot = initial_annotation(gt, num_points=20, brush_radius=3)
    assert np.sum(annot[:, :, 0]) > 0, "should have FG annotations"
    assert np.sum(annot[:, :, 1]) > 0, "should have BG annotations"


def test_initial_annotation_correctness():
    gt = _circle_gt()
    # brush_radius=0 (single pixel) so the disk doesn't bleed across boundaries
    annot = initial_annotation(gt, num_points=30, brush_radius=0, seed=42)
    # every FG-annotated pixel should be on a FG ground truth pixel
    fg_pixels = annot[:, :, 0] > 0
    assert np.all(gt[fg_pixels] == 1), "FG annotation landed on BG"
    # every BG-annotated pixel should be on a BG ground truth pixel
    bg_pixels = annot[:, :, 1] > 0
    assert np.all(gt[bg_pixels] == 0), "BG annotation landed on FG"


def test_corrective_annotation_targets_errors():
    gt = _circle_gt(200, 40)
    # deliberately wrong prediction: shifted circle
    pred = np.zeros_like(gt)
    y, x = np.ogrid[:200, :200]
    pred[(x - 130) ** 2 + (y - 100) ** 2 <= 40 ** 2] = 1

    annot = initial_annotation(gt, num_points=10, brush_radius=0, seed=0)
    corrected = corrective_annotation(gt, pred, annot,
                                      num_points=20, brush_radius=0, seed=0)
    # should have more annotation than before
    assert np.sum(corrected) > np.sum(annot)
    # new FG annotations (corrections) should target false-negative regions
    new_fg = (corrected[:, :, 0] > 0) & (annot[:, :, 0] == 0)
    if np.any(new_fg):
        # these pixels should be in FN region (gt=1, pred=0)
        assert np.all(gt[new_fg] == 1)
    # new BG annotations should target false-positive regions
    new_bg = (corrected[:, :, 1] > 0) & (annot[:, :, 1] == 0)
    if np.any(new_bg):
        assert np.all(gt[new_bg] == 0)


def test_corrective_annotation_preserves_existing():
    gt = _circle_gt()
    pred = np.zeros_like(gt)  # all-zero prediction
    annot = initial_annotation(gt, num_points=10, brush_radius=3, seed=1)
    original_fg = annot[:, :, 0].copy()
    original_bg = annot[:, :, 1].copy()

    corrected = corrective_annotation(gt, pred, annot,
                                      num_points=10, brush_radius=3, seed=2)
    # original annotations should still be present
    assert np.all(corrected[:, :, 0][original_fg > 0] > 0)
    assert np.all(corrected[:, :, 1][original_bg > 0] > 0)
