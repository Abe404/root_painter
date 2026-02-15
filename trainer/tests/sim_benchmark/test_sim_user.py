"""
Unit tests for simulated annotator using synthetic ground truth.
"""
import numpy as np
from scipy.ndimage import binary_erosion
from sim_benchmark.sim_user import initial_annotation, corrective_annotation


def circle_gt(size=200, radius=40):
    """Synthetic ground truth: filled circle centered in a square image."""
    gt = np.zeros((size, size), dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    c = size // 2
    gt[(x - c) ** 2 + (y - c) ** 2 <= radius ** 2] = 1
    return gt


def test_initial_annotation_covers_both_classes():
    gt = circle_gt()
    annot, traj = initial_annotation(gt, coverage=0.05)
    assert np.sum(annot[:, :, 0]) > 0, "should have FG annotations"
    assert np.sum(annot[:, :, 1]) > 0, "should have BG annotations"
    assert len(traj) > 0, "should have trajectory"


def test_initial_annotation_no_cross_class():
    """FG annotations must stay on GT FG, BG annotations must stay on GT BG."""
    gt = circle_gt()
    annot, traj = initial_annotation(gt, coverage=0.05, seed=42)
    fg_pixels = annot[:, :, 0] > 0
    assert np.all(gt[fg_pixels] == 1), "FG annotation landed on BG"
    bg_pixels = annot[:, :, 1] > 0
    assert np.all(gt[bg_pixels] == 0), "BG annotation landed on FG"


def test_initial_annotation_balance_constraint():
    """BG annotation should not exceed ~10x FG annotation."""
    gt = circle_gt(300, radius=20)  # small FG, large BG
    annot, traj = initial_annotation(gt, coverage=0.05, seed=1)
    fg_count = np.sum(annot[:, :, 0] > 0)
    bg_count = np.sum(annot[:, :, 1] > 0)
    if fg_count > 0:
        ratio = bg_count / fg_count
        assert ratio <= 12, f"BG/FG ratio {ratio:.1f} exceeds ~10x limit"


def test_initial_annotation_adapts_to_region_size():
    """Smaller FG region should produce smaller brush marks."""
    gt_big = circle_gt(200, radius=80)
    gt_small = circle_gt(200, radius=10)
    annot_big, t1 = initial_annotation(gt_big, coverage=0.05, seed=1)
    annot_small, t2 = initial_annotation(gt_small, coverage=0.05, seed=1)
    assert np.sum(annot_big[:, :, 0]) > 0
    assert np.sum(annot_small[:, :, 0]) > 0
    assert np.sum(annot_big[:, :, 0] > 0) > np.sum(annot_small[:, :, 0] > 0)


def test_corrective_no_fg_on_bg():
    """Corrective FG annotations must only land on GT foreground pixels."""
    gt = circle_gt(200, 40)
    pred = np.zeros_like(gt)
    y, x = np.ogrid[:200, :200]
    pred[(x - 130) ** 2 + (y - 100) ** 2 <= 40 ** 2] = 1

    annot, traj = corrective_annotation(gt, pred)
    if annot is None:
        return
    fg_pixels = annot[:, :, 0] > 0
    if np.any(fg_pixels):
        assert np.all(gt[fg_pixels] == 1), "FG annotation landed on BG"
    bg_pixels = annot[:, :, 1] > 0
    if np.any(bg_pixels):
        assert np.all(gt[bg_pixels] == 0), "BG annotation landed on FG"


def test_corrective_covers_error_interiors():
    """Interior of error regions (away from edges) should be annotated."""
    gt = circle_gt(200, 40)
    pred = np.zeros_like(gt)  # all-zero prediction -> all FG is FN

    annot, traj = corrective_annotation(gt, pred)
    assert annot is not None

    # The eroded interior of FN should be covered
    fn_mask = (gt == 1) & (pred == 0)
    fn_area = int(np.sum(fn_mask))
    equiv_radius = np.sqrt(fn_area / np.pi)
    brush_radius = max(1, int(equiv_radius / 10))
    eroded_fn = binary_erosion(fn_mask, iterations=brush_radius)
    if np.any(eroded_fn):
        # interior pixels should be annotated
        assert np.any(annot[:, :, 0][eroded_fn] > 0), \
            "interior FN pixels not annotated"


def test_corrective_returns_none_when_perfect():
    gt = circle_gt()
    pred = gt.copy()
    annot, traj = corrective_annotation(gt, pred)
    assert annot is None, "should return None when prediction matches GT"
    assert len(traj) == 0
