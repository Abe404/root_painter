"""
Simulated annotator using CNC-inspired stroke planning.

Same interface as sim_user.py but uses cam_strokes.generate_paths()
for the corrective phase — Shapely polygon offsets generate contour-parallel
stroke paths, then the simulated mouse follows them.

Initial annotation is delegated to the original sim_user module.
"""
import sys
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.ndimage import label as ndimage_label
from skimage.morphology import disk

from sim_benchmark.brush import paint, new_annot
from sim_benchmark.cam_strokes import generate_paths
from sim_benchmark.sim_user import (
    initial_annotation,
    mouse_travel,
    FG_STROKE_DURATION, BG_STROKE_DURATION, STROKE_DURATION_SPREAD,
    INTER_STROKE_GAP, JITTER_RATE,
)


def _follow_path(annot, path, brush_radius, channel, rng, trajectory,
                 stroke_duration):
    """Follow a multi-waypoint path with the brush down.

    Unlike mouse_stroke (straight line start→end), this follows the full
    curved path through all waypoints — essential for contour-parallel
    strokes that trace the error boundary.

    No GT clipping — the stroke planning must keep paths inside the
    correct class. Any spill is a bug in path generation.
    """
    h, w = annot.shape[:2]

    duration = stroke_duration * rng.lognormal(0, STROKE_DURATION_SPREAD)
    dt_per_point = duration / max(1, len(path))

    for pt in path:
        r = int(round(pt[0]))
        c = int(round(pt[1]))
        r = max(0, min(h - 1, r))
        c = max(0, min(w - 1, c))

        paint(annot, (r, c), brush_radius, channel)

        trajectory.append({
            'r': r, 'c': c, 'painting': True,
            'channel': channel, 'brush_radius': brush_radius,
            'dt': dt_per_point,
        })

    last = path[-1]
    return (int(round(last[0])), int(round(last[1])))


def corrective_annotation(ground_truth, prediction):
    """Create corrective annotations using CAM-planned stroke paths.

    Same interface as sim_user.corrective_annotation().
    Returns (annot, trajectory) or (None, []) if no clear errors.
    """
    fn_mask = (ground_truth == 1) & (prediction == 0)
    fp_mask = (ground_truth == 0) & (prediction == 1)

    if not np.any(fn_mask) and not np.any(fp_mask):
        return None, []

    h, w = ground_truth.shape
    annot = new_annot(h, w)
    trajectory = []
    rng = np.random.RandomState(0)
    mouse_pos = (h // 2, w // 2)

    # Merge nearby error blobs
    merge_radius = max(5, min(h, w) // 10)
    errors = []
    for error_mask, gt_class_mask, channel in [
        (fn_mask, ground_truth == 1, 0),
        (fp_mask, ground_truth == 0, 1),
    ]:
        if not np.any(error_mask):
            continue
        merged = binary_dilation(error_mask, structure=disk(merge_radius))
        labeled, num = ndimage_label(merged)
        for i in range(1, num + 1):
            region = error_mask & (labeled == i)
            area = int(np.sum(region))
            if area < 15:
                continue
            region_rows, region_cols = np.where(region)
            centroid = (float(np.mean(region_rows)),
                        float(np.mean(region_cols)))
            errors.append({
                'region': region,
                'area': area,
                'centroid': centroid,
                'gt_class_mask': gt_class_mask,
                'channel': channel,
            })

    if not errors:
        return None, []

    dt_cache = {}

    # Visit errors in nearest-neighbor order
    remaining = list(range(len(errors)))
    while remaining:
        best_idx = None
        best_dist = float('inf')
        for idx in remaining:
            cr, cc = errors[idx]['centroid']
            d = (cr - mouse_pos[0])**2 + (cc - mouse_pos[1])**2
            if d < best_dist:
                best_dist = d
                best_idx = idx
        remaining.remove(best_idx)

        e = errors[best_idx]
        region = e['region']
        gt_class_mask = e['gt_class_mask']
        channel = e['channel']

        # Choose brush radius from GT class depth near the error
        gt_key = channel
        if gt_key not in dt_cache:
            dt_cache[gt_key] = distance_transform_edt(gt_class_mask)
        dt = dt_cache[gt_key]

        error_nbhd = binary_dilation(region, structure=disk(20))
        nearby_dt = dt[error_nbhd & (dt > 0)]
        if len(nearby_dt) > 0:
            brush_radius = max(3, int(np.max(nearby_dt)) - 1)
        else:
            brush_radius = 3

        # Generate CAM stroke paths
        path_groups = generate_paths(region, gt_class_mask, brush_radius)
        if not path_groups:
            continue

        duration_base = FG_STROKE_DURATION if channel == 0 else BG_STROKE_DURATION

        for paths, br in path_groups:
            for path in paths:
                if len(path) < 2:
                    continue

                # Travel to stroke start
                start = (int(round(path[0][0])), int(round(path[0][1])))
                mouse_travel(trajectory, mouse_pos, start, rng)
                mouse_pos = start

                # Follow the full CAM path with the brush
                mouse_pos = _follow_path(
                    annot, path, br, channel, rng,
                    trajectory, duration_base)


    # Check if any annotation was placed
    if not np.any(annot[:, :, 0] > 0) and not np.any(annot[:, :, 1] > 0):
        return None, []

    return annot, trajectory
