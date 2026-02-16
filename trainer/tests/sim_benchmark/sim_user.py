"""
Simulated annotator for corrective annotation benchmarking.

Creates initial and corrective annotations following the RootPainter
protocol using a simulated mouse.

Duration-based time model. Stroke durations are stable across users
even though pixel speeds vary with image resolution and zoom level.

Every event is recorded with a simulated time cost (dt in seconds).
"""
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from sim_benchmark.brush import disk, paint, new_annot

# Stroke durations (seconds).
# FG: slow, careful tracing.  BG: quick confident sweeps.
FG_STROKE_DURATION = 1.4
BG_STROKE_DURATION = 0.75
STROKE_DURATION_SPREAD = 0.3  # log-normal spread for natural variation

# Inter-stroke gap: thinking + repositioning between strokes.
INTER_STROKE_GAP = 1.5

# Jitter: σ in pixels = JITTER_RATE * effective_speed.
# Effective speed is derived from stroke distance / duration, so jitter
# scales naturally — fast sweeps wobble more, slow strokes are precise.
JITTER_RATE = 0.04


def fit_brush(mask, max_radius, min_interior=50):
    """Largest brush radius <= max_radius that fits inside mask.

    Returns (brush_radius, eroded_mask) or (0, None) if nothing fits.
    """
    lo, hi = 1, max_radius
    best_r, best_eroded = 0, None
    while lo <= hi:
        mid = (lo + hi) // 2
        eroded = binary_erosion(mask, structure=disk(mid + 1))
        if np.sum(eroded) >= min_interior:
            best_r, best_eroded = mid, eroded
            lo = mid + 1
        else:
            hi = mid - 1
    return best_r, best_eroded


def principal_axis(mask):
    """Unit vector (dr, dc) along longest dimension of mask via covariance eigendecomposition."""
    rows, cols = np.where(mask)
    if len(rows) < 2:
        return (1.0, 0.0)
    cov = np.cov(rows.astype(float), cols.astype(float))
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]
    length = np.sqrt(axis[0]**2 + axis[1]**2)
    if length < 1e-8:
        return (1.0, 0.0)
    return (float(axis[0] / length), float(axis[1] / length))


def axis_endpoints(mask, direction):
    """Project mask pixels onto direction, return (start, end) at extremes."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None, None
    projections = rows * direction[0] + cols * direction[1]
    return (int(rows[np.argmin(projections)]), int(cols[np.argmin(projections)])), \
           (int(rows[np.argmax(projections)]), int(cols[np.argmax(projections)]))


def mouse_travel(trajectory, from_pos, to_pos, rng):
    """Record mouse-up repositioning from one position to another.

    Uses the inter-stroke gap duration (thinking + travel), distributed
    across interpolated cursor positions for smooth video playback.
    """
    r0, c0 = from_pos
    r1, c1 = to_pos
    dist = np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
    num_steps = max(1, int(dist / 8.0))
    gap = INTER_STROKE_GAP * rng.lognormal(0, 0.3)
    dt = gap / num_steps
    for i in range(1, num_steps + 1):
        t = i / num_steps
        trajectory.append({
            'r': int(round(r0 + (r1 - r0) * t)),
            'c': int(round(c0 + (c1 - c0) * t)),
            'painting': False, 'channel': -1, 'brush_radius': 0, 'dt': dt,
        })


def mouse_stroke(annot, start, end, region, brush_radius, channel, rng,
                  trajectory, stroke_duration):
    """Paint dabs along a directed line from start to end.

    Total time for the stroke is stroke_duration (seconds). Jitter is
    derived from effective speed (distance / duration) so fast sweeps
    wobble more than slow careful strokes.

    Only paints where region is True. Returns number of dabs placed.
    """
    step_size = max(1, brush_radius // 2)

    dr = end[0] - start[0]
    dc = end[1] - start[1]
    dist = max(1.0, np.sqrt(dr ** 2 + dc ** 2))
    num_steps = max(1, int(dist / step_size))
    dt = stroke_duration / num_steps

    # Jitter from effective speed
    effective_speed = dist / stroke_duration
    sigma = effective_speed * JITTER_RATE

    # Perpendicular direction for natural wavering
    perp_r, perp_c = -dc / dist, dr / dist

    h, w = region.shape
    dabs = 0
    for i in range(num_steps):
        t = (i + 1) / num_steps
        waver = rng.normal(0, sigma * 0.08)
        r = start[0] + dr * t + perp_r * waver
        c = start[1] + dc * t + perp_c * waver
        ir, ic = int(round(r)), int(round(c))
        # Paint only where region allows, but always record cursor position
        # (mouse is down for the entire stroke — continuous drag)
        painted = (0 <= ir < h and 0 <= ic < w and region[ir, ic])
        if painted:
            paint(annot, (ir, ic), brush_radius, channel)
            dabs += 1
        trajectory.append({
            'r': max(0, min(ir, h - 1)), 'c': max(0, min(ic, w - 1)),
            'painting': True, 'painted': painted,
            'channel': channel, 'brush_radius': brush_radius, 'dt': dt,
        })
    return dabs


def initial_annotation(ground_truth, seed=None, **_ignored):
    """Create initial annotations using simulated mouse strokes on FG/BG.

    Returns (annot, trajectory).
    """
    rng = np.random.RandomState(seed)
    h, w = ground_truth.shape
    annot = new_annot(h, w)
    trajectory = []

    fg_mask = ground_truth == 1
    bg_mask = ground_truth == 0
    fg_area = int(np.sum(fg_mask))
    bg_area = int(np.sum(bg_mask))

    if fg_area == 0 or bg_area == 0:
        return annot, trajectory

    mouse_pos = (h // 2, w // 2)

    def paint_class(mask, area, channel, max_pixels=None):
        """Fill the safe zone with straight strokes.

        Brush = largest comfortable fit. Strokes are horizontal scan
        lines through the safe zone — simple, long, and predictable.
        Handles non-convex regions naturally since mouse_stroke only
        paints where the safe zone allows.
        """
        nonlocal mouse_pos

        equiv_radius = np.sqrt(area / np.pi)

        # Largest brush that fits comfortably in the region.
        # Require enough interior survives erosion so the brush works
        # across the region, not just in corners of non-convex shapes.
        target_brush = max(1, int(equiv_radius / 2))
        min_interior = max(50, int(area * 0.2))
        brush_radius, eroded = fit_brush(mask, target_brush, min_interior)
        if brush_radius == 0 or eroded is None:
            return 0

        # FG channel=0, BG channel=1
        base_duration = FG_STROKE_DURATION if channel == 0 else BG_STROKE_DURATION

        er_rows, er_cols = np.where(eroded)
        row_min, row_max = int(np.min(er_rows)), int(np.max(er_rows))
        row_extent = row_max - row_min

        stroke_width = brush_radius * 2
        num_strokes = max(1, int(np.ceil(row_extent / stroke_width)))

        # Evenly spaced horizontal scan lines through the safe zone
        scan_ys = np.linspace(row_min, row_max, num_strokes + 2)[1:-1]

        for scan_y in scan_ys:
            if max_pixels is not None:
                painted_so_far = int(np.sum(annot[:, :, channel] > 0))
                if painted_so_far >= max_pixels:
                    break

            iy = int(round(scan_y))
            iy = max(0, min(iy, h - 1))
            row = eroded[iy, :]
            if not np.any(row):
                continue

            cols = np.where(row)[0]
            start = (iy, int(cols[0]))
            end = (iy, int(cols[-1]))

            # Skip very short strokes
            if abs(end[1] - start[1]) < brush_radius:
                continue

            duration = base_duration * rng.lognormal(0, STROKE_DURATION_SPREAD)
            mouse_travel(trajectory, mouse_pos, start, rng)
            mouse_stroke(annot, start, end, eroded, brush_radius, channel,
                          rng, trajectory, duration)

            mouse_pos = (trajectory[-1]['r'], trajectory[-1]['c'])

        return int(np.sum(annot[:, :, channel] > 0))

    # Minority class first, then majority up to 10x
    if fg_area <= bg_area:
        min_mask, min_area, min_ch = fg_mask, fg_area, 0
        maj_mask, maj_area, maj_ch = bg_mask, bg_area, 1
    else:
        min_mask, min_area, min_ch = bg_mask, bg_area, 1
        maj_mask, maj_area, maj_ch = fg_mask, fg_area, 0

    min_annotated = paint_class(min_mask, min_area, min_ch)

    maj_budget = min(10 * max(1, min_annotated), maj_area)
    paint_class(maj_mask, maj_area, maj_ch, max_pixels=maj_budget)

    return annot, trajectory


def _annotate_error_region(annot, error_region, gt_class_mask, channel,
                           mouse_pos, rng, trajectory, h, w):
    """Annotate a single error region with one big stroke.

    Returns updated mouse_pos.
    """
    error_area = int(np.sum(error_region))
    error_equiv = np.sqrt(error_area / np.pi)

    brush_radius = max(3, int(error_equiv * 0.6))

    # Keep brush center far enough from the wrong class that the disk won't spill
    safe = binary_erosion(gt_class_mask, structure=disk(brush_radius + 1))
    paint_region = safe & binary_dilation(error_region,
                                          structure=disk(brush_radius * 2))
    if not np.any(paint_region):
        for r in [brush_radius // 2, max(1, brush_radius // 4), 1]:
            safe = binary_erosion(gt_class_mask, structure=disk(r + 1))
            paint_region = safe & binary_dilation(error_region,
                                                  structure=disk(r * 2))
            if np.any(paint_region):
                brush_radius = r
                break
        else:
            return mouse_pos

    # One stroke along the error region's principal axis.
    # Pick start/end from error pixels so the stroke stays on one side.
    direction = principal_axis(error_region)
    er_rows, er_cols = np.where(error_region)
    axis_proj = (er_rows * direction[0] + er_cols * direction[1]).astype(float)

    start_i = np.argmin(axis_proj)
    end_i = np.argmax(axis_proj)

    base_duration = FG_STROKE_DURATION if channel == 0 else BG_STROKE_DURATION
    duration = base_duration * rng.lognormal(0, STROKE_DURATION_SPREAD)

    dist_est = np.sqrt((er_rows[end_i] - er_rows[start_i])**2 +
                       (er_cols[end_i] - er_cols[start_i])**2)
    sigma = max(1.0, dist_est / duration) * JITTER_RATE

    jr, jc = rng.normal(0, sigma), rng.normal(0, sigma)
    start = (int(er_rows[start_i] + jr), int(er_cols[start_i] + jc))
    end = (int(er_rows[end_i]), int(er_cols[end_i]))

    noise = abs(rng.normal(0, max(1, sigma * 0.5)))
    rad = max(1, brush_radius - int(noise))

    mouse_travel(trajectory, mouse_pos, start, rng)
    mouse_stroke(annot, start, end, paint_region, rad, channel,
                  rng, trajectory, duration)

    mouse_pos = end if not trajectory or not trajectory[-1]['painting'] else \
        (trajectory[-1]['r'], trajectory[-1]['c'])

    return mouse_pos


def corrective_annotation(ground_truth, prediction):
    """Create corrective annotations in the neighborhood of errors.

    Merges nearby error blobs, then sweeps each with a big brush.

    Returns (annot, trajectory) or (None, []) if no clear errors.
    """
    from scipy.ndimage import label as ndimage_label

    fn_mask = (ground_truth == 1) & (prediction == 0)
    fp_mask = (ground_truth == 0) & (prediction == 1)

    if not np.any(fn_mask) and not np.any(fp_mask):
        return None, []

    h, w = ground_truth.shape
    annot = new_annot(h, w)
    trajectory = []
    rng = np.random.RandomState(0)
    mouse_pos = (h // 2, w // 2)

    # Dilate error masks to merge nearby blobs, then label
    merge_radius = max(5, min(h, w) // 10)
    errors = []
    for error_mask, gt_class_mask, channel in [
        (fn_mask, ground_truth == 1, 0),  # FN -> paint FG
        (fp_mask, ground_truth == 0, 1),  # FP -> paint BG
    ]:
        if not np.any(error_mask):
            continue
        merged = binary_dilation(error_mask, structure=disk(merge_radius))
        labeled, num = ndimage_label(merged)
        for i in range(1, num + 1):
            # Use original error pixels within this merged blob
            region = error_mask & (labeled == i)
            area = int(np.sum(region))
            if area < 50:
                continue
            # Skip scattered noise: if error pixels are sparse relative
            # to their bounding box, it's not a coherent error region
            region_rows, region_cols = np.where(region)
            bbox_h = region_rows.max() - region_rows.min() + 1
            bbox_w = region_cols.max() - region_cols.min() + 1
            density = area / max(1, bbox_h * bbox_w)
            if density < 0.05:
                continue
            centroid = (float(np.mean(region_rows)),
                        float(np.mean(region_cols)))
            errors.append({
                'region': region,
                'area': area,
                'centroid': centroid,
                'gt_class_mask': gt_class_mask,
                'channel': channel,
            })

    # Visit each error in nearest-neighbor order from mouse position
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
        mouse_pos = _annotate_error_region(
            annot, e['region'], e['gt_class_mask'], e['channel'],
            mouse_pos, rng, trajectory, h, w)

    if not np.any(annot[:, :, 0] > 0) and not np.any(annot[:, :, 1] > 0):
        return None, []

    return annot, trajectory
