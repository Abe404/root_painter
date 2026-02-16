"""
Simulated annotator for corrective annotation benchmarking.

Creates initial and corrective annotations following the RootPainter
protocol using a simulated mouse.

Duration-based time model. Stroke durations are stable across users
even though pixel speeds vary with image resolution and zoom level.

Every event is recorded with a simulated time cost (dt in seconds).
"""
import sys
from collections import deque
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
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


def _order_waypoints(error_comp, paintable, spacing=15):
    """Order pixels of a thin error component and sample waypoints.

    Walks along the component from one endpoint to the other (BFS on
    8-connected pixels), then samples waypoints at regular spacing and
    snaps them to the paintable region.  This produces curved stroke
    paths that follow boundary error contours.

    Returns list of (row, col) waypoints (at least 2).
    """
    rows, cols = np.where(error_comp)
    if len(rows) < 2:
        pr, pc = np.where(paintable)
        if len(pr) == 0:
            return [(int(rows[0]), int(cols[0]))] * 2
        return [(int(pr[0]), int(pc[0])), (int(pr[-1]), int(pc[-1]))]

    # Build 8-connected adjacency
    pixels = set(zip(rows.tolist(), cols.tolist()))
    adj = {p: [] for p in pixels}
    for r, c in pixels:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) != (0, 0) and (r + dr, c + dc) in pixels:
                    adj[(r, c)].append((r + dr, c + dc))

    # Find an endpoint (fewest neighbors) to start from
    start = min(adj, key=lambda p: len(adj[p]))

    # BFS to order pixels along the component
    visited = {start}
    ordered = [start]
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for n in adj[p]:
            if n not in visited:
                visited.add(n)
                ordered.append(n)
                queue.append(n)

    # Sample waypoints at regular spacing
    waypoints = [ordered[0]]
    for p in ordered[1:]:
        last = waypoints[-1]
        if (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2 >= spacing ** 2:
            waypoints.append(p)
    if waypoints[-1] != ordered[-1]:
        waypoints.append(ordered[-1])

    # Snap each waypoint to nearest paintable pixel
    pr, pc = np.where(paintable)
    if len(pr) == 0:
        return waypoints

    def _snap(point):
        d = (pr - point[0]) ** 2 + (pc - point[1]) ** 2
        i = int(np.argmin(d))
        return (int(pr[i]), int(pc[i]))

    return [_snap(wp) for wp in waypoints]


def _annotate_error_region(annot, error_region, gt_class_mask, channel,
                           mouse_pos, rng, trajectory, h, w):
    """Annotate a single error region with raster zigzag fill.

    Uses parallel scan lines to systematically cover the error, like a
    human coloring in a shape.  Falls back to per-component boundary
    strokes for thin errors near the GT class edge.  Stops when only
    boundary-ambiguous pixels remain (within 4px of opposite class).

    Returns updated mouse_pos.
    """
    from scipy.ndimage import label as ndimage_label

    error_area = int(np.sum(error_region))
    if error_area == 0:
        return mouse_pos

    base_duration = FG_STROKE_DURATION if channel == 0 else BG_STROKE_DURATION
    remaining_error = error_region.copy()
    near_opposite = binary_dilation(~gt_class_mask, structure=disk(2))
    dt = distance_transform_edt(gt_class_mask)

    def _done():
        if not np.any(remaining_error):
            return True
        return not np.any(remaining_error & ~near_opposite)

    def _erode_safe(mask, radius):
        """Erode mask by radius, ignoring image borders.

        The brush can extend outside the image (paint() clips to
        bounds), so we only erode from class boundaries, not image
        edges.  Pad with edge values before eroding, then crop back.
        """
        if radius <= 0:
            return mask.copy()
        padded = np.pad(mask, radius, mode='edge')
        eroded = binary_erosion(padded, structure=disk(radius))
        return eroded[radius:-radius, radius:-radius]

    def _find_brush(err_mask, max_r):
        """Binary search for largest brush <= max_r that fits.

        Erodes gt_class_mask by brush_radius so the full brush disk
        stays inside the correct class.  Safe because paint() uses
        the same disk shape as the erosion SE.
        """
        lo2, hi2 = 1, max_r
        best_r, best_pr = 0, None
        while lo2 <= hi2:
            mid = (lo2 + hi2) // 2
            safe = _erode_safe(gt_class_mask, mid)
            pr = safe & binary_dilation(err_mask, structure=disk(mid))
            if np.any(pr):
                best_r, best_pr = mid, pr
                lo2 = mid + 1
            else:
                hi2 = mid - 1
        return best_r, best_pr

    def _do_stroke(start, end, pr, br, careful=False):
        """Execute a stroke, return dabs_placed.

        careful=True for edge strokes: slower (less jitter), full brush
        radius (need every pixel of reach to cover boundary errors).
        """
        nonlocal mouse_pos
        duration = base_duration * rng.lognormal(0, STROKE_DURATION_SPREAD)
        if careful:
            duration *= 2  # slow down near edges for precision
        dist_est = max(1.0, np.sqrt((end[0]-start[0])**2 +
                                     (end[1]-start[1])**2))
        sigma = max(1.0, dist_est / duration) * JITTER_RATE
        jr, jc = rng.normal(0, sigma), rng.normal(0, sigma)
        jstart = (int(start[0] + jr), int(start[1] + jc))
        if careful:
            rad = br  # full radius to reach edges
        else:
            noise = abs(rng.normal(0, max(1, sigma * 0.5)))
            rad = max(1, br - int(noise))
        mouse_travel(trajectory, mouse_pos, jstart, rng)
        dabs = mouse_stroke(annot, jstart, end, pr, rad, channel,
                             rng, trajectory, duration)
        # Check: did the brush spill into the wrong class?
        spill_mask = (~gt_class_mask) & (annot[:, :, channel] > 0)
        if np.any(spill_mask):
            # Erase the spill — the user notices and fixes it.
            n_spill = int(np.sum(spill_mask))
            annot[spill_mask, channel] = 0
            print(f"  ERASED {n_spill}px of "
                  f"{'FG' if channel == 0 else 'BG'} annotation on "
                  f"{'BG' if channel == 0 else 'FG'} "
                  f"(br={rad}, start={start})", file=sys.stderr)
        mouse_pos = end if not trajectory or not trajectory[-1]['painting'] \
            else (trajectory[-1]['r'], trajectory[-1]['c'])
        return dabs

    # --- Stroke loop ---
    # Each iteration: find the best brush for the remaining error, then
    # fill with raster zigzag scan lines.  When the error becomes a thin
    # boundary strip, fall back to per-component strokes with a small brush.
    total_strokes = 0
    force_boundary = False
    for stroke_i in range(20):
        if _done() or total_strokes >= 20:
            break
        actionable = remaining_error & ~near_opposite
        if not np.any(actionable):
            break

        # Brush sized to the error for raster fill (chunky errors).
        act_rows, act_cols = np.where(actionable)
        act_area = int(np.sum(actionable))
        act_equiv = np.sqrt(act_area / np.pi)
        act_safe = int(np.max(dt[act_rows, act_cols]))
        act_desired = min(act_safe, max(3, int(act_equiv + 0.5)))

        if force_boundary:
            br, pr = 0, None
            force_boundary = False
        else:
            br, pr = _find_brush(actionable, act_desired)

        # Thin errors get curved strokes with a bigger brush — sized
        # to GT class depth nearby, not error size.  One big sweep
        # along the contour with brush edge covering the thin error.
        row_ext = int(np.max(act_rows)) - int(np.min(act_rows))
        col_ext = int(np.max(act_cols)) - int(np.min(act_cols))
        min_ext = min(row_ext, col_ext)
        use_curved = (br == 0 or pr is None)
        if not use_curved and min_ext < br * 2:
            # Thin error — try bigger brush from GT depth nearby
            search_r = min(20, max(h, w) // 10)
            nearby_mask = binary_dilation(actionable, structure=disk(search_r))
            nearby_dt = dt[nearby_mask & (dt > 0)]
            if len(nearby_dt) > 0:
                curved_desired = max(br, int(np.max(nearby_dt)))
                cbr, cpr = _find_brush(actionable, curved_desired)
                if cbr > br:
                    br, pr = cbr, cpr
            use_curved = True
        if use_curved:
            if br == 0 or pr is None:
                # No brush fits — use br=2 with 1px erosion buffer
                br = 2
                safe_boundary = _erode_safe(gt_class_mask, 1)
                curve_pr = safe_boundary & binary_dilation(
                    actionable, structure=disk(br + 1))
                if not np.any(curve_pr):
                    br = 1
                    curve_pr = gt_class_mask & binary_dilation(
                        actionable, structure=disk(2))
                    if not np.any(curve_pr):
                        break
            else:
                # Brush fits — use its paintable region
                curve_pr = pr

            # Per-component curved strokes following the error contour.
            # Waypoint spacing scales with brush — big brush covers more
            # per dab, so waypoints can be further apart.
            wp_spacing = max(15, br * 2)
            labeled, num = ndimage_label(actionable)
            comp_sizes = [(int(np.sum(labeled == ci)), ci)
                          for ci in range(1, num + 1)]
            comp_sizes.sort(reverse=True)
            for _, ci in comp_sizes:
                if _done() or total_strokes >= 20:
                    break
                comp = (labeled == ci)
                comp_pr = curve_pr & binary_dilation(
                    comp, structure=disk(br + 1))
                if not np.any(comp_pr):
                    continue
                waypoints = _order_waypoints(comp, comp_pr, spacing=wp_spacing)
                for wi in range(len(waypoints) - 1):
                    if total_strokes >= 20:
                        break
                    _do_stroke(waypoints[wi], waypoints[wi + 1],
                               comp_pr, br, careful=True)
                    total_strokes += 1
            remaining_error = remaining_error & ~(annot[:, :, channel] > 0)
            continue

        # Normal mode: raster zigzag fill.  Scan lines perpendicular
        # to centroid→farthest, clipped to UNCOVERED area each line.
        # Zigzag alternates sweep direction like a human coloring in.
        centroid_r = float(np.mean(act_rows))
        centroid_c = float(np.mean(act_cols))
        dists_from_centroid = ((act_rows - centroid_r) ** 2
                               + (act_cols - centroid_c) ** 2)
        far_idx = int(np.argmax(dists_from_centroid))
        dr = act_rows[far_idx] - centroid_r
        dc = act_cols[far_idx] - centroid_c
        length = max(1.0, np.sqrt(dr ** 2 + dc ** 2))
        sweep_dir = (float(dr / length), float(dc / length))
        perp_dir = (-sweep_dir[1], sweep_dir[0])

        # Scan line positions from PR extent, spaced at ~85% brush
        # diameter for slight overlap (standard raster fill)
        pr_rows, pr_cols = np.where(pr)
        pr_perp = pr_rows * perp_dir[0] + pr_cols * perp_dir[1]
        perp_min = float(np.min(pr_perp))
        perp_max = float(np.max(pr_perp))
        scan_step = max(1, int(br * 2 * 0.85))
        scan_positions = np.arange(perp_min, perp_max + scan_step,
                                   scan_step)

        scanned_any = False
        zigzag = 1  # alternating sweep direction
        for scan_pos in scan_positions:
            if _done() or total_strokes >= 20:
                break
            # Recompute uncovered strokeable for each scan line
            cur_actionable = remaining_error & ~near_opposite
            if not np.any(cur_actionable):
                break
            uncovered = pr & binary_dilation(
                cur_actionable, structure=disk(br))
            if not np.any(uncovered):
                break
            sr, sc = np.where(uncovered)
            perp_proj = sr * perp_dir[0] + sc * perp_dir[1]
            near = np.abs(perp_proj - scan_pos) <= br
            if not np.any(near):
                continue
            s_sr, s_sc = sr[near], sc[near]
            s_sweep = s_sr * sweep_dir[0] + s_sc * sweep_dir[1]
            if zigzag > 0:
                si, ei = np.argmin(s_sweep), np.argmax(s_sweep)
            else:
                si, ei = np.argmax(s_sweep), np.argmin(s_sweep)
            start = (int(s_sr[si]), int(s_sc[si]))
            end = (int(s_sr[ei]), int(s_sc[ei]))
            _do_stroke(start, end, pr, br)
            total_strokes += 1
            remaining_error = remaining_error & ~(annot[:, :, channel] > 0)
            scanned_any = True
            zigzag *= -1

        if not scanned_any:
            force_boundary = True

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
        traj_before = len(trajectory)
        mouse_pos = _annotate_error_region(
            annot, e['region'], e['gt_class_mask'], e['channel'],
            mouse_pos, rng, trajectory, h, w)
        # Log per-region annotation efficiency (stroke level)
        new_events = trajectory[traj_before:]
        strokes, cur_painted, cur_total = [], 0, 0
        for ev in new_events:
            if ev.get('painting'):
                cur_total += 1
                if ev.get('painted'):
                    cur_painted += 1
            elif cur_total > 0:
                strokes.append((cur_painted, cur_total))
                cur_painted, cur_total = 0, 0
        if cur_total > 0:
            strokes.append((cur_painted, cur_total))
        empty_strokes = sum(1 for p, t in strokes if p == 0)
        if empty_strokes > 0:
            ch_name = 'FG' if e['channel'] == 0 else 'BG'
            print(f"  WARNING: {ch_name} region (area={e['area']}): "
                  f"{empty_strokes}/{len(strokes)} strokes painted nothing",
                  file=sys.stderr)

    if not np.any(annot[:, :, 0] > 0) and not np.any(annot[:, :, 1] > 0):
        return None, []

    return annot, trajectory
